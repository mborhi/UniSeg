import torch 
from torch.distributions import MultivariateNormal, Categorical

# Define GMM class
class GaussianMixtureModel:
    def __init__(self, categorical: Categorical, component_distributions: MultivariateNormal):
        self.categorical = categorical
        self.component_distributions = component_distributions

    def sample(self, sample_shape=torch.Size()):
        component_indices = self.categorical.sample(sample_shape)
        samples = torch.stack([self.component_distributions[i].sample() for i in component_indices])
        return samples

    def log_prob(self, value):
        log_probs = self.component_log_probs(value)
        return torch.logsumexp(log_probs + self.categorical.logits, dim=-1)

    def component_log_probs(self, value):
        log_probs = torch.stack([dist.log_prob(value) for dist in self.component_distributions], dim=-1)
        return log_probs

    def component_log_probs_chunk(self, value):
        log_probs = torch.stack([self.chunk_log_prob(value, dist) for dist in self.component_distributions], dim=-1)
        return log_probs


    # https://discuss.pytorch.org/t/torch-distributions-multivariatenormal-log-prob-throws-runtimeerror-cuda-error-cublas-status-not-supported-for-large-batch-sizes/177977
    # https://discuss.pytorch.org/t/how-to-use-torch-distributions-multivariate-normal-multivariatenormal-in-multi-gpu-mode/135030/3
    def chunk_log_prob(self, value, dist):
        chunk_size = int(262140 *  (2 / value.size(0)))
        if value.size(1) <= chunk_size:
            return dist.log_prob(value)
        else:
            num_chunks = (value.size(1) + chunk_size - 1) // chunk_size
            log_probs = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, value.size(1))
                chunk = value[:, start_idx:end_idx]#.cpu()  # Move to CPU
                log_probs.append(dist.log_prob(chunk))#.cuda())  # Move back to GPU if needed
            return torch.cat(log_probs, dim=1)

    def score(self, value, chunk=True):
        if chunk:
            log_probs = self.component_log_probs_chunk(value)
        else:
            log_probs = self.component_log_probs(value)
        
        return log_probs 
    
    def classify(self, value, chunk=True):
        log_probs = self.score(value, chunk=chunk)
        compt = torch.argmax(log_probs, -1)
        return compt

    def construct_from_dists(self, dists):
        num_dists = len(dists)
        device = dists[0].covariance_matrix.device
        self.component_distributions = dists
        self.categorical = Categorical(torch.ones(num_dists, device=device) / num_dists)
        