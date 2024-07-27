import torch 
import torch.nn as nn
from torch.distributions import Distribution, Normal, MultivariateNormal, kl_divergence
import torch.nn.functional as F
import wandb 

class DynamicDistMatchingLoss(nn.Module):

    def __init__(self, min_dist, cuda_id = -1) -> None:
        super().__init__()

        self.cuda_id = cuda_id
        if not isinstance(min_dist, torch.Tensor):
            min_dist = torch.tensor(min_dist)
        self.min_dist = min_dist

        self.margin = torch.tensor(0.01)
        self.class_weights = [0.5, 2]

    def gnlll(self, features, means, covs, gt_seg, indices):
        num_nan = features.isnan().count_nonzero().item()
        assert num_nan == 0, f"NaN values [{num_nan}] in predicted distribution."
        
        # Target is (typically) the observered data, and network outputs the mean, var
        # Maximize the probability of sampling the observed data from the dist that the mean, var define
        # We flip this: the network outputs data, and we maximize the probability that the it comes from
        # the observed dist. -- sounds a lot like a generative model
        # gnlll_fn = nn.GaussianNLLLoss(reduction='mean')
        batch_size = features.size(0)
        means_tensor = torch.empty(features.size(), dtype=means.dtype, device=features.device)
        vars_tensor = torch.empty(features.size(), dtype=covs.dtype, device=features.device)
        mts, vts = [], []
        for b in range(batch_size):
            mt = means_tensor[b].permute(1, 2, 3, 0)
            vt = vars_tensor[b].permute(1, 2, 3, 0)
            for i, ind in enumerate(indices):
                msk = (gt_seg[b] == i)[0, :]
                mt[msk] = means[ind]
                vt[msk] = covs[ind]
            mts.append(mt.permute(3, 0, 1, 2))
            vts.append(vt.permute(3, 0, 1, 2))

        mts = torch.stack(mts).to(device=features.device)
        vts = torch.stack(vts).to(device=features.device)

        # typically: network_predict: input, var | target --> network_predict: pred_dist | means*, var*
        total_gnlll = F.gaussian_nll_loss(mts, features, vts)
        return total_gnlll

    def vec_gnlll(self, features_cls_extractions, means, covs, indices):
        """
        features_cls_extractions: list[torch.tensor(N, d)]
        means: torch.tensor(k, d)
        covs: torch.tensor(k, d, d)
        """
        total_loss = 0 

        for i, ind in enumerate(indices):
            feature_extraction = features_cls_extractions[i]
            target_mean = means[ind].repeat(feature_extraction.size(0), 1)
            target_cov = covs[ind].repeat(feature_extraction.size(0), 1)
            # homeoscedastic assumption 
            gnlll = F.gaussian_nll_loss(target_mean, feature_extraction, target_cov, reduction='mean')
            total_loss = total_loss + self.class_weights[i] * gnlll

        
        return total_loss

    
    def get_sep_loss(self, means):
        total_loss = 0 
        for i, mean_i in enumerate(means):
            for j in range(i + 1, len(means)): 
                mean_j = means[j]
                dist = torch.norm(mean_i - mean_j, 2)
                if (dist - self.min_dist) + 1e-06 < 0:
                    total_loss = total_loss + (self.min_dist - dist)
        return total_loss

    def get_separability_loss(self, means):
        """
        beta * \sum_{i, j} \min ( || means[i] - means[j] || - C, alpha)

        where beta is a regulatory term w.r.t the total loss, and alpha is a margin
        """
        # NOTE MUST have shape (n, x) where n is the number of means and x is the dimensionality of each mean

        return torch.sum(torch.min(torch.cdist(means, means, 2) - self.min_dist, self.margin).fill_diagonal_(0))
        

    def get_distribution_loss(self, pred_dists, means, covs, indices, handle_nan=False, return_est_dists=False):
        """
        pred_dist = list[torch.tensor(d, N)], d = dimension, N = number of samples
        """
        if return_est_dists: est_dists = []
        total_kl_div = 0
        # for i, pred_dist_samples in enumerate(pred_dists):
        for i, task_label in enumerate(indices):
            pred_dist_samples = pred_dists[i].to(dtype=torch.float64)
            
            num_nan = pred_dist_samples.isnan().count_nonzero().item()
            assert num_nan == 0, f"NaN values [{num_nan}] in predicted distribution."
            
            if handle_nan:
                pred_dist_mean = torch.mean(pred_dist_samples[~pred_dist_samples.isnan()])
                pred_dist_var = (pred_dist_samples - pred_dist_mean).square()
                pred_dist_var = torch.mean(pred_dist_var[~pred_dist_var.isnan()]) + 0.0001 # numerical stability
                # pred_dist_var = torch.var(pred_dist_samples) + 0.0001 # numerical stability
            else:
                pred_dist_mean, pred_dist_var = torch.mean(pred_dist_samples, 1), torch.cov(pred_dist_samples)

            # Clip variance if too large
            pred_dist_var = torch.clamp(pred_dist_var, max=1000.0) + torch.eye(pred_dist_var.shape[-1], device=pred_dist_var.device) * 1e-04

            
            # NOTE Need to generalize to Multivaraite
            # pred_dist = Normal(pred_dist_mean.unsqueeze(-1), torch.sqrt(pred_dist_var).unsqueeze(-1)) 
            # target_dist = Normal(means[task_label], torch.sqrt(covs[task_label]))
            pred_dist = MultivariateNormal(pred_dist_mean, pred_dist_var) 
            target_dist = MultivariateNormal(means[task_label], covs[task_label])
            kl_div = kl_divergence(pred_dist, target_dist)

            total_kl_div = total_kl_div + self.class_weights[i] * kl_div

            if return_est_dists: est_dists.append(pred_dist)

        if return_est_dists:
            return total_kl_div, est_dists
        
        return total_kl_div

    def forward(self, pred_dists, means, covs, indices, handle_nan=False, return_est_dists=False, with_sep_loss=True):
        total_kl_div = self.get_distribution_loss(pred_dists, means, covs, indices, handle_nan=handle_nan, return_est_dists=return_est_dists)

        if return_est_dists:
            total_kl_div, est_dists = total_kl_div

        # sep_loss = self.get_separability_loss(means[indices, :])
        if with_sep_loss:
            sep_loss = self.get_sep_loss(means[indices, :])
        else:
            sep_loss = 0
        
        # total_loss = total_kl_div - sep_loss
        total_loss = total_kl_div + sep_loss
        
        print(f"total loss: {total_loss} = total kl div ({total_kl_div}) + sep loss ({sep_loss})")
        wandb.log({"kl_div": total_kl_div, "sep_loss": sep_loss})
        
        if return_est_dists: return total_loss, est_dists
        
        return total_loss



if __name__ == "__main__":

    pass 