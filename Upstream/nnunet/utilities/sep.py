import torch 
import torch.nn.functional as F
from torch.distributions import kl_divergence, MultivariateNormal, Normal, Categorical, Distribution
import numpy as np

def get_pos_neg_sets(inp, gt, match_dims=True):
    """Extracts the values from `inp` that correspond to the positive labeled values in `gt`.

    Note: This function only works for the scalar-valued feature map case. 
    
    Inputs
    ------
    `inp`: torch.Tensor([B, 1, H, W, D])
    
    `gt`: torch.Tensor([B, 1, a*H, a*W, a*D])
        The ground truth for positive set of voxels. 
        The last three dimensions (H, W, D) scale, `a`, must either be 1 or a multiple of 2. 
    
    `match_dims`=True
        Whether to match the dimensions of `inp` to that of `gt`
    
    Returns
    -------
    tuple[pos_set, neg_set] 
        `pos_set`: list[torch.tensor([N])] 
            List of containing the values in `inp` associated with the positively labeled voxels in `gt` per batch
        `neg_set`: list[torch.tensor([M])]
            List of containing the values in `inp` associated with the negatively labeled voxels in `gt` per batch
    """
    # NOTE: this only works for the scalar feature map case
    if match_dims:
        while inp.size() != gt.size():
            inp = F.interpolate(inp, scale_factor=(2, 2, 2), mode='trilinear')
    
    assert gt.size() == inp.size()
    b = inp.size(0)

    pos_set, neg_set = [], []
    for i in range(b):
        pos_set.append(inp[i, gt[i, :]>0])
        neg_set.append(inp[i, gt[i, :]==0])

    return pos_set, neg_set

def get_task_set(inp, gt, task, match_dims=True):
    """Extracts the values from `inp` that correspond to the positive labeled values in `gt`.

    Note: This function only works for the scalar-valued feature map case. 
    
    Inputs
    ------
    `inp`: torch.Tensor([B, 1, H, W, D])
    
    `gt`: torch.Tensor([B, 1, a*H, a*W, a*D])
        The ground truth for positive set of voxels. 
        The last three dimensions (H, W, D) scale, `a`, must either be 1 or a multiple of 2. 

    `task`: int
        The values to get indices of from `gt`
    
    `match_dims`=True
        Whether to match the dimensions of `inp` to that of `gt`
    
    Returns
    -------
    list[torch.Tensor] 
        List of containing the values in `inp` associated with the voxels labelled as `task` in `gt`, each
        element of the list being the values in a batch
    """
    # NOTE: this only works for the scalar feature map case
    if match_dims:
        while inp.shape[-3:] != gt.shape[-3:]:
            inp = F.interpolate(inp, scale_factor=(2, 2, 2), mode='trilinear')
    
    assert gt.shape[-3:] == inp.shape[-3:]
    b = inp.size(0)

    extraction = []
    for i in range(b):
        extraction.append(inp[i, gt[i, :]==task])

    return torch.cat(extraction)

def extract_task_set(batched_input, batched_gt, task_id, keep_dims=False):

    while batched_input.shape[-3:] != batched_gt.shape[-3:] and batched_input.size(1) == 1:
        batched_input = F.interpolate(batched_input, scale_factor=(2, 2, 2), mode='trilinear')

    extraction_set = []
    for i in range(batched_input.size(0)):
        extraction_voxels = torch.argwhere(batched_gt[i, 0] == task_id)
        
        extraction = batched_input[i, :, extraction_voxels[:, 0], extraction_voxels[:, 1], extraction_voxels[:, 2]] #, pos_voxels[:, 2]]
        extraction_set.append(extraction)
    extraction_set = torch.concat(extraction_set, -1)    

    if keep_dims: 
        return extraction_set
    
    return extraction_set.mean(dim=0)

def extract_correct_task_set(batched_input, batched_gt, batched_seg, task_id, keep_dims=False):
    while batched_input.shape[-3:] != batched_gt.shape[-3:] and batched_input.size(1) == 1:
        batched_input = F.interpolate(batched_input, scale_factor=(2, 2, 2), mode='trilinear')

    msk = batched_seg == batched_gt
    # print(f"tot num correct: {torch.sum(msk)}")
    extraction_set = []
    for i in range(batched_input.size(0)):
        extraction_voxels = torch.argwhere(batched_gt[i, 0] == task_id)
        
        correct_region = msk[i, :, extraction_voxels[:, 0], extraction_voxels[:, 1], extraction_voxels[:, 2]] #, pos_voxels[:, 2]]
        candidate_extreactions = batched_input[i, :, extraction_voxels[:, 0], extraction_voxels[:, 1], extraction_voxels[:, 2]] #, pos_voxels[:, 2]]
        # print(f"num correct: {torch.sum(correct_region)}")
        extraction = candidate_extreactions[:, correct_region[0, :]]
        extraction_set.append(extraction)
    extraction_set = torch.concat(extraction_set, -1)    

    if keep_dims: 
        return extraction_set
    
    return extraction_set.mean(dim=0)


def kl_divs(dists, target_means, target_covs):

    kls = []
    for i, dist in enumerate(dists):
        target_mean, target_cov = target_means[i], target_covs[i]
        # target_std = target_std.repeat(target_mean.size(0), 1)
        # if target_mean.shape[-1] > 1:
        target_dist = MultivariateNormal(target_mean, target_cov)
        # else:
        #     target_dist = Normal(target_mean, target_std)
        
        kls.append(kl_divergence(dist, target_dist))

    return kls

def component_wise_kl_div(means, covs, means_pred, covs_pred, optimal_component_ns=None):

    total_kl_div = 0
    # for p in range(means_pred.shape[0]):
    for p in range(len(means_pred)):
        if optimal_component_ns is not None:
            optimal_n = optimal_component_ns[p]
        else:
            optimal_n = None

        means_p = means_pred[p]
        for c, mean_c in enumerate(means_p):
            if optimal_n is not None and c > optimal_n:
                break
            # cov_c = covs_pred[p, c]
            cov_c = covs_pred[p][c]

            if not is_positive_definite(cov_c):
                total_kl_div = total_kl_div + 100*torch.sum(torch.square(torch.zeros_like(cov_c) - cov_c))
                continue
            
            pred_dist = MultivariateNormal(mean_c, cov_c)
            # dist = MultivariateNormal(means[p, c], covs[p, c])
            dist = MultivariateNormal(means[p][c], covs[p][c])

            l = kl_divergence(dist, pred_dist)

            total_kl_div = total_kl_div + l
            
    return total_kl_div

def is_positive_definite(mat):
    try:
        _ = torch.linalg.cholesky(mat)
        return True
    except RuntimeError:
        return False


def matrix_logm(A):
    """
    Compute the matrix logarithm of a positive definite matrix A.
    
    Args:
        A (torch.Tensor): A positive definite matrix.
        
    Returns:
        torch.Tensor: The matrix logarithm of A.
    """
    # Perform eigenvalue decomposition
    eigvals, eigvecs = torch.linalg.eigh(A)
    
    # Apply the logarithm to the eigenvalues
    log_eigvals = torch.log(eigvals)
    
    # Reconstruct the matrix logarithm
    logA = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-1, -2)
    
    return logA

def riemannian_distance(Sigma1, Sigma2):
    """
    Compute the Riemannian distance between two covariance matrices Sigma1 and Sigma2.
    
    Args:
        Sigma1 (torch.Tensor): A positive definite matrix (covariance matrix).
        Sigma2 (torch.Tensor): Another positive definite matrix (covariance matrix).
        
    Returns:
        torch.Tensor: The Riemannian distance between the two matrices.
    """
    # Ensure Sigma1 and Sigma2 are symmetric
    Sigma1 = 0.5 * (Sigma1 + Sigma1.transpose(-1, -2))
    Sigma2 = 0.5 * (Sigma2 + Sigma2.transpose(-1, -2))
    
    # Compute the square root of Sigma1
    Sigma1_sqrt = torch.linalg.cholesky(Sigma1)
    
    # Compute the inverse of the square root of Sigma1
    Sigma1_sqrt_inv = torch.linalg.inv(Sigma1_sqrt)
    
    # Compute the matrix product Sigma1_sqrt_inv * Sigma2 * Sigma1_sqrt_inv^T
    middle_term = Sigma1_sqrt_inv @ Sigma2 @ Sigma1_sqrt_inv.transpose(-1, -2)
    
    # Compute the matrix logarithm of the middle term
    # log_middle_term = torch.linalg.logm(middle_term)
    log_middle_term = matrix_logm(middle_term)
    
    # Compute the Frobenius norm of the log of the middle term
    distance = torch.norm(log_middle_term, 'fro')
    
    return distance

def measure_change(means, covs, new_means, new_covs):

    mean_changes = []
    cov_changes = []
    # for p in range(new_means.shape[0]):
    for p in range(len(new_means)):
        means_p = new_means[p]
        for c, mean_c in enumerate(means_p):
            # cov_c = new_covs[p, c]
            cov_c = new_covs[p][c]

            mean_changes.append(torch.norm(means[p][c] - mean_c, 2))
            cov_changes.append(riemannian_distance(covs[p][c].float(), cov_c.float()))

    return mean_changes, cov_changes


def sep(means, min_dist):
    total_loss = 0
    for i, mean_i in enumerate(means):
        for j in range(i + 1, len(means)): 
            mean_j = means[j]
            dist = torch.norm(mean_i - mean_j, 2)
            if (dist - min_dist) + 1e-06 < 0:
                total_loss += min_dist - dist

    return total_loss

def est_dist_sep_loss(est_means, min_dist, wrt_target=False, **kwargs):
    est_sep = sep(est_means, min_dist) 
    if wrt_target:
        return est_sep, est_sep - sep(kwargs['mus'], min_dist)
        
    return est_sep

def pen_domain(cls_means, domain=[-1, 1]):
    total = 0
    for i, means in enumerate(cls_means):
        total = total + penalize_out_of_domain(means, domain=domain)

    return total
        

def penalize_out_of_domain(means, domain=[-1, 1]):

    lower = means < domain[0]
    if torch.sum(lower) > 0:
        lower_diff_mean = torch.mean(torch.abs(domain[0] - means[lower]))
    else :
        lower_diff_mean = 0
    higher = means > domain[1]
    if torch.sum(higher) > 0:
        higher_diff_mean = torch.mean(torch.abs(domain[1] - means[higher]))
    else :
        higher_diff_mean = 0

    return 0.5 * (lower_diff_mean + higher_diff_mean)

def get_non_uniform_dynamic_sep_loss( means_pred, covs_pred, min_dists, csep=2):
    total_loss = 0 # torch.tensor([0])
    for p in range(len(means_pred)):
        means_p = means_pred[p]
        for c, mean_c in enumerate(means_p):
            cov_c = covs_pred[p][c]
            cov_c_eval = torch.max(torch.real(torch.linalg.eigvals(cov_c))) # should always be real, since cov is pos-def
            for q in range(p + 1, len(means_pred)):
                means_q = means_pred[q]
                for k, mean_k in enumerate(means_q):
                    cov_k = covs_pred[q][k]
                    cov_k_eval = torch.max(torch.real(torch.linalg.eigvals(cov_k)))

                    pred_min_dist = torch.sqrt(csep * torch.max(cov_c_eval, cov_k_eval)) * len(means_pred)
                    # m = np.min((pred_min_dist.detach().cpu().numpy(), max(min_dists[p], min_dists[q])))
                    m = np.max((pred_min_dist.detach().cpu().numpy(), max(min_dists[p], min_dists[q]))) # NOTE
                    
                    if m < 1e-06:
                        continue

                    dist = torch.norm(mean_c - mean_k, 2)
                    if (dist - m) + 1e-06 < 0:
                        # total_loss = total_loss + (1 - (dist/m))
                        # total_loss = total_loss + torch.float_power(1 - (dist/m), 1/2)
                        total_loss = total_loss + torch.float_power(m - dist, 1)


    return total_loss
    
if __name__ == "__main__":
    # Test
    b = 2
    h = 4
    w = 6
    d = w
    tp = torch.rand(b, 1, h, w, d)
    gt = torch.where(torch.randn(b, 1, 4*h, 4*w, 4*d) > 0, 1, 0)

    pos_set, neg_set = get_pos_neg_sets(tp, gt)

    print(f"pos set: {pos_set}")
    print(f"correct size: {len(pos_set) == b}")
    print(f"sizes per sample: {[s.size() for s in pos_set]}")