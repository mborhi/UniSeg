from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
import torch.nn.functional as F
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, StackedConvLayers, \
    Upsample, Generic_UNet, StackedConvLayers_multi_channel
from nnunet.network_architecture.Prompt_generic_UNet_DP import UniSeg_model
from nnunet.network_architecture.neural_network import SegmentationNetwork
from copy import deepcopy
from torch.distributions import MultivariateNormal, Categorical
from sklearn.mixture._gaussian_mixture import GaussianMixture
from nnunet.utilities.gmm import GaussianMixtureModel
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss
import copy

from nnunet.utilities.sep import kl_divs
from nnunet.utilities.sep import extract_task_set
from nnunet.utilities.tensor_utilities import sum_tensor

import wandb

class DynamicDistributionModel_Fast_DP(nn.Module):
    def __init__(self, feature_space_dim, tp_dim, num_components, reduction="sum", momentum=0.999, queue_size=5000):
        super(DynamicDistributionModel_Fast_DP, self).__init__()
        self.feature_space_dim = feature_space_dim 
        self.num_components = num_components 
        self.tp_dim = tp_dim
        self.reduction = reduction

        hidden_dim = 1000
        
        self.task_mu_modules = nn.ModuleList([
            nn.Sequential(
                # each task's mean of dim feature space (flattented), scalara vars for each task, task prompt dim, task_id
                nn.Linear((feature_space_dim * num_components) + (1 * num_components) + tp_dim + 1, hidden_dim), 
                nn.ReLU(), 
                nn.Linear(hidden_dim, feature_space_dim),
                nn.Tanh(), 
            )
            for t in range(num_components)
        ])

        self.momentum = momentum

        self.queue_size = queue_size
        self.feature_space_qs = [[] for i in range(num_components)]
        

    def update_queues(self, task_repr_elements, tc_inds):
        update_size = 100
        # create mask of accurate feature extractions (highest log-likelihood w.r.t target Gauss.)
        for i, task_id in enumerate(tc_inds):
            # Enqueue and Dequeue
            if len(self.feature_space_qs[task_id]) + update_size > self.queue_size:
                for i in range(len(self.feature_space_qs[task_id]) + update_size - self.queue_size):
                    self.feature_space_qs[task_id].pop(0)
            
            self.feature_space_qs[task_id].append(task_repr_elements[i].reshape(-1, self.feature_space_dim))


    def clear_queues(self):

        for i in range(self.num_components):
            self.feature_space_qs[i] = []


    def distance_bounds(self, k=None, max_var=None, delta=None, m=None, n=1, n_min=1):
        min_num_samples = 500 # min(number of images per class k in batch) * resolution 
        # max_error = 1e-03
        var_1 = var_2 = 1
        # n = n_min = 1 
        w_min = 0.1 
        C = 1344

        def distance_bound(k=None, max_var=None, delta=None, m=None, n=1, n_min=1):
            if m is None: 
                m = min_num_samples

            if max_var is None:
                max_var = var_1

            if k is None:
                k = self.num_components

            # Assume that the min number of required samples is met
            r = np.max((k, C*np.log(n/n_min)))
            # Distance bound:
            min_dist = 14 * max_var * np.power(r * np.log(4*m / delta), 1/4)

            # Test whether the number of samples satisfies the actual min required samples derived under the assumption mu's 
            mu_max = min_dist / 2
            min_req_samples = (np.power(n, 3) / np.power(w_min, 2)) * (np.log(np.power(np.abs(mu_max), 2) / np.power(max_var, 2)) + np.log(n/delta))
            if m < min_req_samples:
                return np.nan 
            
            return min_dist 

        return distance_bound(delta=delta)

    def forward(self, x, means, vars, tc_inds, with_momentum_update=True):
        # Returns: updated means and vars
        
        # self.means = self.means.detach().clone().to(device=x.device)
        # self.vars = self.vars.detach().clone().to(device=x.device)
        if len(means.shape) != 2:
            means = means[0, :]
            vars = vars[0, :]

        means = means.detach().clone().to(device=x.device)
        updated_means = means.detach().clone().to(device=x.device)
        vars = vars.detach().clone().to(device=x.device)

        for t, task_id in enumerate(tc_inds):
            if not isinstance(task_id, torch.Tensor):
                task_id = torch.tensor([task_id])[:, None].to(device=x.device)

            if means.device != x.device:
                means = means.to(device=x.device)
                vars = vars.to(device=x.device)
            
            input_means = means.reshape(-1)[None, :].repeat(x.size(0), 1).detach().to(device=x.device)
            # input_vars = vars.permute(1, 0).repeat(x.size(0), 1).detach().to(device=x.device)
            input_vars = vars.repeat(x.size(0), 1).detach().to(device=x.device)
            task_id = task_id.repeat(x.size(0), 1).detach().to(device=x.device)
            
            input = torch.cat((input_means, input_vars, task_id, x), -1)
            mu_hat_t = self.task_mu_modules[t](input)

            # reduce
            if self.reduction == "mean":
                mu_hat_t = torch.mean(mu_hat_t, dim=0) # averaged along batch
            elif self.reduction == "sum":
                mu_hat_t = torch.sum(mu_hat_t, dim=0)
            else:
                mu_hat_t = torch.mean(mu_hat_t, dim=0) # averaged along batch

            if with_momentum_update:
                updated_mean_t = (1 - self.momentum) * mu_hat_t + (self.momentum * means[t])
                updated_means[t] = updated_mean_t 

            else:
                updated_means[t] = updated_mean_t 
            

        return updated_means[None, :]


class UniSegExtractor_Fast_DP(UniSeg_model):

    def __init__(self, feature_space_dim, num_tasks, class_lst_to_std_mapping, task_id_class_lst_mapping, *args, loss=None, with_wandb=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_tasks = num_tasks
        self.feature_space_dim = feature_space_dim
        
        self.gaussian_mixtures = [GaussianMixture(1) for i in range(num_tasks)]

        self.do_ds = False

        self.queue_min = 1
        self.gmm_fitted = False
        self.feature_space_gmm = None
        self.update_target_dist = False
        self.update_size= 100

        self.class_lst_to_std_mapping = class_lst_to_std_mapping
        self.task_id_class_lst_mapping = task_id_class_lst_mapping
        self.with_wandb = with_wandb

        self.loss = loss
        self.dc_loss = SoftDiceLoss(batch_dice=True, smooth=1e-5, do_bg=True, apply_nonlin=None, do_one_hot=False)


    def init_gmms(self, means, vars):
        # means, vars = self.dynamic_dist.get_mean_var()

        for i in range(self.num_tasks):
            self.gaussian_mixtures[i].means_ = means[i].detach().cpu().numpy()
            self.gaussian_mixtures[i].covariances_ = vars[i].detach().cpu().numpy()#.item() * np.eye(self.feature_space_dim)
    

    def train_gmms(self, feature_space_qs, mus, sigs):
        if not self.gmm_fitted :
            self.init_gmms(mus, sigs)
            self.gmm_fitted = True

        # Train using queue
        trained_indices = []
        for t in range(self.num_tasks):
            task_queue = feature_space_qs[t] 
            if len(task_queue) < self.queue_min:
                continue
            task_queue = torch.vstack(task_queue).detach().cpu().numpy()
            self.gaussian_mixtures[t].fit(task_queue)
            trained_indices.append(t)
        
        # Collect params and construct pytorch dist
        # est_means = torch.from_numpy(np.vstack([self.gaussian_mixtures[t].means_ for t in trained_indices]))
        # est_covs = torch.from_numpy(np.vstack([self.gaussian_mixtures[t].covariances_ for t in trained_indices]))
        est_weights = torch.from_numpy(np.vstack([self.gaussian_mixtures[t].weights_ for t in trained_indices]))
        lower_bounds = np.vstack([self.gaussian_mixtures[t].lower_bound_ for t in trained_indices])

        component_distributions = []
        for t in trained_indices:
            mean_t = torch.from_numpy(self.gaussian_mixtures[t].means_)
            cov_t = torch.from_numpy(self.gaussian_mixtures[t].covariances_)
            if torch.cuda.is_available():
                mean_t = mean_t.cuda()
                cov_t = cov_t.cuda()
                if self.with_wandb:
                    wandb.log({
                        f'gmm_mean_{t}': mean_t,#.item(), 
                        f'gmm_var_{t}': cov_t,#.item(), 
                        f'gmm_lower_bound_{t}': self.gaussian_mixtures[t].lower_bound_,
                    })

            # if len(cov_t.shape) == 2:
            #     cov_t = cov_t.unsqueeze(0).repeat(mean_t.size(0), 1, 1)

            component_distributions.append(MultivariateNormal(mean_t, cov_t))

        # Determine the KL w.r.t these EM-learned dists and targets
        learned_target_kls = kl_divs(component_distributions, mus, sigs)
        if self.with_wandb:
            for i, kl in enumerate(learned_target_kls):
                wandb.log({f'est_gmm_kl_{trained_indices[i]}': kl.item()})

        # https://discuss.pytorch.org/t/how-to-use-torch-distributions-multivariate-normal-multivariatenormal-in-multi-gpu-mode/135030/3
        # component_distributions = [
        #     MultivariateNormal(mean, covariance) for mean, covariance in zip(est_means, est_covs)
        # ]
        categorical = Categorical(est_weights)
        self.feature_space_gmm = GaussianMixtureModel(categorical, component_distributions)

        self.gmm_fitted = True
        # return est_means, est_covs, est_weights, lower_bounds
        return est_weights, lower_bounds

    def construct_torch_gmm(self, means, vars):
        comp_dists = []
        for t in range(self.num_tasks):
            cov_t = vars[t]
            # if len(cov_t.shape) != 3:
            #     cov_t = cov_t.unsqueeze(0).repeat(means[t].size(0), 1, 1)
            
            comp_dists.append(MultivariateNormal(means[t], cov_t))

        categorical = Categorical(torch.ones(self.num_tasks, device=means.device) / self.num_tasks)

        self.feature_space_gmm = GaussianMixtureModel(categorical, comp_dists)

    def forward(self, x, mus, sigs, task_id=None, for_backprop=False, **kwargs):
        """
        if training: return loss
        """
        if self.training or for_backprop:
            return self.forward_train(x, mus[0, :], sigs[0, :], task_id=task_id, for_backprop=for_backprop, **kwargs)
        else :
            return self.forward_inference(x, task_id=task_id, **kwargs)


    def forward_train(self, x, mus, sigs, task_id=None, gt_seg=None, target_classes=None, return_dc_score=False, update_target_dist=False, for_backprop=True, **kwargs):
        if update_target_dist:
            features, _, _, tp_feats, _ = super().forward(x, task_id, get_prompt=True)
        else:
            features = super().forward(x, task_id, get_prompt=False)
        norms = features.reshape(features.size(0), -1).norm(dim=1)
        for norm in norms:
            if self.with_wandb: wandb.log({"output feature space norm (per batch)": norm.item()})
            print(f"output feature space norm (per batch): {norm.item()}")

        # extract + permute dims for DP 
        # gt_extractions = [extract_task_set(features, gt_seg, c, keep_dims=True).permute(1, 0) for c in target_classes]
        gt_extractions = [extract_task_set(features, gt_seg, c, keep_dims=True) for c in target_classes]

        loss_kwargs = {} if self.loss.loss_type == "kl" else {"gt_seg": gt_seg, "features": features}
        if self.loss.loss_type == "kl":
            l, tc_inds = self.get_loss(gt_extractions, mus, sigs, task_id, target_classes, return_est_dists=return_dc_score, with_sep_loss=update_target_dist, **loss_kwargs)
        else:
            l, tc_inds = self.get_loss(gt_extractions, mus, sigs, task_id, target_classes, return_est_dists=return_dc_score, with_sep_loss=update_target_dist, **loss_kwargs)

        # Test the dice score:
        if return_dc_score:
            l, est_dists = l
            with torch.no_grad():
                est_seg = self.segment_from_dists(features, est_dists, self.class_lst_to_std_mapping)
                # NOTE change permute to be (0, num_classes_in_gt_seg, 1, ..., num_classes_in_gt_seg-1 )
                perm_dims = tuple([0, len(features.shape)-1] + list(range(1, len(features.shape)-1)))
                # est_seg = torch.nn.functional.one_hot(est_seg, self.task_class[int(task_id[0])]).permute(0, 4, 1, 2, 3)
                # self.print_to_log_file(f"num classes in gt; {num_classes_in_gt}, {est_seg.unique()}")
                est_seg = torch.nn.functional.one_hot(est_seg, self.num_classes)
                est_seg = torch.permute(est_seg, perm_dims)
                dc_score = self.dc_loss(est_seg, gt_seg)[None] # NOTE
                

        if update_target_dist:
            flat_tp_feats = tp_feats.reshape(x.size(0), -1)
            l = (l, flat_tp_feats)

        if return_dc_score:
            l = l, dc_score

        repr_elements = []
        for i, task_id in enumerate(tc_inds):
            idxs = torch.randint(0, gt_extractions[i].shape[-1], (self.update_size,))
            gt_extractions_ = gt_extractions[i].detach().clone().to(device=gt_extractions[i].device)
            repr_elements.append(gt_extractions_[:, idxs]) # NOTE what is the dimension?
        
        tc_inds = [torch.tensor([ind], device=x.device) for ind in tc_inds]
        return l, repr_elements, tc_inds

    def forward_inference(self, x, task_id=None, gt_seg=None, target_classes=None, **kwargs):
        features = super().forward(x, task_id, get_prompt=False)
        
        if gt_seg is not None:
            # extract + permute dimensions for DP
            gt_extractions = [extract_task_set(features, gt_seg, c, keep_dims=True).permute(1, 0) for c in target_classes]

        #     # Use GMM + Bayes' Rule 
        #     return self.segment(features), gt_extractions

        # return self.segment(features)
            # Use GMM + Bayes' Rule 
            return self.full_segment(features, task_id), gt_extractions

        return self.full_segment(features, task_id)

    def seg_and_eval(self, x, task_id, gt_seg=None, **kwargs):

        seg = self.forward_inference(x, task_id, **kwargs)

        return self.get_hard_tp_fp_fn(seg, gt_seg)

    def segment(self, features: torch.Tensor):

        b = features.size(0)
        h, w, d = tuple(features.shape[2:])
        # bview_features = features.reshape(b, -1, self.feature_space_dim)
        bview_features = features.permute(0, 2, 3, 4, 1).reshape(b, -1, self.feature_space_dim)

        # segmentation = self.feature_space_gmm.classify(bview_features).reshape(features.size())
        feature_log_probs = self.feature_space_gmm.score(bview_features)
        feature_log_probs = feature_log_probs.reshape(b, h, w, d, len(self.feature_space_gmm.component_distributions)).permute(0, 4, 1, 2, 3)
        segmentation = torch.argmax(feature_log_probs, dim=1)

        return segmentation

    def standardize_segmentation(self, segmentation, tc_inds_to_cls):

        vals = torch.unique(segmentation)
        for val in vals:
            msk = segmentation == val
            segmentation[msk] = tc_inds_to_cls[val.item()]

        return segmentation

    def segment_from_dists(self, x, dists, tc_inds_to_cls):
        categorical = Categorical(torch.ones(len(dists), device=x.device) / len(dists))
        sampled_feature_space_gmm = GaussianMixtureModel(categorical=categorical, component_distributions=dists)
        feature_space_gmm = None
        if self.feature_space_gmm is not None:
            feature_space_gmm = copy.copy(self.feature_space_gmm)#.detach().clone()

        self.feature_space_gmm = sampled_feature_space_gmm

        segmentaiton = self.segment(x)
        std_segmentation = self.standardize_segmentation(segmentaiton, tc_inds_to_cls)
        
        self.feature_space_gmm = feature_space_gmm
        
        return std_segmentation

    def full_segment(self, x, task_id):

        output_segmentation = self.segment(x)
        std_output_segmentation = self.standardize_segmentation(output_segmentation, self.class_lst_to_std_mapping)
        # est_seg = torch.nn.functional.one_hot(std_output_segmentation, num_classes_in_gt).permute(0, 4, 1, 2, 3)
        perm_dims = tuple([0, len(x.shape)-1] + list(range(1, len(x.shape)-1)))
        # est_seg = torch.nn.functional.one_hot(std_output_segmentation, len(self.task_id_class_lst_mapping[int(task_id.item())])).permute(perm_dims)
        est_seg = torch.nn.functional.one_hot(std_output_segmentation, self.num_classes)
        est_seg = torch.permute(est_seg, perm_dims)

        return est_seg

    def ood_classify(self, x):
        pass 

    def set_update_target_dist(self, val):
        self.update_target_dist = val

    def get_update_target_dist(self):
        return self.update_target_dist

    def get_loss(self, extractions, mus, sigs, task_id, target_classes, return_est_dists=False, with_sep_loss=False, **kwargs):

        if not isinstance(task_id, int):
            task_id = int(task_id[0].item())

        tc_inds = []
        i, j = 0, 0
        while i < len(extractions) and j < len(target_classes):
            # re-arrange dimensions for loss
            # extractions[i] = extractions[i].permute(1, 0)
            if extractions[i].size(-1) < 2:
                _ = extractions.pop(i)
            else:
                c = target_classes[j]
                tc_inds.append(self.task_id_class_lst_mapping[task_id][int(c)])
                i += 1
            j += 1

        if "gt_seg" in kwargs:
            l =  self.loss(kwargs["features"], mus, sigs, kwargs["gt_seg"], tc_inds, pred_dists=extractions, return_est_dists=return_est_dists, with_sep_loss=with_sep_loss)
        else:
            l =  self.loss(extractions, mus, sigs, tc_inds, return_est_dists=return_est_dists, with_sep_loss=with_sep_loss)

        return l, tc_inds


    def get_hard_tp_fp_fn(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            # output_softmax = softmax_helper(output)
            output_softmax = output
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False)[None]
            fp_hard = fp_hard.sum(0, keepdim=False)[None]
            fn_hard = fn_hard.sum(0, keepdim=False)[None]

            return tp_hard, fp_hard, fn_hard