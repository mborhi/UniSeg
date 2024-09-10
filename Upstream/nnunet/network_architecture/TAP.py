from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
import torch.nn.functional as F
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, StackedConvLayers, \
    Upsample, Generic_UNet, StackedConvLayers_multi_channel
from nnunet.network_architecture.neural_network import SegmentationNetwork
from copy import deepcopy
from torch.distributions import MultivariateNormal, Categorical
from sklearn.mixture._gaussian_mixture import GaussianMixture
from nnunet.utilities.gmm import GaussianMixtureModel
import copy

from nnunet.utilities.sep import kl_divs, est_dist_sep_loss
from nnunet.utilities.sep import extract_task_set

import wandb

class TAP(nn.Module):
    def __init__(self, feature_space_dim, num_components, momentum, queue_size=5000, gmm_comps = 1):
        super(TAP, self).__init__()
        self.feature_space_dim = feature_space_dim 
        self.num_components = num_components 
        self.prompt_out_channel = 1
        self.gmm_comps = gmm_comps

        # self.acc_mus = [[] for _ in range(num_components)]
        # self.acc_covs = [[] for _ in range(num_components)]

        hidden_dim = 1000
        
        self.task_mu_modules = nn.ModuleList([
            nn.Sequential(
                # each task's mean of dim feature space (flattented), scalara vars for each task, task prompt dim, task_id
                nn.Linear((feature_space_dim * (num_components*gmm_comps)) + (1 * (num_components*gmm_comps)) + 1, 1024), 
                nn.PReLU(), 
                nn.Linear(1024, 512),
                nn.PReLU(), 
                nn.Linear(512, feature_space_dim),
                nn.Tanh(), 
            )
            for t in range(num_components * gmm_comps)
            # for t in range(1)
        ])
        self.task_sigma_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear((feature_space_dim * (num_components*gmm_comps)) + (1 * (num_components*gmm_comps)) + 1, 1024), 
                nn.PReLU(), 
                nn.Linear(1024, 512),
                nn.PReLU(), 
                nn.Linear(512, 1),
                nn.ReLU()
            )
            for t in range(num_components * gmm_comps)
        ])

        conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        self.momentum = momentum

        self.queue_size = queue_size
        self.feature_space_qs = [[] for i in range(num_components)]
        self.best_feature_space_qs = None

    def update_queues(self, feature_extracts, tc_inds):
        update_size = 100
        for i, task_id in enumerate(tc_inds):
            idxs = torch.randint(0, feature_extracts[i].shape[-1], (update_size,))
            feature_extracts_ = feature_extracts[i].detach().clone().to(device=feature_extracts[i].device)
            rand_elements = feature_extracts_[:, idxs]
            rand_elements = rand_elements.reshape(-1, self.feature_space_dim)
            
            # Enqueue and Dequeue
            current_queue_size = len(self.feature_space_qs[task_id])
            if current_queue_size + update_size > self.queue_size:
                excess_size = current_queue_size + update_size - self.queue_size
                self.feature_space_qs[task_id] = self.feature_space_qs[task_id][excess_size:]
            
            self.feature_space_qs[task_id].extend(rand_elements)  # assuming you want to store them as numpy arrays

    def clear_queues(self):

        for i in range(self.num_components):
            self.feature_space_qs[i] = []

    def set_best_feature_space_qs(self):

        self.best_feature_space_qs = copy.deepcopy(self.feature_space_qs)

    def use_best_feature_space_qs(self):

        self.feature_space_qs = self.best_feature_space_qs
        

    def forward(self, means, vars, tc_inds, with_momentum_update=True):
        # Returns: updated means and vars
        # self.means = self.means.detach().clone().to(device=x.device)
        # self.vars = self.vars.detach().clone().to(device=x.device)
        # means = means.detach().clone().to(device=x.device)
        # vars = vars.detach().clone().to(device=x.device)
        sigma_hats = []
        # mu_hats = [m for m in means]
        mu_hats = []
        # mu_hats = []
        for t, task_id in enumerate(tc_inds):
            if not isinstance(task_id, torch.Tensor):
                task_id = torch.tensor([task_id])[:, None].to(device=means.device)

            # if means.device != x.device:
            #     means = means.to(device=x.device)
            #     vars = vars.to(device=x.device)
            
            input_means = means.reshape(-1)[None, :]#.repeat(x.size(0), 1).detach().to(device=x.device)
            # input_vars = vars.permute(1, 0).repeat(x.size(0), 1).detach().to(device=x.device)
            input_vars = vars[None, :]#.repeat(x.size(0), 1).detach().to(device=x.device)
            task_id = task_id#.repeat(x.size(0), 1).detach().to(device=x.device)
            
            # input = torch.cat((input_means, input_vars, task_id), -1).to(dtype=torch.float16)
            input = torch.cat((input_means, input_vars, task_id), -1).to(dtype=torch.float32)
            mu_hat_t = torch.mean(self.task_mu_modules[t](input), dim=0) # averaged along batch
            sigma_hat_t = torch.mean(self.task_sigma_modules[t](input), dim=0) # averaged along batch
            # mu_hat_t = torch.mean(self.task_mu_modules[0](input), dim=0) # averaged along batch

            if with_momentum_update:
                # # updated_var_t = (1 - 0.999) * vars[0, t] + (0.999 * sigma_hat_t) # + 0.0001 # for numerical stability
                # sigma_hats.append(updated_var_t)
                # self.vars[t] = updated_var_t 
                # updated_mean_t = (1 - self.momentum) *  means[t] + (self.momentum * mu_hat_t)
                # updated_mean_t = (1 - self.momentum) *  means[t] + (self.momentum * mu_hat_t)
                # [0,..0, 1, ..., 1]
                # 0, 0
                # 1, 0
                #... 4, 0
                # 5, 1, 
                # ...
                # 9, 1 => 9 - (5 * task_id) = 4
                updated_mean_t = (1 - self.momentum) *  means[task_id, t - (self.gmm_comps * task_id)] + (self.momentum * mu_hat_t)
                updated_var_t = (1 - self.momentum) * vars[t - (self.gmm_comps * task_id)] + (self.momentum * sigma_hat_t) # + 0.0001 # for numerical stability
                # print(f"updated mean, var {t}: {updated_mean_t}, {updated_var_t}")
                # print(f"updated mean[{t}]: {updated_mean_t}")
                # print(f"updated sig[{t}]: {updated_var_t}")
                # means[t] = updated_mean_t
                # mu_hats[t] = updated_mean_t 
                # mu_hats[task_id][t - (5 * task_id)] = updated_mean_t 
                mu_hats.append(updated_mean_t) 
                # mu_hats.append(updated_mean_t)
                sigma_hats.append(updated_var_t * torch.eye(self.feature_space_dim, device=updated_var_t.device))

                # NOTE add to acc
                # self.acc_mus[t].append(mu_hat_t)
                # self.acc_covs[t].append(updated_var_t)
            

        # return means, vars
        # return means, sigma_hats
        return mu_hats, sigma_hats