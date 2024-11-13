from nnunet.utilities.nd_softmax import softmax_helper, identity_helper
from torch import nn
import torch
import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.network_architecture.initialization import InitWeights_He
import torch.nn.functional as F
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, StackedConvLayers, \
    Upsample, Generic_UNet, StackedConvLayers_multi_channel
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.TAP import TAP
from copy import deepcopy
from torch.distributions import MultivariateNormal, Categorical
from sklearn.mixture._gaussian_mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from nnunet.utilities.gmm import GaussianMixtureModel
import copy

from nnunet.utilities.sep import kl_divs, est_dist_sep_loss
from nnunet.utilities.sep import extract_task_set
from nnunet.utilities.sep import extract_correct_task_set

import wandb

class DynamicDistributionModel_DP(nn.Module):
    def __init__(self, feature_space_dim, tp_dim, num_components, momentum, queue_size=5000, gmm_comps = 1):
        super(DynamicDistributionModel_DP, self).__init__()
        self.feature_space_dim = feature_space_dim 
        self.num_components = num_components 
        self.tp_dim = tp_dim
        self.prompt_out_channel = 1
        self.gmm_comps = gmm_comps

        # self.acc_mus = [[] for _ in range(num_components)]
        # self.acc_covs = [[] for _ in range(num_components)]

        hidden_dim = 1000
        
        self.task_mu_modules = nn.ModuleList([
            nn.Sequential(
                # each task's mean of dim feature space (flattented), scalara vars for each task, task prompt dim, task_id
                nn.Linear((feature_space_dim * (num_components*gmm_comps)) + (1 * (num_components*gmm_comps)) + (self.prompt_out_channel*tp_dim) + 1, 1024), 
                nn.PReLU(), 
                nn.Linear(1024, 512),
                nn.PReLU(), 
                nn.Linear(512, feature_space_dim),
                nn.Tanh(), 
            )
            for t in range(num_components * gmm_comps)
            # for t in range(1)
        ])
        # self.task_sigma_modules = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear((feature_space_dim * num_components) + (1 * num_components) + (self.prompt_out_channel*tp_dim) + 1, 1024), 
        #         nn.PReLU(), 
        #         nn.Linear(1024, 512),
        #         nn.PReLU(), 
        #         nn.Linear(512, 1),
        #         nn.ReLU()
        #     )
        #     for t in range(num_components)
        # ])

        conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # self.prompt_conv = ConvDropoutNormNonlin(1, 
        #                                          self.prompt_out_channel, 
        #                                          nn.Conv3d, 
        #                                          conv_kwargs=conv_kwargs, 
        #                                          norm_op=nn.InstanceNorm3d, 
        #                                          norm_op_kwargs=norm_op_kwargs,
        #                                          dropout_op=nn.Dropout3d, 
        #                                          dropout_op_kwargs=dropout_op_kwargs
        #                                          )
        bottleneck_feature_channel = 3
        self.prompt_conv = nn.Sequential(
            *([ConvDropoutNormNonlin(1, bottleneck_feature_channel, nn.Conv3d,
                           conv_kwargs,
                           nn.InstanceNorm3d, norm_op_kwargs, nn.Dropout3d, dropout_op_kwargs,
                           nn.LeakyReLU, nonlin_kwargs)] +
              [ConvDropoutNormNonlin(bottleneck_feature_channel, bottleneck_feature_channel, nn.Conv3d,
                           conv_kwargs,
                           nn.InstanceNorm3d, norm_op_kwargs, nn.Dropout3d, dropout_op_kwargs,
                           nn.LeakyReLU, nonlin_kwargs)] +
              [ConvDropoutNormNonlin(bottleneck_feature_channel, self.prompt_out_channel, nn.Conv3d,
                           conv_kwargs,
                           nn.InstanceNorm3d, norm_op_kwargs, nn.Dropout3d, dropout_op_kwargs,
                           nn.LeakyReLU, nonlin_kwargs)]
              ))

        self.momentum = momentum

        self.queue_size = queue_size
        self.feature_space_qs = [[] for i in range(num_components)]
        self.best_feature_space_qs = None
        

    # def update_queues(self, feature_extracts, tc_inds):
    #     update_size = 100
    #     # create mask of accurate feature extractions (highest log-likelihood w.r.t target Gauss.)
    #     for i, task_id in enumerate(tc_inds):
    #         idxs = torch.randint(0, feature_extracts[i].shape[-1], (update_size,))
    #         feature_extracts_ = feature_extracts[i].detach().clone().to(device=feature_extracts[i].device)
    #         rand_elements = feature_extracts_[:, idxs]
    #         # Enqueue and Dequeue
    #         if len(self.feature_space_qs[task_id]) + update_size > self.queue_size:
    #             for _ in range(len(self.feature_space_qs[task_id]) + update_size - self.queue_size):
    #                 self.feature_space_qs[task_id].pop(0)
            
    #         self.feature_space_qs[task_id].append(rand_elements.reshape(-1, self.feature_space_dim))


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
        

    def forward(self, x, means, vars, tc_inds, with_momentum_update=True):
        # Returns: updated means and vars
        x = self.prompt_conv(x)
        x = x.reshape(x.size(0), -1)
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
                task_id = torch.tensor([task_id])[:, None].to(device=x.device)

            if means.device != x.device:
                means = means.to(device=x.device)
                vars = vars.to(device=x.device)
            
            input_means = means.reshape(-1)[None, :].repeat(x.size(0), 1).detach().to(device=x.device)
            # input_vars = vars.permute(1, 0).repeat(x.size(0), 1).detach().to(device=x.device)
            input_vars = vars.repeat(x.size(0), 1).detach().to(device=x.device)
            task_id = task_id.repeat(x.size(0), 1).detach().to(device=x.device)
            
            input = torch.cat((input_means, input_vars, task_id, x), -1).to(dtype=torch.float16)
            mu_hat_t = torch.mean(self.task_mu_modules[t](input), dim=0) # averaged along batch
            # mu_hat_t = torch.mean(self.task_mu_modules[0](input), dim=0) # averaged along batch
            # sigma_hat_t = torch.mean(self.task_sigma_modules[t](input), dim=0) # averaged along batch

            if with_momentum_update:
                # updated_var_t = (1 - 0.999) * vars[t] + (0.999 * sigma_hat_t) # + 0.0001 # for numerical stability
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
                # print(f"updated mean, var {t}: {updated_mean_t}, {updated_var_t}")
                # print(f"updated mean[{t}]: {updated_mean_t}")
                # print(f"updated sig[{t}]: {updated_var_t}")
                # means[t] = updated_mean_t
                # mu_hats[t] = updated_mean_t 
                # mu_hats[task_id][t - (5 * task_id)] = updated_mean_t 
                mu_hats.append(updated_mean_t) 
                # mu_hats.append(updated_mean_t)

                # NOTE add to acc
                # self.acc_mus[t].append(mu_hat_t)
                # self.acc_covs[t].append(updated_var_t)
            

        # return means, vars
        # return means, sigma_hats
        return mu_hats, sigma_hats


class StackedFusionConvLayers(nn.Module):
    def __init__(self, input_feature_channels, bottleneck_feature_channel, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedFusionConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, bottleneck_feature_channel, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(bottleneck_feature_channel, bottleneck_feature_channel, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 2)] +
              [basic_block(bottleneck_feature_channel, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)]
              ))

    def forward(self, x):
        return self.blocks(x)


class UniSeg_model(Generic_UNet):
    def __init__(self, patch_size, task_total_number, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        super(Generic_UNet, self).__init__()
        print("training patch size", patch_size)
        self.num_class =  task_total_number

        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.final_dc_nonlin = lambda x: F.softmax(x, dim=1)

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
            self.input_type = '2D'
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
            self.input_type = '3D'
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        # NOTE 
        # feature_space_dim = 1 #base_num_features * 2 #NOTE
        # feature_space_dim = 3 #base_num_features * 2 #NOTE
        feature_space_dim = num_classes #base_num_features * 2 #NOTE
        # feature_space_dim = 8 #base_num_features * 2 #NOTE
        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            if d == 0:
                self.conv_blocks_context.append(StackedConvLayers_multi_channel(input_features, output_features, num_conv_per_stage,
                                                                  self.conv_op, self.conv_kwargs, self.norm_op,
                                                                  self.norm_op_kwargs, self.dropout_op,
                                                                  self.dropout_op_kwargs, self.nonlin,
                                                                  self.nonlin_kwargs,
                                                                  first_stride, basic_block=basic_block))
            else:
                self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))

            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # Classifier & Global info
        num_conv_stage = 3
        self.intermediate_prompt = nn.Parameter(torch.randn(1, self.num_class, patch_size[0]//16, patch_size[1]//32, patch_size[2]//32))
        self.fusion_layer = StackedFusionConvLayers(final_num_features+task_total_number, (final_num_features+task_total_number)//4, task_total_number, num_conv_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              None, basic_block=basic_block)
        final_num_features = final_num_features + 1
        print("intermediate_prompt size", self.intermediate_prompt.size(), torch.min(self.intermediate_prompt), torch.max(self.intermediate_prompt))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            # self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
            #                                 1, 1, 0, 1, 1, seg_output_use_bias))
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, feature_space_dim,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                # self.upscale_logits_ops.append(lambda x: x)
                self.upscale_logits_ops.append(identity_helper)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        # self.channel_reduction = nn.Conv3d(num_classes, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias)
        # self.final_conv = conv_op(self.conv_blocks_localization[-1][-1].output_channels, base_num_features,
        #                                     1, 1, 0, 1, 1, seg_output_use_bias)

        print("num channels before new block:", self.conv_blocks_localization[-1][-1].output_channels)
        # self.final_extractor = nn.Sequential(StackedConvLayers(self.conv_blocks_localization[-1][-1].output_channels, feature_space_dim, 3,
        #                           self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
        #                           self.dropout_op_kwargs, nn.Tanh, {}, basic_block=basic_block))
        self.final_extractor = nn.Sequential(StackedConvLayers(self.conv_blocks_localization[-1][-1].output_channels, feature_space_dim, 3,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, nn.Identity, {}, basic_block=basic_block))

        self.final_extractors = []
        for ds in range(len(self.conv_blocks_localization)):
            self.final_extractors.append(conv_op(feature_space_dim, 1,
                                            1, 1, 0, 1, 1, seg_output_use_bias))
        self.final_extractors = nn.ModuleList(self.final_extractors)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

        # Init module for getting embeddings 
        self.extractor = nn.ModuleDict({
            "identity": nn.Identity()
        })
        
        # NOTE
        # feature_space_dim = 128
        self.final_tanh = nn.Tanh()
        self.final_sigmoid = nn.Sigmoid()


    def forward(self, x, task_id, get_prompt=False):
        skips = []
        seg_outputs = []

        bs = x.size()[0]
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            # print(x.size())
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)
        # print(now_prompt.size())
        now_prompt = self.intermediate_prompt.repeat(bs,1,1,1,1)
        dynamic_prompt = self.fusion_layer(torch.cat([x, now_prompt], dim=1))
        task_prompt = torch.index_select(dynamic_prompt, 1, task_id[0])
        task_prompt = self.extractor["identity"](task_prompt) # for debugging forward hook
        if get_prompt:
            temp_x = x.detach().clone()
        x = torch.cat([x, task_prompt], dim=1)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.seg_outputs[u](x)) # NOTE current standard
            
            # seg_outputs.append(self.final_extractors[u](self.seg_outputs[u](x))) # NOTE
            # seg_outputs.append(self.final_tanh(self.seg_outputs[u](x))) # NOTE
            # seg_outputs.append(self.final_sigmoid(self.seg_outputs[u](x))) # NOTE
            # seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x))) # NOTE
            # seg_outputs.append(self.final_dc_nonlin(self.seg_outputs[u](x))) # NOTE

            ## assumes feature dim independence, then models distribution of each feature dim
            # seg_outputs.append(torch.amax(self.seg_outputs[u](x), 1, keepdim=True)) # NOTE
            # if u + 1 == len(self.tu):
            #     seg_outputs.append(self.seg_outputs[u](x))
            #     # seg_outputs.append(self.final_tanh(x))
            #     # seg_outputs.append(self.final_tanh(self.final_conv(x)))
            #     # seg_outputs.append(F.normalize(self.final_extractor(x), dim=1))
            #     # seg_outputs.append(self.final_extractor(x)) # NOTE
            #     # seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x))) # NOTE
            #     seg_outputs.append(self.final_tanh(self.seg_outputs[u](x))) # NOTE
            #     # seg_outputs.append(self.final_dc_nonlin(self.seg_outputs[u](x))) # NOTE
            #     # seg_outputs.append(self.final_sigmoid(self.seg_outputs[u](x))) # NOTE
            # else:
            #     seg_outputs.append(self.final_dc_nonlin(self.seg_outputs[u](x)))
            #     # seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            #     # seg_outputs.append(self.final_dc_nonlin(self.seg_outputs[u](x)))
            #     seg_outputs.append(self.final_tanh(self.seg_outputs[u](x)))
            #     # seg_outputs.append(self.seg_outputs[u](x))
            #     # seg_outputs.append(self.final_sigmoid(self.seg_outputs[u](x)))

            assert x.isnan().count_nonzero() == 0 
        
        final_out = seg_outputs[-1]
        # print(f"pct non-positive: {torch.sum(final_out <= 0) / torch.numel(final_out) }")

        if get_prompt and self._deep_supervision and self.do_ds:
            return list([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]), \
                    self.intermediate_prompt.detach().clone(), \
                    dynamic_prompt.detach().clone(), \
                    task_prompt.detach().clone(), \
                    temp_x
        elif self._deep_supervision and self.do_ds:
            return list([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        
        elif get_prompt:
            return final_out, self.intermediate_prompt.detach().clone(), dynamic_prompt.detach().clone(), task_prompt.detach().clone(), temp_x

        else:
            return final_out


class UniSegExtractor_DP(UniSeg_model):

    def __init__(self, feature_space_dim, num_tasks, class_lst_to_std_mapping, task_id_class_lst_mapping, *args, with_wandb=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_tasks = num_tasks
        self.feature_space_dim = feature_space_dim
        
        self.gaussian_mixtures = [GaussianMixture(1, tol=1e-4, max_iter=5) for i in range(num_tasks)]

        # self.do_ds = True

        self.queue_min = 1
        self.gmm_fitted = False
        self.feature_space_gmm = None
        self.update_target_dist = False

        self.class_lst_to_std_mapping = class_lst_to_std_mapping
        self.task_id_class_lst_mapping = task_id_class_lst_mapping
        self.with_wandb = with_wandb


    def init_gmms(self, means, vars):
        # means, vars = self.dynamic_dist.get_mean_var()

        for i in range(self.num_tasks):
            self.gaussian_mixtures[i].means_ = means[i].detach().cpu().numpy()
            self.gaussian_mixtures[i].covariances_ = vars[i].detach().cpu().numpy()#.item() * np.eye(self.feature_space_dim)

        self.gmm_fitted = True
    
    def gmm_analysis(self, feature_space_qs, mus, sigs):
        sizes = [0.10, 0.25, 0.50, 0.8, 0.9, 1]
        comp_errors = []
        for t in range(self.num_tasks):
            task_queue = feature_space_qs[t] 
            if len(task_queue) < self.queue_min:
                continue
            size_errors = []
            task_queue = torch.vstack(task_queue).detach().cpu().numpy()
            for i, size in enumerate(sizes):
                N = int(task_queue.shape[0] * sizes[i])
                task_queue_ = task_queue[:N, :]
                gmm = GaussianMixture(1)
                gmm.fit(task_queue_)

                dist = np.linalg.norm(mus[t].detach().cpu().numpy() - gmm.means_)
                size_errors.append(dist)
                if wandb.run is not None:
                    wandb.log({f"comp_{t}_error_{size}": dist})
                    wandb.log({f"comp_{t}_size_{size}": N})

            comp_errors.append(size_errors)


    def train_gmms(self, feature_space_qs, mus, sigs):
        # if not self.gmm_fitted :
        #     self.init_gmms(mus, sigs)

        trained_indices = self.gmm_analysis(feature_space_qs, mus, sigs)
        # Train using queue
        trained_indices = []
        for t in range(self.num_tasks):
            task_queue = feature_space_qs[t] 
            if len(task_queue) < self.queue_min:
                continue
            task_queue = torch.vstack(task_queue).detach().cpu().numpy()
            self.gaussian_mixtures[t].fit(task_queue)
            trained_indices.append(t)


        if self.with_wandb:
            wandb.log({"trained_inds": trained_indices})
        print(f"Trained inds: {trained_indices}")
        # Collect params and construct pytorch dist
        # est_means = torch.from_numpy(np.vstack([self.gaussian_mixtures[t].means_ for t in trained_indices]))
        # est_covs = torch.from_numpy(np.vstack([self.gaussian_mixtures[t].covariances_ for t in trained_indices]))
        est_weights = torch.from_numpy(np.vstack([self.gaussian_mixtures[t].weights_ for t in trained_indices]))
        lower_bounds = np.vstack([self.gaussian_mixtures[t].lower_bound_ for t in trained_indices])

        est_means = []
        component_distributions = []
        for t in trained_indices:
            mean_t = torch.from_numpy(self.gaussian_mixtures[t].means_)
            cov_t = torch.from_numpy(self.gaussian_mixtures[t].covariances_)
            print(f"Stat. diff {t}: {torch.norm(mus[t].detach().cpu() - mean_t)}")
            wandb.log({f"Stat. diff {t}": torch.norm(mus[t].detach().cpu() - mean_t)})
            # print(f"Est cov {t}: {cov_t}")
            if torch.cuda.is_available():
                mean_t = mean_t.cuda()
                cov_t = cov_t.cuda()
                if self.with_wandb:
                    wandb.log({
                        f'gmm_mean_{t}': mean_t,#.item(), 
                        f'gmm_var_{t}': cov_t,#.item(), 
                        f'gmm_lower_bound_{t}': self.gaussian_mixtures[t].lower_bound_,
                    })

                    est_means.append(mean_t)

            # if len(cov_t.shape) == 2:
            #     cov_t = cov_t.unsqueeze(0).repeat(mean_t.size(0), 1, 1)

            # component_distributions.append(MultivariateNormal(mean_t, cov_t))
            component_distributions.append(self.construct_torch_gmm(mean_t, cov_t))

        # Determine the KL w.r.t these EM-learned dists and targets
        est_dist, wrt_target = est_dist_sep_loss(est_means, self.min_dist, wrt_target=True, **{'mus': mus[trained_indices]})
        if self.with_wandb:
            wandb.log({'est_sep_dist':est_dist, 'sep_wrt_target': wrt_target})
            learned_target_kls = kl_divs(component_distributions, mus, sigs)
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
        if means.shape[0] == 1:
            return MultivariateNormal(means, vars)

        comp_dists = []
        # for t in range(self.num_tasks):
        for t in range(self.num_classes):
            cov_t = vars[t]
            # if len(cov_t.shape) != 3:
            #     cov_t = cov_t.unsqueeze(0).repeat(means[t].size(0), 1, 1)
            
            comp_dists.append(MultivariateNormal(means[t], cov_t))

        categorical = Categorical(torch.ones(means.shape[0], device=means.device) / means.shape[0])

        # self.feature_space_gmm = GaussianMixtureModel(categorical, comp_dists)
        dist = GaussianMixtureModel(categorical, comp_dists)

        return dist
        
    def forward(self, x, task_id=None, for_loss=False, uniseg_only=False, **kwargs):
        if uniseg_only:
            return super().forward(x, task_task=task_id, get_prompt=True)

        if self.training or for_loss:
            return self.forward_train(x, task_id=task_id, **kwargs)
        else :
            return self.forward_inference(x, task_id=task_id, **kwargs)


    def forward_train(self, x, task_id=None, gt_seg=None, target_classes=None, update_target_dist=False, **kwargs):
        if update_target_dist:
            features, _, _, tp_feats, _ = super().forward(x, task_id, get_prompt=True)
        else:
            features = super().forward(x, task_id, get_prompt=False)
        norms = features[0].reshape(features[0].size(0), -1).norm(dim=1)
        for norm in norms:
            if self.with_wandb: wandb.log({"output feature space norm (per input)": norm.item()})
            print(f"output feature space norm (per input): {norm.item()}")
            # print(f"features: {features[0].mean(dim=1)}, max: {features[0].max(dim=1)}, min: {features[0].min(dim=1)}")

        # extract + permute dims for DP 
        lst_gt_extractions = []
        # print(f"out features size: {features[0].size()}")
        # print(f"out features target classes: {target_classes[0]}")
        for i, feature in enumerate(features):
            gt_extractions = [extract_task_set(feature, gt_seg[i], c, keep_dims=True).permute(1, 0) for c in target_classes[i]]
            lst_gt_extractions.append(gt_extractions)

        if update_target_dist:
            flat_tp_feats = tp_feats#tp_feats.reshape(x.size(0), -1)
            return (features, flat_tp_feats), lst_gt_extractions
        
        return features, lst_gt_extractions#, mus, sigs

    def forward_inference(self, x, task_id=None, gt_seg=None, target_classes=None, **kwargs):
        features = super().forward(x, task_id, get_prompt=False)

        if gt_seg is not None:
            # extract + permute dimensions for DP
            gt_extractions = [extract_task_set(features, gt_seg, c, keep_dims=True).permute(1, 0) for c in target_classes]
            # lst_gt_extractions = []
            # for i, feature in enumerate(features):
            #     lst_gt_extractions.append(gt_extractions)
        #     # Use GMM + Bayes' Rule 
        #     return self.segment(features), gt_extractions

        # return self.segment(features)
            # Use GMM + Bayes' Rule 
            return self.full_segment(features, task_id), gt_extractions

        return self.full_segment(features, task_id)

    def segment(self, features: torch.Tensor, component_indices=None):

        b = features.size(0)
        h, w, d = tuple(features.shape[2:])
        # bview_features = features.reshape(b, -1, self.feature_space_dim)
        bview_features = features.permute(0, 2, 3, 4, 1).reshape(b, -1, self.feature_space_dim)

        # segmentation = self.feature_space_gmm.classify(bview_features).reshape(features.size())
        feature_log_probs = self.feature_space_gmm.score(bview_features, component_indices=component_indices)
        # feature_log_probs = self.feature_space_gmm.score(bview_features, component_indices=None)
        # feature_log_probs = feature_log_probs.reshape(b, h, w, d, len(self.feature_space_gmm.component_distributions)).permute(0, 4, 1, 2, 3)
        feature_log_probs = feature_log_probs.reshape(b, h, w, d, len(component_indices)).permute(0, 4, 1, 2, 3)
        # segmentation = torch.argmax(feature_log_probs, dim=1)
        segmentation = self.bayes(feature_log_probs)

        return segmentation

    def bayes(self, log_probs):
        # print("log probs", log_probs)
        # probs = torch.exp(log_probs)
        # scaled = probs / probs.sum(dim=1, keepdim=True)  # shape: [batch, num_components, height, width, depth]
        # # scaled = torch.exp(scaled)
        # max_prb, seg = torch.max(scaled, dim=1)
        # seg += 1
        # seg[max_prb <= 0.1] = 0
        # return seg
        scaled_log_probs = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)

        seg = torch.argmax(scaled_log_probs, dim=1)

        return seg

    def standardize_segmentation(self, segmentation, tc_inds_to_cls):

        vals = torch.unique(segmentation)
        for val in vals:
            msk = segmentation == val
            segmentation[msk] = tc_inds_to_cls[val.item()]

        return segmentation

    def segment_from_dists(self, x, dists, tc_inds_to_cls):
        1/ 0
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
        component_indices = self.task_id_class_lst_mapping[task_id[0].item()] if self.task_id_class_lst_mapping is not None else None
        # component_indices = None # self.task_id_class_lst_mapping[task_id[0].item()] if self.task_id_class_lst_mapping is not None else None
        std_output_segmentation = self.segment(x, component_indices=component_indices)
        # output_segmentation = self.segment(x, component_indices=component_indices)
        # std_output_segmentation = self.standardize_segmentation(output_segmentation, self.class_lst_to_std_mapping)
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



class TAPFeatureExtractor_DP(UniSeg_model):

    def __init__(self, feature_space_dim, gmm_comps, num_classes, class_lst_to_std_mapping, task_id_class_lst_mapping, *args, with_wandb=False, ood_detection_mode=False, **kwargs):
        super().__init__(*args, **kwargs)

        num_tasks = len(task_id_class_lst_mapping.keys())
        self.task_to_num_classes = {tidx : len(cls_lst) for tidx, cls_lst in task_id_class_lst_mapping.items()}
        self.num_tasks = num_tasks
        # self.tap = DynamicDistributionModel_DP(feature_space_dim, int(self.intermediate_prompt.numel() / self.num_class), num_classes, momentum=0.99, queue_size=5000, gmm_comps=gmm_comps)
        # self.feature_space_dim, self.num_components, 0.995, queue_size=5000, gmm_comps = self.max_gmm_comps
        # self.tap = TAP(feature_space_dim, num_classes, num_tasks, momentum=0.99, queue_size=5000, gmm_comps=10)

        # self.gaussian_mixtures = [GaussianMixture(gmm_comps, tol=1e-4) for i in range(num_classes)]
        self.mus = None 
        self.covs = None

        # self.num_classes = num_classes
        self.feature_space_dim = feature_space_dim
        self.gmm_comps = gmm_comps
        
        # self.do_ds = True

        self.queue_min = 1
        self.gmm_fitted = False
        self.feature_space_gmm = None
        self.update_target_dist = False

        self.task_feature_space_gmm = [None for _ in range(num_tasks)]
        self.task_taps = nn.ModuleList([
            # TAP(feature_space_dim, num_classes, num_tasks, momentum=0.99, queue_size=10000, val_q_size=2500, gmm_comps=10) 
            TAP(feature_space_dim, self.task_to_num_classes[tidx], num_tasks, momentum=0.99, queue_size=10000, val_q_size=2500, gmm_comps=self.gmm_comps) 
            for tidx in range(num_tasks)
        ])
        # self.task_taps = [
        #     TAP(feature_space_dim, num_classes, num_tasks, momentum=0.99, queue_size=5000, gmm_comps=10) 
        #     for _ in range(num_tasks)
        # ]
        self.tasks_mus = [None for _ in range(num_tasks)]
        self.tasks_covs = [None for _ in range(num_tasks)]
        self.tasks_weights = [None for _ in range(num_tasks)]

        self.class_lst_to_std_mapping = class_lst_to_std_mapping
        self.task_id_class_lst_mapping = task_id_class_lst_mapping
        self.with_wandb = with_wandb
        
        # self.var_gmm = BayesianGaussianMixture(n_components=10, covariance_type="diag", warm_start=True, max_iter=30, random_state=42)
        # self.var_gmm = BayesianGaussianMixture(n_components=10, covariance_type="full", warm_start=True, max_iter=30, random_state=42)

        # self.task_var_gmms = [
        #     BayesianGaussianMixture(n_components=10, covariance_type="diag", warm_start=True, max_iter=30, random_state=42)
        #     for _ in range(num_tasks)
        # ]
        self.task_var_gmms = [
            [ # NOTE
            # BayesianGaussianMixture(n_components=gmm_comps, covariance_type="diag", warm_start=True, max_iter=400, random_state=42)
            BayesianGaussianMixture(n_components=gmm_comps, covariance_type="full", warm_start=True, max_iter=400, random_state=42)
            for _ in range(self.task_to_num_classes[tidx])
            # for _ in range(num_classes)
            ]
            for tidx in range(num_tasks)
        ]

        self.q_update_size = 25

        self.ood_detection_mode = ood_detection_mode

        if ood_detection_mode:
            self.thresholds = [None for tidx in range(num_tasks)]
            # self.thresholds = [0.46685976679755287 for tidx in range(num_tasks)]

    def fit_task_gmms(self, X, task, cidx):

        # self.task_var_gmms[task].fit(X)
        self.task_var_gmms[task][cidx].fit(X)

        return self.task_var_gmms[task][cidx], self.gmm_comps, []
    
    def fit_gmms(self, X, **kwargs):
        # gmm = BayesianGaussianMixture(n_components=10, covariance_type="diag", **kwargs)
        # gmm = BayesianGaussianMixture(n_components=5)
        # gmm.fit(X)
        self.var_gmm.fit(X)

        return self.var_gmm, self.gmm_comps, []

    def find_optimal_components(self, X, min_components=1, max_components=10):
        bics = []
        best_gmm = None
        lowest_bic = np.inf
        for n in range(min_components, max_components+1):
            gmm = GaussianMixture(n_components=n)
            gmm.fit(X)
            if gmm.bic(X) < lowest_bic:
                best_gmm = gmm
            bics.append(gmm.bic(X))
        
        optimal_n = np.argmin(bics) + min_components
        return best_gmm, optimal_n, bics

    def set_feature_space_distribution_parameters(self, mus, covs, weights, task=None):
        if task is not None:
            self.tasks_mus[task] = mus
            self.tasks_covs[task] = covs
            self.tasks_weights[task] = weights
        else:
            self.mus = mus 
            self.covs = covs
            self.weights = weights

    def reestimate_components(self):
        pass 

    def adjust_dist_params(self, tp_feats, task_id, tc_inds):
        tc_inds_rep = np.repeat(tc_inds, self.gmm_comps)
        # self.covs = [3, 5, dim, dim]
        input_sigs = []
        for cov_comp_set in self.covs:
            for cov in cov_comp_set:
                input_sigs.append(torch.max(torch.real(torch.linalg.eigvals(cov))))

        input_sigs = torch.stack(input_sigs)

        # input_sigs = torch.tensor([torch.max(torch.real(torch.linalg.eigvals(cov))) for cov in self.covs])
        # mu_prime, _ = self.tap(tp_feats, self.mus.reshape(-1), input_sigs, tc_inds)
        mu_prime, _ = self.tap(tp_feats, self.mus, input_sigs, tc_inds_rep)

        # component_indices = self.task_id_class_lst_mapping[task_id[0].item()] if self.task_id_class_lst_mapping is not None else None
        
        mu_prime = torch.stack(mu_prime).reshape(len(tc_inds), self.gmm_comps, -1)
        
        for t, comp_idx in enumerate(tc_inds):
            self.feature_space_gmm.set_component_loc(comp_idx, mu_prime[t])

    def forward_train(self, x, task_id=None, gt_seg=None, target_classes=None, update_target_dist=False, **kwargs):
        features, _, _, tp_feats, _ = super().forward(x, task_id, get_prompt=True)

        # apply normalizations
        # NOTE check dimensions here
        # for i, feature in enumerate(features):
        #     feature = feature.permute(0, 2, 3, 4, 1)
        #     # nfeature = self.feat_norm(feature) # (b h w) c
        #     nfeature = F.normalize(feature, p=2, dim=-1)
        #     features[i] = nfeature.permute(0, 4, 1, 2, 3,)

        
        # with torch.no_grad():
        norms = features[0].reshape(features[0].size(0), -1).norm(dim=1)
        for norm in norms:
            if self.with_wandb: wandb.log({"output feature space norm (per input)": norm.item()})
            print(f"output feature space norm (per input): {norm.item()}")
            # print(f"features: {features[0].mean(dim=1)}, max: {features[0].max(dim=1)}, min: {features[0].min(dim=1)}")

        # NOTE
        # extract + permute dims for DP 
        lst_gt_extractions = []
        for i, feature in enumerate(features): # for each level of the deep supervision
            gt_extractions = [extract_task_set(feature, gt_seg[i], c, keep_dims=True).permute(1, 0) for c in target_classes[i]]
            lst_gt_extractions.append(gt_extractions)
        
        tc_inds = self.update_queues(lst_gt_extractions, target_classes, task_id, update_size=5, task_specific=True)
        # self.adjust_dist_params(tp_feats, task_id, tc_inds)

        # seg = self.full_segment(features[0], task_id)
        # features[0] = seg
        segs, segs_probs = [], []
        for i, feature in enumerate(features): # for each level of the deep supervision
            # features[i] = self.full_segment(feature, task_id)
            seg, seg_probs = self.full_segment(feature, task_id, return_probs=True)
            segs.append(seg)
            # if i != 0:
            #     seg_probs = feature
            segs_probs.append(seg_probs)

        # tc_inds = target_classes 
        # lst_gt_extractions = segs
        
        lst_gt_correct_extractions = []
        for i, feature in enumerate(features):
            gt_correct_extractions = [extract_correct_task_set(feature, gt_seg[i], c, segs[i], keep_dims=True).permute(1, 0) for c in target_classes[i]]
            lst_gt_correct_extractions.append(gt_correct_extractions)

        lst_gt_extractions = lst_gt_correct_extractions # NOTE
        tc_inds = self.update_queues(lst_gt_correct_extractions, target_classes, task_id, task_specific=True, update_size=self.q_update_size)
        
        # return features, lst_gt_extractions[0], tc_inds
        # return features, lst_gt_extractions, tc_inds
        return segs, lst_gt_extractions, tc_inds, segs_probs, features

    def forward_inference(self, x, task_id=None, gt_seg=None, target_classes=None, **kwargs):
        features, _, _, tp_feats, _ = super().forward(x, task_id, get_prompt=True)
        tc_inds = self.task_id_class_lst_mapping[task_id[0].item()]
        # self.adjust_dist_params(tp_feats, task_id, tc_inds)

        # features = features.permute(0, 2, 3, 4, 1)
        # features = F.normalize(features, p=2, dim=-1)
        # features = features.permute(0, 4, 1, 2, 3)
        
        if gt_seg is not None:
            # extract + permute dimensions for DP
            gt_extractions = [extract_task_set(features, gt_seg, c, keep_dims=True).permute(1, 0) for c in target_classes]

            return self.full_segment(features, task_id), gt_extractions

        # return self.full_segment(features, task_id)
        # return features
        seg, seg_prob = self.full_segment(features, task_id, return_probs=True)
        return seg_prob

    def update_queues(self, lst_extractions, lst_target_classes, task_id, update_size=10, task_specific=False):
        lst_tc_inds = []
        k = 0
        while k < len(lst_extractions):
        # for k, extractions in enumerate(lst_extractions):
            extractions = lst_extractions[k] # feature extractions at the k-th depth
            target_classes = lst_target_classes[k] # target classes at the k-th depth
            tc_inds = []
            i, j = 0, 0
            while i < len(extractions) and j < len(target_classes):
                # re-arrange dimensions for loss
                extractions[i] = extractions[i].permute(1, 0) # extreactions for the i-th class
                if extractions[i].size(-1) < 2:
                    _ = extractions.pop(i)
                else:
                    c = target_classes[j]
                    tc_inds.append(self.task_id_class_lst_mapping[int(task_id[0])][int(c)])
                    # tc_inds.append(self.task_id_class_lst_mapping[int(task_id[0])][int(c) - 1])
                    i += 1
                j += 1
            if len(tc_inds) > 0:
                lst_tc_inds.append(tc_inds)
                k+=1
            else:
                lst_extractions.pop(k) 

        # print(f"tc inds: {lst_tc_inds}")
        if sum([len(l) for l in lst_tc_inds]) == 0:
            return 0.

        # Updated Queues using the final feature space outputs
        # pick random element from extraction 
        # self.tap.update_queues(lst_extractions[0], lst_tc_inds[0], update_size=25)
        t = int(task_id[0])
        if task_specific:
            # self.task_taps[t].update_task_queue(t, lst_extractions[0], lst_tc_inds[0], update_size=update_size)
            self.task_taps[t].update_queues(lst_extractions[0], lst_tc_inds[0], update_size=update_size)
        else:
            self.tap.update_queues(lst_extractions[0], lst_tc_inds[0], update_size=update_size)

        # return lst_tc_inds[0]
        return lst_tc_inds

    def init_gmms(self, means, covs):
        # means, covs = self.dynamic_dist.get_mean_var()

        for i in range(self.num_classes):
            self.gaussian_mixtures[i].means_ = means[i].detach().cpu().numpy()
            self.gaussian_mixtures[i].covariances_ = covs[i].detach().cpu().numpy()#.item() * np.eye(self.feature_space_dim)

        self.gmm_fitted = True
    
    def gmm_analysis(self, feature_space_qs, mus, sigs):
        sizes = [0.10, 0.25, 0.50, 0.8, 0.9, 1]
        comp_errors = []
        for t in range(self.num_classes):
            task_queue = feature_space_qs[t] 
            if len(task_queue) < self.queue_min:
                continue
            size_errors = []
            task_queue = torch.vstack(task_queue).detach().cpu().numpy()
            for i, size in enumerate(sizes):
                N = int(task_queue.shape[0] * sizes[i])
                task_queue_ = task_queue[:N, :]
                gmm = GaussianMixture(1)
                gmm.fit(task_queue_)

                dist = np.linalg.norm(mus[t].detach().cpu().numpy() - gmm.means_)
                size_errors.append(dist)
                if wandb.run is not None:
                    wandb.log({f"comp_{t}_error_{size}": dist})
                    wandb.log({f"comp_{t}_size_{size}": N})

            comp_errors.append(size_errors)


    def train_gmms(self, feature_space_qs, mus, sigs):
        # if not self.gmm_fitted :
        #     self.init_gmms(mus, sigs)

        trained_indices = self.gmm_analysis(feature_space_qs, mus, sigs)
        # Train using queue
        trained_indices = []
        for t in range(self.num_classes):
            task_queue = feature_space_qs[t] 
            if len(task_queue) < self.queue_min:
                continue
            task_queue = torch.vstack(task_queue).detach().cpu().numpy()
            self.gaussian_mixtures[t].fit(task_queue)
            trained_indices.append(t)


        if self.with_wandb:
            wandb.log({"trained_inds": trained_indices})
        print(f"Trained inds: {trained_indices}")
        # Collect params and construct pytorch dist
        # est_means = torch.from_numpy(np.vstack([self.gaussian_mixtures[t].means_ for t in trained_indices]))
        # est_covs = torch.from_numpy(np.vstack([self.gaussian_mixtures[t].covariances_ for t in trained_indices]))
        est_weights = torch.from_numpy(np.vstack([self.gaussian_mixtures[t].weights_ for t in trained_indices]))
        lower_bounds = np.vstack([self.gaussian_mixtures[t].lower_bound_ for t in trained_indices])

        est_means = []
        component_distributions = []
        for t in trained_indices:
            mean_t = torch.from_numpy(self.gaussian_mixtures[t].means_)
            cov_t = torch.from_numpy(self.gaussian_mixtures[t].covariances_)
            print(f"Stat. diff {t}: {torch.norm(mus[t].detach().cpu() - mean_t)}")
            wandb.log({f"Stat. diff {t}": torch.norm(mus[t].detach().cpu() - mean_t)})
            # print(f"Est cov {t}: {cov_t}")
            if torch.cuda.is_available():
                mean_t = mean_t.cuda()
                cov_t = cov_t.cuda()
                if self.with_wandb:
                    wandb.log({
                        f'gmm_mean_{t}': mean_t,#.item(), 
                        f'gmm_var_{t}': cov_t,#.item(), 
                        f'gmm_lower_bound_{t}': self.gaussian_mixtures[t].lower_bound_,
                    })

                    est_means.append(mean_t)

            # if len(cov_t.shape) == 2:
            #     cov_t = cov_t.unsqueeze(0).repeat(mean_t.size(0), 1, 1)

            # component_distributions.append(MultivariateNormal(mean_t, cov_t))
            component_distributions.append(self.construct_torch_gmm(mean_t, cov_t))

        # Determine the KL w.r.t these EM-learned dists and targets
        est_dist, wrt_target = est_dist_sep_loss(est_means, self.min_dist, wrt_target=True, **{'mus': mus[trained_indices]})
        if self.with_wandb:
            wandb.log({'est_sep_dist':est_dist, 'sep_wrt_target': wrt_target})
            learned_target_kls = kl_divs(component_distributions, mus, sigs)
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
        if means.shape[0] == 1:
            return MultivariateNormal(means, vars)

        comp_dists = []
        for t in range(means.shape[0]):
            cov_t = vars[t]
            # if len(cov_t.shape) != 3:
            #     cov_t = cov_t.unsqueeze(0).repeat(means[t].size(0), 1, 1)
            
            comp_dists.append(MultivariateNormal(means[t], cov_t))

        categorical = Categorical(torch.ones(means.shape[0], device=means.device) / means.shape[0])

        # self.feature_space_gmm = GaussianMixtureModel(categorical, comp_dists)
        dist = GaussianMixtureModel(categorical, comp_dists)

        return dist

    def construct_task_feature_space_gmm_implicit(self, task):
        task_mus, task_covs, task_weights = self.tasks_mus[task], self.tasks_covs[task], self.tasks_weights[task]
        task_gmm_component_distributions = []
        # for i in range(self.num_classes): # i is class; idx for distribution representing class i of `task`
        num_classes = self.task_to_num_classes[task]
        for i in range(num_classes): # i is class; idx for distribution representing class i of `task`
            mu, cov = task_mus[i], task_covs[i]
            if mu.shape[0] == 1:
                dist =  MultivariateNormal(mu, cov)
                task_gmm_component_distributions.append(dist)
                continue

            comp_dists = []
            for t in range(mu.shape[0]):
                comp_dists.append(MultivariateNormal(mu[t], cov[t])) 

            categorical = Categorical(task_weights[i])

            # self.feature_space_gmm = GaussianMixtureModel(categorical, comp_dists)
            dist = GaussianMixtureModel(categorical, comp_dists)
            task_gmm_component_distributions.append(dist)

        # categorical = Categorical(torch.ones(self.num_classes, device=self.mus[0].device) / self.num_classes)
        # categorical = Categorical(torch.ones(self.num_classes, device=task_mus[0].device) / self.num_classes)
        categorical = Categorical(torch.ones(num_classes, device=task_mus[0].device) / num_classes)
        self.task_feature_space_gmm[task] = GaussianMixtureModel(categorical, task_gmm_component_distributions)
            

    def construct_feature_space_gmm_implicit(self):
        component_distributions = []
        for i in range(self.num_classes):
            mu, cov = self.mus[i], self.covs[i]
            if mu.shape[0] == 1:
                dist =  MultivariateNormal(mu, cov)
                component_distributions.append(dist)
                continue

            comp_dists = []
            for t in range(mu.shape[0]):
                
                comp_dists.append(MultivariateNormal(mu[t], cov[t]))

            # categorical = Categorical(torch.ones(mu.shape[0], device=mu.device) / mu.shape[0])
            categorical = Categorical(self.weights[i])

            # self.feature_space_gmm = GaussianMixtureModel(categorical, comp_dists)
            dist = GaussianMixtureModel(categorical, comp_dists)
            component_distributions.append(dist)

        # categorical = Categorical(torch.ones(self.num_classes, device=mus.device) / self.num_classes)
        categorical = Categorical(torch.ones(self.num_classes, device=self.mus[0].device) / self.num_classes)
        self.feature_space_gmm = GaussianMixtureModel(categorical, component_distributions)


    def construct_feature_space_gmm(self, mus, covs, task=None):
        if task is not None:
            num_classes = self.task_to_num_classes[task]
        else: 
            num_classes = self.num_classes 

        component_distributions = []
        for i in range(num_classes):
            mu, cov = mus[i], covs[i]
            component_distributions.append(self.construct_torch_gmm(mu, cov))

        # categorical = Categorical(torch.ones(self.num_classes, device=mus.device) / self.num_classes)
        categorical = Categorical(torch.ones(num_classes, device=mus[0].device) / num_classes)
        if task is not None:
            self.task_feature_space_gmm[task] = GaussianMixtureModel(categorical, component_distributions)
        else:
            self.feature_space_gmm = GaussianMixtureModel(categorical, component_distributions)

    def forward(self, x, task_id=None, for_loss=False, **kwargs):

        if self.training or for_loss:
            return self.forward_train(x, task_id=task_id, **kwargs)
        else :
            return self.forward_inference(x, task_id=task_id, **kwargs)

    def segment(self, features: torch.Tensor, component_indices=None, task=None, return_probs=False):
        b = features.size(0)
        h, w, d = tuple(features.shape[2:])
        bview_features = features.permute(0, 2, 3, 4, 1).reshape(b, -1, self.feature_space_dim)

        if task is not None:
            task = task[0].item()
            feature_log_probs = self.task_feature_space_gmm[task].score(bview_features, component_indices=component_indices)
        else:    
            feature_log_probs = self.feature_space_gmm.score(bview_features, component_indices=component_indices)
        
        # feature_log_probs = feature_log_probs.reshape(b, h, w, d, len(component_indices)).permute(0, 4, 1, 2, 3)
        feature_log_probs = feature_log_probs.reshape(b, h, w, d, len(component_indices)).permute(0, 4, 1, 2, 3) # len(component_indices) = num_classes for given task

        if len(component_indices) < self.num_classes:
            # add dummy
            difference = self.num_classes - len(component_indices)
            # dummy_dim = torch.zeros(b, 1, h, w, d, device=feature_log_probs.device)
            dummy_dim = torch.zeros(b, difference, h, w, d, device=feature_log_probs.device)
            feature_log_probs = torch.cat((feature_log_probs, dummy_dim), 1)
            # feature_log_probs = torch.cat((dummy_dim, feature_log_probs), 1)
        # segmentation = torch.argmax(feature_log_probs, dim=1)
        # segmentation = self.bayes(feature_log_probs)
        # segmentation = torch.argmax(F.softmax(feature_log_probs, dim=1), dim=1)
        segmentation = torch.argmax(feature_log_probs, dim=1)

        if return_probs:
            # for CE: 
            # input shape = (batch_size, num_classes, d1, d2, ..., dK), 
            # target shape = (batch_size, d1, d2, ..., dK) , contianing class indices
            return segmentation, feature_log_probs

        return segmentation

    def bayes(self, log_probs):
        scaled_log_probs = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)

        seg = torch.argmax(scaled_log_probs, dim=1)

        return seg

    def full_segment(self, x, task_id, return_probs=False):
        component_indices = self.task_id_class_lst_mapping[task_id[0].item()] if self.task_id_class_lst_mapping is not None else None
        # component_indices = None # self.task_id_class_lst_mapping[task_id[0].item()] if self.task_id_class_lst_mapping is not None else None
        seg_output = self.segment(x, component_indices=component_indices, task=task_id, return_probs=return_probs)
        if return_probs:
            std_output_segmentation, seg_probabilities = seg_output
        else :
            std_output_segmentation = seg_output
        
        # output_segmentation = self.segment(x, component_indices=component_indices)
        # std_output_segmentation = self.standardize_segmentation(output_segmentation, self.class_lst_to_std_mapping)
        # est_seg = torch.nn.functional.one_hot(std_output_segmentation, num_classes_in_gt).permute(0, 4, 1, 2, 3)
        perm_dims = (0, 4, 1, 2, 3) # tuple([0, len(x.shape)-1] + list(range(1, len(x.shape)-1)))
        # est_seg = torch.nn.functional.one_hot(std_output_segmentation, len(self.task_id_class_lst_mapping[int(task_id.item())])).permute(perm_dims)
        est_seg = F.one_hot(std_output_segmentation, self.num_classes)
        est_seg = torch.permute(est_seg, perm_dims).to(dtype=torch.float64)

        if return_probs:
            return est_seg, seg_probabilities

        return est_seg

    def component_wise_logp(self, x, task):
        """
        -- univariate -- 
        x shape: (Batch, Class, H, W, D)
        task: int
        """
        batch_size, channels, h, w, d = tuple(x.shape)
        cls_gmms = self.task_univar_feature_space_gmm[task]
        task_num_classes = self.task_to_num_classes[task]
        output_probs = []
        for batch_idx in range(batch_size):
            class_view_batch_features = x[batch_idx].reshape(task_num_classes, -1, channels) # feature space dim = 1
            batch_probs = []
            for class_idx in range(task_num_classes):
                cls_gmm = cls_gmms[class_idx]
                class_probs = cls_gmm.log_prob(class_view_batch_features[class_idx]) # (-1, self.feaure_space_dim)
                batch_probs.append(class_probs.reshape(h, w, d, self.feature_space_dim))
            batch_probs = torch.vstack(batch_probs) # (num_classes, h, w, d, self.feature_space_dim)
            output_probs.append(batch_probs)

        output_probs = torch.vstack(output_probs) # (b, num_classes, h, w, d, self.feature_space_dim)
        out = output_probs.permute(0, 4, 1, 2, 3)

        return out

    def calculate_task_ood_threshold(self, task_id):
        self.thresholds[task_id] = None
        task_gmm = self.task_feature_space_gmm[task_id]
        task_queues = self.task_taps[task_id].feature_space_qs
        # NOTE check shape of features
        correct_classification_probs_avg = []
        for class_idx in range(self.task_to_num_classes[task_id]):
            # unsqueeze to "batch"
            # need to apply softmax
            # class_probs = torch.vstack([task_gmm.score(sample.unsqueeze(0), component_indices=None)[..., class_idx] for sample in task_queues[class_idx]])
            class_probs = torch.vstack([F.softmax(task_gmm.score(sample.unsqueeze(0), component_indices=None), dim=1)[..., class_idx] for sample in task_queues[class_idx]])
            # avg_class_prob = torch.mean(torch.exp(class_probs))
            avg_class_prob = torch.mean(class_probs)
            print(f"Class {class_idx}, average prob: {avg_class_prob}")
            correct_classification_probs_avg.append(avg_class_prob.item())

        self.thresholds[task_id] = np.mean(correct_classification_probs_avg)

    def determine_ood(self, predicted_probabilities, task_id):
        task_id = int(task_id[0].item())
        if self.thresholds[task_id] is None:
            self.calculate_task_ood_threshold(task_id)
        ood_mask = predicted_probabilities < self.thresholds[task_id]
        segmentation_prediction = predicted_probabilities.argmax(0)
        segmentation_prediction[ood_mask.all(0)] = -1
        # anomaly_probabilities = 1 - np.amax(predicted_probabilities, 0)
        anomaly_probabilities = 1 - predicted_probabilities
        return segmentation_prediction, anomaly_probabilities

    def _internal_predict_3D_3Dconv_tiled(self, x: np.ndarray, task_id: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool):
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
                if verbose: print("done")
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

            #predict on cpu if cuda not available
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())

        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], task_id, mirror_axes, do_mirroring,
                        gaussian_importance_map)[0]

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        aggregated_results /= aggregated_nb_of_predictions
        del aggregated_nb_of_predictions

        if self.ood_detection_mode:
            predicted_segmentation, anomaly_probabilities = self.determine_ood(aggregated_results, task_id)
            return predicted_segmentation, (aggregated_results, anomaly_probabilities)

        if regions_class_order is None:
            predicted_segmentation = aggregated_results.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = aggregated_results.detach().cpu().numpy()
            else:
                class_probabilities_here = aggregated_results
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            aggregated_results = aggregated_results.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, aggregated_results