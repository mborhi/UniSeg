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

class DynamicDistributionModel(nn.Module):
    def __init__(self, feature_space_dim, tp_dim, num_components, momentum):
        super(DynamicDistributionModel, self).__init__()
        self.feature_space_dim = feature_space_dim 
        self.num_components = num_components 
        self.tp_dim = tp_dim

        max_var = 0.01
        delta=1-(1e-03)
        min_dist = self.distance_bounds(k=num_components, delta=delta, max_var=max_var)
        unscaled_means = torch.arange(0, num_components*min_dist, min_dist, requires_grad=False)[:, None] # for feature dim == 1
        # scale to [-1, 1]^{feature_space_dim}
        unscaled_means_norms = torch.norm(unscaled_means, dim=-1)
        max_norm = torch.max(unscaled_means_norms)
        self.means = unscaled_means / max_norm
        
        self.vars = torch.full_like(self.means, fill_value=max_var, requires_grad=False)
        # self.vars = (unscaled_vars - 1e-05) / (1e-03 - 1e-05)

        self.min_dist = min_dist / max_norm.item()

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
        self.task_sigma_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear((feature_space_dim * num_components) + (1 * num_components) + tp_dim + 1, hidden_dim), 
                nn.ReLU(), 
                nn.Linear(hidden_dim, 1),
                nn.ReLU()
            )
            for t in range(num_components)
        ])

        self.momentum = momentum

    def get_mean_var(self):
        return self.means, self.vars


    def distance_bounds(self, k=None, max_var=None, delta=None, m=None, n=1, n_min=1):
        min_num_samples = 5000 * 28 * 28 # min(number of images per class k) * resolution 
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

    def forward(self, x, tc_inds, with_momentum_update=True):
        # Returns: updated means and vars
        
        self.means = self.means.detach().clone().to(device=x.device)
        self.vars = self.vars.detach().clone().to(device=x.device)

        for t, task_id in enumerate(tc_inds):
            if not isinstance(task_id, torch.Tensor):
                task_id = torch.tensor([task_id])[:, None].to(device=x.device)

            if self.means.device != x.device:
                self.means = self.means.to(device=x.device)
                self.vars = self.vars.to(device=x.device)
            
            input_means = self.means.permute(1, 0).repeat(x.size(0), 1).detach().to(device=x.device)
            input_vars = self.vars.permute(1, 0).repeat(x.size(0), 1).detach().to(device=x.device)
            task_id = task_id.repeat(x.size(0), 1)
            
            input = torch.cat((input_means, input_vars, task_id, x), -1)
            mu_hat_t = self.task_mu_modules[t](input).mean(0) # averaged along batch
            sigma_hat_t = self.task_sigma_modules[t](input).mean(0) # averaged along batch

            if with_momentum_update:
                updated_var_t = (1 - self.momentum) * sigma_hat_t + (self.momentum * self.vars[t]) + 0.0001 # for numerical stability
                self.vars[t] = updated_var_t 
                updated_mean_t = (1 - self.momentum) * mu_hat_t + (self.momentum * self.means[t])
                print(f"updated mean, var {t}: {updated_mean_t}, {updated_var_t}")
                self.means[t] = updated_mean_t 
            

        return self.means, self.vars


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
            self.seg_outputs.append(nn.Conv3d(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))
            # self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
            #                                 1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

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

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

        # Init module for getting embeddings 
        self.extractor = nn.ModuleDict({
            "identity": nn.Identity()
        })
        
        self.channel_reduction = nn.Conv3d(num_classes, 1, 1, 1, 0, 1, 1, seg_output_use_bias)
        self.final_tanh = nn.Tanh()


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
        task_prompt = self.extractor["identity"](task_prompt)
        if get_prompt:
            temp_x = x.detach().clone()
        x = torch.cat([x, task_prompt], dim=1)
        # print(x.size())

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        
        # final_out = self.final_tanh(seg_outputs[-1].mean(dim=1, keepdim=True))
        final_out = self.final_tanh(self.channel_reduction(seg_outputs[-1]))
        seg_outputs[-1] = final_out

        if get_prompt:
            return final_out, self.intermediate_prompt.detach().clone(), dynamic_prompt.detach().clone(), task_prompt.detach().clone(), temp_x

        if self._deep_supervision and self.do_ds:
            return list([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
    
    def forward_with_extraction(self, x, task_id):
        """
        Standard forward function, returns the task_prompt as well:
        ```
        task_prompt = torch.index_select(self.fusion_layer(torch.cat([x, now_prompt], dim=1)), 1, task_id[0])
        ```
        This method always returns the outputs of the Decoder as well, even with `self._deep_supervision` is False
        Returns:
        -------
        torch.Tensor([batch_size, len(task_id[0]), 4, 6, 6])
        """
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

        x = torch.cat([x, task_prompt], dim=1)
        # print(x.size())

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))


        return (list([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], 
                                                               seg_outputs[:-1][::-1])]), 
                task_prompt)

        if self._deep_supervision and self.do_ds:
            return list([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]



# class TaskPromptFeatureExtractor(nn.Module):
class TaskPromptFeatureExtractor(SegmentationNetwork):

    def __init__(self, feature_space_dim, feature_extractor, num_tasks, queue_size, momentum=0.90, *args, **kwargs):
        super().__init__()

        self.num_tasks = num_tasks
        self.feature_space_dim = feature_space_dim
        tp_dim = int(feature_extractor.intermediate_prompt.numel() / feature_extractor.num_class)
        self.dynamic_dist = DynamicDistributionModel(feature_space_dim, tp_dim, num_tasks, momentum)

        # self.feature_extractor = UniSeg_model(final_nonlin=lambda x: x, *args, **kwargs)
        self.feature_extractor = feature_extractor

        # TODO: rename to task prompt embeddings
        self.task_prompt_module_embeddings = None

        def get_embeddings_hook(module, input, output: torch.Tensor):
            
            self.task_prompt_module_embeddings = output.detach().clone().to(device=output.device).requires_grad_(False)
        
        self.get_embeddings_hook_handle = self.feature_extractor.extractor["identity"].register_forward_hook(get_embeddings_hook)

        self.queue_size = queue_size
    

        self.feature_space_qs = [[] for i in range(num_tasks)]
        self.gaussian_mixtures = [GaussianMixture(1) for i in range(num_tasks)]

        self.do_ds = False

        self.queue_min = queue_size // 10
        self.gmm_fitted = False


    def init_gmms(self):
        means, vars = self.dynamic_dist.get_mean_var()

        for i in range(self.num_tasks):
            self.gaussian_mixtures[i].means_ = means[i].detach().cpu().numpy()
            self.gaussian_mixtures[i].covariances_ = vars[i].detach().cpu().numpy().item() * np.eye(self.feature_space_dim)
    

    def train_gmms(self):
        # Train using queue
        trained_indices = []
        for t in range(self.num_tasks):
            task_queue = self.feature_space_qs[t] 
            if len(task_queue) <= self.queue_min:
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

            if len(cov_t.shape) == 2:
                cov_t = cov_t.unsqueeze(0).repeat(mean_t.size(0), 1, 1)

            component_distributions.append(MultivariateNormal(mean_t, cov_t))

        # https://discuss.pytorch.org/t/how-to-use-torch-distributions-multivariate-normal-multivariatenormal-in-multi-gpu-mode/135030/3
        # component_distributions = [
        #     MultivariateNormal(mean, covariance) for mean, covariance in zip(est_means, est_covs)
        # ]
        categorical = Categorical(est_weights)
        self.feature_space_gmm = GaussianMixtureModel(categorical, component_distributions)

        self.gmm_fitted = True
        # return est_means, est_covs, est_weights, lower_bounds
        return est_weights, lower_bounds

    def forward(self, x, task_id, tc_inds):
        if self.train:
            return self.forward_train(x, task_id, tc_inds)
        else :
            return self.forward_inference(x, task_id)


    def forward_train(self, x, task_id, tc_inds):
        features, _, _, _, _ = self.feature_extractor(x, task_id, get_prompt=True)

        # Validate that hook activated 
        if self.task_prompt_module_embeddings is not None:

            # Flatten
            flat_feat_extracts = self.task_prompt_module_embeddings.reshape(x.size(0), -1)

            mus, sigs = self.dynamic_dist(flat_feat_extracts, tc_inds, with_momentum_update=True)

        return features, mus, sigs

    def forward_inference(self, x, task_id):
        features, _, _, _, _ = self.feature_extractor(x, task_id, get_prompt=False)

        # Use GMM + Bayes' Rule 
        return self.segment(features)

    def get_distance_bounds(self):
        return self.dynamic_dist.min_dist

    def segment(self, features: torch.Tensor):

        b = features.size(0)
        bview_features = features.reshape(b, -1, self.feature_space_dim)

        segmentation = self.feature_space_gmm.classify(bview_features).reshape(features.size())

        return segmentation

    def standardize_segmentation(self, segmentation, tc_inds_to_cls):

        vals = torch.unique(segmentation)
        for val in vals:
            msk = segmentation == val
            segmentation[msk] = tc_inds_to_cls[val.item()]

        return segmentation
    
    def update_queues(self, feature_extracts, tc_inds):
        
        for i, task_id in enumerate(tc_inds):
            idxs = torch.randint(0, feature_extracts[i].shape[-1], (10,))
            rand_elements = feature_extracts[i][idxs]
            # Enqueue and Dequeue
            if len(self.feature_space_qs[task_id]) + 10 > self.queue_size:
                self.feature_space_qs[task_id].pop(0)
            
            self.feature_space_qs[task_id].append(rand_elements.reshape(-1, self.feature_space_dim))
