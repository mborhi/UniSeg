#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn
import torch.nn.functional as F


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l

class MixedDSLoss(nn.Module):
    def __init__(self, main_loss, ds_loss, weight_factors=None, dice_loss=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MixedDSLoss, self).__init__()
        self.weight_factors = weight_factors
        self.main_loss = main_loss
        self.ds_loss = ds_loss
        self.dice_loss = dice_loss

    def forward(self, outputs, targets, *args, **kwargs):
        assert isinstance(outputs, (tuple, list)), "x must be either tuple or list"
        assert isinstance(targets, (tuple, list)), "y must be either tuple or list"
        #if self.loss_type == "kl":
            # l =  self.loss(extractions, self.mus, self.sigs, tc_inds, return_est_dists=self.return_est_dists, with_sep_loss=False)
        # else:
        #     l =  self.loss(output, self.mus, self.sigs, target[0], tc_inds, pred_dists=extractions, return_est_dists=self.return_est_dists, with_sep_loss=False)
                
        depth = len(outputs)
        if self.weight_factors is None:
            weights = [1] * len(depth)
        else:
            weights = self.weight_factors

        main_l = self.main_loss(*args, **kwargs)
        if "return_est_dists" in kwargs and kwargs["return_est_dists"]:
            main_l, est_dists = main_l
        l = weights[0] * main_l
        for i in range(1, depth):
            if weights[i] != 0:
                print(f"outputs[{i}] for ds: {outputs[i].shape}")
                l += weights[i] * self.ds_loss(outputs[i], targets[i])
        if "return_est_dists" in kwargs and kwargs["return_est_dists"]:
            return l, est_dists
        return l

    def forward_dist_dc_ds(self, outputs, targets, features, lst_tc_inds, mus, sigs, **kwargs):
    # def forward_dist_dc_ds(self, outputs, targets, features, *args, **kwargs):
        assert isinstance(outputs, (tuple, list)), "x must be either tuple or list"
        assert isinstance(targets, (tuple, list)), "y must be either tuple or list"

        depth = len(outputs)
        if self.weight_factors is None:
            weights = [1] * len(depth)
        else:
            weights = self.weight_factors

        main_l = self.main_loss(features[0], mus, sigs, lst_tc_inds[0], **kwargs)
        dc_l = self.dice_loss(outputs[0], targets[0])
        if "return_est_dists" in kwargs and kwargs["return_est_dists"]:
            main_l, est_dists = main_l
        l = weights[0] * (main_l + dc_l)
        for i in range(1, depth):
            if weights[i] != 0:
                print(f"outputs[{i}] for ds: {outputs[i].shape}")
                ce_l = self.main_loss(features[i], mus, sigs, lst_tc_inds[i], **kwargs)
                if "return_est_dists" in kwargs and kwargs["return_est_dists"]:
                    ce_l, _  = ce_l
                l += weights[i] * (ce_l + self.dice_loss(outputs[i], targets[i]))
        if "return_est_dists" in kwargs and kwargs["return_est_dists"]:
            return l, est_dists
        return l



    def forward_ce_dc(self, outputs, targets, *args, **kwargs):
        assert isinstance(outputs, (tuple, list)), "x must be either tuple or list"
        assert isinstance(targets, (tuple, list)), "y must be either tuple or list"
        #if self.loss_type == "kl":
            # l =  self.loss(extractions, self.mus, self.sigs, tc_inds, return_est_dists=self.return_est_dists, with_sep_loss=False)
        # else:
        #     l =  self.loss(output, self.mus, self.sigs, target[0], tc_inds, pred_dists=extractions, return_est_dists=self.return_est_dists, with_sep_loss=False)
                
        depth = len(outputs)
        if self.weight_factors is None:
            weights = [1] * len(depth)
        else:
            weights = self.weight_factors

        main_l = self.main_loss(*args, **kwargs)
        dc_l = self.dice_loss(outputs[0], targets[0])
        if "return_est_dists" in kwargs and kwargs["return_est_dists"]:
            main_l, est_dists = main_l
        l = weights[0] * (main_l + dc_l)
        for i in range(1, depth):
            if weights[i] != 0:
                print(f"outputs[{i}] for ds: {outputs[i].shape}")
                l += weights[i] * self.ds_loss(outputs[i], targets[i])
        if "return_est_dists" in kwargs and kwargs["return_est_dists"]:
            return l, est_dists
        return l
    
    def forward_dist_ds(self, lst_extractions, mus, sigs, lst_tc_inds, return_est_dists=True, with_sep_loss=False):
        assert isinstance(lst_extractions, (tuple, list)), "x must be either tuple or list"
                
        depth = len(lst_extractions)
        if self.weight_factors is None:
            weights = [1] * len(depth)
        else:
            weights = self.weight_factors

        l = 0
        for i in range(0, depth):
            ds_l = self.ds_loss(lst_extractions[i], mus, sigs, lst_tc_inds[i], return_est_dists=True, with_sep_loss=False)
            if return_est_dists:
                ds_l, est_dists = ds_l
                if i == 0: est_dist = est_dists
            if weights[i] != 0:
                l += weights[i] * ds_l
        
        if return_est_dists:
            return l, est_dist
        return l
