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
    def __init__(self, main_loss, ds_loss, weight_factors=None):
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

        l = weights[0] * self.main_loss(*args, **kwargs)
        for i in range(1, depth):
            if weights[i] != 0:
                l += weights[i] * self.ds_loss(outputs[i], targets[i])
        return l
