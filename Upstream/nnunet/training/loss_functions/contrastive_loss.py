import torch
from torch import nn
import torch.nn.functional as F

class TaskPromptLoss(nn.Module):
    def __init__(self, scaling_weight=1.):
        """
        """
        super(TaskPromptLoss, self).__init__()
        self.scaling_weight = scaling_weight
        self.bce_loss = nn.BCELoss()
        self.cos_sim = nn.CosineSimilarity()

    def average_along_value(vol: torch.Tensor, msk:torch.Tensor, type='zeros'):
        vol_total = vol.sum(dim=(-3, -2, -1))
        msk_nonzero = msk.count_nonzero(dim=(-3, -2, -1))
        if type == 'zeros':
            msk_nonzero = (msk == 0).sum(dim=(-3, -2, -1))
        avg = vol_total / msk_nonzero
        return avg


    def forward(self, x_encoded, task_prompt, gt_mask):
        # Take the average in non zero region of the mask (tumor + organ)
        b_size = x_encoded.size(0)
        f_pos = self.average_along_value(x_encoded, gt_mask)
        f_neg = self.average_along_value(x_encoded, gt_mask, type='zeros')
        p_loss = self.bce_loss(self.cos_sim(f_pos, task_prompt).unsqueeze(1), torch.ones(b_size, 1, device=x_encoded.device))
        n_loss = self.bce_loss(self.cos_sim(f_neg, task_prompt).unsqueeze(1), torch.ones(b_size, 1, device=x_encoded.device))
        
        return self.scaling_weight * (p_loss + n_loss) 