import torch 
import torch.nn.functional as F

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