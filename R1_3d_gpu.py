import torch

def compute_R1_3d_gpu(phantom_size=128, device='cuda'):
    """
    Compute a 3D R1 field on a regular grid.
    Returns a torch tensor of shape (shape,shape,shape,3).
    """
    n = phantom_size
    yy = torch.linspace(-1.7, 0.3, n, device=device)   
    xy = torch.ones(n, device=device)                
    half = n // 2
    xy[:half]  = torch.linspace(0.5, 1.0, half, device=device)
    xy[-half:] = torch.linspace(1.0, 0.5, half, device=device)
    yx = xy.clone()                                 
    xx = torch.linspace(-1.0, 1.0, n, device=device)

    zz = torch.linspace(-1.0, 1.0, n, device=device)
    zy = torch.ones(n, device=device)
    zy[:half]  = torch.linspace(0.5, 1.0, half, device=device)
    zy[-half:] = torch.linspace(1.0, 0.5, half, device=device)

    R1_x = zy[:, None, None] * yx[None, :, None] * xx[None, None, :]
    R1_y = zy[:, None, None] * yy[None, :, None] * xy[None, None, :]
    R1_z = zz[:, None, None] * torch.ones(n, device=device)[None, :, None] * \
           torch.ones(n, device=device)[None, None, :]

    # Stack into final tensor (Z, Y, X, 3)
    R1_local = torch.stack([R1_z, R1_y, R1_x], dim=-1)
    return R1_local
