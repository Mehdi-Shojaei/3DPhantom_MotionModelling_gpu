import torch

def compute_R1_3d_gpu(phantom_size=128, device='cuda'):
    """
    Extend the 2D R1_local pattern into 3D by introducing
    variation along Z as well. Returns a torch tensor of shape
    (Z, Y, X, 3), with local R1 components in (x, y, z).
    """
    n = phantom_size
    # 1) Build 1D profiles along each axis (on GPU)
    yy = torch.linspace(-1.7, 0.3, n, device=device)   # shape (n,)
    xy = torch.ones(n, device=device)                 # shape (n,)
    half = n // 2
    xy[:half]  = torch.linspace(0.5, 1.0, half, device=device)
    xy[-half:] = torch.linspace(1.0, 0.5, half, device=device)
    yx = xy.clone()                                   # shape (n,)

    # X-axis profile
    xx = torch.linspace(-1.0, 1.0, n, device=device)

    # Z-axis profile (simple linear ramp)
    zz = torch.linspace(-1.0, 1.0, n, device=device)
    zy = torch.ones(n, device=device)
    zy[:half]  = torch.linspace(0.5, 1.0, half, device=device)
    zy[-half:] = torch.linspace(1.0, 0.5, half, device=device)

    # 2) Broadcast-multiply to get three 3D volumes (on GPU)
    # R1_x varies in Z by zy, in Y by yx, in X by xx
    R1_x = zy[:, None, None] * yx[None, :, None] * xx[None, None, :]
    # R1_y varies in Z by zy, in Y by yy, in X by xy
    R1_y = zy[:, None, None] * yy[None, :, None] * xy[None, None, :]
    # R1_z varies in Z by zz, in Y by yy, in X by xx (unity in Y,X)
    R1_z = zz[:, None, None] * torch.ones(n, device=device)[None, :, None] * \
           torch.ones(n, device=device)[None, None, :]

    # 3) Stack into final tensor (Z, Y, X, 3)
    R1_local = torch.stack([R1_z, R1_y, R1_x], dim=-1)
    return R1_local
