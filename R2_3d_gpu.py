import torch

def compute_R2_3d_gpu(shape=128, frac=40.0, scale=(1.6, 1.3, 6.0), device='cuda'):
    """
    Compute a 3D R2 field on a regular grid (torch GPU).
    Returns a torch tensor of shape (shape,shape,shape,3).
    """
    # Coordinate grid on GPU
    coord = torch.arange(shape, device=device, dtype=torch.float32)
    Z, Y, X = torch.meshgrid(coord, coord, coord, indexing='ij')
    # Centers
    cx = (shape - 1) / 2.0
    cy = (shape - 1) / 2.0
    cz = float(shape - 1)
    sx, sy, sz = scale

    # Compute normalized distance (on GPU)
    dist = torch.sqrt(((X - cx)/sx)**2 + ((Y - cy)/sy)**2 + ((Z - cz)/sz)**2)
    # Mask inside radius
    mask = dist < frac

    # Build R2_x and other components (initialize zeros on GPU)
    R2_x = torch.zeros((shape, shape, shape), device=device)
    R2_x[mask] = 1.0 - dist[mask] / frac

    R2_y = -R2_x / 3.0
    R2_z =  R2_x * 2.0

    # Stack in (z, y, x) order on a new last axis
    R2_3d = torch.stack([R2_z, R2_y, R2_x], dim=-1)
    return R2_3d
