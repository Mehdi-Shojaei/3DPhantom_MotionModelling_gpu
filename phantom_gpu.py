import torch

def phantom3d_gpu(definition=None, n=128, device='cuda'):
    """
    3D ellipsoid phantom, with per-ellipsoid rotation.
    definition: Tensor of ellipsoid parameters (m x 10).
    n: volume size.
    """
   
    if definition is None:
        E = torch.tensor([
            [1140, 0.9,  0.6,   5,  0,   0, 0, 0, 0, 0],
            [320, 0.85, 0.55, 5, 0,   0, 0, 0, 0, 0],
            [1140, 0.75, 0.45, 5, 0,   0, 0, 0, 0, 0],
            [600, 0.2, 0.1,   0.75, 0.4,   0.18, -0.1, 20, 0, 0],
            [600, 0.2, 0.1,   0.75,-0.35,   0.18, -0.15,-20,-8,0],
            [2000, 0.05, 0.05, 5,   0,   0.5,  0,    0, 0, 0],
            [700, 0.35, 0.2,   1,  -0.45, -0.1,  0.2, 45, 2, 0],
            [850, 0.35, 0.1,   0.5,  0.25, -0.1,  0.7,-45, 2, 0],
            [800, 0.1,  0.1,   0.4,    0, -0.05,  0.5,  0,10, 5]
        ], dtype=torch.float32, device=device)
    else:
        E = torch.tensor(definition, dtype=torch.float32, device=device)

    if E.ndim != 2 or E.shape[1] != 10:
        raise ValueError("Need shape (m,10): [A,a,b,c,x0,y0,z0,phi,theta,psi]")

    # Create normalized grid (coords in [-1,1]) 
    coords = torch.linspace(-1, 1, n, device=device)
    Zg, Yg, Xg = torch.meshgrid(coords, coords, coords, indexing='ij')
    pts = torch.stack([Xg.flatten(), Yg.flatten(), Zg.flatten()], dim=1)  # (n^3, 3)
    vol = torch.zeros(n**3, dtype=torch.float32, device=device)

    deg2rad = torch.tensor(torch.pi/180.0, device=device)
    for ell in E:
        A, a, b, c, x0, y0, z0, phi, theta, psi = ell.tolist()
        # Build rotation matrices
        p = phi * deg2rad; t = theta * deg2rad; s = psi * deg2rad
        Rz = torch.tensor([[ torch.cos(p), -torch.sin(p), 0],
                           [ torch.sin(p),  torch.cos(p), 0],
                           [         0,           0,      1]], device=device)
        Ry = torch.tensor([[ torch.cos(t),  0, torch.sin(t)],
                           [         0,      1,         0],
                           [-torch.sin(t),  0, torch.cos(t)]], device=device)
        Rx = torch.tensor([[1,         0,          0],
                           [0, torch.cos(s), -torch.sin(s)],
                           [0, torch.sin(s),  torch.cos(s)]], device=device)
        R = Rx @ Ry @ Rz  # Combine rotations

        # Translate & rotate points
        centered = pts - torch.tensor([x0, y0, z0], device=device)
        primed = centered @ R.T  # shape (n^3, 3)

        # Ellipsoid test: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
        vals = (primed[:,0]/a)**2 + (primed[:,1]/b)**2 + (primed[:,2]/c)**2
        mask = vals <= 1.0
        vol[mask] = A  # set amplitude where inside ellipsoid

    return vol.reshape((n, n, n))
