import torch
import torch.nn.functional as F
from monai.networks.blocks import Warp


def deformImWithModel_gpu(I_ref, R1, R2, s1, s2, device='cuda', def_xs = None, def_ys = None, def_zs = None):
    """
    Deform image I_ref using motion model (R1, R2) and surrogates (s1, s2).
    Returns (I_def, T_Z, T_Y, T_X) on GPU.
    """
    # D, H, W = I_ref.shape
    # # Build coordinate grids (float)
    # zs = torch.arange(D, device=device)
    # ys = torch.arange(H, device=device)
    # xs = torch.arange(W, device=device)
    # Zg, Yg, Xg = torch.meshgrid(zs, ys, xs, indexing='ij')  # int grid

    # # Compute displacement fields 
    # T_X = s1 * R1[...,2] + s2 * R2[...,2]
    # T_Y = s1 * R1[...,1] + s2 * R2[...,1]
    # T_Z = s1 * R1[...,0] + s2 * R2[...,0]

    # # Compute warped floating coordinates (float grid)
    # def_X = Xg.float() + T_X
    # def_Y = Yg.float() + T_Y
    # def_Z = Zg.float() + T_Z

    # # Normalize coordinates to [-1,1] for grid_sample
    # # Coordinates are in [0, W-1], [0,H-1], [0,D-1]; normalize
    # def_X_norm = 2.0 * def_X / (W-1) - 1.0
    # def_Y_norm = 2.0 * def_Y / (H-1) - 1.0
    # def_Z_norm = 2.0 * def_Z / (D-1) - 1.0

    # # Stack into grid shape (D,H,W,3)
    # grid = torch.stack((def_X_norm, def_Y_norm, def_Z_norm), dim=-1)
    # grid = grid.unsqueeze(0)  # add batch dim

    # # Prepare input for grid_sample: (N=1, C=1, D, H, W)
    # I_in = I_ref.unsqueeze(0).unsqueeze(0)
    # # Perform interpolation (trilinear, zeros outside)
    # # I_def = F.grid_sample(I_in, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    # I_real_def = F.grid_sample(I_in.real, grid, mode='bilinear', 
    #                           padding_mode='zeros', align_corners=True)[0,0]
    # I_imag_def = F.grid_sample(I_in.imag, grid, mode='bilinear', 
    #                           padding_mode='zeros', align_corners=True)[0,0]

    
    # I_def = torch.complex(I_real_def, I_imag_def)

    device = torch.device(device)
    
    D, H, W = I_ref.shape[:3]
    if (def_xs is None) != (def_ys is None):
        raise ValueError("Must specify both def_xs and def_ys or neither.")
    if def_xs is None:
        def_xs = torch.arange(W, device=device)
        def_ys = torch.arange(H, device=device)
        def_zs = torch.arange(D, device=device)
    else:
        def_xs = torch.tensor(def_xs, device=device)
        def_ys = torch.tensor(def_ys, device=device)
        def_zs = torch.tensor(def_zs, device=device)
        
        
    # Coordinate grid for region
    Zg, Yg, Xg = torch.meshgrid(def_zs, def_ys, def_xs, indexing='ij') 
    Zg = Zg.to("cuda")
    Yg = Yg.to("cuda")
    Xg = Xg.to("cuda")
    # sample the motion fields
    T_X = (s1 * R1[Zg, Yg, Xg, 2] +
           s2 * R2[Zg, Yg, Xg, 2])
    T_Y = (s1 * R1[Zg, Yg, Xg, 1] +
           s2 * R2[Zg, Yg, Xg, 1])
    T_Z = (s1 * R1[Zg, Yg, Xg, 0] +
           s2 * R2[Zg, Yg, Xg, 0])
    
    
    # 3) compute deformed coordinates
    def_X0 = Xg.float() + T_X
    def_Y0 = Yg.float() + T_Y
    def_Z0 = Zg.float() + T_Z
    
    
    # map_coordinates expects a shape (3, Npoints) array in the same
    #    order as the volume axes: (z_indices, y_indices, x_indices)
    def_X0 = 2.0 * def_X0 / (W-1) - 1.0
    def_Y0 = 2.0 * def_Y0 / (H-1) - 1.0
    def_Z0 = 2.0 * def_Z0 / (D-1) - 1.0    
    
    
    coords = torch.stack([
        def_X0,
        def_Y0,
        def_Z0
        
        
    ], dim=-1).unsqueeze(0)
    
    
    # sample the (complex) phantom volume
    I_in = I_ref.permute(2,1,0).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
    if torch.is_complex(I_in):
        Re = I_in.real
        Im = I_in.imag
    else:
        Re = I_in
        Im = torch.zeros_like(I_in)

    
    real_part = F.grid_sample(Re, coords, mode='bilinear', 
                              padding_mode='zeros', align_corners=True)[0,0]
    imag_part = F.grid_sample(Im, coords, mode='bilinear', 
                              padding_mode='zeros', align_corners=True)[0,0]   
    
    
    # wrp = Warp(mode="bilinear",
    #             padding_mode="zeros",
                
    #             ).to(device) 
    
    
    # real_part = wrp(image=I_in.real, ddf=coords)[0, 0]
    # imag_part = wrp(image=I_in.imag, ddf=coords)[0, 0]
    I_def = torch.complex(real_part, imag_part)
    I_def = I_def.permute(2,1,0)
    return I_def, T_Z, T_Y, T_X