import torch
import torch.nn.functional as F
from monai.networks.blocks import Warp


def deformImWithModel_gpu(I_ref, R1, R2, s1, s2, device='cuda', def_xs = None, def_ys = None, def_zs = None):
    """
    Deform image I_ref using motion model (R1, R2) and surrogates (s1, s2).
    Returns (I_def, T_Z, T_Y, T_X).
    """


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

    Zg, Yg, Xg = torch.meshgrid(def_zs, def_ys, def_xs, indexing='ij') 
    Zg = Zg.to("cuda")
    Yg = Yg.to("cuda")
    Xg = Xg.to("cuda")
    
    T_X = (s1 * R1[Zg, Yg, Xg, 2] +
           s2 * R2[Zg, Yg, Xg, 2])
    T_Y = (s1 * R1[Zg, Yg, Xg, 1] +
           s2 * R2[Zg, Yg, Xg, 1])
    T_Z = (s1 * R1[Zg, Yg, Xg, 0] +
           s2 * R2[Zg, Yg, Xg, 0])

    def_X0 = Xg.float() + T_X
    def_Y0 = Yg.float() + T_Y
    def_Z0 = Zg.float() + T_Z
  
    # print(f"Grid sample range: {def_X0.min().item():.3f} to {def_X0.max().item():.3f} for def_X0\
    #     Grid sample range: {def_Y0.min().item():.3f} to {def_Y0.max().item():.3f} for def_Y0\
    #     Grid sample range: {def_Z0.min().item():.3f} to {def_Z0.max().item():.3f} for def_Z0\
    #         They are all normalized to 0-1 range!\
    #         ______________________________________")
    
    def_X0 = 2.0 * def_X0 / (W-1) - 1.0
    def_Y0 = 2.0 * def_Y0 / (H-1) - 1.0
    def_Z0 = 2.0 * def_Z0 / (D-1) - 1.0  
    
    coords = torch.stack([
        def_X0,
        def_Y0,
        def_Z0
    ], dim=-1).unsqueeze(0)
    
    
    
    
    I_in = I_ref.unsqueeze(0).unsqueeze(0).to(device)  
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
    
    # ddf = torch.stack([T_Z, T_Y, T_X], dim=0).unsqueeze(0)
    # wrp = Warp(mode="bilinear",
    #             padding_mode="zeros",
                
    #             ).to(device) 
    
    
    # real_part = wrp(image=I_in.real, ddf=ddf)[0, 0]
    # imag_part = wrp(image=I_in.imag, ddf=ddf)[0, 0]
    
    
    I_def = torch.complex(real_part, imag_part)
    # I_def = I_def.permute(2,1,0)
    return I_def, T_Z, T_Y, T_X