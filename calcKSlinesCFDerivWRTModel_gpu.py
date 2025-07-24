import torch
from deformImWithModel_gpu import deformImWithModel_gpu  

def calcKSlinesCFDerivWRTModel_gpu(diff_KS, I_ref, R1, R2, S1, S2, KS_lines, lambda_reg:float=0.01, device='cuda'):
    """
    3D cost-function derivative w.r.t. motion model (R1, R2).
    diff_KS: (D x H x N) complex torch tensor of k-space residuals.
    I_ref: (D x H x W) torch tensor.
    R1, R2: (D x H x W x 3) torch tensors.
    S1, S2: lists or 1D tensors of length N.
    lambda_reg_mag : float, optional
    Weight of the L2 magnitude regularization term: lambda * (||R1||² + ||R2||²).
    Default: 1e-4.
    KS_lines: list of x-indices (length N).
    Returns: (dC_by_dR1, dC_by_dR2) as torch tensors (D x H x W x 3).
    """
    D, H, W = I_ref.shape
    # spatial gradients of I_ref
    I_ref_dz, I_ref_dy, I_ref_dx = torch.gradient(I_ref, dim=(0,1,2))
    dC_by_dR1 = torch.zeros((D, H, W, 3), dtype=torch.float32, device=device)
    dC_by_dR2 = torch.zeros_like(dC_by_dR1)

    for n, ky in enumerate(KS_lines):
        
        # dP/dI . dS/dP ==> A*(dS/dP)
        F_diff = torch.zeros((D, H, W), dtype=torch.cfloat, device=device)
        # F_diff[:, :, ky] = diff_KS[:, :, n] * -2 # not sure about this ds/dM = -2Diff_kspace
        F_diff[:, :, ky] = diff_KS[:, :, n]
        
        F_un = torch.fft.ifftshift(F_diff, dim=(0,1,2))
        I_diff = torch.fft.ifftn(F_un) 
        Rd = I_diff.real * -2; Id = I_diff.imag * -2 # not sure about this ds/dM = -2Diff_kspace

        # dI/dM
        I_def_dx, _, _, _ = deformImWithModel_gpu(I_ref_dx, R1, R2, S1[n], S2[n])
        I_def_dy, _, _, _ = deformImWithModel_gpu(I_ref_dy, R1, R2, S1[n], S2[n])
        I_def_dz, _, _, _ = deformImWithModel_gpu(I_ref_dz, R1, R2, S1[n], S2[n])

        Rdx, I_dx = I_def_dx.real, I_def_dx.imag
        Rdy, I_dy = I_def_dy.real, I_def_dy.imag
        Rdz, I_dz = I_def_dz.real, I_def_dz.imag

        # accumulate gradient w.r.t R1 and R2 - dM/dR [dI/dM ∘ A*(-2diff_kspace) (1-lambda) + 2 lambda Reg]
        
        if lambda_reg > 0:
            dC_by_dR1[..., 2] += (Rd*Rdx*(1-lambda_reg) + Id*I_dx*(1-lambda_reg) + 2*lambda_reg*R1[..., 2]) * S1[n]
            dC_by_dR1[..., 1] += (Rd*Rdy*(1-lambda_reg) + Id*I_dy*(1-lambda_reg) + 2*lambda_reg*R1[..., 1]) * S1[n]
            dC_by_dR1[..., 0] += (Rd*Rdz*(1-lambda_reg) + Id*I_dz*(1-lambda_reg) + 2*lambda_reg*R1[..., 0]) * S1[n]

            dC_by_dR2[..., 2] += (Rd*Rdx*(1-lambda_reg) + Id*I_dx*(1-lambda_reg) + 2*lambda_reg*R2[..., 2]) * S2[n]
            dC_by_dR2[..., 1] += (Rd*Rdy*(1-lambda_reg) + Id*I_dy*(1-lambda_reg) + 2*lambda_reg*R2[..., 1]) * S2[n]
            dC_by_dR2[..., 0] += (Rd*Rdz*(1-lambda_reg) + Id*I_dz*(1-lambda_reg) + 2*lambda_reg*R2[..., 0]) * S2[n]
            
            

        else:
            dC_by_dR1[..., 2] += (Rd*Rdx + Id*I_dx) * S1[n]
            dC_by_dR1[..., 1] += (Rd*Rdy + Id*I_dy) * S1[n]
            dC_by_dR1[..., 0] += (Rd*Rdz + Id*I_dz) * S1[n]

            dC_by_dR2[..., 2] += (Rd*Rdx + Id*I_dx) * S2[n]
            dC_by_dR2[..., 1] += (Rd*Rdy + Id*I_dy) * S2[n]
            dC_by_dR2[..., 0] += (Rd*Rdz + Id*I_dz) * S2[n]
            
            
    return dC_by_dR1, dC_by_dR2
