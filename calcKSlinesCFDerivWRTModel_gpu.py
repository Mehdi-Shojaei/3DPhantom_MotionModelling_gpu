import torch
from deformImWithModel_gpu import deformImWithModel_gpu  

def calcKSlinesCFDerivWRTModel_gpu(diff_KS, I_ref, R1, R2, S1, S2, KS_lines, device='cuda'):
    """
    3D cost-function derivative w.r.t. motion model (R1, R2) using torch GPU.
    diff_KS: (D x H x N) complex torch tensor of k-space residuals.
    I_ref: (D x H x W) torch tensor.
    R1, R2: (D x H x W x 3) torch tensors.
    S1, S2: lists or 1D tensors of length N.
    KS_lines: list of x-indices (length N).
    Returns: (dC_by_dR1, dC_by_dR2) as torch tensors (D x H x W x 3).
    """
    D, H, W = I_ref.shape
    # Compute spatial gradients of I_ref on GPU
    I_ref_dz, I_ref_dy, I_ref_dx = torch.gradient(I_ref, dim=(0,1,2))
    dC_by_dR1 = torch.zeros((D, H, W, 3), dtype=torch.float32, device=device)
    dC_by_dR2 = torch.zeros_like(dC_by_dR1)

    # Loop over lines (could be parallelized)
    for n, ky in enumerate(KS_lines):
        # Embed diff_KS[:, :, n] into a 3D k-space volume
        F_diff = torch.zeros((D, H, W), dtype=torch.cfloat, device=device)
        F_diff[:, :, ky] = diff_KS[:, :, n]

        # Undo shift and iFFT on GPU
        F_un = torch.fft.ifftshift(F_diff, dim=(0,1,2))
        I_diff = torch.fft.ifftn(F_un)
        Rd = I_diff.real; Id = I_diff.imag

        # Warp gradient volumes forward (GPU)
        I_def_dx, _, _, _ = deformImWithModel_gpu(I_ref_dx, R1, R2, S1[n], S2[n])
        I_def_dy, _, _, _ = deformImWithModel_gpu(I_ref_dy, R1, R2, S1[n], S2[n])
        I_def_dz, _, _, _ = deformImWithModel_gpu(I_ref_dz, R1, R2, S1[n], S2[n])

        # Convert to real/imag components
        Rdx, I_dx = I_def_dx.real, I_def_dx.imag
        Rdy, I_dy = I_def_dy.real, I_def_dy.imag
        Rdz, I_dz = I_def_dz.real, I_def_dz.imag

        # Accumulate gradient w.r.t R1 and R2
        dC_by_dR1[..., 2] += (Rd*Rdx + Id*I_dx) * S1[n]
        dC_by_dR1[..., 1] += (Rd*Rdy + Id*I_dy) * S1[n]
        dC_by_dR1[..., 0] += (Rd*Rdz + Id*I_dz) * S1[n]

        dC_by_dR2[..., 2] += (Rd*Rdx + Id*I_dx) * S2[n]
        dC_by_dR2[..., 1] += (Rd*Rdy + Id*I_dy) * S2[n]
        dC_by_dR2[..., 0] += (Rd*Rdz + Id*I_dz) * S2[n]

    return dC_by_dR1, dC_by_dR2
