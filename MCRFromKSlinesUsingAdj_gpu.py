import torch
from deformImWithModelAdjoint_gpu import deformImWithModelAdjoint_gpu

def MCRFromKSlinesUsingAdj_gpu(KS_acq, R1, R2, S1, S2, KS_lines, anim=False):
    """
    GPU‐accelerated MCR from k‐space lines via adjoint.
    KS_acq : torch.complex64, shape (D,H,N)
    R1, R2 : torch.float32, shape (D,H,W,3)
    S1,S2  : lists or 1D tensors length N
    KS_lines: list of N ints
    anim   : bool → collects frames of central slice

    Returns I_rec (complex64, D,H,W) and frames list of HxW RGB arrays.
    """
    device = KS_acq.device
    D, H, N = KS_acq.shape
    W = R1.shape[2]

    # prepare output accumulators
    I_rec = torch.zeros((D,H,W), dtype=torch.complex64, device=device)
    frames = []

    for n, ky in enumerate(KS_lines):
        # put line back into k-space volume
        Fvol = torch.zeros((D,H,W), dtype=torch.complex64, device=device)
        Fvol[:,:,ky] = KS_acq[:,:,n]

        # undo shift + invert
        Fu = torch.fft.ifftshift(Fvol, dim=(0,1,2))
        Ivol = torch.fft.ifftn(Fu, dim=(0,1,2))

        # run adjoint warp + gather weights
        I_def, weights = deformImWithModelAdjoint_gpu(Ivol, R1, R2, float(S1[n]), float(S2[n]))

        # avoid division by zero
        w = weights.clone()
        w[w==0] = 1.0

        I_rec += I_def / w

        # optionally collect an animation frame
        if anim:
            zmid = D//2
            img = I_rec[zmid].abs()
            mx  = img.max()
            rgb = torch.stack([img/mx]*3, dim=-1).cpu().numpy()
            frames.append((rgb*255).astype('uint8'))

    return I_rec, frames
