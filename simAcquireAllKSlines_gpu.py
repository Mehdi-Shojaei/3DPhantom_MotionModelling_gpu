import torch
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from deformImWithModel_gpu import deformImWithModel_gpu
import numpy as np 

def simAcquireAllKSlines_gpu(I_ref, R1, R2, S1, S2, KS_lines, noise=0, anim=False, save=False):
    device = I_ref.device
    D, H, W = I_ref.shape[:3]
    KS_idx = [int(x) for x in torch.tensor(KS_lines).flatten()]
    N = len(KS_idx)

    s1 = S1.detach().clone().to(device=device, dtype=torch.float32).flatten()
    s2 = S2.detach().clone().to(device=device, dtype=torch.float32).flatten()
    while s1.numel() < N:
        s1 = torch.cat((s1, s1))
        s2 = torch.cat((s2, s2))
    s1 = s1[:N]
    s2 = s2[:N]

    KS_acq = torch.zeros((D, H, N), dtype=torch.complex64, device=device)
    F_acq  = torch.zeros((D, H, W), dtype=torch.complex64, device=device)
    frames = []

    if anim:
        # plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plt.tight_layout()

    for n in tqdm(range(N)):

        I_def, _, _, _ = deformImWithModel_gpu(I_ref, R1, R2, float(s1[n].item()), float(s2[n].item()))
        F_def = torch.fft.fftn(I_def, dim=(0, 1, 2))
        F_def = torch.fft.fftshift(F_def, dim=(0, 1, 2))

        if noise:
            std_r = (noise / 100.) * torch.max(torch.abs(F_def.real))
            std_i = (noise / 100.) * torch.max(torch.abs(F_def.imag))
            noise_real = torch.randn_like(F_def.real) * std_r
            noise_imag = torch.randn_like(F_def.imag) * std_i
            F_def = F_def + noise_real + 1j * noise_imag

        ky = KS_idx[n]
        KS_acq[:, :, n] = F_def[:, :, ky]

        cnt = KS_idx.count(ky)
        F_acq[:, :, ky] += KS_acq[:, :, n] / cnt

        if anim:
            os.makedirs('frames', exist_ok=True)
            zmid = D // 2

            ax1.clear()
            ax1.imshow(torch.abs(I_def[zmid]).cpu().numpy(), cmap='gray')
            ax1.set_title(f"Deformed Image (z={zmid})")
            ax1.axis('off')

            ax2.clear()
            ax2.imshow(torch.log(torch.abs(F_def[zmid]).cpu() + 1e-6).numpy(), cmap='gray')
            ax2.axvline(ky - 0.5, color='r'); ax2.axvline(ky + 0.5, color='r')
            ax2.set_title("K-space (log|F|)")
            ax2.axis('off')

            ax3.clear()
            ax3.imshow(torch.log(torch.abs(KS_acq[zmid]).cpu() + 1e-6).numpy(), aspect='auto', cmap='gray')
            ax3.set_title("Acquired Lines")
            ax3.axis('off')

            ax4.clear()
            Fp = torch.fft.ifftshift(F_acq, dim=(0, 1, 2))
            recon = torch.fft.ifftn(Fp, dim=(0, 1, 2))
            ax4.imshow(torch.abs(recon[zmid]).cpu().numpy(), cmap='gray')
            ax4.set_title("Partial Recon")
            ax4.axis('off')

            if save:
                fig.savefig(f'frames/frame_{n:04d}.png', dpi=150)

            fig.canvas.draw()
            fig.canvas.flush_events()

            buf = fig.canvas.buffer_rgba()
            h, w = buf.shape[:2]
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            frame_rgb = arr[..., :3].copy()
            frames.append(frame_rgb)

    return KS_acq, frames
