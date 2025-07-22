import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simAcquireAllKSlines_gpu import simAcquireAllKSlines_gpu
from MCRFromKSlinesUsingAdj_gpu import MCRFromKSlinesUsingAdj_gpu
from calcKSlinesCFDerivWRTModel_gpu import calcKSlinesCFDerivWRTModel_gpu
from save_anim_gpu import save_animation_gpu
# from gaussian import gaussian_3D
import numpy as np
# from monai.transforms import GaussianSmooth
from scipy.ndimage import gaussian_filter


def fitModelAndNonIterMCRToKSlines_gpu(KS_acq, S1, S2, KS_lines,
                                   anim=True,
                                   lims_R1_x=None, lims_R1_y=None, lims_R1_z=None,
                                   lims_R2_x=None, lims_R2_y=None, lims_R2_z=None,
                                   I_size=None, num_lev=3, sigma=5,
                                   max_iter=1000,
                                   step_sizes_R=None, C_thresh=0.001):

    device = torch.device('cuda')

    S1_arr = S1.detach().clone().to(device=device, dtype=torch.float32).flatten()
    S2_arr = S2.detach().clone().to(device=device, dtype=torch.float32).flatten()
    N = KS_acq.shape[2]
    while S1_arr.numel() < N:
        S1_arr = torch.cat((S1_arr, S1_arr))
        S2_arr = torch.cat((S2_arr, S2_arr))
    S1_arr = S1_arr[:N]
    S2_arr = S2_arr[:N]

    D, H = KS_acq.shape[0], KS_acq.shape[1]
    W = I_size[2] if I_size is not None else H

    if step_sizes_R is None:
        step_sizes_R = [2.0/x for x in (2,4,8,16,32,64,128,256)]

    KS_acq = KS_acq.to(device)
    anim_frames = []
    if anim:
        # plt.ion()
        fig, axes = plt.subplots(2, 5, figsize=(10,6))
        gs = axes[0,0].get_gridspec()
        for ax in axes[:, :2].flatten():
            ax.remove()
        ax_img  = fig.add_subplot(gs[:, :2])
        ax_R1x  = fig.add_subplot(gs[0,2])
        ax_R1y  = fig.add_subplot(gs[0,3])
        ax_R1z  = fig.add_subplot(gs[0,4])
        ax_R2x  = fig.add_subplot(gs[1,2])
        ax_R2y  = fig.add_subplot(gs[1,3])
        ax_R2z  = fig.add_subplot(gs[1,4])
        plt.tight_layout()
    
    loss = []

    for lev in range(1, num_lev+1):
        print(f"\n=== Level {lev} ===============================")
        if lev == 1:
            R1_fit = torch.zeros((D, H, W, 3), device=device)
            R2_fit = torch.zeros((D, H, W, 3), device=device)
            I_rec, _ = MCRFromKSlinesUsingAdj_gpu(KS_acq, R1_fit, R2_fit, S1_arr, S2_arr, KS_lines, anim=True)

        sigma_level = sigma / 2**(lev-2)
        KS_sim, anim1 = simAcquireAllKSlines_gpu(I_rec, R1_fit, R2_fit, S1_arr, S2_arr, KS_lines, noise=0, anim=True)
        save_animation_gpu(anim1, f'I_rec_test_GPU_level_{lev}/anim.mp4')
        diff_KS = KS_acq - KS_sim
        C = torch.sum(diff_KS.real**2 + diff_KS.imag**2).item()
        print(f"[Level {lev}] initial loss C = {C:.6e}")


        if anim:
            mid_z = R1_fit.shape[0] // 2
            ax_img.clear(); ax_img.imshow((torch.dstack([torch.abs(I_rec[mid_z, :, :] / torch.abs(I_rec).max())]*3).cpu())); ax_img.axis('off')
            ax_R1x.clear(); ax_R1x.imshow((R1_fit[mid_z,:,:,2]).cpu(), vmin=lims_R1_x[0], vmax=lims_R1_x[1]); ax_R1x.axis('off'); ax_R1x.set_title("R1_fit X")
            
            ax_R1y.clear(); ax_R1y.imshow((R1_fit[mid_z,:,:,1]).cpu(), vmin=lims_R1_y[0], vmax=lims_R1_y[1]); ax_R1y.axis('off'); ax_R1y.set_title("R1_fit Y")
            
            ax_R1z.clear(); ax_R1z.imshow((R1_fit[mid_z,:,:,0]).cpu(), vmin=lims_R1_z[0], vmax=lims_R1_z[1]); ax_R1z.axis('off'); ax_R1z.set_title("R1_fit Z")
            
            ax_R2x.clear(); ax_R2x.imshow((R2_fit[mid_z,:,:,2]).cpu(), vmin=lims_R2_x[0], vmax=lims_R2_x[1]); ax_R2x.axis('off'); ax_R2x.set_title("R2_fit X")
            
            ax_R2y.clear(); ax_R2y.imshow((R2_fit[mid_z,:,:,1]).cpu(), vmin=lims_R2_y[0], vmax=lims_R2_y[1]); ax_R2y.axis('off'); ax_R2y.set_title("R2_fit Y")
            
            ax_R2z.clear(); ax_R2z.imshow((R2_fit[mid_z,:,:,0]).cpu(), vmin=lims_R2_z[0], vmax=lims_R2_z[1]); ax_R2z.axis('off'); ax_R2z.set_title("R2_fit Z")
            
            
            fig.suptitle(f"Level {lev} - Initial Fit")
            fig.savefig(f'I_rec_test_GPU_level_{lev}/Rs_I.png')
            fig.canvas.draw()
            buf, (w, h) = fig.canvas.print_to_buffer()
            img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            img_rgb  = img_rgba[:, :, :3]
            anim_frames.append(img_rgb)
            
            plt.close(fig)


        iter_count = 0
        C_prev = 2 * C

        while (C_prev - C) > C_prev*C_thresh:
            print(f"\n[Level {lev}] Iteration {iter_count}")
            print(f"  -> current loss C = {C:.6e}")
            iter_count += 1
            if iter_count > max_iter:
                print("  !! Reached max iterations")
                break

            C_prev = C
            loss.append(C)

            dC_dR1, dC_dR2 = calcKSlinesCFDerivWRTModel_gpu(diff_KS, I_rec, R1_fit, R2_fit, S1_arr, S2_arr, KS_lines)
            dC_dR1[torch.isnan(dC_dR1)] = 0
            dC_dR2[torch.isnan(dC_dR2)] = 0
            
            
            if sigma_level > 0:
                dC_dR1 = dC_dR1.detach().cpu().numpy()
                dC_dR2 = dC_dR2.detach().cpu().numpy()
                for c in range(3):
                    dC_dR1[:,:,:,c] = gaussian_filter(dC_dR1[:,:,:,c], sigma_level)
                    dC_dR2[:,:,:,c] = gaussian_filter(dC_dR2[:,:,:,c], sigma_level)
            # Normalize
                    # gauss = GaussianSmooth(sigma=sigma_level, approx='scalespace')
                    # dC_dR1[..., c] = gauss(dC_dR1[..., c].unsqueeze(0)).squeeze()
                    # dC_dR2[..., c] = gauss(dC_dR2[..., c].unsqueeze(0)).squeeze()
                    # dC_dR1[..., c] = gaussian_3D(dC_dR1[..., c].unsqueeze(0).unsqueeze(0), kernel_size=5 , sigma= sigma_level).squeeze()
                    # dC_dR2[..., c] = gaussian_3D(dC_dR2[..., c].unsqueeze(0).unsqueeze(0), kernel_size=5 , sigma= sigma_level).squeeze()
                dC_dR1 = torch.from_numpy(dC_dR1).to(device)
                dC_dR2 = torch.from_numpy(dC_dR2).to(device)
            maxg = max(dC_dR1.abs().max().item(), dC_dR2.abs().max().item())
            if maxg != 0:
                dC_dR1 /= maxg
                dC_dR2 /= maxg

            for step in step_sizes_R:
                print(f"   Testing step size {step}")
                improvement = True
                while improvement:
                    R1_new = R1_fit + step * dC_dR1
                    R2_new = R2_fit + step * dC_dR2
                    KS_sim_new, anim2 = simAcquireAllKSlines_gpu(I_rec, R1_new, R2_new, S1_arr, S2_arr, KS_lines, noise=0, anim=True)
                    
                    save_animation_gpu(anim2, f'I_rec_test_GPU_level_{lev}/anim_{step}.mp4')
                    diff_new = KS_acq - KS_sim_new
                    C_new = torch.sum(diff_new.real**2 + diff_new.imag**2).item()
                    if C_new < C:
                        C = C_new
                        R1_fit = R1_new
                        R2_fit = R2_new
                        diff_KS = diff_new
                        print(f"     -> improved! new loss = {C:.6e}")
                    else:
                        improvement = False
                        print("\033[1;31mNo improvement in this iteration\033[0m")
                        print(f"   Improvement below threshold: ΔC = {C_prev-C:.6e}")
                        break

            if C < C_prev:
                print("  Reconstructing image with updated motion...")
                I_rec, _ = MCRFromKSlinesUsingAdj_gpu(KS_acq, R1_fit, R2_fit, S1_arr, S2_arr, KS_lines, anim=False)
                KS_sim, anim3 = simAcquireAllKSlines_gpu(I_rec, R1_fit, R2_fit, S1_arr, S2_arr, KS_lines, noise=0, anim=True)
                save_animation_gpu(anim3, f'I_rec_test_GPU_level_{lev}/anim_updated_I_rec.mp4')
                diff_KS = KS_acq - KS_sim
                C = torch.sum(diff_KS.real**2 + diff_KS.imag**2).item()
            else:
                print("The image is not reconstructed nor updated dude!")
            print(f"  -> post-recon loss = {C:.6e}")

        print(f"[Level {lev}] Final loss = {C:.6e}")

    return (I_rec, R1_fit, R2_fit, anim_frames, loss)
