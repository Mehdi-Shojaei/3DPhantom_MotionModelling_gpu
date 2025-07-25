import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simAcquireAllKSlines_gpu import simAcquireAllKSlines_gpu
from MCRFromKSlinesUsingAdj_gpu import MCRFromKSlinesUsingAdj_gpu
from calcKSlinesCFDerivWRTModel_gpu import calcKSlinesCFDerivWRTModel_gpu
from save_anim_gpu import save_animation_gpu
from gaussian import gaussian_blur_3d
import numpy as np
# from monai.transforms import GaussianSmooth
# from scipy.ndimage import gaussian_filter
from utils import reg_magnitude, huber_loss
from display_Rs import display
import SimpleITK as sitk

def fitModelAndNonIterMCRToKSlines_gpu(KS_acq, S1, S2, KS_lines, step_test=True,
                                   anim=True, lambda_reg_mag = 1e-4,
                                   lims_R1_x=None, lims_R1_y=None, lims_R1_z=None,
                                   lims_R2_x=None, lims_R2_y=None, lims_R2_z=None,
                                   I_size=None, num_lev=3, sigma=5,
                                   max_iter=100, loss_f = None,
                                   step_sizes_R=None, C_thresh=0.001):

    """
    Fit a two-surrogate, 3D motion model to acquired k-space lines and reconstruct a
    motion-compensated reference image using a multi-resolution, gradient-descent scheme.

    The routine alternates between:
      1) Forward simulation of k-space lines from the current image and motion model.
      2) Computing a data loss (L1/L2/Huber) plus optional L2 magnitude regularization.
      3) Computing descent directions for the motion fields (R1, R2) via an adjoint-based
         gradient.
      4) Line-search updates of R1/R2.
      5) Reconstructing the reference image with the updated motion.
    This is repeated over multiple levels (coarse → fine) with Gaussian-smoothed
    gradients.

    Parameters
    ----------
    KS_acq : torch.Tensor (D, H, N) or (D, H, W_lines)
        Complex acquired k-space lines (Fourier samples) stacked along the last dimension.
        D, H are k-space plane dimensions; N is the number of acquired lines/time points.
    S1, S2 : torch.Tensor (N,) or broadcastable to N
        Surrogate signals for each acquired line. They are flattened internally and
        repeated if needed to match N.
    KS_lines : sequence[int]
        Indices (ky positions) of the acquired k-space lines corresponding to each time point.
    anim : bool, optional
        If True, saves intermediate animations of the reconstruction
        and motion fields at each level/iteration. Default: True.
    lambda_reg_mag : float, optional
        Weight of the L2 magnitude regularization term: lambda * (||R1||² + ||R2||²).
        Default: 1e-4.
    lims_R1_x, lims_R1_y, lims_R1_z, lims_R2_x, lims_R2_y, lims_R2_z : tuple(float, float) or None
        Display ranges for the motion components when `anim=True`. If None, autoscaling is used.
    I_size : tuple(int, int, int) or None
        Desired (D, H, W) size of the reconstructed image. If None, W defaults to H.
    num_lev : int, optional
        Number of pyramid levels. Default: 3.
    sigma : float, optional
        Base Gaussian smoothinga sigma applied to motion gradients. Effective sigma is
        scaled per level (larger at coarse levels). Default: 5.
    max_iter : int, optional
        Maximum iterations per level. Default: 100.
    loss_f : {"L2", "L1", "HuberLoss"} or None
        Data-term loss function in k-space:
            "L2"        -> sum(|diff|^2) --> P.S.: Now only L2 works. The rest need more modification to the code.
            "L1"        -> sum(|diff|)
            "HuberLoss" -> huber_loss(|diff|) (see utils.huber_loss)
        If None, you must set it before calling; a ValueError is raised otherwise.
    step_sizes_R : list[float] or None
        Step sizes for the line search. If None, a default geometric list is used.
    C_thresh : float, optional
        Relative convergence threshold per level.
        

    Returns
    -------
    I_rec : torch.Tensor (D, H, W), complex
        Final reconstructed reference image after motion compensation.
    R1_fit : torch.Tensor (D, H, W, 3), float
        Fitted first motion parameter (Z,Y,X components in the last dim).
    R2_fit : torch.Tensor (D, H, W, 3), float
        Fitted second motion parameter (Z,Y,X components).
    anim_frames : list[np.ndarray]
        RGB frames captured during optimization (only if `anim=True`), for later video saving.
    loss : list[float]
        History of cost values across iterations and levels.
    """

    device = torch.device('cuda')
    loss_func = {
                "L2": lambda x: torch.sum(x.real**2 + x.imag**2),
                "L1": lambda x: torch.sum(torch.abs(x)),
                "HuberLoss": lambda x: huber_loss(torch.abs(x))
                }
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

    if step_sizes_R is None and step_test is False:
        step_sizes_R = [2.0/x for x in (2,4,8,16,32,64,128,256)]

    KS_acq = KS_acq.to(device)
    anim_frames = []
    loss = []
    
    R1_fit = torch.zeros((D, H, W, 3), device=device)
    R2_fit = torch.zeros((D, H, W, 3), device=device)
    
    if step_test:
        m_R1 = torch.zeros_like(R1_fit, device=device)
        m_R2 = torch.zeros_like(R1_fit, device=device)
        v_R1 = torch.zeros_like(R1_fit, device=device)
        v_R2 = torch.zeros_like(R1_fit, device=device)
        lr=0.001 
        beta1=0.9 
        beta2=0.999 
        eps=1e-8 
        step_sizes_R=1000
        
    for lev in range(1, num_lev+1):
        print(f"\n=== Level {lev} ===============================")
        if lev == 1:

            I_rec, _ = MCRFromKSlinesUsingAdj_gpu(KS_acq, R1_fit, R2_fit, S1_arr, S2_arr, KS_lines, anim=False)
            sigma_level = sigma*(2**(num_lev-2))
                      
        else:
            sigma_level = sigma/(2**(lev))
        # sigma_level = sigma / 2**(lev-2)
        KS_sim, anim1 = simAcquireAllKSlines_gpu(I_rec, R1_fit, R2_fit, S1_arr, S2_arr, KS_lines, noise=0, anim=anim)
        if anim:
            save_animation_gpu(anim1, f'I_rec_test_GPU_level_{lev}/anim.mp4')
        
        iter_count = 0
        if anim:
            
            RGB = display(I_rec, R1_fit, R2_fit,
                    lims_R1_x, lims_R1_y, lims_R1_z,
                    lims_R2_x, lims_R2_y, lims_R2_z,
                    lev, 0, iter_count,
                    save_dir='I_rec_test_GPU_level')
            anim_frames.append(RGB)

        diff_KS = KS_acq - KS_sim
        
        
        if lambda_reg_mag > 0:
            reg_loss = lambda_reg_mag * reg_magnitude(R1_fit, R2_fit)
            
            try:
                C = (1 - lambda_reg_mag) * loss_func[loss_f](diff_KS)
            except KeyError as exc:
                raise ValueError("loss_f must be 'L1', 'L2' or 'HuberLoss'") from exc
            
            # if loss_f == "L2":
            #     C = (1 - lambda_reg_mag) * torch.sum(diff_KS.real**2 + diff_KS.imag**2).item()
            # elif loss_f == "L1":
            #     C = (1 - lambda_reg_mag) * torch.sum(torch.abs(diff_KS)).item()
            # else:
            #     print("You should choose either L1 or L2!")
            C += reg_loss
        else:
            # C = torch.sum(diff_KS.real**2 + diff_KS.imag**2).item()
            try:
                C = loss_func[loss_f](diff_KS)
            except KeyError as exc:
                raise ValueError("loss_f must be 'L1', 'L2' or 'HuberLoss'") from exc
        
        print(f"[Level {lev}] initial loss C = {C:.6e}")
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

            dC_dR1, dC_dR2 = calcKSlinesCFDerivWRTModel_gpu(diff_KS, I_rec, R1_fit, R2_fit, S1_arr, S2_arr, KS_lines, lambda_reg=lambda_reg_mag)
            dC_dR1[torch.isnan(dC_dR1)] = 0
            dC_dR2[torch.isnan(dC_dR2)] = 0
            
            # if lambda_reg_mag > 0:
            #     dC_dR1 = (1 - lambda_reg_mag) * dC_dR1 - 2.0 * lambda_reg_mag * R1_fit
            #     dC_dR2 = (1 - lambda_reg_mag) * dC_dR2 - 2.0 * lambda_reg_mag * R2_fit
            
            
            # if step_test:
            #     # Based on gradient norm
            #     norm_grad = torch.norm(dC_dR1) + torch.norm(dC_dR2)
            #     alpha_base = 1e+9/norm_grad  # starting scale
            #     step_sizes_R = [alpha_base / (2 ** i) for i in range(1,5)]
                

            if sigma_level > 0:
                for c in range(3):
                    dC_dR1[:,:,:,c] = gaussian_blur_3d(dC_dR1[:,:,:,c], sigma = sigma_level)
                    dC_dR2[:,:,:,c] = gaussian_blur_3d(dC_dR2[:,:,:,c], sigma = sigma_level)
            maxg = max(dC_dR1.abs().max().item(), dC_dR2.abs().max().item())
            if maxg != 0:
                dC_dR1 /= maxg
                dC_dR2 /= maxg

            # for step in step_sizes_R: # commented for the sake of Adam
            for step in range(len(step_sizes_R)):
               
                print(f"   Testing step size {step_sizes_R[step]}")
                improvement = True
                while improvement:
                    
                    if step_test:
                        m_R1 = beta1 * m_R1 + (1 - beta1) * dC_dR1 #first momentum (mean). EMA
                        m_R2 = beta1 * m_R2 + (1 - beta1) * dC_dR2 
                        
                        v_R1 = beta2 * v_R1 + (1 - beta2) * dC_dR1 ** 2 # second momentum (var)
                        v_R2 = beta2 * v_R2 + (1 - beta2) * dC_dR2 ** 2
                        
                        m_R1_hat = m_R1 / (1 - beta1 ** step_sizes_R[step]) # inital bias
                        m_R2_hat = m_R2 / (1 - beta1 ** step_sizes_R[step])
                        
                        v_R1_hat = v_R1 / (1 - beta2 ** step_sizes_R[step])
                        v_R2_hat = v_R2 / (1 - beta2 ** step_sizes_R[step])
                        
                        R1_new = R1_fit - lr * m_R1_hat / (torch.sqrt(v_R1_hat) + eps)
                        R2_new = R2_fit - lr * m_R2_hat / (torch.sqrt(v_R2_hat) + eps)
                        
                    else:
                        R1_new = R1_fit - step_sizes_R[step] * dC_dR1 # changed to negative since -2diff is already accounted for
                        R2_new = R2_fit - step_sizes_R[step] * dC_dR2
                        
                        
                    # restrict if needed just for testing
                    # R1_new = torch.clamp(R1_new, -50, 50)
                    # R2_new = torch.clamp(R2_new, -50, 50)
                    
                    
                    KS_sim_new, anim2 = simAcquireAllKSlines_gpu(I_rec, R1_new, R2_new, S1_arr, S2_arr, KS_lines, noise=0, anim=anim)
                    if anim:
                        save_animation_gpu(anim2, f'I_rec_test_GPU_level_{lev}/anim_{step_sizes_R[step]}.mp4')
                    diff_new = KS_acq - KS_sim_new
                    
                    
                    if lambda_reg_mag > 0:
                        reg_loss = lambda_reg_mag * reg_magnitude(R1_new, R2_new)
                        
                        try:
                            C_new = (1 - lambda_reg_mag) * loss_func[loss_f](diff_new)
                        except KeyError as exc:
                            raise ValueError("loss_f must be 'L1', 'L2' or 'HuberLoss'") from exc
                        # C_new = (1 - lambda_reg_mag) * torch.sum(diff_new.real**2 + diff_new.imag**2).item()
                        C_new += reg_loss
                    else:
                        # C_new = torch.sum(diff_new.real**2 + diff_new.imag**2).item()
                        try:
                            C_new = loss_func[loss_f](diff_KS)
                        except KeyError as exc:
                            raise ValueError("loss_f must be 'L1', 'L2' or 'HuberLoss'") from exc
                    loss.append(C_new)
                    print(f"R step size: {step_sizes_R[step]:.4f}, New cost function: {C_new:g}")

                    if C_new < C:
                        C = C_new
                        R1_fit = R1_new
                        R2_fit = R2_new
                        diff_KS = diff_new
                        print(f"     -> improved! new loss = {C:.6e}")
                        
                        if anim:
                            RGB = display(I_rec, R1_fit, R2_fit,
                                    lims_R1_x, lims_R1_y, lims_R1_z,
                                    lims_R2_x, lims_R2_y, lims_R2_z,
                                    lev, step_sizes_R[step], iter_count,
                                    save_dir='I_rec_test_GPU_level')
                            anim_frames.append(RGB)
                    else:
                        improvement = False
                        # print("\033[1;31mNo improvement in this iteration\033[0m")
                        # print(f"   Improvement below threshold: ΔC = {C_prev-C:.6e}")
                        
                        break

            if C < C_prev:
                print("  Reconstructing image with updated motion...")
                I_rec, _ = MCRFromKSlinesUsingAdj_gpu(KS_acq, R1_fit, R2_fit, S1_arr, S2_arr, KS_lines, anim=False)
                I_rec_mag = np.abs(I_rec.cpu())
                
                I_rec_mag_img = sitk.GetImageFromArray(I_rec_mag)
                sitk.WriteImage(I_rec_mag_img, f"I_rec_Lev{lev}_It{iter_count}.nii.gz")
                
                
                
                KS_sim, anim3 = simAcquireAllKSlines_gpu(I_rec, R1_fit, R2_fit, S1_arr, S2_arr, KS_lines, noise=0, anim=anim)
                
                if anim:
                    RGB = display(I_rec, R1_fit, R2_fit,
                            lims_R1_x, lims_R1_y, lims_R1_z,
                            lims_R2_x, lims_R2_y, lims_R2_z,
                            lev, step_sizes_R[step], iter_count,
                            save_dir='I_rec_test_GPU_level')
                    anim_frames.append(RGB)
                if anim:
                    save_animation_gpu(anim3, f'I_rec_test_GPU_level_{lev}/anim_updated_I_rec.mp4')
                diff_KS = KS_acq - KS_sim
                
                
                if lambda_reg_mag > 0:
                    reg_loss = lambda_reg_mag * reg_magnitude(R1_fit, R2_fit)
                    
                    try:
                        C = (1 - lambda_reg_mag) * loss_func[loss_f](diff_KS)
                    except KeyError as exc:
                        raise ValueError("loss_f must be 'L1', 'L2' or 'HuberLoss'") from exc
                    
                    # C = (1 - lambda_reg_mag) * torch.sum(diff_KS.real**2 + diff_KS.imag**2).item()
                    C += reg_loss
                else:
                    # C = torch.sum(diff_KS.real**2 + diff_KS.imag**2).item()
                    try:
                        C = loss_func[loss_f](diff_KS)
                    except KeyError as exc:
                        raise ValueError("loss_f must be 'L1', 'L2' or 'HuberLoss'") from exc
                loss.append(C)
                
            else:
                print("The image is not reconstructed or updated dude!")
            print(f"  -> post-recon loss = {C:.6e}")

        print(f"[Level {lev}] Final loss = {C:.6e}")

    return (I_rec, R1_fit, R2_fit, anim_frames, loss)
