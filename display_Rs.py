import numpy as np
import torch
import matplotlib.pyplot as plt

def display(I_rec, R1_fit, R2_fit,
            lims_R1_x, lims_R1_y, lims_R1_z,
            lims_R2_x, lims_R2_y, lims_R2_z,
            lev, step, iter_count,
            # anim_frames=None, 
            save_dir='I_rec_test_GPU_level'):
    """
    Create the custom 2x5 layout, plot slices and save a PNG.
    """
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    gs = axes[0, 0].get_gridspec()

  
    for ax in axes[:, :2].flatten():
        ax.remove()

    ax_img = fig.add_subplot(gs[:, :2])
    ax_R1x = fig.add_subplot(gs[0, 2])
    ax_R1y = fig.add_subplot(gs[0, 3])
    ax_R1z = fig.add_subplot(gs[0, 4])
    ax_R2x = fig.add_subplot(gs[1, 2])
    ax_R2y = fig.add_subplot(gs[1, 3])
    ax_R2z = fig.add_subplot(gs[1, 4])

    mid_z = R1_fit.shape[0] // 2

    
    ax_img.clear()
    ax_img.imshow(torch.dstack([torch.abs(I_rec[mid_z]) / torch.abs(I_rec).max()]*3).cpu())
    ax_img.axis('off')

    
    def _imshow(ax, arr, lims, title):
        ax.clear()
        ax.imshow(arr.cpu(), vmin=lims[0], vmax=lims[1])
        ax.axis('off')
        ax.set_title(title, fontsize=8)

    _imshow(ax_R1x, R1_fit[mid_z, :, :, 2], lims_R1_x, "R1_fit X")
    _imshow(ax_R1y, R1_fit[mid_z, :, :, 1], lims_R1_y, "R1_fit Y")
    _imshow(ax_R1z, R1_fit[mid_z, :, :, 0], lims_R1_z, "R1_fit Z")
    _imshow(ax_R2x, R2_fit[mid_z, :, :, 2], lims_R2_x, "R2_fit X")
    _imshow(ax_R2y, R2_fit[mid_z, :, :, 1], lims_R2_y, "R2_fit Y")
    _imshow(ax_R2z, R2_fit[mid_z, :, :, 0], lims_R2_z, "R2_fit Z")

    fig.suptitle(f"Level {lev}, Iter {iter_count} - Initial Fit")
    fig.tight_layout(rect=[0, 0, 1, 0.95])  


    png_path = f'{save_dir}_{lev}/Rs_I_{step}_{iter_count}.png'
    fig.savefig(png_path, dpi=150)

    
    fig.canvas.draw()
    buf, (w, h) = fig.canvas.print_to_buffer()
    img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    img_rgb = img_rgba[:, :, :3]

    # if anim_frames is not None:
    #     anim_frames.append(img_rgb)

    plt.close(fig)
    return img_rgb
