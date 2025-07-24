import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter
mpl.rcParams["animation.ffmpeg_path"] = r"D:\Software\ffmpeg\ffmpeg\bin\ffmpeg.exe"
def save_animation_gpu(anim_mov_acq, output_path='anim.mp4', fps=10, dpi=300):
    fig, ax = plt.subplots()
    im = ax.imshow(anim_mov_acq[0], cmap='gray')
    ax.axis('off'); plt.tight_layout()
    def update(frame_index):
        im.set_data(anim_mov_acq[frame_index])
        return (im,)
    ani = FuncAnimation(fig, update, frames=len(anim_mov_acq), blit=True, interval=1000/fps)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved MP4 to {output_path}")
