import torch
import torch.nn.functional as F

def _gauss1d(sigma, device):
    radius = int(3.0 * sigma + 0.5)
    radius = max(radius, 1)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k

def gaussian_blur_3d(x, sigma):
    """
    x: [D,H,W] or [D,H,W,3] float tensor on CUDA
    sigma: float
    """
    if sigma <= 0:
        return x

    device = x.device
    k1d = _gauss1d(sigma, device) 
    ksz = k1d.numel()
    pad = ksz // 2

    if x.dim() == 4:  # [D,H,W,3]
        x_ = x.permute(3, 0, 1, 2).unsqueeze(0)  # [1,C,D,H,W]
        C = x_.shape[1]
        # build grouped weights
        wz = k1d.view(1, 1, ksz, 1, 1).repeat(C, 1, 1, 1, 1)   # [C,1,K,1,1]
        wy = k1d.view(1, 1, 1, ksz, 1).repeat(C, 1, 1, 1, 1)
        wx = k1d.view(1, 1, 1, 1, ksz).repeat(C, 1, 1, 1, 1)
        # depth
        x_ = F.conv3d(x_, wz, padding=(pad, 0, 0), groups=C)
        # height
        x_ = F.conv3d(x_, wy, padding=(0, pad, 0), groups=C)
        # width
        x_ = F.conv3d(x_, wx, padding=(0, 0, pad), groups=C)
        x_ = x_.squeeze(0).permute(1, 2, 3, 0)  # back to [D,H,W,3]
        return x_
    else:  # [D,H,W]
        x_ = x.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        wz = k1d.view(1, 1, ksz, 1, 1)
        wy = k1d.view(1, 1, 1, ksz, 1)
        wx = k1d.view(1, 1, 1, 1, ksz)
        x_ = F.conv3d(x_, wz, padding=(pad, 0, 0))
        x_ = F.conv3d(x_, wy, padding=(0, pad, 0))
        x_ = F.conv3d(x_, wx, padding=(0, 0, pad))
        return x_.squeeze(0).squeeze(0)
