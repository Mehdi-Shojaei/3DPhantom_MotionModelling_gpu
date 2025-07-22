import torch
import torch.nn.functional as F


def gaussian_kernel1d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel.

    Args:
        kernel_size (int): Length of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian.
        device (torch.device): Device to create the kernel on.

    Returns:
        torch.Tensor: 1D kernel of shape [kernel_size].
    """
    # Create a symmetric range centered at zero
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_3D(
    x: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> torch.Tensor:
    """
    Applies a 3D Gaussian blur to a 5D tensor using separable 1D convolutions.

    Args:
        x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
        kernel_size (int): Size of the Gaussian kernel in each dimension (must be odd).
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: Blurred tensor of the same shape as input.
    """
    
    # print(f"Gaussian filter is being applied onto the tensor with size {x.shape}")
    
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    B, C, D, H, W = x.shape
    device = x.device

    # Create 1D Gaussian kernel
    k1d = gaussian_kernel1d(kernel_size, sigma, device)  # [K]

    # Reshape for separable convolution
    kz = k1d.view(1, 1, kernel_size, 1, 1).repeat(C, 1, 1, 1, 1)
    ky = k1d.view(1, 1, 1, kernel_size, 1).repeat(C, 1, 1, 1, 1)
    kx = k1d.view(1, 1, 1, 1, kernel_size).repeat(C, 1, 1, 1, 1)

    pad = kernel_size // 2

    # Apply depth (D) blur
    out = F.conv3d(x, weight=kz, padding=(pad, 0, 0), groups=C)
    # Apply height (H) blur
    out = F.conv3d(out, weight=ky, padding=(0, pad, 0), groups=C)
    # Apply width (W) blur
    out = F.conv3d(out, weight=kx, padding=(0, 0, pad), groups=C)

    return out