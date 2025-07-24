import torch


def reg_magnitude(R1, R2):
    return torch.sum(R1*R1) + torch.sum(R2*R2)

def lap_grad_term(R):
    """Discrete laplacian gradient helper for smoothness penalty."""
    g = torch.zeros_like(R)
    for shift in [(1,0,0),(0,1,0),(0,0,1)]:
        R_s = torch.roll(R, shifts=shift, dims=(0,1,2))
        diff = R - R_s
        g += 2*diff
        g -= 2*torch.roll(diff, shifts=(-shift[0], -shift[1], -shift[2]), dims=(0,1,2))
    return g

def huber_loss(r, delta=1.0, reduction="sum"):
    
    abs_r = r.abs()
    quad = torch.clamp(abs_r, max=delta)
    loss = 0.5 * quad**2 + delta * (abs_r - quad)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")


