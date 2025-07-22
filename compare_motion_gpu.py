import torch

def compute_rmse_gpu(true, est):
    """
    Per-component RMSE on GPU.
    true, est: torch tensors shape (...,3)
    Returns torch tensor shape (3,)
    """
    diff = est - true
    mse = (diff**2).mean(dim=list(range(diff.ndim-1)))
    return torch.sqrt(mse)

def report_motion_error_gpu(R1_true, R1_est, R2_true, R2_est):
    """
    Print RMSE for (z,y,x) components.
    All inputs can be either numpy arrays or torch tensors.
    """
    if not torch.is_tensor(R1_true):
        R1_true = torch.tensor(R1_true, device='cuda')
        R1_est = torch.tensor(R1_est, device='cuda')
        R2_true = torch.tensor(R2_true, device='cuda')
        R2_est = torch.tensor(R2_est, device='cuda')

    rmse1 = compute_rmse_gpu(R1_true, R1_est)
    rmse2 = compute_rmse_gpu(R2_true, R2_est)
    print(f"R1 RMSE (z,y,x): {rmse1.cpu().tolist()}")
    print(f"R2 RMSE (z,y,x): {rmse2.cpu().tolist()}")
