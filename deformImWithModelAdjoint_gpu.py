import torch
import numpy as np
def deformImWithModelAdjoint_gpu(I, R1, R2, s1, s2, def_xs=None, def_ys=None, def_zs=None):
    """
    Adjoint deformation (pull-based) of a 3D complex volume I using motion model fields R1, R2
    and surrogate signals s1, s2. GPU-compatible via PyTorch.

    Parameters:
    -----------
    I    : torch.Tensor, shape (D, H, W), complex-valued
        The itorchut volume to be back-warped (after k-space adjoint).
    R1   : torch.Tensor, shape (D, H, W, 3)
        Motion displacement field associated with surrogate s1.
    R2   : torch.Tensor, shape (D, H, W, 3)
        Motion displacement field associated with surrogate s2.
    s1   : float or torch scalar
        Value of first surrogate signal for this deformation.
    s2   : float or torch scalar
        Value of second surrogate signal.
    def_xs, def_ys, def_zs : 1D index arrays or None
        Optional rectangular ROI indices along x, y, z. If None, full range is used.

    Returns:
    --------
    I_def   : torch.Tensor, shape (D, H, W)
        The adjoint-deformed volume.
    weights : torch.Tensor, shape (D, H, W)
        Sum of interpolation weights at each voxel (for normalization).
    """
    D, H, W = I.shape

    # default to full volume if no region specified
    if ((def_xs is None) != (def_ys is None)) \
    or ((def_xs is None) != (def_zs is None)):
        raise ValueError("Must specify all of def_xs, def_ys, def_zs or none.")
    if def_xs is None:
        # set all three to full ranges
        def_xs = torch.arange(W)
        def_ys = torch.arange(H)
        def_zs = torch.arange(D)
    else:
        # safely convert all three to int arrays
        def_xs = torch.tensor(def_xs, int)
        def_ys = torch.tensor(def_ys, int)
        def_zs = torch.tensor(def_zs, int)

    # build 3D meshgrid in (Z,Y,X) order
    Zg, Yg, Xg = torch.meshgrid(def_zs, def_ys, def_xs, indexing='ij')
    Zg = Zg.to("cuda")
    Yg = Yg.to("cuda")
    Xg = Xg.to("cuda")

    # compute total displacements
    TX = s1 * R1[Zg, Yg, Xg, 2] + s2 * R2[Zg, Yg, Xg, 2]
    TY = s1 * R1[Zg, Yg, Xg, 1] + s2 * R2[Zg, Yg, Xg, 1]
    TZ = s1 * R1[Zg, Yg, Xg, 0] + s2 * R2[Zg, Yg, Xg, 0]

    # warped floating-point coordinates
    def_X0 = Xg + TX
    def_Y0 = Yg + TY
    def_Z0 = Zg + TZ

    # out-of-bounds mask
    oob = (
        (def_X0 < 0) | (def_X0 >= W-1) |
        (def_Y0 < 0) | (def_Y0 >= H-1) |
        (def_Z0 < 0) | (def_Z0 >= D-1)
    )

    # flatten arrays
    def_Xf = def_X0.flatten()
    def_Yf = def_Y0.flatten()
    def_Zf = def_Z0.flatten()
    I_flat = I.flatten()
    oob_flat = oob.flatten()

    # only keep valid points
    valid = ~oob_flat
    Xf = def_Xf[valid]
    Yf = def_Yf[valid]
    Zf = def_Zf[valid]
    Ivals = I_flat[valid]

    # integer floors
    Xi = torch.floor(Xf).long()
    Yi = torch.floor(Yf).long()
    Zi = torch.floor(Zf).long()

    # fractional parts
    wx = Xf - Xi
    wy = Yf - Yi
    wz = Zf - Zi

    # base index in flattened volume
    base = Zi * (H*W) + Yi * W + Xi

    # compute the 8 corner indices
    idx000 = base
    idx100 = base + 1
    idx010 = base + W
    idx110 = base + W + 1
    idx001 = base + H*W
    idx101 = base + H*W + 1
    idx011 = base + H*W + W
    idx111 = base + H*W + W + 1

    # trilinear weights
    w000 = (1 - wx) * (1 - wy) * (1 - wz)
    w100 =  wx      * (1 - wy) * (1 - wz)
    w010 = (1 - wx) * wy      * (1 - wz)
    w110 =  wx      * wy      * (1 - wz)
    w001 = (1 - wx) * (1 - wy) * wz
    w101 =  wx      * (1 - wy) * wz
    w011 = (1 - wx) * wy      * wz
    w111 =  wx      * wy      * wz

    # allocate flattened outputs
    size = D * H * W
    I_def_flat   = torch.zeros(size, dtype=I.dtype, device = "cuda")
    weights_flat = torch.zeros(size, dtype=torch.float32, device="cuda")

    # scatter-add contributions
# For your volume
    I_def_flat.scatter_add_(0, idx000, Ivals * w000)
    I_def_flat.scatter_add_(0, idx100, Ivals * w100)
    I_def_flat.scatter_add_(0, idx010, Ivals * w010)
    I_def_flat.scatter_add_(0, idx110, Ivals * w110)
    I_def_flat.scatter_add_(0, idx001, Ivals * w001)
    I_def_flat.scatter_add_(0, idx101, Ivals * w101)
    I_def_flat.scatter_add_(0, idx011, Ivals * w011)
    I_def_flat.scatter_add_(0, idx111, Ivals * w111)

    # For your weights
    weights_flat.scatter_add_(0, idx000, w000)
    weights_flat.scatter_add_(0, idx100, w100)
    weights_flat.scatter_add_(0, idx010, w010)
    weights_flat.scatter_add_(0, idx110, w110)
    weights_flat.scatter_add_(0, idx001, w001)
    weights_flat.scatter_add_(0, idx101, w101)
    weights_flat.scatter_add_(0, idx011, w011)
    weights_flat.scatter_add_(0, idx111, w111)


    # reshape back to 3D
    I_def   = I_def_flat.reshape((D, H, W))
    weights = weights_flat.reshape((D, H, W))

    return I_def, weights
