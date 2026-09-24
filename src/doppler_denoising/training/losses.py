"""Masked reconstruction, gradient and Hessian objectives."""
import torch

def loss_terms(prediction, target, mask, objective="article"):
    """Equations 6, 14, 18, 19, with every derivative stencil fully masked."""
    dims = (1, 2, 3)
    count = mask.sum(dims)
    if torch.any(count == 0):
        raise ValueError("Empty reconstruction mask")
    error = prediction - target
    if objective not in ("article", "l1", "l2", "l1_grad_hessian", "l2_grad_hessian"):
        raise ValueError(f"Unknown objective: {objective}")
    reconstruction = ((error.square() if objective in ("l2", "l2_grad_hessian") else error.abs()) * mask).sum(dims) / count
    zero = torch.zeros_like(reconstruction)
    gradient, hessian = zero, zero
    if objective in ("article", "l1_grad_hessian", "l2_grad_hessian"):
        mx = mask[..., 1:] * mask[..., :-1]
        my = mask[..., 1:, :] * mask[..., :-1, :]
        gx = error[..., 1:] - error[..., :-1]
        gy = error[..., 1:, :] - error[..., :-1, :]
        gradient = ((gx.abs()*mx).sum(dims) + (gy.abs()*my).sum(dims)) / (mx.sum(dims)+my.sum(dims)+1e-8)
        mxx = mask[..., 2:]*mask[..., 1:-1]*mask[..., :-2]
        myy = mask[..., 2:, :]*mask[..., 1:-1, :]*mask[..., :-2, :]
        mxy = mask[..., 1:, 1:]*mask[..., :-1, 1:]*mask[..., 1:, :-1]*mask[..., :-1, :-1]
        xx = error[..., 2:] - 2*error[..., 1:-1] + error[..., :-2]
        yy = error[..., 2:, :] - 2*error[..., 1:-1, :] + error[..., :-2, :]
        xy = error[..., 1:, 1:] - error[..., :-1, 1:] - error[..., 1:, :-1] + error[..., :-1, :-1]
        hessian = ((xx.abs()*mxx).sum(dims)+(yy.abs()*myy).sum(dims)+2*(xy.abs()*mxy).sum(dims)) / (
            mxx.sum(dims)+myy.sum(dims)+2*mxy.sum(dims)+1e-8)
    total = reconstruction + .10*gradient + .05*hessian
    return dict(total=total, reconstruction=reconstruction, gradient=gradient, hessian=hessian)



