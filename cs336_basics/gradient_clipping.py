import torch
from collections.abc import Iterable


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6
):
    """
    Given parameters and a maximum norm, clip the gradients to the maximum norm.

    Args:
        parameters: Iterable of parameters whose gradients should be clipped
        max_l2_norm: Maximum L2 norm for gradient clipping
        eps: Small epsilon value to avoid division by zero
    """
    # Collect all gradients that are not None
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad)

    if not grads:
        return

    # Calculate total norm across all gradients
    total_norm = torch.norm(torch.cat([g.flatten() for g in grads]))

    # Clip gradients if total norm exceeds max_l2_norm
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.data.mul_(clip_coef)  # in-place modification
