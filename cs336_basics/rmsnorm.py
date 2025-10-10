import torch
import torch.nn as nn


class RmsNorm(nn.Module):
    """
    Construct the RMSNorm module.

    Args:
        d_model (int): Hidden dimension of the model.
        eps (float, optional): Epsilon value for numerical stability.
        Default is 1e-5.
        device (torch.device, optional): Device to store the parameters on.
        Default is None.
        dtype (torch.dtype, optional): Data type of the parameters.
        Default is None.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.randn(d_model))
        torch.nn.init.trunc_normal_(self.weight, std=0.02)

    """
    Process an input tensor of shape
    (batch_size, sequence_length, d_model) and return a tensor of the same
    shape.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x * self.weight / rms
        return x.to(in_dtype)
