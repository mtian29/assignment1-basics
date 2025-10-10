# flake8: noqa

import torch
import torch.nn as nn
from jaxtyping import Float


class Linear(nn.Module):
    """Linear layer without bias"""

    # in_features: int final dimension of the input
    # out_features: int final dimension of the output
    # device: torch.device | None = None Device to store the parameters on
    # dtype: torch.dtype | None = None Data type of the parameters
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: str | None = None,
        dtype: str | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        # construct and store your parameter as W (not W ⊤) for memory ordering reasons, putting it in an nn.Parameter
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # For initializations, use the settings from above along with torch.nn.init.trunc_normal_ to initialize the weights
        torch.nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(
        self, x: Float[torch.Tensor, " ... in_features"]
    ) -> Float[torch.Tensor, " ... out_features"]:
        return x @ self.weight.T
