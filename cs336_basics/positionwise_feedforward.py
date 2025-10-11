import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    d_model: int Dimension of the model
    d_ff: int Dimension of the feedforward network
    dropout: float Dropout rate
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        # x shape: (..., d_model)
        # Apply linear transformations
        w1x = self.w1(x)  # (..., d_ff)
        w3x = self.w3(x)  # (..., d_ff)
        silu_gate = torch.sigmoid(w1x) * w1x  # SiLU activation
        return self.w2(silu_gate * w3x)  # (..., d_model)


# This works as well.
# class PositionwiseFeedForward(nn.Module):
#     """
#     d_model: int Dimension of the model
#     d_ff: int Dimension of the feedforward network
#     dropout: float Dropout rate
#     """

#     def __init__(self, d_model, d_ff):
#         super().__init__()
#         self.d_model = d_model
#         self.d_ff = d_ff
#         self.w1 = nn.Parameter(torch.randn(self.d_ff, self.d_model))
#         self.w2 = nn.Parameter(torch.randn(self.d_model, self.d_ff))
#         self.w3 = nn.Parameter(torch.randn(self.d_ff, self.d_model))

#     def forward(self, x):
#         # x shape: (..., d_model)
#         # Apply linear transformations along the last dimension
#         w1x = torch.matmul(x, self.w1.T)  # (..., d_ff)
#         w3x = torch.matmul(x, self.w3.T)  # (..., d_ff)
#         silu_gate = torch.sigmoid(w1x) * w1x  # SiLU activation
#         return torch.matmul(silu_gate * w3x, self.w2.T)  # (..., d_model)


# test code
# swiglu = PositionwiseFeedForward(d_model, d_ff)
# swiglu.w1 = nn.Parameter(w1_weight)
# swiglu.w2 = nn.Parameter(w2_weight)
# swiglu.w3 = nn.Parameter(w3_weight)
# return swiglu(in_features)
