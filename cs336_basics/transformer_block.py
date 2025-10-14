import torch.nn as nn

from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.positionwise_feedforward import PositionwiseFeedForward
from cs336_basics.rmsnorm import RmsNorm


class TransformerBlock(nn.Module):
    """
    Transformer block module.

    Args:
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
    """

    def __init__(self, d_model, num_heads, d_ff, rope):
        super().__init__()

        self.norm1 = RmsNorm(d_model)
        self.attention = MultiheadSelfAttention(d_model, num_heads, rope=rope)
        self.norm2 = RmsNorm(d_model)
        self.feedforward = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x, token_positions):
        # First residual connection: x + MultiheadSelfAttention(RMSNorm(x))
        attn_output = self.attention(self.norm1(x), token_positions)
        x = x + attn_output

        # Second residual connection: x + PositionwiseFeedForward(RMSNorm(x))
        ff_output = self.feedforward(self.norm2(x))
        x = x + ff_output

        return x
