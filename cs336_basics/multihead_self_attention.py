import torch
import torch.nn as nn
from einops import rearrange
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class MultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention module.

    Args:
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        rope (RotaryPositionalEmbedding): RoPE module to apply to the query and key.
    """

    def __init__(self, d_model, num_heads, rope=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.weightQ = nn.Linear(d_model, d_model, bias=False)
        self.weightK = nn.Linear(d_model, d_model, bias=False)
        self.weightV = nn.Linear(d_model, d_model, bias=False)
        self.weightO = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope

    def forward(self, x, token_positions=None):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(-2)

        # Linear projections
        q = self.weightQ(x)  # (batch_size, seq_len, d_model)
        k = self.weightK(x)  # (batch_size, seq_len, d_model)
        v = self.weightV(x)  # (batch_size, seq_len, d_model)

        # Reshape for multi-head attention using einops
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RoPE if it is provided
        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Create causal mask using torch.triu
        # mask should be 1 for allowed positions, 0 for masked positions
        # torch.triu(torch.ones(seq_len, seq_len), diagonal=1) creates a matrix like:
        # [[0, 1, 1, 1],
        # [0, 0, 1, 1],
        # [0, 0, 0, 1],
        # [0, 0, 0, 0]]

        # This allows each token to attend to itself and all previous tokens, but not future tokens (causal/autoregressive attention).
        # [[True,  False, False, False],
        # [True,  True,  False, False],
        # [True,  True,  True,  False],
        # [True,  True,  True,  True ]]
        # print(torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1))
        causal_mask = (
            torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1) == 0
        )

        # Apply scaled dot product attention
        attn = scaled_dot_product_attention(
            q, k, v, mask=causal_mask
        )  # (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads using einops
        attn = rearrange(attn, "b h s d -> b s (h d)")

        # Final linear projection
        return self.weightO(attn)
