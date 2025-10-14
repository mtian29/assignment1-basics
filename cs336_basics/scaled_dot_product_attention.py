import torch
from einops import einsum
from cs336_basics.softmax import softmax


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot product attention using einops.

    Args:
        query: Query tensor of shape (..., seq_len_q, d_k)
        key: Key tensor of shape (..., seq_len_k, d_k)
        value: Value tensor of shape (..., seq_len_v, d_v)
        mask: Optional mask tensor

    Returns:
        Attention output of shape (..., seq_len_q, d_v)
    """
    # Get the dimension for scaling
    d_k = query.size(-1)

    # Compute attention scores: Q * K^T / sqrt(d_k)
    # Using einops einsum for the matrix multiplication
    scores = einsum(query, key, "... i d, ... j d -> ... i j") / torch.sqrt(
        torch.tensor(d_k, dtype=query.dtype)
    )

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Apply softmax to get attention weights
    attn_weights = softmax(scores, dim=-1)

    # Apply attention weights to values: softmax(scores) * V
    output = einsum(attn_weights, value, "... i j, ... j d -> ... i d")

    return output
