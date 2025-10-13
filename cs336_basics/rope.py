import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Construct the RoPE module and create buffers if needed.

    theta: float Θ value for the RoPE
    d_k: int dimension of query and key vectors
    max_seq_len: int Maximum sequence length that will be inputted
    device: torch.device | None = None Device to store the buffer on
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute the frequency values for RoPE
        # inv_freq has shape (d_k // 2,)
        # torch.arange(0, d_k, 2) is [0, 2, 4, ..., d_k - 2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

        # Create position indices for the maximum sequence length
        # positions has shape (max_seq_len,): [0, 1, 2, 3, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len).float()

        # Compute the frequency matrix: positions[:, None] * inv_freq[None, :]
        # freqs has shape (max_seq_len, d_k // 2)
        # theta_i_k
        freqs = torch.outer(positions, inv_freq)

        # Register cos and sin buffers with precomputed values
        # shape (max_seq_len, d_k // 2)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def rotate_half(self, u):
        # last dimension of u is [u1, u2, u3, u4, ...]
        orig_shape = u.shape
        d_k = orig_shape[-1]
        # reshape last dim to (..., d_k//2, 2)
        u = u.view(*orig_shape[:-1], d_k // 2, 2)
        u1 = u[..., 0]
        u2 = u[..., 1]
        # stack (-u2, u1) along the last axis, then reshape back
        u_rot = torch.stack((-u2, u1), dim=-1)
        return u_rot.reshape(*orig_shape)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension.
        """

        # print("x.shape, token_positions.shape", x.shape, token_positions.shape)

        cos_cached = self.cos_cached[token_positions, :]  # (..., seq_len, d_k // 2)
        sin_cached = self.sin_cached[token_positions, :]  # (..., seq_len, d_k // 2)

        # print("cos_cached.shape, sin_cached.shape", cos_cached.shape, sin_cached.shape)

        cos_cached_expanded = cos_cached.repeat_interleave(2, dim=-1)
        sin_cached_expanded = sin_cached.repeat_interleave(2, dim=-1)

        # print("cos_cached.shape, sin_cached.shape", cos_cached_expanded.shape, sin_cached_expanded.shape)
        # print("cos_cached", cos_cached_expanded[3])
        # print("sin_cached", sin_cached_expanded[3])

        # print("x[0]", x[0][:2])
        # print("rotated x[0]", self.rotate_half(x[0])[:2])

        return x * cos_cached_expanded + self.rotate_half(x) * sin_cached_expanded
