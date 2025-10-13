import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation.

    RoPE encodes positional information by rotating query and key vectors in pairs
    of dimensions. Instead of adding positional embeddings, RoPE multiplies the
    embeddings by rotation matrices that depend on the position.

    For a vector [x1, x2, x3, x4, ...], RoPE groups dimensions in pairs:
    - (x1, x2) gets rotated by angle θ₁ * position
    - (x3, x4) gets rotated by angle θ₂ * position
    - etc.

    The rotation angles θᵢ decrease with higher dimensions, allowing the model
    to capture both local and global positional relationships.

    Args:
        theta: Base frequency parameter (typically 10000.0)
        d_k: Dimension of query/key vectors (must be even), d_model
        max_seq_len: Maximum sequence length for precomputing rotations
        device: Device to store cached tensors on
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Step 1: Compute inverse frequencies for each dimension pair
        # RoPE formula: θᵢ = θ^(-2i/d) where i ∈ [0, d/2)
        # For d_k=64: i = [0, 2, 4, ..., 62] → θᵢ = [θ^0, θ^(-2/64), θ^(-4/64), ...]
        # inv_freq has shape (d_k // 2,) = (32,) for d_k=64
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

        # Step 2: Create position indices for all possible positions
        # positions has shape (max_seq_len,): [0, 1, 2, 3, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len).float()

        # Step 3: Compute rotation angles for each (position, dimension_pair) combination
        # freqs[pos, dim_pair] = position * θ_dim_pair
        # freqs has shape (max_seq_len, d_k // 2)
        # This gives us the rotation angle for each dimension pair at each position
        freqs = torch.outer(positions, inv_freq)

        # Step 4: Precompute cos and sin values for efficiency
        # During forward pass, we'll use: x_rotated = x * cos + rotate_half(x) * sin
        # cos_cached, sin_cached both have shape (max_seq_len, d_k // 2)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def rotate_half(self, u):
        """
        Rotate vector by 90 degrees in each dimension pair.

        This function implements the rotation part of RoPE by swapping and negating
        elements in each pair. For a 2D rotation by 90°:
        [x, y] → [-y, x]

        For higher dimensions, we apply this to each consecutive pair:
        [x1, x2, x3, x4, ...] → [-x2, x1, -x4, x3, ...]

        Args:
            u: Input tensor with shape (..., d_k)

        Returns:
            Rotated tensor with same shape as input
        """
        orig_shape = u.shape
        d_k = orig_shape[-1]

        # Reshape to group consecutive pairs: (..., d_k//2, 2)
        # This groups [x1,x2], [x3,x4], etc. into separate pairs
        u = u.view(*orig_shape[:-1], d_k // 2, 2)

        # Extract first and second elements of each pair
        u1 = u[..., 0]  # [x1, x3, x5, ...] - shape (..., d_k//2)
        u2 = u[..., 1]  # [x2, x4, x6, ...] - shape (..., d_k//2)

        # Apply 90° rotation: [x1, x2] → [-x2, x1]
        # Stack creates (..., d_k//2, 2) with pairs [-x2, x1], [-x4, x3], etc.
        u_rot = torch.stack((-u2, u1), dim=-1)

        # Reshape back to original dimensions
        return u_rot.reshape(*orig_shape)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension.

        Rotate every token's embedding in the pair.

        Apply rotary positional embedding to input tensor.

        The core RoPE formula for each dimension pair (x₁, x₂) at position m is:
        [x₁'] = [cos(mθ)  -sin(mθ)] [x₁]
        [x₂']   [sin(mθ)   cos(mθ)] [x₂]

        # print("x.shape, token_positions.shape", x.shape, token_positions.shape)

        This can be rewritten as:
        x₁' = x₁ * cos(mθ) - x₂ * sin(mθ)
        x₂' = x₂ * cos(mθ) + x₁ * sin(mθ)

        Or in vector form: x' = x * cos + rotate_half(x) * sin
        where rotate_half swaps and negates pairs: [x₁, x₂] → [-x₂, x₁]

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices of shape (..., seq_len)

        Returns:
            Rotated tensor of same shape as input

        """

        # Step 1: Extract cos/sin values for the specific positions in this sequence
        # cos_cached[pos] gives cos values for all dimension pairs at position 'pos'
        cos_cached = self.cos_cached[token_positions, :]  # (seq_len, d_k // 2)
        sin_cached = self.sin_cached[token_positions, :]  # (seq_len, d_k // 2)

        # Step 2: Expand cos/sin to match input dimensions
        # Each cos/sin value applies to a pair of dimensions, so we repeat each value twice
        # [cos₁, cos₂, cos₃] → [cos₁, cos₁, cos₂, cos₂, cos₃, cos₃]
        cos_cached_expanded = cos_cached.repeat_interleave(2, dim=-1)  # (seq_len, d_k)
        sin_cached_expanded = sin_cached.repeat_interleave(2, dim=-1)  # (seq_len, d_k)

        # Step 3: Apply RoPE rotation formula
        # x' = x * cos + rotate_half(x) * sin
        # This applies the 2D rotation matrix to each consecutive pair of dimensions
        # print("cos_cached_expanded", cos_cached_expanded.shape)  # ( seq_len, d_k)
        # print("sin_cached_expanded", sin_cached_expanded.shape)  # (seq_len, d_k)
        # print("x", x.shape)  # (batch_size, seq_len, d_k)

        return x * cos_cached_expanded + self.rotate_half(x) * sin_cached_expanded
