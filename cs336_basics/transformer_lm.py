import torch
import torch.nn as nn
from einops import repeat
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RmsNorm
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        vocab_size,
        context_length,
        rope_theta,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.embedding = Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(
            rope_theta, d_model // num_heads, context_length
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, self.rope)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RmsNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        token_positions = repeat(
            torch.arange(x.shape[1]),
            "seq -> batch seq",
            batch=x.shape[0],
        )
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
