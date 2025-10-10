# token id => look up in embedding table (vocab_size, d_model) => d_model vector
# (batch_size, sequence_length) => (batch_size, sequence_length, d_model)
# sequence_length is is a list of token ids

# Suppose you have 10 words in your vocab, each represented by a 5-dimensional vector
# embedding = nn.Embedding(num_embeddings=10, embedding_dim=5)
# # Suppose you input word indices [1, 2, 4, 8]
# input_indices = torch.LongTensor([1, 2, 4, 8])
# output = embedding(input_indices)

# print(output.shape)  # torch.Size([4, 5])

# flake8: noqa

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    num_embeddings: int Size of the vocabulary
    embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        torch.nn.init.trunc_normal_(self.embedding_matrix, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # print(self.embedding_matrix.shape) # torch.Size([10000, 64])
        # print(token_ids.shape) # torch.Size([4, 12])
        # print(token_ids)
        # print(self.embedding_matrix[token_ids].shape) # torch.Size([4, 12, 64])
        # batch size is 4, sequence length is 12, embedding dimension is 64, vocab size is 10000
        return self.embedding_matrix[token_ids]
