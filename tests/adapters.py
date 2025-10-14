# pylint: disable=unused-import,line-too-long
# pyright: reportUnusedImport=false
# flake8: noqa

from __future__ import annotations
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RmsNorm
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.positionwise_feedforward import PositionwiseFeedForward
from cs336_basics.rope import RotaryPositionalEmbedding
from einops import repeat
from .utils import GPT2_PRETOKENIZER_PATTERN
from .tokenizer import Tokenizer
import concurrent.futures
from collections import Counter
from cs336_basics.softmax import softmax
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.transformer_block import TransformerBlock

import os
import pathlib
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor

import regex as re
from typing import Iterable
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    layer = Linear(d_in, d_out)
    layer.weight = nn.Parameter(weights)
    return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = Embedding(vocab_size, d_model)
    embedding.embedding_matrix = nn.Parameter(weights)
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    swiglu = PositionwiseFeedForward(d_model, d_ff)
    swiglu.w1.weight.data = w1_weight
    swiglu.w2.weight.data = w2_weight
    swiglu.w3.weight.data = w3_weight
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    multihead_self_attention = MultiheadSelfAttention(d_model, num_heads)
    multihead_self_attention.weightQ.weight.data = q_proj_weight
    multihead_self_attention.weightK.weight.data = k_proj_weight
    multihead_self_attention.weightV.weight.data = v_proj_weight
    multihead_self_attention.weightO.weight.data = o_proj_weight
    return multihead_self_attention(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
    multihead_self_attention = MultiheadSelfAttention(d_model, num_heads, rope=rope)

    multihead_self_attention.weightQ.weight.data = q_proj_weight
    multihead_self_attention.weightK.weight.data = k_proj_weight
    multihead_self_attention.weightV.weight.data = v_proj_weight
    multihead_self_attention.weightO.weight.data = o_proj_weight
    return multihead_self_attention(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)

    transformer_block = TransformerBlock(d_model, num_heads, d_ff, rope=rope)
    transformer_block.attention.weightQ.weight.data = weights["attn.q_proj.weight"]
    transformer_block.attention.weightK.weight.data = weights["attn.k_proj.weight"]
    transformer_block.attention.weightV.weight.data = weights["attn.v_proj.weight"]
    transformer_block.attention.weightO.weight.data = weights["attn.output_proj.weight"]
    transformer_block.norm1.weight.data = weights["ln1.weight"]
    transformer_block.norm2.weight.data = weights["ln2.weight"]
    transformer_block.feedforward.w1.weight.data = weights["ffn.w1.weight"]
    transformer_block.feedforward.w2.weight.data = weights["ffn.w2.weight"]
    transformer_block.feedforward.w3.weight.data = weights["ffn.w3.weight"]

    token_positions = repeat(
        torch.arange(in_features.shape[1]),
        "seq -> batch seq",
        batch=in_features.shape[0],
    )
    # print(token_positions)
    # # tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
    # #         [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
    # #         [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
    # #         [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
    # print(token_positions.shape) #torch.Size([4, 12])
    # print(in_features.shape) # torch.Size([4, 12, 64])

    return transformer_block(in_features, token_positions)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = RmsNorm(d_model)
    rmsnorm.weight = nn.Parameter(weights)
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


def _find_pretokens(text_list: list[str]):
    """
    Find the pretokens in the text.
    Pre-tokenization and then count the frequency of each pretoken.

    if you have a corpus (or chunk) like [Doc 1]<|endoftext|>[Doc
    2], you should split on the special token <|endoftext|>, and pre-tokenize [Doc 1] and [Doc 2] separately,
    so that no merging can occur across the document boundary.
    """
    logging.info(f"Pre-tokenizing the text of length {len(text_list)}")
    result = Counter()
    for text in text_list:
        pretokens = Counter(re.findall(GPT2_PRETOKENIZER_PATTERN, text))
        result = (
            result + pretokens
        )  # sum the two Counters by merging the counts for each key
    return result


def _read_text_file(input_path: str, num_worker: int, special_tokens: Iterable[str]):
    """
    Read the text file at the given path.
    Return the text as pretoken frequency table.
    """

    # Read the input text file
    with open(input_path, "r") as file:
        text = file.read()

    # Remove special tokens from the text (?)
    text = re.split("|".join(special_tokens), text)  # a list of strings

    logging.info("Initializing pretoken frequency table")
    if num_worker == 1:
        pretokens = _find_pretokens(text)  # ' for' -> 237 times
    else:
        # count each chuck's pretoken frequency and sum them up
        chunk_size = len(text) // num_worker
        text_chunks = [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
            pretokens = executor.map(_find_pretokens, text_chunks)
        pretokens = sum(pretokens, Counter())
    # convert pretoken to tuple of bytes e.g. 'iron' -> (b'i', b'r', b'o',
    # b'n')

    def gen_tuple_of_bytes(pretoken):
        return tuple([bytes([b]) for b in pretoken.encode("utf-8")])

    pretoken_freq = {}
    for pretoken, freq in pretokens.items():
        pretoken_freq[gen_tuple_of_bytes(pretoken)] = freq

    return pretoken_freq


def _update_byte_tuple(byte_tuple: Iterable[bytes], merge_loc: int):
    """
    Merge the byte tuple at the merge location.
    (b' ', b't', b'h', b'e') and merge_loc = 0
    return (b' t', b'h', b'e'), (), (b'h', b'e')
    """
    assert (
        len(byte_tuple) > 1
    ), "Cannot merge a byte tuple with length less than 2."  # length should be 2
    prefix = byte_tuple[:merge_loc]  # (b' ')
    tomerge = byte_tuple[merge_loc : merge_loc + 2]  # (b' ', b't')
    suffix = byte_tuple[merge_loc + 2 :]  # (b'h', b'e')
    new_byte_tuple = prefix + (b"".join(tomerge),) + suffix  # (b' t', b'h', b'e')
    # (b' t', b'h', b'e'), (), (b'h', b'e')
    return new_byte_tuple, prefix, suffix


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Here is a stylized example from Sennrich et al. [2016]. Consider a corpus consisting of the following text
    low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest
    and the vocabulary has a special token <|endoftext|>.
    Vocabulary values.
    We initialize our vocabulary with our special token <|endoftext|> and the 256 byte
    Pre-tokenization For simplicity and to focus on the merge procedure, we assume in this example
    that pretokenization simply splits on whitespace. When we pretokenize and count, we end up with the
    frequency table.
    {low: 5, lower: 2, widest: 3, newest: 6}

    It is convenient to represent this as a dict[tuple[bytes], int], e.g. {(l,o,w): 5 …}. Note that even
    a single byte is a bytes object in Python. There is no byte type in Python to represent a single byte,
    just as there is no char type in Python to represent a single character.
    Merges We first look at every successive pair of bytes and sum the frequency of the words where they
    appear {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}. The pair ('es')
    and ('st') are tied, so we take the lexicographically greater pair, ('st'). We would then merge the
    pre-tokens so that we end up with {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}.
    In the second round, we see that (e, st) is the most common pair (with a count of 9) and we would
    merge into {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}. Continuing this, the
    sequence of merges we get in the end will be ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e',
    'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r'].
    If we take 6 merges, we have ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e'] and our vocab-
    ulary elements would be [<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne].
    With this vocabulary and set of merges, the word newest would tokenize as [ne, west]. (<-- different from the pretokenization)

    Without pretokenization, the above could have a token like "ow lo" in the final vocabulary.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    progress_bar = kwargs.get("progress_bar", False)
    num_workers = kwargs.get("num_workers", 1)

    vocab = {i: bytes([i]) for i in range(256)}  # 0 -> b'\x00
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")  # 256 -> b'<|endoftext|>'

    pretoken_freq = _read_text_file(input_path, num_workers, special_tokens)
    # (b'i', b'r', b'o', b'n') -> 2 times

    logging.info("Initializing byte pair frequency table")
    pair_freq = Counter()
    for pretoken_tuple, freq in tqdm(pretoken_freq.items(), disable=not progress_bar):
        for i in range(
            len(pretoken_tuple) - 1
        ):  # pretoken_tuple = (b'i', b'r', b'o', b'n')
            pair = pretoken_tuple[i : i + 2]  # pair = (b'i', b'r')
            if pair not in pair_freq:
                pair_freq[pair] = 0
            pair_freq[pair] += freq

    # pretoken_freq: (b'i', b'r', b'o', b'n') : 2 times
    # pair_freq: Counter({(b'i', b'r'): 2, (b'r', b'o'): 2, (b'o', b'n'): 2})
    # now we have all the byte pairs and their frequencies

    logging.info("Performing BPE algorithm")
    pre_merge_vocab_size = len(vocab)
    pbar = tqdm(total=vocab_size - pre_merge_vocab_size) if progress_bar else None
    merges = []
    while len(vocab) < vocab_size:  # quit after we reached the desired vocab size
        # Find the most frequent pair, if tie, choose the lexicographically largest one
        # (OR use a max_heap to find the most frequent pair)
        most_freq_pair = max(
            pair_freq, key=lambda k: (pair_freq[k], k)
        )  # (b' ', b't') -> 2940 times

        # Add the pair to the merges list
        merges.append(most_freq_pair)  # [(b' ', b't')]

        # Update the vocab
        new_id = max(vocab.keys()) + 1  # 257
        vocab[new_id] = b"".join(most_freq_pair)  # vocab: 257 -> b' t'

        # Update the pre-token frequency table pretoken_freq (pretoken_tuple -> times) and pair frequency table pair_freq (pair -> times)
        # need pair_freq to find the max pair; need pretoken_freq's key to merge the tuple after finding the max pair
        # {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6} this is pair_freq
        # {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6} this is pretoken_freq
        new_pretoken_freq = {}
        for (
            pretoken_tuple,
            freq,
        ) in pretoken_freq.items():  # (b' ', b't', b'h', b'e') : 1279 times
            i = 0
            while i < len(pretoken_tuple):
                pair = pretoken_tuple[i : i + 2]  # pair = (b' ', b't')
                if pair == most_freq_pair:
                    # pretoken_tuple = (b' ', b't', b'h', b'e'), i = 0; pair = (b' ', b't');
                    # prefix
                    pretoken_tuple, prefix, suffix = _update_byte_tuple(
                        pretoken_tuple, i
                    )
                    # pretoken_tuple = (b' t', b'h', b'e') <-- this is the new
                    # merged tuple, prefix = (), suffix = (b'h', b'e')

                    # Update the pair frequency table; https://github.com/marta1994/efficient_bpe_explanation
                    # for the new merged ' t'; we form new pairs with the prefix and suffix (add_pair)
                    # update the pair frequency table for the new pairs
                    # (add_pair (b' t', b'h')) and delete the old pair
                    # (del_pair)  (b't', b'h')
                    if prefix:
                        add_pair = (prefix[-1], vocab[new_id])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        del_pair = (prefix[-1], most_freq_pair[0])
                        pair_freq[del_pair] -= freq
                    if suffix:
                        add_pair = (vocab[new_id], suffix[0])  # (b' t', b'h')
                        pair_freq[add_pair] = (
                            pair_freq.get(add_pair, 0) + freq
                        )  # (b' t', b'h') : 1279 times
                        del_pair = (
                            most_freq_pair[1],
                            suffix[0],
                        )  # (b't', b'h'), this pair is deleted;
                        # cannot set it to 0, should minus the frequency;
                        # because ' th' is gone but 'xth' is still there
                        pair_freq[del_pair] -= freq
                    pair_freq[most_freq_pair] -= freq
                i += 1
            # Update the pre-token frequency table; pretoken_tuple = (b' t',
            # b'h', b'e')
            new_pretoken_freq[pretoken_tuple] = freq
        pretoken_freq = new_pretoken_freq
        (
            pbar.update(len(vocab) - pre_merge_vocab_size - pbar.n)
            if progress_bar
            else None
        )
    pbar.close() if progress_bar else None

    # vocab: {0: b'\x00', 1: b'<|endoftext|>', 257: b' t', 258: b' a', ...}
    # merges: [(b' ', b't'), (b' ', b'a')]
    return vocab, merges


FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"
CURRENT_PATH = pathlib.Path(__file__).resolve().parent


def run_bpe():
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    # input_path = CURRENT_PATH  / "../data/TinyStoriesV2-GPT4-train.txt"
    print(input_path)
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        progress_bar=True,
        num_workers=10,
    )
    # Write vocab and merges to text files (not pickle)
    vocab_path = CURRENT_PATH / "outputs/generated_vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for k, v in vocab.items():
            # Write key and value as int and bytes literal
            f.write(f"{k}\t{v!r}\n")

    merges_path = CURRENT_PATH / "outputs/generated_merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            # Write each merge as a tuple of bytes literals
            f.write(f"{merge[0]!r} {merge[1]!r}\n")


if __name__ == "__main__":
    # python -m cProfile -o bpe.prof tests/adapters.py && snakeviz bpe.prof
    # OR scalene tests/adapters.py
    run_bpe()
