import torch


def cross_entropy(predicted_logits, targets):
    """
    Calculate the cross-entropy loss between predicted logits and targets.

    Args:
        predicted_logits: Tensor of shape ( (batch_size * context_length), vocab_size) containing the predicted logits. (they did a reshape from (B, C, V) -> (B * C, V))
        targets: Tensor of shape (batch_size *context_length) containing the target class indices.

    Returns:
        Float: The average cross-entropy loss across examples.

    For cross entropy loss ℓᵢ = -log softmax(oᵢ)[xᵢ₊₁], we can derive:
    softmax(oᵢ)[j] = exp(oᵢ[j]) / Σₖ exp(oᵢ[k])
    log softmax(oᵢ)[j] = oᵢ[j] - log(Σₖ exp(oᵢ[k]))
    So: ℓᵢ = -(oᵢ[xᵢ₊₁] - log_sum_exp(oᵢ)) = -oᵢ[xᵢ₊₁] + log_sum_exp(oᵢ)
    """
    # Use einops to find the maximum value in the last dimension of predicted_logits
    from einops import reduce

    # Subtract max for numerical stability

    max_vals = reduce(predicted_logits, "... vocab -> ... 1", "max")  # (BC, 1)
    stable_logits = predicted_logits - max_vals  # (BC, vocab_size)

    # Compute log_sum_exp: log(sum(exp(stable_logits)))
    exp_logits = torch.exp(stable_logits)  # (BC, vocab_size)
    sum_exp = reduce(exp_logits, "... vocab -> ... 1", "sum")  # (BC, 1)
    log_sum_exp = torch.log(sum_exp)  # (BC, 1)

    # Get the logits corresponding to target classes using advanced indexing (stable_logits[batch_indices, targets])
    batch_indices = torch.arange(stable_logits.shape[0])
    target_logits = stable_logits[
        batch_indices, targets
    ]  # (BC,), # stable_logits[:, targets] is wrong
    target_logits = target_logits.unsqueeze(-1)  # (BC, 1)

    # Cross entropy: -target_logit + log_sum_exp
    cross_entropy_losses = -target_logits + log_sum_exp  # (BC, 1)

    # Return average across batch
    return reduce(cross_entropy_losses, "... -> ", "mean")
