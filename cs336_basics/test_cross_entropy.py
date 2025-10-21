import torch
import torch.nn.functional as F

# Test data from the test case
inputs = torch.tensor(
    [
        [
            [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
            [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
            [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
            [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
        ],
        [
            [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
            [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
            [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
            [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
        ],
    ]
)
targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])

# Reshape to match test
inputs_flat = inputs.view(-1, inputs.size(-1))  # (8, 5)
targets_flat = targets.view(-1)  # (8,)

print("Input shape:", inputs_flat.shape)
print("Target shape:", targets_flat.shape)
print("Targets:", targets_flat)

# Expected result
expected = F.cross_entropy(inputs_flat, targets_flat)
print("Expected cross entropy:", expected.item())

# cross entropy for first row
cross_entropy = F.cross_entropy(inputs_flat[0], targets_flat[0])
print("!!! Cross entropy for first row:", cross_entropy.item())

# Let's manually compute one example to understand the formula
print("\\nManual computation for first example:")
logits = inputs_flat[0]  # [0.1088, 0.1060, 0.6683, 0.5131, 0.0645]
target = targets_flat[0]  # 1
print("Logits:", logits)
print("Target class:", target.item())

# Softmax computation
max_val = torch.max(logits)
print("Max logit:", max_val.item())
stable_logits = logits - max_val
print("Stable logits:", stable_logits)
exp_logits = torch.exp(stable_logits)
print("Exp of stable logits:", exp_logits)
sum_exp = torch.sum(exp_logits)
print("Sum of exp:", sum_exp.item())
log_sum_exp = torch.log(sum_exp)
print("Log sum exp:", log_sum_exp.item())

# Cross entropy for this example
target_logit = stable_logits[target]
print("Target logit (stable):", target_logit.item())
cross_entropy_single = -target_logit + log_sum_exp
print("!!! Cross entropy for this example:", cross_entropy_single.item())
