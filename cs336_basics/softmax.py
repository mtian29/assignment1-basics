import math
import torch


def softmax(x, dim):
    """

    Write a function to apply the softmax operation on a tensor. Your function should
    take two parameters: a tensor and a dimension i, and apply softmax to the i-th dimension of the input
    tensor. The output tensor should have the same shape as the input tensor, but its i-th dimension will
    now have a normalized probability distribution. Use the trick of subtracting the maximum value in
    the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues.
    """
    max_val = torch.max(x, dim=dim, keepdim=True).values
    return torch.exp(x - max_val) / torch.sum(
        torch.exp(x - max_val), dim=dim, keepdim=True
    )


# Original tensor:
# [[1, 2, 3],
#  [4, 5, 6]]

# dim=0 (↓): Compare vertically
# [[1, 2, 3],    →  Each column sums to 1
#  [4, 5, 6]]       [0.047+0.953=1, 0.047+0.953=1, 0.047+0.953=1]

# dim=1 (→): Compare horizontally
# [[1, 2, 3],    →  Each row sums to 1
#  [4, 5, 6]]       [0.090+0.245+0.665=1, 0.090+0.245+0.665=1]
# # Example tensor: 2x3 matrix
# x = torch.tensor([[1, 2, 3],
#                   [4, 5, 6]], dtype=torch.float32)

# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print("Original tensor:")
# print(x)
# print("Shape:", x.shape)  # Shape is [2, 3] - 2 rows, 3 columns
# print()

# print("=== DIMENSION 0 (along rows) ===")
# print("Softmax along dim=0 - normalizes ACROSS ROWS (vertically)")
# result_dim0 = softmax(x, 0)
# print("Result:")
# print(result_dim0)
# print("Each COLUMN sums to 1:")
# print("Column sums:", torch.sum(result_dim0, dim=0))
# print()

# print("=== DIMENSION 1 (along columns) ===")
# print("Softmax along dim=1 - normalizes ACROSS COLUMNS (horizontally)")
# result_dim1 = softmax(x, 1)
# print("Result:")
# print(result_dim1)
# print("Each ROW sums to 1:")
# print("Row sums:", torch.sum(result_dim1, dim=1))
# print()

# print("Manual calculation for dim=1, first row [1,2,3]:")
# print(
#     "exp(1-3)/(exp(1-3)+exp(2-3)+exp(3-3)) =",
#     math.exp(1 - 3) / (math.exp(1 - 3) + math.exp(2 - 3) + math.exp(3 - 3)),
# )
# print(
#     "exp(2-3)/(exp(1-3)+exp(2-3)+exp(3-3)) =",
#     math.exp(2 - 3) / (math.exp(1 - 3) + math.exp(2 - 3) + math.exp(3 - 3)),
# )
# print(
#     "exp(3-3)/(exp(1-3)+exp(2-3)+exp(3-3)) =",
#     math.exp(3 - 3) / (math.exp(1 - 3) + math.exp(2 - 3) + math.exp(3 - 3)),
# )
