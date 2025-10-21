from collections.abc import Callable
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.

                # Initialize state if not present, the m and v are initialized to 0 for each parameter
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data, requires_grad=False)
                    state["v"] = torch.zeros_like(p.data, requires_grad=False)

                t = state["t"]  # Get iteration number from the state.
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                # Update momentum terms
                state["m"] = betas[0] * state["m"] + (1 - betas[0]) * grad
                state["v"] = betas[1] * state["v"] + (1 - betas[1]) * grad**2

                # Bias correction
                lr_corrected = (
                    lr * math.sqrt(1 - betas[1] ** (t + 1)) / (1 - betas[0] ** (t + 1))
                )

                # Update parameters
                p.data -= (
                    lr_corrected * state["m"] / (torch.sqrt(state["v"]) + eps)
                    + lr * weight_decay * p.data
                )

                # Increment iteration number
                state["t"] = t + 1
        return loss
