"""Neural Autoregressive Distribution Estimator."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class NADE(nn.Module):
    """NADE for binary vectors with one shared hidden layer."""

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim: int = 128,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if init_std < 0:
            raise ValueError("init_std must be non-negative")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_std = init_std

        # W[:, i] updates the shared hidden state after observing x_i.
        self.input_weight = nn.Parameter(torch.empty(hidden_dim, input_dim))
        # V[i] maps the hidden representation for position i to its logit.
        self.output_weight = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.visible_bias = nn.Parameter(torch.zeros(input_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.input_weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.output_weight, mean=0.0, std=self.init_std)
        nn.init.zeros_(self.hidden_bias)
        nn.init.zeros_(self.visible_bias)

    def forward(self, x: Tensor) -> Tensor:
        """Return all conditional Bernoulli logits in parallel."""
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"expected shape [batch, {self.input_dim}], got {tuple(x.shape)}"
            )

        contributions = x.unsqueeze(1) * self.input_weight.unsqueeze(0)
        prefix_sum = torch.cumsum(contributions, dim=2)
        initial = torch.zeros(
            x.shape[0], self.hidden_dim, 1, dtype=x.dtype, device=x.device
        )
        hidden_preactivation = torch.cat(
            [initial, prefix_sum[:, :, :-1]], dim=2
        )
        hidden = torch.sigmoid(
            hidden_preactivation + self.hidden_bias.view(1, -1, 1)
        )
        logits = torch.einsum("bhd,dh->bd", hidden, self.output_weight)
        return logits + self.visible_bias

    def log_prob(self, x: Tensor) -> Tensor:
        """Return log p(x) in nats for each item in the batch."""
        logits = self(x)
        return -F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)

    def nll(
        self,
        x: Tensor,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> Tensor:
        """Return negative log-likelihood in nats."""
        losses = -self.log_prob(x)
        if reduction == "none":
            return losses
        if reduction == "mean":
            return losses.mean()
        if reduction == "sum":
            return losses.sum()
        raise ValueError(f"unsupported reduction: {reduction}")

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        *,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Generate binary vectors one dimension at a time."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        device = self.visible_bias.device
        samples = torch.zeros(num_samples, self.input_dim, device=device)
        hidden_preactivation = self.hidden_bias.expand(num_samples, -1).clone()
        for index in range(self.input_dim):
            hidden = torch.sigmoid(hidden_preactivation)
            logits = self.visible_bias[index] + (
                hidden * self.output_weight[index]
            ).sum(dim=1)
            probabilities = torch.sigmoid(logits / temperature)
            current = torch.bernoulli(probabilities, generator=generator)
            samples[:, index] = current
            hidden_preactivation.add_(
                current.unsqueeze(1) * self.input_weight[:, index]
            )
        return samples

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"parameters={sum(parameter.numel() for parameter in self.parameters()):,}"
        )


def bits_per_dimension(nll: float, input_dim: int) -> float:
    """Convert NLL in nats/example to bits/dimension."""
    return nll / (input_dim * math.log(2.0))
