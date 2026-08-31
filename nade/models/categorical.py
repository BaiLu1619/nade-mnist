"""Categorical Neural Autoregressive Distribution Estimator."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CategoricalNADE(nn.Module):
    """NADE whose variables follow categorical rather than Bernoulli distributions."""

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim: int = 32,
        num_categories: int = 256,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 2 <= num_categories <= 256:
            raise ValueError("num_categories must be between 2 and 256")
        if init_std < 0:
            raise ValueError("init_std must be non-negative")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_categories = num_categories
        self.init_std = init_std

        # W[i, k] is the hidden-state contribution when x_i has category k.
        self.input_embedding = nn.Parameter(
            torch.empty(input_dim, num_categories, hidden_dim)
        )
        # A[i, k] maps the hidden state to the logit for category k at position i.
        self.output_weight = nn.Parameter(
            torch.empty(input_dim, num_categories, hidden_dim)
        )
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.visible_bias = nn.Parameter(torch.zeros(input_dim, num_categories))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.input_embedding, mean=0.0, std=self.init_std)
        nn.init.normal_(self.output_weight, mean=0.0, std=self.init_std)
        nn.init.zeros_(self.hidden_bias)
        nn.init.zeros_(self.visible_bias)

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"expected shape [batch, {self.input_dim}], got {tuple(x.shape)}"
            )
        if x.dtype != torch.long:
            raise TypeError("categorical inputs must use torch.long")
        if x.numel() and (x.min().item() < 0 or x.max().item() >= self.num_categories):
            raise ValueError(
                f"categorical values must be in [0, {self.num_categories - 1}]"
            )

    def forward(self, x: Tensor) -> Tensor:
        """Return logits with shape [batch, input_dim, num_categories]."""
        self._validate_input(x)
        positions = torch.arange(self.input_dim, device=x.device).view(1, -1)
        contributions = self.input_embedding[positions, x]
        prefix_sum = torch.cumsum(contributions, dim=1)
        initial = torch.zeros(
            x.shape[0], 1, self.hidden_dim,
            dtype=self.input_embedding.dtype,
            device=x.device,
        )
        hidden_preactivation = torch.cat([initial, prefix_sum[:, :-1]], dim=1)
        hidden = torch.sigmoid(hidden_preactivation + self.hidden_bias.view(1, 1, -1))
        logits = torch.einsum("bdh,dkh->bdk", hidden, self.output_weight)
        return logits + self.visible_bias

    def log_prob(self, x: Tensor) -> Tensor:
        """Return log p(x) in nats for each item in the batch."""
        logits = self(x)
        losses = F.cross_entropy(
            logits.reshape(-1, self.num_categories),
            x.reshape(-1),
            reduction="none",
        ).view(x.shape[0], self.input_dim)
        return -losses.sum(dim=1)

    def nll(
        self,
        x: Tensor,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> Tensor:
        """Return categorical negative log-likelihood in nats."""
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
        """Generate categorical vectors one position at a time."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        device = self.visible_bias.device
        samples = torch.zeros(
            num_samples, self.input_dim, dtype=torch.long, device=device
        )
        hidden_preactivation = self.hidden_bias.expand(num_samples, -1).clone()
        for index in range(self.input_dim):
            hidden = torch.sigmoid(hidden_preactivation)
            logits = torch.einsum(
                "bh,kh->bk", hidden, self.output_weight[index]
            ) + self.visible_bias[index]
            probabilities = torch.softmax(logits / temperature, dim=1)
            current = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            ).squeeze(1)
            samples[:, index] = current
            hidden_preactivation.add_(self.input_embedding[index, current])
        return samples

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_categories={self.num_categories}, "
            f"parameters={sum(parameter.numel() for parameter in self.parameters()):,}"
        )
