"""Real-valued NADE with equally weighted Gaussian mixture conditionals."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RNADE(nn.Module):
    """NADE for continuous vectors using K equally weighted Gaussians.

    For every position i, the network produces K means and K positive standard
    deviations. Mixture weights are fixed to 1 / K, matching the simple RNADE
    formulation used by this project.
    """

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim: int = 128,
        num_components: int = 10,
        min_std: float = 1e-3,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_components <= 0:
            raise ValueError("num_components must be positive")
        if min_std <= 0:
            raise ValueError("min_std must be positive")
        if init_std < 0:
            raise ValueError("init_std must be non-negative")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.min_std = min_std
        self.init_std = init_std

        # W[:, i] updates the shared hidden state after observing real value x_i.
        self.input_weight = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.mean_weight = nn.Parameter(
            torch.empty(input_dim, num_components, hidden_dim)
        )
        self.scale_weight = nn.Parameter(
            torch.empty(input_dim, num_components, hidden_dim)
        )
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.mean_bias = nn.Parameter(torch.zeros(input_dim, num_components))
        self.scale_bias = nn.Parameter(torch.zeros(input_dim, num_components))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.input_weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.mean_weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.scale_weight, mean=0.0, std=self.init_std)
        nn.init.zeros_(self.hidden_bias)
        nn.init.zeros_(self.mean_bias)
        nn.init.zeros_(self.scale_bias)

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"expected shape [batch, {self.input_dim}], got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError("continuous inputs must use a floating-point dtype")

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return means and positive standard deviations with shape [B, D, K]."""
        self._validate_input(x)
        contributions = x.unsqueeze(1) * self.input_weight.unsqueeze(0)
        prefix_sum = torch.cumsum(contributions, dim=2)
        initial = torch.zeros(
            x.shape[0], self.hidden_dim, 1, dtype=x.dtype, device=x.device
        )
        hidden_preactivation = torch.cat([initial, prefix_sum[:, :, :-1]], dim=2)
        hidden = torch.sigmoid(
            hidden_preactivation + self.hidden_bias.view(1, -1, 1)
        ).transpose(1, 2)
        means = torch.einsum("bdh,dkh->bdk", hidden, self.mean_weight)
        raw_scales = torch.einsum("bdh,dkh->bdk", hidden, self.scale_weight)
        means = means + self.mean_bias
        scales = F.softplus(raw_scales + self.scale_bias) + self.min_std
        return means, scales

    def log_prob(self, x: Tensor) -> Tensor:
        """Return log p(x) in nats for each item in the batch."""
        means, scales = self(x)
        standardized = (x.unsqueeze(-1) - means) / scales
        component_log_prob = (
            -0.5 * standardized.square()
            - torch.log(scales)
            - 0.5 * math.log(2.0 * math.pi)
        )
        conditional_log_prob = torch.logsumexp(component_log_prob, dim=-1)
        conditional_log_prob = conditional_log_prob - math.log(self.num_components)
        return conditional_log_prob.sum(dim=1)

    def nll(
        self,
        x: Tensor,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> Tensor:
        """Return Gaussian-mixture negative log-likelihood in nats."""
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
        """Generate continuous vectors one dimension at a time."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        device = self.mean_bias.device
        samples = torch.zeros(num_samples, self.input_dim, device=device)
        hidden_preactivation = self.hidden_bias.expand(num_samples, -1).clone()
        rows = torch.arange(num_samples, device=device)
        for index in range(self.input_dim):
            hidden = torch.sigmoid(hidden_preactivation)
            means = torch.einsum(
                "bh,kh->bk", hidden, self.mean_weight[index]
            ) + self.mean_bias[index]
            raw_scales = torch.einsum(
                "bh,kh->bk", hidden, self.scale_weight[index]
            ) + self.scale_bias[index]
            scales = F.softplus(raw_scales) + self.min_std
            components = torch.randint(
                self.num_components,
                (num_samples,),
                device=device,
                generator=generator,
            )
            noise = torch.randn(num_samples, device=device, generator=generator)
            current = means[rows, components] + (
                temperature * scales[rows, components] * noise
            )
            samples[:, index] = current
            hidden_preactivation.add_(
                current.unsqueeze(1) * self.input_weight[:, index]
            )
        return samples

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_components={self.num_components}, min_std={self.min_std}, "
            f"parameters={sum(parameter.numel() for parameter in self.parameters()):,}"
        )
