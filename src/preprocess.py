"""MNIST preprocessing: convert grayscale pixels to binary values."""

from typing import Literal

import torch
from torch import Tensor

BinarizationMode = Literal["fixed", "stochastic"]


class Binarize:
    """Convert a [0, 1] image tensor to a binary tensor."""

    def __init__(self, mode: BinarizationMode = "fixed", threshold: float = 0.5):
        if mode not in {"fixed", "stochastic"}:
            raise ValueError("mode must be 'fixed' or 'stochastic'")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.mode = mode
        self.threshold = threshold

    def __call__(self, image: Tensor) -> Tensor:
        if self.mode == "fixed":
            return (image >= self.threshold).to(torch.float32)
        return torch.bernoulli(image).to(torch.float32)
