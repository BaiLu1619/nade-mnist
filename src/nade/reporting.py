"""Save generated tabular samples and distribution summaries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor


def collect_vectors(loader: Iterable) -> Tensor:
    """Collect every feature vector from a deterministic data loader."""
    batches = [batch[0].detach().cpu() for batch in loader]
    if not batches:
        raise RuntimeError("data loader is empty")
    return torch.cat(batches, dim=0)


def save_samples_csv(
    samples: Tensor,
    feature_names: tuple[str, ...],
    output: str | Path,
) -> Path:
    """Save generated samples in their original feature units."""
    if samples.ndim != 2 or samples.shape[1] != len(feature_names):
        raise ValueError("sample shape does not match feature names")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(feature_names)
        writer.writerows(samples.detach().cpu().tolist())
    return destination


def save_statistics_csv(
    real_values: Tensor,
    generated_values: Tensor,
    feature_names: tuple[str, ...],
    output: str | Path,
) -> Path:
    """Compare marginal means and standard deviations in original units."""
    expected_dim = len(feature_names)
    for name, values in (
        ("real_values", real_values),
        ("generated_values", generated_values),
    ):
        if values.ndim != 2 or values.shape[1] != expected_dim:
            raise ValueError(f"{name} shape does not match feature names")

    real_values = real_values.detach().cpu()
    generated_values = generated_values.detach().cpu()
    real_mean = real_values.mean(dim=0)
    real_std = real_values.std(dim=0, unbiased=False)
    generated_mean = generated_values.mean(dim=0)
    generated_std = generated_values.std(dim=0, unbiased=False)

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "feature",
                "real_mean",
                "real_std",
                "generated_mean",
                "generated_std",
            )
        )
        for index, feature in enumerate(feature_names):
            writer.writerow(
                (
                    feature,
                    real_mean[index].item(),
                    real_std[index].item(),
                    generated_mean[index].item(),
                    generated_std[index].item(),
                )
            )
    return destination
