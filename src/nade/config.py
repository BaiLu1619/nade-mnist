"""Configuration loading and reproducibility utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the flat YAML configuration without an extra dependency."""
    config: dict[str, Any] = {}
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"invalid config line {line_number}: {raw_line}")
        key, raw_value = (part.strip() for part in line.split(":", 1))
        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            value: Any = lowered == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value.strip("\"'")
        config[key] = value
    if not config:
        raise ValueError("configuration is empty")
    return config


def seed_everything(seed: int) -> None:
    """Seed all random number generators used by the project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
