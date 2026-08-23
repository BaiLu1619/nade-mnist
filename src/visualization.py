"""Create grids that compare real MNIST images and generated samples."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


def collect_images(loader: Iterable, num_images: int) -> Tensor:
    """Collect the first images from a deterministic data loader."""
    if num_images <= 0:
        raise ValueError("num_images must be positive")
    batches = []
    remaining = num_images
    for images, _ in loader:
        batches.append(images[:remaining].cpu())
        remaining -= min(remaining, images.shape[0])
        if remaining == 0:
            break
    if not batches or remaining:
        raise RuntimeError(f"data loader contains fewer than {num_images} images")
    return torch.cat(batches, dim=0)


def save_comparison_grid(
    real_images: Tensor,
    generated_images: Tensor,
    output: str | Path,
    *,
    nrow: int = 8,
    model_label: str = "NADE",
) -> None:
    """Save labeled real/generated grids side by side."""
    if real_images.ndim != 4 or generated_images.ndim != 4:
        raise ValueError("images must have shape [N, C, H, W]")
    count = min(real_images.shape[0], generated_images.shape[0])
    if count == 0:
        raise ValueError("at least one real and generated image is required")
    if nrow <= 0:
        raise ValueError("nrow must be positive")

    grid_options = {
        "nrow": min(nrow, count),
        "padding": 2,
        "pad_value": 1.0,
    }
    real_grid = to_pil_image(make_grid(real_images[:count], **grid_options)).convert("RGB")
    generated_grid = to_pil_image(
        make_grid(generated_images[:count], **grid_options)
    ).convert("RGB")

    title_height = 28
    section_gap = 12
    width = real_grid.width + section_gap + generated_grid.width
    height = title_height + max(real_grid.height, generated_grid.height)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), f"Real (n={count})", fill="black")
    canvas.paste(real_grid, (0, title_height))
    generated_x = real_grid.width + section_gap
    draw.text((generated_x + 8, 7), f"{model_label} (n={count})", fill="black")
    canvas.paste(generated_grid, (generated_x, title_height))

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
