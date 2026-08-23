"""Train, evaluate, and sample from NADE."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.categorical_model import CategoricalNADE
from src.dataset import DATASET_TITLES, ImageLoaders, get_image_loaders
from src.model import NADE, bits_per_dimension
from src.utils import seed_everything
from src.visualization import collect_images, save_comparison_grid


def run_epoch(
    model: NADE | CategoricalNADE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> dict[str, float]:
    """Train for one epoch, or evaluate when optimizer is omitted."""
    training = optimizer is not None
    model.train(training)
    total_nll = 0.0
    total_items = 0

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, _ in loader:
            inputs = images.flatten(1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = model.nll(inputs)
            if training:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_nll += loss.item() * inputs.shape[0]
            total_items += inputs.shape[0]

    mean_nll = total_nll / total_items
    return {
        "nll": mean_nll,
        "bits_per_dim": bits_per_dimension(mean_nll, model.input_dim),
    }


def run_pipeline(config: dict) -> None:
    """Train, test, sample, and save a comparison figure."""
    seed_everything(config["seed"])
    dataset_name = config.get("dataset", "mnist")
    representation = config.get("representation", "binary")
    num_categories = config.get("num_categories", 256)
    if representation == "categorical" and num_categories != 256:
        raise ValueError(
            "categorical grayscale mode preserves the original pixels and "
            "therefore requires num_categories: 256"
        )
    loaders: ImageLoaders = get_image_loaders(
        config["data_dir"],
        dataset_name=dataset_name,
        representation=representation,
        batch_size=config["batch_size"],
        validation_size=config["validation_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
        binarization=config["binarization"],
        threshold=config["threshold"],
    )
    if representation == "binary":
        model: NADE | CategoricalNADE = NADE(
            hidden_dim=config["hidden_dim"],
            init_std=config["init_std"],
        )
        model_label = "Binary NADE"
    elif representation == "categorical":
        model = CategoricalNADE(
            hidden_dim=config["hidden_dim"],
            num_categories=num_categories,
            init_std=config["init_std"],
        )
        model_label = f"Categorical NADE ({num_categories} levels)"
    else:
        raise ValueError("representation must be 'binary' or 'categorical'")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    print(
        f"dataset: {DATASET_TITLES[dataset_name]} | "
        f"model: {model_label} | device: cpu"
    )
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(
            model,
            loaders.train,
            optimizer,
            config["grad_clip"],
        )
        validation_metrics = run_epoch(model, loaders.validation)
        print(
            f"epoch {epoch:02d}/{config['epochs']} | "
            f"train NLL {train_metrics['nll']:.3f} | "
            f"validation NLL {validation_metrics['nll']:.3f} | "
            f"validation bits/dim {validation_metrics['bits_per_dim']:.4f}"
        )

    test_metrics = run_epoch(model, loaders.test)
    print(
        f"test NLL {test_metrics['nll']:.3f} | "
        f"test bits/dim {test_metrics['bits_per_dim']:.4f}"
    )

    model.eval()
    generator = torch.Generator().manual_seed(config["seed"] + 10_000)
    samples = model.sample(
        config["num_samples"], generator=generator
    ).view(-1, 1, 28, 28)
    real_images = collect_images(loaders.test, config["num_samples"])
    if representation == "categorical":
        scale = float(num_categories - 1)
        real_images = real_images.float() / scale
        samples = samples.float() / scale
    save_comparison_grid(
        real_images,
        samples,
        "comparison.png",
        nrow=max(1, int(math.sqrt(config["num_samples"]))),
        model_label=model_label,
    )
    print("comparison image: comparison.png")
