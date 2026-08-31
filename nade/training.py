"""Train, evaluate, and sample from NADE."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from nade.config import seed_everything
from nade.data import DATASET_TITLES, ImageLoaders, get_image_loaders
from nade.models import CategoricalNADE, NADE, RNADE, bits_per_dimension
from nade.reporting import collect_vectors, save_samples_csv, save_statistics_csv
from nade.tabular import TabularLoaders, get_white_wine_loaders
from nade.visualization import collect_images, save_comparison_grid


def run_epoch(
    model: NADE | CategoricalNADE | RNADE,
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
    """Train, test, sample, and save representation-appropriate outputs."""
    seed_everything(config["seed"])
    dataset_name = config["dataset"]
    representation = config["representation"]
    expected_pairs = {
        "mnist": "binary",
        "fashion_mnist": "categorical",
        "white_wine": "continuous",
    }
    expected_representation = expected_pairs.get(dataset_name)
    if expected_representation is None:
        choices = ", ".join(expected_pairs)
        raise ValueError(f"unknown dataset {dataset_name!r}; choose from: {choices}")
    if representation != expected_representation:
        raise ValueError(
            f"{dataset_name} requires representation: {expected_representation}"
        )
    num_categories = config.get("num_categories", 256)
    num_components = config.get("num_components", 10)
    if representation == "categorical" and num_categories != 256:
        raise ValueError(
            "categorical grayscale mode preserves the original pixels and "
            "therefore requires num_categories: 256"
        )
    tabular = dataset_name == "white_wine"
    if tabular:
        if representation != "continuous":
            raise ValueError("white_wine requires representation: continuous")
        loaders: ImageLoaders | TabularLoaders = get_white_wine_loaders(
            config["data_dir"],
            batch_size=config["batch_size"],
            validation_size=config["validation_size"],
            train_size=config.get("train_size"),
            test_size=config.get("test_size", 500),
            num_workers=config["num_workers"],
            seed=config["seed"],
        )
        input_dim = len(loaders.feature_names)
    else:
        loaders = get_image_loaders(
            config["data_dir"],
            dataset_name=dataset_name,
            representation=representation,
            batch_size=config["batch_size"],
            validation_size=config["validation_size"],
            train_size=config.get("train_size"),
            test_size=config.get("test_size"),
            num_workers=config["num_workers"],
            seed=config["seed"],
            binarization=config.get("binarization", "fixed"),
            threshold=config.get("threshold", 0.5),
        )
        input_dim = 28 * 28
    if representation == "binary":
        model: NADE | CategoricalNADE | RNADE = NADE(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            init_std=config["init_std"],
        )
        model_label = "Binary NADE"
    elif representation == "categorical":
        model = CategoricalNADE(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_categories=num_categories,
            init_std=config["init_std"],
        )
        model_label = f"Categorical NADE ({num_categories} levels)"
    elif representation == "continuous":
        model = RNADE(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_components=num_components,
            min_std=config.get("min_std", 1e-3),
            init_std=config["init_std"],
        )
        model_label = f"RNADE ({num_components} Gaussian components)"
    else:
        raise ValueError(
            "representation must be 'binary', 'categorical', or 'continuous'"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best_epoch = 0
    best_validation_nll = math.inf
    best_state: dict[str, torch.Tensor] | None = None

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
        if validation_metrics["nll"] < best_validation_nll:
            best_epoch = epoch
            best_validation_nll = validation_metrics["nll"]
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"epoch {epoch:02d}/{config['epochs']} | "
            f"train NLL {train_metrics['nll']:.3f} | "
            f"validation NLL {validation_metrics['nll']:.3f} | "
            f"validation bits/dim {validation_metrics['bits_per_dim']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("training completed without a valid model state")
    model.load_state_dict(best_state)
    print(
        f"selected epoch {best_epoch:02d} | "
        f"best validation NLL {best_validation_nll:.3f}"
    )

    test_metrics = run_epoch(model, loaders.test)
    print(
        f"test NLL {test_metrics['nll']:.3f} | "
        f"test bits/dim {test_metrics['bits_per_dim']:.4f}"
    )

    model.eval()
    generator = torch.Generator().manual_seed(config["seed"] + 10_000)
    samples = model.sample(config["num_samples"], generator=generator)
    if tabular:
        assert isinstance(loaders, TabularLoaders)
        standardized_real = collect_vectors(loaders.test)
        real_values = loaders.standardizer.inverse_transform(standardized_real)
        generated_values = loaders.standardizer.inverse_transform(samples.cpu())
        samples_path = save_samples_csv(
            generated_values,
            loaders.feature_names,
            config.get("samples_path", "outputs/rnade_samples.csv"),
        )
        statistics_path = save_statistics_csv(
            real_values,
            generated_values,
            loaders.feature_names,
            config.get("statistics_path", "outputs/rnade_statistics.csv"),
        )
        print(f"generated samples: {samples_path}")
        print(f"feature statistics: {statistics_path}")
        return

    samples = samples.view(-1, 1, 28, 28)
    real_images = collect_images(loaders.test, config["num_samples"])
    if representation == "categorical":
        scale = float(num_categories - 1)
        real_images = real_images.float() / scale
        samples = samples.float() / scale
    elif representation == "continuous":
        samples = samples.clamp(0.0, 1.0)
    save_comparison_grid(
        real_images,
        samples,
        "comparison.png",
        nrow=max(1, int(math.sqrt(config["num_samples"]))),
        model_label=model_label,
    )
    print("comparison image: comparison.png")
