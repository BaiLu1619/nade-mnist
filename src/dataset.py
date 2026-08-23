"""Download grayscale image datasets and build reproducible data loaders."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.preprocess import BinarizationMode, Binarize, ToCategorical

DatasetName = Literal["mnist", "fashion_mnist", "kmnist"]
Representation = Literal["binary", "categorical"]


class _FlatFolders:
    """Store IDX files directly below each dataset's directory."""

    @property
    def raw_folder(self) -> str:
        return self.root

    @property
    def processed_folder(self) -> str:
        return self.root


class MirroredMNIST(_FlatFolders, datasets.MNIST):
    """MNIST with maintained HTTPS mirrors before torchvision's defaults."""

    mirrors = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://azureopendatastorage.blob.core.windows.net/mnist/",
        *datasets.MNIST.mirrors,
    ]



class FlatFashionMNIST(_FlatFolders, datasets.FashionMNIST):
    """Fashion-MNIST stored below data/fashion_mnist."""


class FlatKMNIST(_FlatFolders, datasets.KMNIST):
    """Kuzushiji-MNIST stored below data/kmnist."""


DATASET_CLASSES = {
    "mnist": MirroredMNIST,
    "fashion_mnist": FlatFashionMNIST,
    "kmnist": FlatKMNIST,
}

DATASET_TITLES = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "kmnist": "KMNIST",
}


@dataclass
class ImageLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    train_generator: torch.Generator


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _transform(
    representation: Representation,
    binarization: BinarizationMode,
    threshold: float,
):
    if representation == "binary":
        return transforms.Compose(
            [transforms.ToTensor(), Binarize(binarization, threshold)]
        )
    if representation == "categorical":
        return transforms.Compose([transforms.PILToTensor(), ToCategorical()])
    raise ValueError("representation must be 'binary' or 'categorical'")


def _dataset_root(data_dir: str | Path, dataset_name: DatasetName) -> Path:
    root = Path(data_dir)
    return root if dataset_name == "mnist" else root / dataset_name


def _dataset_class(dataset_name: DatasetName):
    try:
        return DATASET_CLASSES[dataset_name]
    except KeyError as error:
        choices = ", ".join(DATASET_CLASSES)
        raise ValueError(f"unknown dataset {dataset_name!r}; choose from: {choices}") from error


def download_dataset(
    data_dir: str | Path = "data",
    dataset_name: DatasetName = "mnist",
) -> None:
    """Download and verify both splits of a supported grayscale dataset."""
    dataset_class = _dataset_class(dataset_name)
    destination = _dataset_root(data_dir, dataset_name)
    title = DATASET_TITLES[dataset_name]
    print(f"downloading {title} to: {destination.resolve()}")
    dataset_class(root=str(destination), train=True, download=True)
    dataset_class(root=str(destination), train=False, download=True)
    print(f"{title} is ready.")


def download_mnist(data_dir: str | Path = "data") -> None:
    """Backward-compatible MNIST download helper."""
    download_dataset(data_dir, "mnist")


def _load_dataset(
    data_dir: str | Path,
    *,
    dataset_name: DatasetName,
    train: bool,
    transform,
    download: bool,
) -> datasets.VisionDataset:
    dataset_class = _dataset_class(dataset_name)
    root = _dataset_root(data_dir, dataset_name)
    try:
        return dataset_class(
            root=str(root), train=train, transform=transform, download=download
        )
    except RuntimeError as error:
        proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        proxy_hint = (
            f" The current HTTPS proxy is {proxy!r}; make sure that proxy is running, "
            "or unset http_proxy and https_proxy before retrying."
            if proxy
            else " Check DNS/network access and retry."
        )
        raise RuntimeError(
            f"Unable to download a checksum-valid {DATASET_TITLES[dataset_name]} "
            f"copy from the configured torchvision mirrors.{proxy_hint}"
        ) from error


def get_image_loaders(
    data_dir: str | Path,
    *,
    dataset_name: DatasetName = "mnist",
    representation: Representation = "binary",
    batch_size: int = 128,
    validation_size: int = 10_000,
    num_workers: int = 0,
    seed: int = 42,
    binarization: BinarizationMode = "fixed",
    threshold: float = 0.5,
    download: bool = True,
) -> ImageLoaders:
    """Build deterministic train/validation/test splits and loaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    transform = _transform(representation, binarization, threshold)
    train_dataset = _load_dataset(
        data_dir,
        dataset_name=dataset_name,
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = _load_dataset(
        data_dir,
        dataset_name=dataset_name,
        train=False,
        transform=transform,
        download=download,
    )
    if not 0 < validation_size < len(train_dataset):
        raise ValueError(
            f"validation_size must be between 1 and {len(train_dataset) - 1}"
        )

    split_generator = torch.Generator().manual_seed(seed)
    train_size = len(train_dataset) - validation_size
    train_subset, validation_subset = random_split(
        train_dataset, [train_size, validation_size], generator=split_generator
    )
    train_generator = torch.Generator().manual_seed(seed + 1)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": _seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_subset, shuffle=True, generator=train_generator, **common
    )
    validation_loader = DataLoader(validation_subset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return ImageLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        train_generator=train_generator,
    )


# Keep the original API available for the binary MNIST workflow.
MNISTLoaders = ImageLoaders


def get_mnist_loaders(
    data_dir: str | Path,
    **kwargs,
) -> ImageLoaders:
    return get_image_loaders(
        data_dir,
        dataset_name="mnist",
        representation="binary",
        **kwargs,
    )
