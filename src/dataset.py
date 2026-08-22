"""Download MNIST and build reproducible data loaders."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.preprocess import BinarizationMode, Binarize


class MirroredMNIST(datasets.MNIST):
    """MNIST with maintained HTTPS mirrors before torchvision's defaults."""

    mirrors = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://azureopendatastorage.blob.core.windows.net/mnist/",
        *datasets.MNIST.mirrors,
    ]

    @property
    def raw_folder(self) -> str:
        return self.root

    @property
    def processed_folder(self) -> str:
        return self.root


@dataclass
class MNISTLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    train_generator: torch.Generator


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _transform(mode: BinarizationMode, threshold: float):
    return transforms.Compose([transforms.ToTensor(), Binarize(mode, threshold)])


def download_mnist(data_dir: str | Path = "data") -> None:
    """Download and verify both MNIST splits without starting training."""
    destination = Path(data_dir)
    print(f"downloading MNIST to: {destination.resolve()}")
    MirroredMNIST(root=str(destination), train=True, download=True)
    MirroredMNIST(root=str(destination), train=False, download=True)
    print("MNIST is ready (60,000 train images and 10,000 test images).")


def _load_mnist(
    data_dir: str | Path,
    *,
    train: bool,
    transform,
    download: bool,
) -> MirroredMNIST:
    try:
        return MirroredMNIST(
            root=str(data_dir), train=train, transform=transform, download=download
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
            "Unable to download a checksum-valid MNIST copy from the CVDF, Azure, "
            f"or torchvision mirrors.{proxy_hint}"
        ) from error


def get_mnist_loaders(
    data_dir: str | Path,
    *,
    batch_size: int = 128,
    validation_size: int = 10_000,
    num_workers: int = 0,
    seed: int = 42,
    binarization: BinarizationMode = "fixed",
    threshold: float = 0.5,
    download: bool = True,
) -> MNISTLoaders:
    """Build deterministic train/validation/test splits and loaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 < validation_size < 60_000:
        raise ValueError("validation_size must be between 1 and 59,999")

    transform = _transform(binarization, threshold)
    train_dataset = _load_mnist(
        data_dir, train=True, transform=transform, download=download
    )
    test_dataset = _load_mnist(
        data_dir, train=False, transform=transform, download=download
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
    return MNISTLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        train_generator=train_generator,
    )
