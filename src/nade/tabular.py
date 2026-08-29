"""White Wine data loading for real-valued RNADE experiments."""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

WHITE_WINE_URLS = (
    (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-white.csv"
    ),
    "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
)
WHITE_WINE_FILENAME = "winequality-white.csv"
WHITE_WINE_ROWS = 4_898
WHITE_WINE_COLUMNS = (
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
)


@dataclass(frozen=True)
class Standardizer:
    """Feature-wise statistics fitted on the training split only."""

    mean: Tensor
    std: Tensor

    def transform(self, values: Tensor) -> Tensor:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: Tensor) -> Tensor:
        return values * self.std + self.mean


@dataclass
class TabularLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    train_generator: torch.Generator
    feature_names: tuple[str, ...]
    standardizer: Standardizer


def _csv_from_download(payload: bytes, url: str) -> bytes:
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            matches = [
                name
                for name in archive.namelist()
                if name.endswith(WHITE_WINE_FILENAME)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"archive must contain exactly one {WHITE_WINE_FILENAME}"
                )
            return archive.read(matches[0])
    return payload


def _parse_white_wine(
    payload: bytes,
    *,
    expected_rows: int = WHITE_WINE_ROWS,
) -> tuple[tuple[str, ...], Tensor]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("White Wine data must be UTF-8 text") from error

    reader = csv.reader(io.StringIO(text), delimiter=";")
    try:
        columns = tuple(next(reader))
    except StopIteration as error:
        raise ValueError("White Wine CSV is empty") from error
    if columns != WHITE_WINE_COLUMNS:
        raise ValueError(f"unexpected White Wine columns: {columns}")

    rows = list(reader)
    if len(rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} White Wine rows, found {len(rows)}"
        )
    try:
        values = np.asarray(rows, dtype=np.float32)
    except ValueError as error:
        raise ValueError("White Wine CSV contains non-numeric values") from error
    if values.shape != (expected_rows, len(WHITE_WINE_COLUMNS)):
        raise ValueError(f"unexpected White Wine shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("White Wine CSV contains missing or non-finite values")

    # The ordinal quality score is not a continuous input to the density model.
    return columns[:-1], torch.from_numpy(values[:, :-1].copy())


def download_white_wine(data_dir: str | Path = "data") -> Path:
    """Download the official UCI White Wine CSV and validate its schema."""
    destination = Path(data_dir) / "white_wine" / WHITE_WINE_FILENAME
    if destination.exists():
        _parse_white_wine(destination.read_bytes())
        return destination

    errors: list[str] = []
    for url in WHITE_WINE_URLS:
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "nade/0.1 (+https://archive.ics.uci.edu/)"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                csv_payload = _csv_from_download(response.read(), url)
            _parse_white_wine(csv_payload)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".csv.part")
            temporary.write_bytes(csv_payload)
            temporary.replace(destination)
            return destination
        except (
            OSError,
            urllib.error.URLError,
            ValueError,
            zipfile.BadZipFile,
        ) as error:
            errors.append(f"{url}: {error}")

    details = "\n".join(errors)
    raise RuntimeError(
        "Unable to download schema-valid UCI White Wine data. "
        "Check DNS/proxy settings and retry. Attempts:\n" + details
    )


def get_white_wine_loaders(
    data_dir: str | Path,
    *,
    batch_size: int = 128,
    train_size: int | None = None,
    validation_size: int = 500,
    test_size: int = 500,
    num_workers: int = 0,
    seed: int = 42,
    download: bool = True,
) -> TabularLoaders:
    """Build deterministic standardized splits for White Wine density estimation."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    path = (
        download_white_wine(data_dir)
        if download
        else Path(data_dir) / "white_wine" / WHITE_WINE_FILENAME
    )
    if not path.exists():
        raise FileNotFoundError(
            f"White Wine data not found at {path}; run with --download-data"
        )
    feature_names, features = _parse_white_wine(path.read_bytes())

    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be positive")
    available_train_size = len(features) - validation_size - test_size
    if available_train_size <= 0:
        raise ValueError("validation and test splits leave no training data")
    if train_size is None:
        train_size = available_train_size
    if not 0 < train_size <= available_train_size:
        raise ValueError(
            f"train_size must be between 1 and {available_train_size}"
        )

    split_generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(features), generator=split_generator)
    train_indices = order[:train_size]
    validation_start = available_train_size
    validation_indices = order[
        validation_start : validation_start + validation_size
    ]
    test_indices = order[-test_size:]

    raw_train = features[train_indices]
    mean = raw_train.mean(dim=0)
    std = raw_train.std(dim=0, unbiased=False)
    if torch.any(std <= torch.finfo(std.dtype).eps):
        raise ValueError("White Wine training split contains a constant feature")
    standardizer = Standardizer(mean=mean, std=std)

    def dataset(indices: Tensor) -> TensorDataset:
        values = standardizer.transform(features[indices])
        labels = torch.zeros(len(indices), dtype=torch.long)
        return TensorDataset(values, labels)

    train_generator = torch.Generator().manual_seed(seed + 1)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    return TabularLoaders(
        train=DataLoader(
            dataset(train_indices),
            shuffle=True,
            generator=train_generator,
            **common,
        ),
        validation=DataLoader(
            dataset(validation_indices), shuffle=False, **common
        ),
        test=DataLoader(dataset(test_indices), shuffle=False, **common),
        train_generator=train_generator,
        feature_names=feature_names,
        standardizer=standardizer,
    )
