"""Command-line interface for training and sampling NADE models."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from nade.config import load_config
from nade.data import download_dataset
from nade.training import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nade",
        description=(
            "Train and sample binary, categorical, or continuous Neural "
            "Autoregressive Distribution Estimators."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="experiment configuration",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="download and verify the configured dataset, then exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.download_data:
        try:
            download_dataset(
                config["data_dir"],
                config["dataset"],
            )
        except RuntimeError as error:
            raise SystemExit(str(error)) from None
        return
    run_pipeline(config)
