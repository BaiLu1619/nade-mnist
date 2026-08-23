"""Run the complete NADE-MNIST pipeline with one command."""

import argparse

from src.dataset import download_dataset
from src.train import run_pipeline
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="NADE-MNIST")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="download and verify the configured dataset, then exit",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.download_data:
        download_dataset(
            config["data_dir"],
            config.get("dataset", "mnist"),
        )
    else:
        run_pipeline(config)


if __name__ == "__main__":
    main()
