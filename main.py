"""Run the complete NADE-MNIST pipeline with one command."""

import argparse

from src.dataset import download_mnist
from src.train import run_pipeline
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="NADE-MNIST")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="download and verify MNIST, then exit",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.download_data:
        download_mnist(config["data_dir"])
    else:
        run_pipeline(config)


if __name__ == "__main__":
    main()
