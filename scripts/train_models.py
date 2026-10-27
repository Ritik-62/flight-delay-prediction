#!/usr/bin/env python3
"""Train baseline models independently."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flight_delay.config import Config
from flight_delay.pipeline import Phase1Pipeline


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Models")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(PROJECT_ROOT / args.config, base_dir=PROJECT_ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pipeline = Phase1Pipeline(config=cfg)
    pipeline.run()
    logging.info("Model training pipeline finished.")


if __name__ == "__main__":
    main()
