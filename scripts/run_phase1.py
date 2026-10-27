#!/usr/bin/env python3
"""Main entry point to execute the complete Phase 1 pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flight_delay.config import Config
from flight_delay.pipeline import Phase1Pipeline


def setup_logging(logs_dir: Path) -> None:
    """Configure console and file logging."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "phase1_execution.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Execute Phase 1 Flight Delay Prediction Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1.yaml",
        help="Path to YAML configuration file (default: configs/phase1.yaml)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override sample size (default: use value from config)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    cfg = Config.from_yaml(config_path, base_dir=PROJECT_ROOT)
    if args.sample_size is not None:
        cfg.sample_size = args.sample_size

    setup_logging(cfg.logs_dir)
    logging.info(f"Loaded configuration from: {config_path}")

    pipeline = Phase1Pipeline(config=cfg)
    result = pipeline.run()

    logging.info("=" * 70)
    logging.info("PHASE 1 EXECUTION COMPLETED SUCCESSFULLY!")
    logging.info(f"PDF Report: {result['pdf_report']}")
    logging.info(f"Markdown Report: {result['md_report']}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
