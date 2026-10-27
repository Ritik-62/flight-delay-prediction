#!/usr/bin/env python3
"""Run Exploratory Data Analysis independently."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flight_delay.config import Config
from flight_delay.data_loader import DataLoader
from flight_delay.data_validation import DataValidator
from flight_delay.eda import EDARunner


def main():
    parser = argparse.ArgumentParser(description="Run EDA on Flight Data")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(PROJECT_ROOT / args.config, base_dir=PROJECT_ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    loader = DataLoader(cfg.raw_path, cfg.airlines_path, cfg.airports_path)
    raw_df = loader.load_data(sample_size=cfg.sample_size, random_seed=cfg.seed)
    airlines_df = loader.load_airlines()

    validator = DataValidator(target_name=cfg.target_name, raw_target_col=cfg.raw_target_col)
    clean_df, dq_report = validator.clean_and_validate(raw_df, save_report_path=cfg.results_dir / "data_quality" / "data_quality_report.json")

    eda_runner = EDARunner(cfg.results_dir / "eda", cfg.results_dir / "figures")
    eda_runner.run_eda(clean_df, target_col=cfg.target_name, airlines_df=airlines_df)
    logging.info("EDA completed successfully.")


if __name__ == "__main__":
    main()
