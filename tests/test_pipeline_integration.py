"""Integration test running small end-to-end synthetic pipeline to verify software correctness."""

import tempfile
from pathlib import Path
import yaml
from flight_delay.config import Config
from flight_delay.pipeline import Phase1Pipeline


def test_end_to_end_synthetic_pipeline(synthetic_flights_df):
    """Run full pipeline on synthetic dataset in temporary sandbox directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        raw_path = tmp_root / "synthetic_flights.parquet"
        synthetic_flights_df.to_parquet(raw_path)

        config_dict = {
            "project": {"name": "test-pipeline", "phase": "Phase 1 Test", "version": "0.1", "seed": 42},
            "data": {
                "raw_path": str(raw_path),
                "sample_size": None,
                "chunk_size": 1000,
            },
            "target": {"name": "ARR_DEL15", "raw_target_col": "ARRIVAL_DELAY", "threshold_minutes": 15},
            "prediction_timestamp": {"minutes_before_departure": 60, "description": "T-60 min"},
            "leakage": {
                "denylist": ["ARR_DELAY", "DEP_DELAY", "TAXI_OUT", "WHEELS_OFF", "CANCELLED", "DIVERTED"],
                "allowlist": ["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL", "SCHEDULED_TIME", "DISTANCE"],
            },
            "temporal_split": {"train_ratio": 0.60, "val_ratio": 0.20, "test_ratio": 0.20, "embargo_days": 0, "date_column": "FLIGHT_DATE"},
            "historical_features": {"smoothing_weight": 5.0, "min_samples": 2, "entities": ["AIRLINE", "ORIGIN_AIRPORT"]},
            "models": {
                "logistic_regression": {"max_iter": 50, "C": 1.0},
                "xgboost": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1},
            },
            "evaluation": {"target_precision": 0.60, "calibration_method": "sigmoid", "n_bins_calibration": 5},
            "paths": {
                "results_dir": str(tmp_root / "results"),
                "models_dir": str(tmp_root / "models"),
                "reports_dir": str(tmp_root / "reports"),
                "logs_dir": str(tmp_root / "logs"),
                "artifacts_dir": str(tmp_root / "artifacts"),
            },
        }

        cfg_file = tmp_root / "test_config.yaml"
        with open(cfg_file, "w") as f:
            yaml.safe_dump(config_dict, f)

        cfg = Config.from_yaml(cfg_file, base_dir=tmp_root)
        pipeline = Phase1Pipeline(cfg)
        result = pipeline.run()

        assert result["status"] == "SUCCESS"
        assert Path(result["pdf_report"]).exists()
        assert Path(result["pdf_report"]).stat().st_size > 0
        assert Path(result["md_report"]).exists()
