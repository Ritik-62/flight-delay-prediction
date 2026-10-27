"""Configuration management for Flight Delay Prediction pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class Config:
    """Project configuration container."""

    raw_dict: Dict[str, Any]

    # Project
    project_name: str
    phase: str
    version: str
    seed: int

    # Data
    raw_path: Path
    airlines_path: Optional[Path]
    airports_path: Optional[Path]
    sample_size: Optional[int]
    chunk_size: int

    # Target
    target_name: str
    raw_target_col: str
    threshold_minutes: int

    # Prediction Timestamp
    minutes_before_departure: int
    prediction_timestamp_description: str

    # Leakage
    denylist: List[str]
    allowlist: List[str]

    # Split
    train_ratio: float
    val_ratio: float
    test_ratio: float
    embargo_days: int
    date_column: str

    # Historical Features
    smoothing_weight: float
    min_samples: int
    historical_entities: List[str]

    # Models
    lr_params: Dict[str, Any]
    xgb_params: Dict[str, Any]

    # Evaluation
    target_precision: float
    calibration_method: str
    n_bins_calibration: int

    # Paths
    results_dir: Path
    models_dir: Path
    reports_dir: Path
    logs_dir: Path
    artifacts_dir: Path

    @classmethod
    def from_yaml(cls, path: str | Path, base_dir: Optional[Path] = None) -> "Config":
        """Load configuration from a YAML file."""
        config_path = Path(path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if base_dir is None:
            # Assume base_dir is repository root if config is under configs/
            base_dir = config_path.parent.parent

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        project_cfg = cfg.get("project", {})
        data_cfg = cfg.get("data", {})
        target_cfg = cfg.get("target", {})
        pred_cfg = cfg.get("prediction_timestamp", {})
        leakage_cfg = cfg.get("leakage", {})
        split_cfg = cfg.get("temporal_split", {})
        hist_cfg = cfg.get("historical_features", {})
        models_cfg = cfg.get("models", {})
        eval_cfg = cfg.get("evaluation", {})
        paths_cfg = cfg.get("paths", {})

        def resolve_p(p_str: Optional[str]) -> Optional[Path]:
            if not p_str:
                return None
            p = Path(p_str)
            return p if p.is_absolute() else (base_dir / p).resolve()

        return cls(
            raw_dict=cfg,
            project_name=project_cfg.get("name", "flight-delay-prediction"),
            phase=project_cfg.get("phase", "Phase 1"),
            version=project_cfg.get("version", "1.0.0"),
            seed=project_cfg.get("seed", 42),
            raw_path=resolve_p(data_cfg.get("raw_path", "data/raw/flights.parquet")),
            airlines_path=resolve_p(data_cfg.get("airlines_path")),
            airports_path=resolve_p(data_cfg.get("airports_path")),
            sample_size=data_cfg.get("sample_size"),
            chunk_size=data_cfg.get("chunk_size", 100000),
            target_name=target_cfg.get("name", "ARR_DEL15"),
            raw_target_col=target_cfg.get("raw_target_col", "ARRIVAL_DELAY"),
            threshold_minutes=target_cfg.get("threshold_minutes", 15),
            minutes_before_departure=pred_cfg.get("minutes_before_departure", 60),
            prediction_timestamp_description=pred_cfg.get("description", ""),
            denylist=leakage_cfg.get("denylist", []),
            allowlist=leakage_cfg.get("allowlist", []),
            train_ratio=split_cfg.get("train_ratio", 0.70),
            val_ratio=split_cfg.get("val_ratio", 0.15),
            test_ratio=split_cfg.get("test_ratio", 0.15),
            embargo_days=split_cfg.get("embargo_days", 1),
            date_column=split_cfg.get("date_column", "FLIGHT_DATE"),
            smoothing_weight=hist_cfg.get("smoothing_weight", 15.0),
            min_samples=hist_cfg.get("min_samples", 5),
            historical_entities=hist_cfg.get("entities", ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE"]),
            lr_params=models_cfg.get("logistic_regression", {}),
            xgb_params=models_cfg.get("xgboost", {}),
            target_precision=eval_cfg.get("target_precision", 0.70),
            calibration_method=eval_cfg.get("calibration_method", "sigmoid"),
            n_bins_calibration=eval_cfg.get("n_bins_calibration", 10),
            results_dir=resolve_p(paths_cfg.get("results_dir", "results")),
            models_dir=resolve_p(paths_cfg.get("models_dir", "models")),
            reports_dir=resolve_p(paths_cfg.get("reports_dir", "reports")),
            logs_dir=resolve_p(paths_cfg.get("logs_dir", "logs")),
            artifacts_dir=resolve_p(paths_cfg.get("artifacts_dir", "artifacts")),
        )

    def ensure_directories(self) -> None:
        """Create all required project output directories."""
        for p in [self.results_dir, self.models_dir, self.reports_dir, self.logs_dir, self.artifacts_dir]:
            p.mkdir(parents=True, exist_ok=True)
        # Subdirectories
        for sub in ["eda", "data_quality", "metrics", "predictions", "feature_importance", "calibration", "error_analysis", "figures"]:
            (self.results_dir / sub).mkdir(parents=True, exist_ok=True)
        (self.models_dir / "logistic_regression").mkdir(parents=True, exist_ok=True)
        (self.models_dir / "xgboost").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "manifests").mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "figures").mkdir(parents=True, exist_ok=True)
