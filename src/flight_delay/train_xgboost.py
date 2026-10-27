"""XGBoost baseline model trainer with early stopping and class weighting."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False


class XGBoostTrainer:
    """Trains and evaluates an XGBoost gradient boosting classifier."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        scale_pos_weight: Any = "auto",
        eval_metric: str = "logloss",
        early_stopping_rounds: int = 25,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": eval_metric,
            "early_stopping_rounds": early_stopping_rounds,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self.model: Optional[Any] = None
        self.training_metadata: Dict[str, Any] = {}

    def is_available(self) -> bool:
        """Check if xgboost is installed and importable."""
        return XGBOOST_AVAILABLE

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Fit XGBClassifier on training data using validation for early stopping."""
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is not installed in the environment.")

        fit_params = dict(self.params)
        
        # Calculate scale_pos_weight from training labels if set to 'auto'
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        if fit_params["scale_pos_weight"] == "auto":
            calculated_ratio = float(neg_count / max(1, pos_count))
            fit_params["scale_pos_weight"] = round(calculated_ratio, 3)
            logger.info(f"Auto-calculated scale_pos_weight from train class ratio: {calculated_ratio:.3f}")

        logger.info(f"Training XGBoost with params: {fit_params} on {X_train.shape} samples...")
        start_time = time.time()

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

        # Instantiate XGBClassifier
        early_stop = fit_params.pop("early_stopping_rounds", 25)
        self.model = xgb.XGBClassifier(
            **fit_params,
            early_stopping_rounds=early_stop if eval_set else None,
        )

        if eval_set:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

        duration = time.time() - start_time
        logger.info(f"XGBoost training completed in {duration:.2f}s.")

        best_iteration = getattr(self.model, "best_iteration", None)
        best_score = getattr(self.model, "best_score", None)

        self.training_metadata = {
            "model_type": "XGBoost",
            "hyperparameters": {**fit_params, "early_stopping_rounds": early_stop},
            "training_duration_seconds": round(duration, 3),
            "n_samples": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "best_iteration": int(best_iteration) if best_iteration is not None else None,
            "best_score": float(best_score) if best_score is not None else None,
        }
        return self.model, self.training_metadata

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Output predicted probabilities for positive class (ARR_DEL15=1)."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, model_dir: Path) -> Path:
        """Save model and metadata to directory."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"

        joblib.dump(self.model, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.training_metadata, f, indent=2)

        logger.info(f"Saved XGBoost model to {model_path}.")
        return model_path

    @classmethod
    def load(cls, model_dir: Path) -> "XGBoostTrainer":
        """Load serialized model from directory."""
        model_dir = Path(model_dir)
        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        trainer = cls(**meta["hyperparameters"])
        trainer.model = joblib.load(model_path)
        trainer.training_metadata = meta
        return trainer
