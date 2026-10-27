"""Logistic Regression baseline model trainer."""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class LogisticRegressionTrainer:
    """Trains, tunes, and serializes a Logistic Regression baseline classifier."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.params = {
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "random_state": random_state,
        }
        self.model: Optional[LogisticRegression] = None
        self.training_metadata: Dict[str, Any] = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[LogisticRegression, Dict[str, Any]]:
        """Fit Logistic Regression on training array."""
        logger.info(f"Training Logistic Regression with params: {self.params} on {X_train.shape} samples...")
        self.model = LogisticRegression(**self.params)

        start_time = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.model.fit(X_train, y_train)
            convergence_warnings = [str(item.message) for item in w if "converge" in str(item.message).lower()]

        duration = time.time() - start_time
        logger.info(f"Logistic Regression training completed in {duration:.2f}s.")

        self.training_metadata = {
            "model_type": "LogisticRegression",
            "hyperparameters": self.params,
            "training_duration_seconds": round(duration, 3),
            "n_samples": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_iter_": int(self.model.n_iter_[0]) if hasattr(self.model, "n_iter_") else None,
            "intercept_": float(self.model.intercept_[0]) if hasattr(self.model, "intercept_") else None,
            "convergence_warnings": convergence_warnings,
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

        logger.info(f"Saved Logistic Regression model to {model_path}.")
        return model_path

    @classmethod
    def load(cls, model_dir: Path) -> "LogisticRegressionTrainer":
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
