"""LightGBM model trainer with validation early stopping and hyperparameter tuning."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    logger.warning(f"LightGBM import failed: {e}")
    lgb = None
    LIGHTGBM_AVAILABLE = False


class LightGBMTrainer:
    """Trains, tunes, and evaluates a LightGBM gradient boosting classifier."""

    def __init__(
        self,
        n_estimators: int = 350,
        num_leaves: int = 31,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        scale_pos_weight: Any = "auto",
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_samples": min_child_samples,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "objective": "binary",
            "verbose": -1,
        }
        self.model: Optional[Any] = None
        self.training_metadata: Dict[str, Any] = {}

    def is_available(self) -> bool:
        return LIGHTGBM_AVAILABLE

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """Tune key hyperparameters using validation PR AUC."""
        if not LIGHTGBM_AVAILABLE:
            return self.params

        try:
            import optuna
            from sklearn.metrics import average_precision_score
        except ImportError:
            logger.info("Optuna not available; using default hyperparameters.")
            return self.params

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        auto_ratio = float(neg_count / max(1, pos_count))

        def objective(trial):
            num_leaves = trial.suggest_int("num_leaves", 15, 63)
            max_depth = trial.suggest_int("max_depth", 4, 10)
            lr = trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
            colsample = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)

            clf = lgb.LGBMClassifier(
                n_estimators=150,
                num_leaves=num_leaves,
                max_depth=max_depth,
                learning_rate=lr,
                colsample_bytree=colsample,
                subsample=subsample,
                subsample_freq=1,
                scale_pos_weight=auto_ratio,
                random_state=42,
                n_jobs=1,
                verbose=-1,
            )
            clf.fit(X_train, y_train)
            val_probs = clf.predict_proba(X_val)[:, 1]
            return average_precision_score(y_val, val_probs)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        logger.info(f"LightGBM tuning completed. Best trial PR AUC: {study.best_value:.4f}, params: {best_params}")
        self.params.update(best_params)
        return self.params

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        tune: bool = False,
        n_tuning_trials: int = 10,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Fit LGBMClassifier on training data."""
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM is not available in the environment.")

        if tune and X_val is not None and y_val is not None:
            self.tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=n_tuning_trials)

        fit_params = dict(self.params)
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        if fit_params.get("scale_pos_weight") == "auto":
            calculated_ratio = float(neg_count / max(1, pos_count))
            fit_params["scale_pos_weight"] = round(calculated_ratio, 3)

        start_time = time.time()
        self.model = lgb.LGBMClassifier(**fit_params)

        if X_val is not None and y_val is not None:
            callbacks = [lgb.early_stopping(stopping_rounds=25, verbose=False)]
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss",
                callbacks=callbacks,
            )
        else:
            self.model.fit(X_train, y_train)

        duration = time.time() - start_time
        logger.info(f"LightGBM training completed in {duration:.2f}s.")

        best_iteration = getattr(self.model, "best_iteration_", None)
        self.training_metadata = {
            "model_type": "LightGBM",
            "hyperparameters": fit_params,
            "training_duration_seconds": round(duration, 3),
            "n_samples": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "best_iteration": int(best_iteration) if best_iteration is not None else None,
        }
        return self.model, self.training_metadata

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Output predicted probabilities for ARR_DEL15=1."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, model_dir: Path) -> Path:
        """Serialize model to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"

        joblib.dump(self.model, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.training_metadata, f, indent=2)

        logger.info(f"Saved LightGBM model to {model_path}.")
        return model_path

    @classmethod
    def load(cls, model_dir: Path) -> "LightGBMTrainer":
        """Load serialized model."""
        model_dir = Path(model_dir)
        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        trainer = cls(**meta.get("hyperparameters", {}))
        trainer.model = joblib.load(model_path)
        trainer.training_metadata = meta
        return trainer
