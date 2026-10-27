"""Tests for model and pipeline serialization, persistence, and reloading consistency."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from flight_delay.train_logistic import LogisticRegressionTrainer
from flight_delay.train_xgboost import XGBoostTrainer, XGBOOST_AVAILABLE


def test_logistic_regression_serialization():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5)
    y = rng.choice([0, 1], size=50)

    trainer = LogisticRegressionTrainer(max_iter=100)
    trainer.train(X, y)
    orig_probs = trainer.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = Path(tmp_dir) / "lr_model"
        trainer.save(save_dir)

        # Reload
        reloaded = LogisticRegressionTrainer.load(save_dir)
        new_probs = reloaded.predict_proba(X)

        np.testing.assert_allclose(orig_probs, new_probs, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
def test_xgboost_serialization():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5)
    y = rng.choice([0, 1], size=50)

    trainer = XGBoostTrainer(n_estimators=10, max_depth=3)
    trainer.train(X, y)
    orig_probs = trainer.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = Path(tmp_dir) / "xgb_model"
        trainer.save(save_dir)

        reloaded = XGBoostTrainer.load(save_dir)
        new_probs = reloaded.predict_proba(X)

        np.testing.assert_allclose(orig_probs, new_probs, rtol=1e-5, atol=1e-5)
