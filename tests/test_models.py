"""Unit tests verifying model training, outputs, and probability boundaries."""

import numpy as np
import pytest
from flight_delay.train_logistic import LogisticRegressionTrainer
from flight_delay.train_xgboost import XGBoostTrainer, XGBOOST_AVAILABLE


def test_logistic_regression_fit_and_predict_bounds():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    y = rng.choice([0, 1], size=100)

    trainer = LogisticRegressionTrainer(max_iter=100)
    model, meta = trainer.train(X, y)

    assert meta["model_type"] == "LogisticRegression"
    probs = trainer.predict_proba(X)
    assert len(probs) == 100
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
def test_xgboost_fit_and_predict_bounds():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    y = rng.choice([0, 1], size=100)

    trainer = XGBoostTrainer(n_estimators=10, max_depth=3)
    model, meta = trainer.train(X, y)

    assert meta["model_type"] == "XGBoost"
    probs = trainer.predict_proba(X)
    assert len(probs) == 100
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
