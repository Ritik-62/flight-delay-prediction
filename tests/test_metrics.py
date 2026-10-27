"""Unit tests for metric calculations, threshold selection, and probability calibration."""

import numpy as np
import pytest
from flight_delay.calibration import ModelCalibrator, compute_expected_calibration_error
from flight_delay.evaluation import compute_comprehensive_metrics, find_operational_threshold


def test_toy_metric_calculation():
    # 2 TP, 1 FP, 1 FN, 1 TN
    y_true = np.array([1, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.1])
    
    metrics = compute_comprehensive_metrics(y_true, y_prob, threshold=0.50)
    cm = metrics["confusion_matrix"]
    assert cm["true_positives"] == 2
    assert cm["false_positives"] == 1
    assert cm["true_negatives"] == 1
    assert cm["false_negatives"] == 1
    assert np.isclose(metrics["precision"], 2 / 3, atol=1e-3)
    assert np.isclose(metrics["recall"], 2 / 3, atol=1e-3)


def test_single_class_edge_case():
    # Only class 0 present
    y_true = np.array([0, 0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.1, 0.3, 0.2])

    metrics = compute_comprehensive_metrics(y_true, y_prob, threshold=0.50)
    assert metrics["confusion_matrix"]["true_positives"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_operational_threshold_search():
    # Synthetic validation set where high threshold achieves high precision
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.linspace(0.1, 0.95, 100)

    thresh, info = find_operational_threshold(y_true, y_prob, target_precision=0.75)
    assert thresh > 0.35
    assert info["validation_precision_at_threshold"] >= 0.75


def test_calibration_ece_computation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    
    ece, stats = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece >= 0.0
    assert ece <= 1.0
    assert len(stats["bins"]) == 5
