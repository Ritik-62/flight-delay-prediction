"""Probability calibration module implementing Platt (sigmoid) scaling and ECE computation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, Dict[str, Any]]:
    """Compute Expected Calibration Error (ECE) and binning statistics.
    
    ECE = sum_b (|acc(b) - conf(b)| * |b| / N)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    bin_stats = []
    n_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count > 0:
            avg_prob = float(np.mean(y_prob[mask]))
            empirical_pos = float(np.mean(y_true[mask]))
            abs_diff = abs(empirical_pos - avg_prob)
            ece += (count / n_samples) * abs_diff
            bin_stats.append({
                "bin_index": i,
                "bin_range": [round(bins[i], 2), round(bins[i+1], 2)],
                "count": count,
                "mean_predicted_prob": round(avg_prob, 4),
                "empirical_fraction_pos": round(empirical_pos, 4),
                "calibration_error": round(abs_diff, 4),
            })
        else:
            bin_stats.append({
                "bin_index": i,
                "bin_range": [round(bins[i], 2), round(bins[i+1], 2)],
                "count": 0,
                "mean_predicted_prob": None,
                "empirical_fraction_pos": None,
                "calibration_error": 0.0,
            })

    return round(float(ece), 4), {"ece": round(float(ece), 4), "bins": bin_stats}


class ModelCalibrator:
    """Calibrator fitted on validation set raw predictions to refine probability estimates."""

    def __init__(self, method: str = "sigmoid", n_bins: int = 10):
        self.method = method
        self.n_bins = n_bins
        self.calibrator: Optional[Any] = None
        self.fitted: bool = False

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "ModelCalibrator":
        """Fit sigmoid (logistic) calibrator on validation probabilities."""
        logger.info(f"Fitting {self.method} probability calibrator on validation predictions...")
        # For sigmoid calibration, train 1D logistic regression on log-odds or raw probabilities
        epsilon = 1e-6
        clipped_probs = np.clip(val_probs, epsilon, 1 - epsilon)
        logit_features = np.log(clipped_probs / (1 - clipped_probs)).reshape(-1, 1)

        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        self.calibrator.fit(logit_features, y_val)
        self.fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to raw predicted probabilities."""
        if not self.fitted or self.calibrator is None:
            return probs
        epsilon = 1e-6
        clipped = np.clip(probs, epsilon, 1 - epsilon)
        logit_feats = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        return self.calibrator.predict_proba(logit_feats)[:, 1]

    def save(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "ModelCalibrator":
        return joblib.load(filepath)


def plot_reliability_diagram(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    calibrated_probs: Optional[np.ndarray],
    save_path: Path,
    model_name: str = "Model",
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Plot publication-quality reliability curve comparing uncalibrated vs calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Perfectly calibrated reference
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # Uncalibrated curve
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, raw_probs, n_bins=n_bins, strategy="uniform")
    brier_raw = brier_score_loss(y_true, raw_probs)
    ece_raw, _ = compute_expected_calibration_error(y_true, raw_probs, n_bins=n_bins)
    ax.plot(mean_pred_raw, frac_pos_raw, "s-", color="#d95f02", label=f"Raw (Brier={brier_raw:.4f}, ECE={ece_raw:.4f})")

    brier_cal = None
    ece_cal = None
    if calibrated_probs is not None:
        frac_pos_cal, mean_pred_cal = calibration_curve(y_true, calibrated_probs, n_bins=n_bins, strategy="uniform")
        brier_cal = brier_score_loss(y_true, calibrated_probs)
        ece_cal, _ = compute_expected_calibration_error(y_true, calibrated_probs, n_bins=n_bins)
        ax.plot(mean_pred_cal, frac_pos_cal, "o-", color="#1b9e77", label=f"Calibrated (Brier={brier_cal:.4f}, ECE={ece_cal:.4f})")

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Empirical Fraction of Positives (Delayed)", fontsize=11)
    ax.set_title(f"Reliability Diagram: {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.grid(True)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return {
        "raw_brier": round(float(brier_raw), 5),
        "raw_ece": ece_raw,
        "calibrated_brier": round(float(brier_cal), 5) if brier_cal is not None else None,
        "calibrated_ece": ece_cal,
    }
