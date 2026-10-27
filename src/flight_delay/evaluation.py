"""Model evaluation, operational threshold optimization, and metric reporting."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def find_operational_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = 0.70,
) -> Tuple[float, Dict[str, Any]]:
    """Select decision threshold using validation data only.
    
    Aims to satisfy operational precision target (e.g., 70%).
    If unachievable, selects the threshold maximizing the validation F1 score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Avoid div by zero in f1
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)

    # Search for thresholds meeting target precision
    qualifying_indices = np.where(precisions[:-1] >= target_precision)[0]
    
    if len(qualifying_indices) > 0:
        # Choose the threshold that maximizes recall while achieving precision >= target
        best_idx = qualifying_indices[np.argmax(recalls[qualifying_indices])]
        selected_threshold = float(thresholds[best_idx])
        rule = f"Target precision of >= {target_precision*100:.0f}% met on validation set"
        achieved_target = True
    else:
        # Fallback to threshold that maximizes F1
        best_idx = int(np.argmax(f1_scores[:-1]))
        selected_threshold = float(thresholds[best_idx])
        rule = f"Target precision {target_precision*100:.0f}% unachievable; selected threshold maximizing validation F1"
        achieved_target = False

    sel_p = float(precisions[best_idx])
    sel_r = float(recalls[best_idx])
    sel_f1 = float(f1_scores[best_idx])

    result = {
        "selected_threshold": round(selected_threshold, 4),
        "rule_applied": rule,
        "target_precision_requested": target_precision,
        "achieved_target_on_validation": achieved_target,
        "validation_precision_at_threshold": round(sel_p, 4),
        "validation_recall_at_threshold": round(sel_r, 4),
        "validation_f1_at_threshold": round(sel_f1, 4),
    }
    logger.info(f"Threshold Optimization: {result}")
    return selected_threshold, result


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute full suite of classification and probabilistic evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_total = len(y_true)
    prevalence = float(np.mean(y_true))

    # ROC AUC & PR AUC (Average Precision)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = prevalence

    # Threshold-dependent metrics
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Probabilistic losses
    brier = float(brier_score_loss(y_true, y_prob))
    try:
        loss = float(log_loss(y_true, y_prob))
    except Exception:
        loss = None

    metrics = {
        "n_samples": n_total,
        "prevalence": round(prevalence, 4),
        "threshold": round(threshold, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc_average_precision": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "brier_score": round(brier, 5),
        "log_loss": round(loss, 5) if loss is not None else None,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    return metrics


def benchmark_inference_speed(
    model: Any,
    X_sample: np.ndarray,
    n_runs: int = 5,
) -> Dict[str, float]:
    """Measure inference latency in milliseconds per 1,000 samples."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(X_sample)
        else:
            _ = model.predict(X_sample)
        times.append(time.perf_counter() - t0)

    median_sec = np.median(times)
    ms_per_1k = (median_sec / len(X_sample)) * 1000 * 1000
    return {
        "inference_ms_per_1000_samples": round(float(ms_per_1k), 2),
        "benchmark_sample_size": len(X_sample),
    }


def plot_evaluation_curves(
    y_true: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    figures_dir: Path,
    title_suffix: str = "Test Set",
) -> None:
    """Generate and save publication-quality ROC, PR curves, and Confusion Matrices."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    prevalence = float(np.mean(y_true))

    colors = {"Logistic Regression": "#1f77b4", "XGBoost": "#ff7f0e"}

    # 1. ROC Curves
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)")
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        c = colors.get(name, None)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=c, linewidth=2)

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title(f"Receiver Operating Characteristic (ROC) — {title_suffix}", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.savefig(figures_dir / "roc_curves.png", dpi=300)
    plt.close(fig)

    # 2. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.axhline(prevalence, color="gray", linestyle="--", label=f"Baseline Prevalence ({prevalence*100:.1f}%)")
    for name, probs in probs_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        c = colors.get(name, None)
        ax.plot(rec, prec, label=f"{name} (PR AUC = {ap:.3f})", color=c, linewidth=2)

    ax.set_xlabel("Recall (Coverage of Delayed Flights)", fontsize=11)
    ax.set_ylabel("Precision (Accuracy of Delay Predictions)", fontsize=11)
    ax.set_title(f"Precision-Recall (PR) Curves — {title_suffix}", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.savefig(figures_dir / "precision_recall_curves.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrices(
    y_true: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    """Generate confusion matrix visual heatmaps."""
    figures_dir = Path(figures_dir)
    n_models = len(preds_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), dpi=300)
    if n_models == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cax = ax.matshow(cm, cmap="Blues", alpha=0.8)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / np.sum(cm) * 100
                ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=11)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["On-Time (0)", "Delayed (1)"])
        ax.set_yticklabels(["On-Time (0)", "Delayed (1)"])
        ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_title(f"Confusion Matrix: {name}", fontsize=11, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png", dpi=300)
    plt.close(fig)
