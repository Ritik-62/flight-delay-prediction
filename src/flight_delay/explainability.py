"""Feature importance extraction and interpretability module."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Extracts, ranks, and visualizes feature importance across models."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def explain_logistic_regression(
        self,
        lr_model: Any,
        output_dir: Path,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Extract and rank Logistic Regression normalized coefficients."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        coefs = lr_model.coef_[0]
        n_features = min(len(self.feature_names), len(coefs))
        df_coefs = pd.DataFrame({
            "Feature": self.feature_names[:n_features],
            "Coefficient": coefs[:n_features],
            "Abs_Coefficient": np.abs(coefs[:n_features]),
            "Direction": np.where(coefs[:n_features] > 0, "Increases Delay Risk", "Reduces Delay Risk"),
        }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)

        csv_path = output_dir / "logistic_regression_coefficients.csv"
        df_coefs.to_csv(csv_path, index=False)
        logger.info(f"Saved Logistic Regression coefficients to {csv_path}.")

        # Plot top N coefficients
        top_df = df_coefs.head(top_n).sort_values("Coefficient", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        colors = ["#d95f02" if c > 0 else "#2b5c8f" for c in top_df["Coefficient"]]
        ax.barh(top_df["Feature"], top_df["Coefficient"], color=colors)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Logistic Regression Coefficient (Log-Odds Impact)", fontsize=10)
        ax.set_title(f"Top {top_n} Logistic Regression Feature Coefficients", fontsize=11, fontweight="bold", pad=12)
        ax.grid(axis="x")

        fig.tight_layout()
        fig.savefig(output_dir / "lr_coefficients.png", dpi=300)
        plt.close(fig)
        return df_coefs

    def explain_xgboost(
        self,
        xgb_model: Any,
        output_dir: Path,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Extract and rank XGBoost feature importances (gain-based)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        importances = getattr(xgb_model, "feature_importances_", None)
        if importances is None:
            logger.warning("XGBoost model does not have feature_importances_ attribute.")
            return pd.DataFrame()

        n_features = min(len(self.feature_names), len(importances))
        df_imp = pd.DataFrame({
            "Feature": self.feature_names[:n_features],
            "Importance_Gain": importances[:n_features],
        }).sort_values("Importance_Gain", ascending=False).reset_index(drop=True)

        csv_path = output_dir / "xgboost_feature_importance.csv"
        df_imp.to_csv(csv_path, index=False)
        logger.info(f"Saved XGBoost feature importance to {csv_path}.")

        # Plot top N
        top_df = df_imp.head(top_n).sort_values("Importance_Gain", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.barh(top_df["Feature"], top_df["Importance_Gain"], color="#ff7f0e", edgecolor="#b35806")
        ax.set_xlabel("Feature Importance (Relative Information Gain)", fontsize=10)
        ax.set_title(f"Top {top_n} XGBoost Feature Importances", fontsize=11, fontweight="bold", pad=12)
        ax.grid(axis="x")

        fig.tight_layout()
        fig.savefig(output_dir / "xgb_feature_importance.png", dpi=300)
        plt.close(fig)
        return df_imp

    def compute_permutation_importance(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_dir: Path,
        model_name: str = "Model",
        max_samples: int = 3000,
        n_repeats: int = 3,
        random_state: int = 42,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """Calculate permutation importance on validation subset to measure true generalization impact."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(X_val) > max_samples:
            rng = np.random.RandomState(random_state)
            indices = rng.choice(len(X_val), size=max_samples, replace=False)
            X_sub = X_val[indices]
            y_sub = y_val[indices]
        else:
            X_sub = X_val
            y_sub = y_val

        logger.info(f"Computing permutation importance for {model_name} on {len(X_sub)} validation samples...")
        perm_res = permutation_importance(
            model,
            X_sub,
            y_sub,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )

        n_features = min(len(self.feature_names), len(perm_res.importances_mean))
        df_perm = pd.DataFrame({
            "Feature": self.feature_names[:n_features],
            "Mean_Importance_Decrease": perm_res.importances_mean[:n_features],
            "Std_Importance": perm_res.importances_std[:n_features],
        }).sort_values("Mean_Importance_Decrease", ascending=False).reset_index(drop=True)

        clean_name = model_name.lower().replace(" ", "_")
        csv_path = output_dir / f"permutation_importance_{clean_name}.csv"
        df_perm.to_csv(csv_path, index=False)

        # Plot top N
        top_df = df_perm.head(top_n).sort_values("Mean_Importance_Decrease", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
        ax.barh(top_df["Feature"], top_df["Mean_Importance_Decrease"], xerr=top_df["Std_Importance"], color="#2ca02c", alpha=0.85)
        ax.set_xlabel("Mean Decrease in ROC AUC Score", fontsize=10)
        ax.set_title(f"Validation Permutation Importance: {model_name}", fontsize=11, fontweight="bold", pad=12)
        ax.grid(axis="x")

        fig.tight_layout()
        fig.savefig(output_dir / f"permutation_importance_{clean_name}.png", dpi=300)
        plt.close(fig)
        return df_perm
