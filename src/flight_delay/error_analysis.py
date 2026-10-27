"""Error analysis module investigating false positives, false negatives, and operational risks."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Performs subgroup failure analysis across airlines, airports, and temporal dimensions."""

    def __init__(self, output_dir: Path, figures_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir) if figures_dir else self.output_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def analyze_errors(
        self,
        df_meta: pd.DataFrame,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        model_name: str = "XGBoost",
        min_subgroup_size: int = 30,
    ) -> Dict[str, pd.DataFrame]:
        """Compute error breakdown across critical operational subgroups."""
        logger.info(f"Running error analysis for {model_name} at decision threshold {threshold:.4f}...")
        y_pred = (y_prob >= threshold).astype(int)

        df_eval = df_meta.copy().reset_index(drop=True)
        df_eval["y_true"] = y_true
        df_eval["y_prob"] = y_prob
        df_eval["y_pred"] = y_pred

        # Classify prediction outcome
        df_eval["Outcome"] = "TN"
        df_eval.loc[(df_eval["y_true"] == 0) & (df_eval["y_pred"] == 1), "Outcome"] = "FP"
        df_eval.loc[(df_eval["y_true"] == 1) & (df_eval["y_pred"] == 0), "Outcome"] = "FN"
        df_eval.loc[(df_eval["y_true"] == 1) & (df_eval["y_pred"] == 1), "Outcome"] = "TP"

        results = {}
        clean_model_name = model_name.lower().replace(" ", "_")

        # 1. Error Analysis by Airline
        airline_err = df_eval.groupby("AIRLINE").agg(
            Total_Flights=("y_true", "count"),
            Actual_Delays=("y_true", "sum"),
            Predicted_Delays=("y_pred", "sum"),
            False_Positives=("Outcome", lambda s: (s == "FP").sum()),
            False_Negatives=("Outcome", lambda s: (s == "FN").sum()),
            True_Positives=("Outcome", lambda s: (s == "TP").sum()),
            True_Negatives=("Outcome", lambda s: (s == "TN").sum()),
        ).reset_index()

        airline_err["Actual_Delay_Rate"] = round(airline_err["Actual_Delays"] / airline_err["Total_Flights"], 4)
        airline_err["Subgroup_Precision"] = round(airline_err["True_Positives"] / np.maximum(1, airline_err["Predicted_Delays"]), 4)
        airline_err["Subgroup_Recall"] = round(airline_err["True_Positives"] / np.maximum(1, airline_err["Actual_Delays"]), 4)
        airline_err["FPR"] = round(airline_err["False_Positives"] / np.maximum(1, airline_err["Total_Flights"] - airline_err["Actual_Delays"]), 4)
        airline_err["FNR"] = round(airline_err["False_Negatives"] / np.maximum(1, airline_err["Actual_Delays"]), 4)

        airline_err = airline_err.sort_values("Total_Flights", ascending=False)
        airline_csv = self.output_dir / f"error_analysis_airline_{clean_model_name}.csv"
        airline_err.to_csv(airline_csv, index=False)
        results["airline"] = airline_err

        # 2. Error Analysis by Origin Airport (Top 30 Hubs)
        origin_err = df_eval.groupby("ORIGIN_AIRPORT").agg(
            Total_Flights=("y_true", "count"),
            Actual_Delays=("y_true", "sum"),
            Predicted_Delays=("y_pred", "sum"),
            False_Positives=("Outcome", lambda s: (s == "FP").sum()),
            False_Negatives=("Outcome", lambda s: (s == "FN").sum()),
            True_Positives=("Outcome", lambda s: (s == "TP").sum()),
        ).reset_index()
        origin_err = origin_err[origin_err["Total_Flights"] >= min_subgroup_size].copy()
        origin_err["Subgroup_Precision"] = round(origin_err["True_Positives"] / np.maximum(1, origin_err["Predicted_Delays"]), 4)
        origin_err["Subgroup_Recall"] = round(origin_err["True_Positives"] / np.maximum(1, origin_err["Actual_Delays"]), 4)
        origin_err = origin_err.sort_values("Total_Flights", ascending=False).head(30)
        origin_csv = self.output_dir / f"error_analysis_origin_airport_{clean_model_name}.csv"
        origin_err.to_csv(origin_csv, index=False)
        results["origin_airport"] = origin_err

        # 3. Error Analysis by Scheduled Departure Hour
        if "SCHED_DEP_HOUR" in df_eval.columns:
            hour_err = df_eval.groupby("SCHED_DEP_HOUR").agg(
                Total_Flights=("y_true", "count"),
                Actual_Delays=("y_true", "sum"),
                Predicted_Delays=("y_pred", "sum"),
                False_Positives=("Outcome", lambda s: (s == "FP").sum()),
                False_Negatives=("Outcome", lambda s: (s == "FN").sum()),
                True_Positives=("Outcome", lambda s: (s == "TP").sum()),
                True_Negatives=("Outcome", lambda s: (s == "TN").sum()),
            ).reset_index()
            hour_err["Subgroup_Precision"] = round(hour_err["True_Positives"] / np.maximum(1, hour_err["Predicted_Delays"]), 4)
            hour_err["Subgroup_Recall"] = round(hour_err["True_Positives"] / np.maximum(1, hour_err["Actual_Delays"]), 4)
            hour_csv = self.output_dir / f"error_analysis_departure_hour_{clean_model_name}.csv"
            hour_err.to_csv(hour_csv, index=False)
            results["departure_hour"] = hour_err

            # Plot error distribution across departure hours
            fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=300)
            ax1.bar(hour_err["SCHED_DEP_HOUR"] - 0.2, hour_err["False_Positives"], width=0.4, label="False Positives (FP)", color="#fc8d62", alpha=0.9)
            ax1.bar(hour_err["SCHED_DEP_HOUR"] + 0.2, hour_err["False_Negatives"], width=0.4, label="False Negatives (FN)", color="#8da0cb", alpha=0.9)
            ax1.set_xlabel("Scheduled Departure Hour", fontsize=11)
            ax1.set_ylabel("Error Counts (Flights)", fontsize=11)
            ax1.set_xticks(range(0, 24))
            ax1.set_title(f"Subgroup Error Distribution Across Departure Hours — {model_name}", fontsize=12, fontweight="bold", pad=12)
            ax1.legend(loc="upper left")
            ax1.grid(True)
            fig.tight_layout()
            fig.savefig(self.figures_dir / "error_analysis.png", dpi=300)
            plt.close(fig)

        # 4. Error Analysis by Day of Week
        if "DAY_OF_WEEK" in df_eval.columns:
            dow_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
            dow_err = df_eval.groupby("DAY_OF_WEEK").agg(
                Total_Flights=("y_true", "count"),
                Actual_Delays=("y_true", "sum"),
                Predicted_Delays=("y_pred", "sum"),
                False_Positives=("Outcome", lambda s: (s == "FP").sum()),
                False_Negatives=("Outcome", lambda s: (s == "FN").sum()),
                True_Positives=("Outcome", lambda s: (s == "TP").sum()),
            ).reset_index()
            dow_err["Day_Name"] = dow_err["DAY_OF_WEEK"].map(dow_names)
            dow_err["Subgroup_Precision"] = round(dow_err["True_Positives"] / np.maximum(1, dow_err["Predicted_Delays"]), 4)
            dow_err["Subgroup_Recall"] = round(dow_err["True_Positives"] / np.maximum(1, dow_err["Actual_Delays"]), 4)
            dow_csv = self.output_dir / f"error_analysis_day_of_week_{clean_model_name}.csv"
            dow_err.to_csv(dow_csv, index=False)
            results["day_of_week"] = dow_err

        logger.info(f"Error analysis completed. Subgroup breakdowns saved under {self.output_dir}.")
        return results
