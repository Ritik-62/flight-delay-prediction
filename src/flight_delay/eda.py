"""Exploratory Data Analysis (EDA) module for Flight Delay Prediction."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EDARunner:
    """Performs rigorous exploratory data analysis, generates publication figures and CSV aggregates."""

    def __init__(self, output_dir: Path, figures_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir) if figures_dir else self.output_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Matplotlib styling for publication quality
        plt.rcParams.update({
            "font.sans-serif": "DejaVu Sans",
            "font.family": "sans-serif",
            "figure.autolayout": True,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#e0e0e0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
        })

    def run_eda(
        self,
        df: pd.DataFrame,
        target_col: str = "ARR_DEL15",
        airlines_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Execute full suite of EDA computations, save tables and figures."""
        logger.info("Executing comprehensive Exploratory Data Analysis...")
        results: Dict[str, pd.DataFrame] = {}

        # 1. Target Distribution
        target_counts = df[target_col].value_counts().reset_index()
        target_counts.columns = ["Target_Value", "Count"]
        target_counts["Percentage"] = target_counts["Count"] / len(df) * 100
        target_counts.to_csv(self.output_dir / "target_distribution.csv", index=False)
        results["target_distribution"] = target_counts

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        bars = ax.bar(["On-Time / Minor (<15m)", "Delayed (>=15m)"], target_counts["Count"], color=["#2b5c8f", "#d95f02"], width=0.5)
        ax.set_ylabel("Flight Count", fontsize=11)
        ax.set_title("Arrival Delay Target Class Distribution (ARR_DEL15)", fontsize=12, fontweight="bold", pad=12)
        ax.grid(axis="y")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:,}\n({h/len(df)*100:.1f}%)", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
        fig.savefig(self.figures_dir / "target_distribution.png", dpi=300)
        plt.close(fig)

        # 2. Delay Rate by Airline
        airline_agg = df.groupby("AIRLINE")[target_col].agg(["count", "mean"]).reset_index()
        airline_agg.columns = ["AIRLINE", "Flight_Count", "Delay_Rate"]
        if airlines_df is not None and "IATA_CODE" in airlines_df.columns and "AIRLINE" in airlines_df.columns:
            name_map = dict(zip(airlines_df["IATA_CODE"], airlines_df["AIRLINE"]))
            airline_agg["Airline_Name"] = airline_agg["AIRLINE"].map(name_map).fillna(airline_agg["AIRLINE"])
        else:
            airline_agg["Airline_Name"] = airline_agg["AIRLINE"]
        airline_agg = airline_agg.sort_values(by="Delay_Rate", ascending=False)
        airline_agg.to_csv(self.output_dir / "delay_rate_by_airline.csv", index=False)
        results["delay_by_airline"] = airline_agg

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        bars = ax.bar(airline_agg["AIRLINE"], airline_agg["Delay_Rate"] * 100, color="#1f77b4", edgecolor="#0e4b75")
        ax.axhline(df[target_col].mean() * 100, color="red", linestyle="--", linewidth=1.5, label=f"Average Delay Rate ({df[target_col].mean()*100:.1f}%)")
        ax.set_xlabel("Airline Carrier Code", fontsize=11)
        ax.set_ylabel("Arrival Delay Rate (%)", fontsize=11)
        ax.set_title("Arrival Delay Rate (ARR_DEL15 >= 15 min) by Airline Carrier", fontsize=12, fontweight="bold", pad=12)
        ax.legend()
        ax.grid(axis="y")
        fig.savefig(self.figures_dir / "delay_rate_by_carrier.png", dpi=300)
        plt.close(fig)

        # 3. Delay Rate by Scheduled Departure Hour
        if "SCHEDULED_DEPARTURE" in df.columns:
            sched_hour = df["SCHEDULED_DEPARTURE"].astype(str).str.zfill(4).str[:2].astype(int)
            hour_agg = df.assign(Sched_Dep_Hour=sched_hour).groupby("Sched_Dep_Hour")[target_col].agg(["count", "mean"]).reset_index()
            hour_agg.columns = ["Departure_Hour", "Flight_Count", "Delay_Rate"]
            hour_agg.to_csv(self.output_dir / "delay_rate_by_departure_hour.csv", index=False)
            results["delay_by_hour"] = hour_agg

            fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=300)
            color = "#d95f02"
            ax1.set_xlabel("Scheduled Departure Hour (Local 24h Clock)", fontsize=11)
            ax1.set_ylabel("Delay Rate (%)", color=color, fontsize=11)
            ax1.plot(hour_agg["Departure_Hour"], hour_agg["Delay_Rate"] * 100, marker="o", color=color, linewidth=2.2, label="Delay Rate")
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.set_xticks(range(0, 24))
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.bar(hour_agg["Departure_Hour"], hour_agg["Flight_Count"], alpha=0.25, color="#2b5c8f", width=0.6, label="Flight Volume")
            ax2.set_ylabel("Scheduled Flight Volume", color="#2b5c8f", fontsize=11)
            ax2.tick_params(axis="y", labelcolor="#2b5c8f")

            plt.title("Arrival Delay Rate Progression Across Scheduled Departure Hours", fontsize=12, fontweight="bold", pad=12)
            fig.savefig(self.figures_dir / "delay_rate_by_departure_hour.png", dpi=300)
            plt.close(fig)

        # 4. Delay Rate by Day of Week
        dow_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
        dow_agg = df.groupby("DAY_OF_WEEK")[target_col].agg(["count", "mean"]).reset_index()
        dow_agg.columns = ["Day_Of_Week", "Flight_Count", "Delay_Rate"]
        dow_agg["Day_Name"] = dow_agg["Day_Of_Week"].map(dow_names)
        dow_agg.to_csv(self.output_dir / "delay_rate_by_day_of_week.csv", index=False)
        results["delay_by_dow"] = dow_agg

        # 5. Delay Rate by Month
        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        month_agg = df.groupby("MONTH")[target_col].agg(["count", "mean"]).reset_index()
        month_agg.columns = ["Month", "Flight_Count", "Delay_Rate"]
        month_agg["Month_Name"] = month_agg["Month"].map(month_names)
        month_agg.to_csv(self.output_dir / "delay_rate_by_month.csv", index=False)
        results["delay_by_month"] = month_agg

        # Combined Day of Week and Month Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
        ax1.bar(dow_agg["Day_Name"], dow_agg["Delay_Rate"] * 100, color="#377eb8", width=0.55)
        ax1.axhline(df[target_col].mean() * 100, color="red", linestyle="--", label="Overall Avg")
        ax1.set_xlabel("Day of Week", fontsize=11)
        ax1.set_ylabel("Delay Rate (%)", fontsize=11)
        ax1.set_title("Delay Rate by Day of Week", fontsize=11, fontweight="bold")
        ax1.legend()
        ax1.grid(axis="y")

        ax2.plot(month_agg["Month_Name"], month_agg["Delay_Rate"] * 100, marker="s", color="#e41a1c", linewidth=2)
        ax2.axhline(df[target_col].mean() * 100, color="gray", linestyle="--")
        ax2.set_xlabel("Month of Year", fontsize=11)
        ax2.set_ylabel("Delay Rate (%)", fontsize=11)
        ax2.set_title("Seasonal Delay Rate by Month", fontsize=11, fontweight="bold")
        ax2.grid(True)
        fig.savefig(self.figures_dir / "delay_rate_by_weekday_and_month.png", dpi=300)
        plt.close(fig)

        # 6. Delay Rate by Origin Airport (Top 20 by Volume)
        top_origins = df["ORIGIN_AIRPORT"].value_counts().head(20).index
        origin_agg = df[df["ORIGIN_AIRPORT"].isin(top_origins)].groupby("ORIGIN_AIRPORT")[target_col].agg(["count", "mean"]).reset_index()
        origin_agg.columns = ["ORIGIN_AIRPORT", "Flight_Count", "Delay_Rate"]
        origin_agg = origin_agg.sort_values(by="Delay_Rate", ascending=False)
        origin_agg.to_csv(self.output_dir / "delay_rate_by_origin_airport.csv", index=False)
        results["delay_by_origin"] = origin_agg

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        bars = ax.bar(origin_agg["ORIGIN_AIRPORT"], origin_agg["Delay_Rate"] * 100, color="#4daf4a", edgecolor="#2d772a")
        ax.axhline(df[target_col].mean() * 100, color="red", linestyle="--", label="Overall Avg")
        ax.set_xlabel("Origin Airport (Top 20 by Volume)", fontsize=11)
        ax.set_ylabel("Delay Rate (%)", fontsize=11)
        ax.set_title("Arrival Delay Rate by Origin Airport (Top 20 Busiest Hubs)", fontsize=12, fontweight="bold", pad=12)
        ax.legend()
        ax.grid(axis="y")
        fig.savefig(self.figures_dir / "delay_rate_by_airport.png", dpi=300)
        plt.close(fig)

        # 7. Delay Prevalence Over Time (Daily Time Series across 365 days)
        if "FLIGHT_DATE" in df.columns:
            daily_agg = df.groupby("FLIGHT_DATE")[target_col].agg(["count", "mean"]).reset_index()
            daily_agg.columns = ["FLIGHT_DATE", "Flight_Count", "Delay_Rate"]
            daily_agg.to_csv(self.output_dir / "daily_delay_prevalence.csv", index=False)
            results["daily_prevalence"] = daily_agg

            # 7-day rolling average for smoother visualization
            daily_agg["7D_Rolling_Delay_Rate"] = daily_agg["Delay_Rate"].rolling(7, min_periods=1).mean()

            fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
            ax.plot(daily_agg["FLIGHT_DATE"], daily_agg["Delay_Rate"] * 100, alpha=0.35, color="#1f78b4", label="Daily Delay Rate")
            ax.plot(daily_agg["FLIGHT_DATE"], daily_agg["7D_Rolling_Delay_Rate"] * 100, color="#e31a1c", linewidth=2.0, label="7-Day Rolling Trend")
            ax.axhline(df[target_col].mean() * 100, color="black", linestyle="--", linewidth=1.0, label="Annual Mean")
            ax.set_xlabel("Flight Date", fontsize=11)
            ax.set_ylabel("Arrival Delay Rate (%)", fontsize=11)
            ax.set_title("Temporal Volatility of Arrival Delays Across Calendar Year", fontsize=12, fontweight="bold", pad=12)
            ax.legend(loc="upper right")
            ax.grid(True)
            fig.savefig(self.figures_dir / "delay_prevalence_over_time.png", dpi=300)
            plt.close(fig)

        # 8. Missing Values Summary
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing_Count"]
        missing_df["Missing_Percentage"] = missing_df["Missing_Count"] / len(df) * 100
        missing_df.to_csv(self.output_dir / "missing_values_summary.csv", index=False)
        results["missing_values"] = missing_df

        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
        top_missing = missing_df[missing_df["Missing_Percentage"] > 0].sort_values("Missing_Percentage", ascending=True)
        if len(top_missing) > 0:
            ax.barh(top_missing["Column"], top_missing["Missing_Percentage"], color="#984ea3")
            ax.set_xlabel("Missing Percentage (%)", fontsize=11)
            ax.set_title("Missing Values by Column in Analyzed Dataset", fontsize=12, fontweight="bold")
            ax.grid(axis="x")
        else:
            ax.text(0.5, 0.5, "No Missing Values in Cleaned Predictor Dataset", ha="center", va="center", fontsize=12)
        fig.savefig(self.figures_dir / "missing_values.png", dpi=300)
        plt.close(fig)

        logger.info(f"EDA successfully completed. Aggregates saved to {self.output_dir}, figures to {self.figures_dir}.")
        return results
