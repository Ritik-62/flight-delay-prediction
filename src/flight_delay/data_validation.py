"""Data cleaning and validation module for flight records."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates raw flight dataset, cleans records, logs quality issues, and constructs target."""

    REQUIRED_RAW_COLUMNS = [
        "YEAR",
        "MONTH",
        "DAY",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE",
        "SCHEDULED_ARRIVAL",
        "ARRIVAL_DELAY",
    ]

    def __init__(
        self,
        target_name: str = "ARR_DEL15",
        raw_target_col: str = "ARRIVAL_DELAY",
        threshold_minutes: int = 15,
    ):
        self.target_name = target_name
        self.raw_target_col = raw_target_col
        self.threshold_minutes = threshold_minutes
        self.quality_report: Dict[str, Any] = {}

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Verify that all required columns exist in the DataFrame."""
        missing = [col for col in self.REQUIRED_RAW_COLUMNS if col not in df.columns]
        if missing:
            msg = f"Missing required columns in dataset: {missing}"
            logger.error(msg)
            return False, missing
        return True, []

    def clean_and_validate(
        self,
        df: pd.DataFrame,
        save_report_path: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform comprehensive data quality checks, exclusions, and target creation."""
        initial_rows = len(df)
        exclusions: Dict[str, int] = {}

        # 1. Schema check
        valid, missing_cols = self.validate_schema(df)
        if not valid:
            raise ValueError(f"Dataset schema invalid. Missing required columns: {missing_cols}")

        # 2. Missing values overview
        missing_counts = df.isnull().sum().to_dict()

        # 3. Handle Missing Target (Cancellations, Diversions, Unrecorded delays)
        # Flights that were cancelled or diverted do not complete their scheduled arrival
        # and therefore have no valid ARRIVAL_DELAY measurement.
        null_target_mask = df[self.raw_target_col].isnull()
        null_target_count = int(null_target_mask.sum())
        exclusions["missing_target_or_cancelled_diverted"] = null_target_count

        df_valid = df[~null_target_mask].copy()
        logger.info(f"Excluded {null_target_count:,} records with missing {self.raw_target_col}.")

        # 4. Construct Target Variable ARR_DEL15
        # 1 if ARRIVAL_DELAY >= threshold (default 15 minutes), else 0
        df_valid[self.target_name] = (df_valid[self.raw_target_col] >= self.threshold_minutes).astype(int)

        # 5. Deduplication policy:
        # A duplicate flight is defined as identical date, airline, flight number (if present),
        # origin, destination, and scheduled departure.
        dedup_cols = ["YEAR", "MONTH", "DAY", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE"]
        if "FLIGHT_NUMBER" in df_valid.columns:
            dedup_cols.append("FLIGHT_NUMBER")

        dup_mask = df_valid.duplicated(subset=dedup_cols, keep="first")
        dup_count = int(dup_mask.sum())
        exclusions["duplicate_records"] = dup_count
        if dup_count > 0:
            df_valid = df_valid[~dup_mask].copy()
            logger.info(f"Removed {dup_count:,} duplicate flight records.")

        # 6. Validate dates and construct FLIGHT_DATE
        # Check invalid year, month (1-12), day (1-31)
        invalid_dates_mask = (
            (df_valid["MONTH"] < 1) | (df_valid["MONTH"] > 12) |
            (df_valid["DAY"] < 1) | (df_valid["DAY"] > 31)
        )
        invalid_dates_count = int(invalid_dates_mask.sum())
        exclusions["invalid_date_values"] = invalid_dates_count
        if invalid_dates_count > 0:
            df_valid = df_valid[~invalid_dates_mask].copy()

        # Construct unified FLIGHT_DATE string and datetime
        df_valid["FLIGHT_DATE"] = pd.to_datetime(
            df_valid["YEAR"].astype(str) + "-" +
            df_valid["MONTH"].astype(str).str.zfill(2) + "-" +
            df_valid["DAY"].astype(str).str.zfill(2),
            errors="coerce",
        )
        unparseable_date_count = int(df_valid["FLIGHT_DATE"].isnull().sum())
        exclusions["unparseable_dates"] = unparseable_date_count
        if unparseable_date_count > 0:
            df_valid = df_valid[df_valid["FLIGHT_DATE"].notnull()].copy()

        # 7. Check impossible or suspicious numeric values
        # e.g., negative distance or negative scheduled time
        suspicious_numeric_mask = pd.Series(False, index=df_valid.index)
        if "DISTANCE" in df_valid.columns:
            suspicious_numeric_mask |= (df_valid["DISTANCE"] <= 0)
        if "SCHEDULED_TIME" in df_valid.columns:
            suspicious_numeric_mask |= (df_valid["SCHEDULED_TIME"] <= 0)

        suspicious_numeric_count = int(suspicious_numeric_mask.sum())
        exclusions["suspicious_nonpositive_distance_or_time"] = suspicious_numeric_count
        if suspicious_numeric_count > 0:
            df_valid = df_valid[~suspicious_numeric_mask].copy()

        # 8. Clean scheduled departure and arrival time representations
        # Format can be integer (e.g. 5, 2359) or string ('0005', '2359')
        def format_time_hhmm(val: Any) -> Optional[str]:
            if pd.isnull(val):
                return None
            s = str(val).split(".")[0].strip()
            if not s.isdigit():
                return None
            return s.zfill(4)[-4:]

        df_valid["SCHEDULED_DEPARTURE"] = df_valid["SCHEDULED_DEPARTURE"].apply(format_time_hhmm)
        df_valid["SCHEDULED_ARRIVAL"] = df_valid["SCHEDULED_ARRIVAL"].apply(format_time_hhmm)

        invalid_times_mask = df_valid["SCHEDULED_DEPARTURE"].isnull() | df_valid["SCHEDULED_ARRIVAL"].isnull()
        invalid_times_count = int(invalid_times_mask.sum())
        exclusions["invalid_scheduled_times"] = invalid_times_count
        if invalid_times_count > 0:
            df_valid = df_valid[~invalid_times_mask].copy()

        # 9. Clean airport codes (string, stripped, uppercase)
        df_valid["ORIGIN_AIRPORT"] = df_valid["ORIGIN_AIRPORT"].astype(str).str.strip().str.upper()
        df_valid["DESTINATION_AIRPORT"] = df_valid["DESTINATION_AIRPORT"].astype(str).str.strip().str.upper()
        df_valid["AIRLINE"] = df_valid["AIRLINE"].astype(str).str.strip().str.upper()

        # Create ROUTE feature
        df_valid["ROUTE"] = df_valid["ORIGIN_AIRPORT"] + "_" + df_valid["DESTINATION_AIRPORT"]

        final_rows = len(df_valid)
        total_dropped = initial_rows - final_rows

        # 10. Class balance summary
        target_counts = df_valid[self.target_name].value_counts().to_dict()
        pos_count = target_counts.get(1, 0)
        neg_count = target_counts.get(0, 0)
        prevalence = pos_count / final_rows if final_rows > 0 else 0.0

        report = {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "total_dropped": total_dropped,
            "drop_rate_pct": round(total_dropped / initial_rows * 100, 2) if initial_rows > 0 else 0.0,
            "exclusions_breakdown": exclusions,
            "target_distribution": {
                "negative_class_count_0": neg_count,
                "positive_class_count_1": pos_count,
                "prevalence_delayed_pct": round(prevalence * 100, 2),
                "imbalance_ratio": round(neg_count / max(1, pos_count), 2),
            },
            "date_range": {
                "min_date": str(df_valid["FLIGHT_DATE"].min().date()),
                "max_date": str(df_valid["FLIGHT_DATE"].max().date()),
                "unique_calendar_dates": int(df_valid["FLIGHT_DATE"].nunique()),
            },
            "cardinality": {
                "unique_airlines": int(df_valid["AIRLINE"].nunique()),
                "unique_origin_airports": int(df_valid["ORIGIN_AIRPORT"].nunique()),
                "unique_destination_airports": int(df_valid["DESTINATION_AIRPORT"].nunique()),
                "unique_routes": int(df_valid["ROUTE"].nunique()),
            },
            "missing_values_initial": {k: int(v) for k, v in missing_counts.items()},
        }
        self.quality_report = report

        if save_report_path:
            save_path = Path(save_report_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved data quality report to {save_path}.")

        return df_valid, report
