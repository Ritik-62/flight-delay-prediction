"""Chronological data splitting module to prevent future-looking leakage."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalSplitter:
    """Partitions dataset strictly by calendar dates to prevent temporal leakage."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        date_column: str = "FLIGHT_DATE",
        embargo_days: int = 0,
    ):
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Split ratios must sum to 1.0, got: {train_ratio} + {val_ratio} + {test_ratio} = {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.date_column = date_column
        self.embargo_days = embargo_days
        self.manifest: Dict[str, Any] = {}

    def split(
        self,
        df: pd.DataFrame,
        target_col: str = "ARR_DEL15",
        save_manifest_path: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Split dataframe into train, validation, and test subsets based on calendar dates."""
        if self.date_column not in df.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in DataFrame.")

        # Ensure datetime type
        if not np.issubdtype(df[self.date_column].dtype, np.datetime64):
            df = df.copy()
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        # Get sorted unique calendar dates
        unique_dates = np.sort(df[self.date_column].dt.date.unique())
        n_dates = len(unique_dates)

        if n_dates < 3:
            raise ValueError(f"Insufficient unique dates ({n_dates}) for 3-way temporal split.")

        n_train_dates = int(np.floor(n_dates * self.train_ratio))
        n_val_dates = int(np.floor(n_dates * self.val_ratio))
        # Test gets all remaining dates
        train_dates = set(unique_dates[:n_train_dates])
        val_dates = set(unique_dates[n_train_dates : n_train_dates + n_val_dates])
        test_dates = set(unique_dates[n_train_dates + n_val_dates :])

        logger.info(
            f"Temporal Split: {len(train_dates)} train dates, {len(val_dates)} val dates, {len(test_dates)} test dates "
            f"out of {n_dates} total unique dates."
        )

        # Boolean masks ensures all records on the same calendar date belong to the exact same partition
        df_dates = df[self.date_column].dt.date
        train_mask = df_dates.isin(train_dates)
        val_mask = df_dates.isin(val_dates)
        test_mask = df_dates.isin(test_dates)

        train_df = df[train_mask].copy().sort_values(by=[self.date_column, "SCHEDULED_DEPARTURE"]).reset_index(drop=True)
        val_df = df[val_mask].copy().sort_values(by=[self.date_column, "SCHEDULED_DEPARTURE"]).reset_index(drop=True)
        test_df = df[test_mask].copy().sort_values(by=[self.date_column, "SCHEDULED_DEPARTURE"]).reset_index(drop=True)

        def partition_stats(part_df: pd.DataFrame, dates_set: set) -> Dict[str, Any]:
            if len(part_df) == 0:
                return {}
            pos_count = int(part_df[target_col].sum()) if target_col in part_df.columns else 0
            total_count = len(part_df)
            sorted_d = sorted(list(dates_set))
            return {
                "start_date": str(sorted_d[0]),
                "end_date": str(sorted_d[-1]),
                "num_calendar_dates": len(dates_set),
                "record_count": total_count,
                "record_percentage": round(total_count / len(df) * 100, 2),
                "positive_count": pos_count,
                "negative_count": total_count - pos_count,
                "prevalence_delayed_pct": round(pos_count / total_count * 100, 2) if total_count > 0 else 0.0,
            }

        manifest = {
            "total_records": len(df),
            "total_calendar_dates": n_dates,
            "date_range": {
                "start": str(unique_dates[0]),
                "end": str(unique_dates[-1]),
            },
            "train": partition_stats(train_df, train_dates),
            "validation": partition_stats(val_df, val_dates),
            "test": partition_stats(test_df, test_dates),
        }
        self.manifest = manifest

        if save_manifest_path:
            save_path = Path(save_manifest_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Saved temporal split manifest to {save_path}.")

        return train_df, val_df, test_df, manifest
