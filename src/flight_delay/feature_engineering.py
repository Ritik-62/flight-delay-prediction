"""Feature engineering module for calendar, cyclical, route, and static flight predictors."""

import logging
from typing import List, Optional, Tuple, Set
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Constructs leakage-safe temporal, cyclical, route, and static features."""

    DEFAULT_DENYLIST = [
        "ARR_DELAY",
        "ARR_DELAY_NEW",
        "ARR_DEL15",
        "ARR_DELAY_GROUP",
        "DEP_DELAY",
        "DEP_DELAY_NEW",
        "DEP_DEL15",
        "DEP_DELAY_GROUP",
        "DEPARTURE_TIME",
        "ARRIVAL_TIME",
        "WHEELS_OFF",
        "WHEELS_ON",
        "TAXI_OUT",
        "TAXI_IN",
        "ELAPSED_TIME",
        "AIR_TIME",
        "CANCELLATION_REASON",
        "AIR_SYSTEM_DELAY",
        "SECURITY_DELAY",
        "AIRLINE_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "WEATHER_DELAY",
        "DIVERTED",
        "CANCELLED",
    ]

    def __init__(self, denylist: Optional[List[str]] = None):
        self.denylist: Set[str] = set(denylist or self.DEFAULT_DENYLIST)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create leakage-safe tabular features from flight records."""
        df_feats = df.copy()

        # 1. Calendar & Date features
        if "FLIGHT_DATE" in df_feats.columns:
            date_series = pd.to_datetime(df_feats["FLIGHT_DATE"])
            df_feats["DAY_OF_YEAR"] = date_series.dt.dayofyear
            # Day of week: 1=Mon, 7=Sun
            if "DAY_OF_WEEK" not in df_feats.columns:
                df_feats["DAY_OF_WEEK"] = date_series.dt.dayofweek + 1
        elif "YEAR" in df_feats.columns and "MONTH" in df_feats.columns and "DAY" in df_feats.columns:
            date_series = pd.to_datetime(
                df_feats["YEAR"].astype(str) + "-" +
                df_feats["MONTH"].astype(str).str.zfill(2) + "-" +
                df_feats["DAY"].astype(str).str.zfill(2)
            )
            df_feats["DAY_OF_YEAR"] = date_series.dt.dayofyear
            if "DAY_OF_WEEK" not in df_feats.columns:
                df_feats["DAY_OF_WEEK"] = date_series.dt.dayofweek + 1

        # Weekend indicator (Saturday=6, Sunday=7)
        df_feats["IS_WEEKEND"] = df_feats["DAY_OF_WEEK"].isin([6, 7]).astype(int)

        # 2. Scheduled Departure and Arrival time parsing (HHMM format)
        def parse_hh_mm(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
            s_str = series.astype(str).str.split(".").str[0].str.strip().str.zfill(4)
            hour = pd.to_numeric(s_str.str[:2], errors="coerce").fillna(0).astype(int).clip(0, 23)
            minute = pd.to_numeric(s_str.str[2:4], errors="coerce").fillna(0).astype(int).clip(0, 59)
            return hour, minute

        if "SCHEDULED_DEPARTURE" in df_feats.columns:
            dep_hour, dep_min = parse_hh_mm(df_feats["SCHEDULED_DEPARTURE"])
            df_feats["SCHED_DEP_HOUR"] = dep_hour
            df_feats["SCHED_DEP_MINUTE"] = dep_min

        if "SCHEDULED_ARRIVAL" in df_feats.columns:
            arr_hour, arr_min = parse_hh_mm(df_feats["SCHEDULED_ARRIVAL"])
            df_feats["SCHED_ARR_HOUR"] = arr_hour
            df_feats["SCHED_ARR_MINUTE"] = arr_min

        # Overnight flight indicator: scheduled arrival hour < scheduled departure hour
        if "SCHED_DEP_HOUR" in df_feats.columns and "SCHED_ARR_HOUR" in df_feats.columns:
            # Flight is overnight if arrival hour is significantly earlier than departure hour
            df_feats["IS_OVERNIGHT"] = (df_feats["SCHED_ARR_HOUR"] < df_feats["SCHED_DEP_HOUR"]).astype(int)

        # 3. Cyclical encodings (Departure Hour, Day of Week, Month)
        if "SCHED_DEP_HOUR" in df_feats.columns:
            dep_fraction = (df_feats["SCHED_DEP_HOUR"] + df_feats.get("SCHED_DEP_MINUTE", 0) / 60.0) / 24.0
            df_feats["DEP_HOUR_SIN"] = np.sin(2 * np.pi * dep_fraction)
            df_feats["DEP_HOUR_COS"] = np.cos(2 * np.pi * dep_fraction)

        if "DAY_OF_WEEK" in df_feats.columns:
            dow_fraction = (df_feats["DAY_OF_WEEK"] - 1) / 7.0
            df_feats["DOW_SIN"] = np.sin(2 * np.pi * dow_fraction)
            df_feats["DOW_COS"] = np.cos(2 * np.pi * dow_fraction)

        if "MONTH" in df_feats.columns:
            month_fraction = (df_feats["MONTH"] - 1) / 12.0
            df_feats["MONTH_SIN"] = np.sin(2 * np.pi * month_fraction)
            df_feats["MONTH_COS"] = np.cos(2 * np.pi * month_fraction)

        # 4. Route identifier
        if "ORIGIN_AIRPORT" in df_feats.columns and "DESTINATION_AIRPORT" in df_feats.columns:
            if "ROUTE" not in df_feats.columns:
                df_feats["ROUTE"] = df_feats["ORIGIN_AIRPORT"].astype(str) + "_" + df_feats["DESTINATION_AIRPORT"].astype(str)

        return df_feats

    def assert_zero_leakage(self, feature_columns: List[str]) -> None:
        """Enforce strict compile-time / runtime audit: forbidden columns must never enter feature matrix."""
        violating = [col for col in feature_columns if col in self.denylist]
        if violating:
            raise ValueError(
                f"DATA LEAKAGE DETECTED! The following forbidden post-departure / target columns "
                f"were discovered in the model feature matrix: {violating}"
            )
        logger.info(f"Leakage audit passed: 0 forbidden columns found across {len(feature_columns)} features.")
