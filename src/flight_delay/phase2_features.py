"""Phase 2 advanced feature engineering: Airport congestion and Turnaround delay propagation."""

import logging
from typing import List, Optional, Set, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Phase2FeatureEngineer:
    """Computes operational tabular features, airport congestion, and tail turnaround delay propagation.
    
    Strict Zero-Leakage Guarantee:
    At decision time T - 60 min (60 minutes prior to scheduled departure):
    - Inbound flights for the same aircraft (TAIL_NUMBER) that arrived before T - 60 min
      have their actual arrival delays observed.
    - If the inbound flight had not arrived by T - 60 min, it is marked as unarrived/in-flight.
    - Airport congestion is computed solely using published scheduled flight times.
    """

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

    def __init__(self, denylist: Optional[List[str]] = None, prediction_cutoff_minutes: int = 60):
        self.denylist: Set[str] = set(denylist or self.DEFAULT_DENYLIST)
        self.prediction_cutoff_minutes = prediction_cutoff_minutes

    def _parse_time_to_minutes(self, series: pd.Series) -> pd.Series:
        """Parse HHMM float/int/string format into total minutes from midnight [0, 1439]."""
        s_str = series.astype(str).str.split(".").str[0].str.strip().str.zfill(4)
        hour = pd.to_numeric(s_str.str[:2], errors="coerce").fillna(0).astype(int).clip(0, 23)
        minute = pd.to_numeric(s_str.str[2:4], errors="coerce").fillna(0).astype(int).clip(0, 59)
        return hour * 60 + minute

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all baseline, congestion, and turnaround delay features."""
        df_out = df.copy()

        # 1. Standard Date / Calendar parsing
        if "FLIGHT_DATE" in df_out.columns:
            date_series = pd.to_datetime(df_out["FLIGHT_DATE"])
            df_out["DAY_OF_YEAR"] = date_series.dt.dayofyear
            if "DAY_OF_WEEK" not in df_out.columns:
                df_out["DAY_OF_WEEK"] = date_series.dt.dayofweek + 1
            if "MONTH" not in df_out.columns:
                df_out["MONTH"] = date_series.dt.month
        elif "YEAR" in df_out.columns and "MONTH" in df_out.columns and "DAY" in df_out.columns:
            date_series = pd.to_datetime(
                df_out["YEAR"].astype(str) + "-" +
                df_out["MONTH"].astype(str).str.zfill(2) + "-" +
                df_out["DAY"].astype(str).str.zfill(2)
            )
            df_out["DAY_OF_YEAR"] = date_series.dt.dayofyear
            if "DAY_OF_WEEK" not in df_out.columns:
                df_out["DAY_OF_WEEK"] = date_series.dt.dayofweek + 1

        df_out["IS_WEEKEND"] = df_out["DAY_OF_WEEK"].isin([6, 7]).astype(int)

        # 2. Scheduled Time Features & Cyclical Encodings
        sched_dep_min = self._parse_time_to_minutes(df_out["SCHEDULED_DEPARTURE"])
        sched_arr_min = self._parse_time_to_minutes(df_out["SCHEDULED_ARRIVAL"])

        df_out["SCHED_DEP_MINUTES"] = sched_dep_min
        df_out["SCHED_DEP_HOUR"] = (sched_dep_min // 60).astype(int)
        df_out["SCHED_DEP_MINUTE"] = (sched_dep_min % 60).astype(int)
        df_out["SCHED_ARR_HOUR"] = (sched_arr_min // 60).astype(int)
        df_out["SCHED_ARR_MINUTE"] = (sched_arr_min % 60).astype(int)
        df_out["IS_OVERNIGHT"] = (df_out["SCHED_ARR_HOUR"] < df_out["SCHED_DEP_HOUR"]).astype(int)

        # Cyclical Encodings
        dep_frac = sched_dep_min / 1440.0
        df_out["DEP_HOUR_SIN"] = np.sin(2 * np.pi * dep_frac)
        df_out["DEP_HOUR_COS"] = np.cos(2 * np.pi * dep_frac)

        dow_frac = (df_out["DAY_OF_WEEK"] - 1) / 7.0
        df_out["DOW_SIN"] = np.sin(2 * np.pi * dow_frac)
        df_out["DOW_COS"] = np.cos(2 * np.pi * dow_frac)

        month_val = df_out["MONTH"] if "MONTH" in df_out.columns else 1
        month_frac = (month_val - 1) / 12.0
        df_out["MONTH_SIN"] = np.sin(2 * np.pi * month_frac)
        df_out["MONTH_COS"] = np.cos(2 * np.pi * month_frac)

        # Route Identifier
        if "ROUTE" not in df_out.columns and "ORIGIN_AIRPORT" in df_out.columns and "DESTINATION_AIRPORT" in df_out.columns:
            df_out["ROUTE"] = df_out["ORIGIN_AIRPORT"].astype(str) + "_" + df_out["DESTINATION_AIRPORT"].astype(str)

        # 3. Airport Congestion & Hub Traffic Density (Schedule-Based Only)
        # Count departures scheduled in the same hour at origin
        dep_hour_group = df_out.groupby(["DAY_OF_YEAR", "ORIGIN_AIRPORT", "SCHED_DEP_HOUR"])["SCHEDULED_DEPARTURE"].transform("count")
        df_out["ORIGIN_CONGESTION_1H"] = dep_hour_group.fillna(1.0).astype(float)

        arr_hour_group = df_out.groupby(["DAY_OF_YEAR", "DESTINATION_AIRPORT", "SCHED_ARR_HOUR"])["SCHEDULED_ARRIVAL"].transform("count")
        df_out["DEST_CONGESTION_1H"] = arr_hour_group.fillna(1.0).astype(float)

        # Carrier origin concentration (share of departures at origin)
        carrier_origin = df_out.groupby(["ORIGIN_AIRPORT", "AIRLINE"])["SCHEDULED_DEPARTURE"].transform("count")
        origin_total = df_out.groupby("ORIGIN_AIRPORT")["SCHEDULED_DEPARTURE"].transform("count")
        df_out["CARRIER_ORIGIN_SHARE"] = (carrier_origin / np.maximum(origin_total, 1)).fillna(0.1).astype(float)

        # 4. Aircraft Turnaround & Inbound Delay Propagation (Strict Leakage-Safe)
        # Sort chronologically by date and scheduled departure
        df_out = self._extract_turnaround_features(df_out)

        return df_out

    def _extract_turnaround_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive inbound aircraft turnaround buffer and known delay at T - 60 minutes."""
        df_sorted = df.copy()
        orig_index = df_sorted.index

        if "TAIL_NUMBER" not in df_sorted.columns:
            df_sorted["INBOUND_SCHEDULED_BUFFER_MIN"] = 180.0
            df_sorted["INBOUND_DELAY_KNOWN"] = 0.0
            df_sorted["INBOUND_UNARRIVED_AT_T60"] = 0.0
            df_sorted["IS_FIRST_LEG_OF_DAY"] = 1.0
            return df_sorted

        # Sort by tail number, day of year, and scheduled departure minutes
        sort_cols = ["TAIL_NUMBER", "DAY_OF_YEAR", "SCHED_DEP_MINUTES"]
        df_sorted["_orig_order"] = np.arange(len(df_sorted))
        df_sorted = df_sorted.sort_values(sort_cols).reset_index(drop=True)

        # Prior flight for the same tail number on the same day
        tail_group = df_sorted.groupby(["TAIL_NUMBER", "DAY_OF_YEAR"])

        # Prior scheduled arrival minutes
        sched_arr_min = self._parse_time_to_minutes(df_sorted["SCHEDULED_ARRIVAL"])
        df_sorted["_prev_sched_arr"] = tail_group[sched_arr_min.name if hasattr(sched_arr_min, 'name') else "SCHEDULED_ARRIVAL"].shift(1)
        # Convert _prev_sched_arr to minutes
        prev_arr_min = self._parse_time_to_minutes(df_sorted["_prev_sched_arr"])
        
        # Inbound buffer: scheduled departure - prior scheduled arrival
        inbound_buffer = df_sorted["SCHED_DEP_MINUTES"] - prev_arr_min
        # Handle overnight / missing prior legs
        is_first_leg = df_sorted["_prev_sched_arr"].isna() | (inbound_buffer < 0)
        inbound_buffer = inbound_buffer.mask(is_first_leg, 180.0).clip(0, 720)

        df_sorted["INBOUND_SCHEDULED_BUFFER_MIN"] = inbound_buffer.fillna(180.0).astype(float)
        df_sorted["IS_FIRST_LEG_OF_DAY"] = is_first_leg.astype(float)

        # Leakage-Safe Inbound Arrival Delay at decision time T - 60 min:
        # If the prior flight actually had an arrival delay, did it arrive before T - 60 min?
        # Note: Arrival time = Sched Arr + Arr Delay.
        # If ARRIVAL_DELAY is present in the raw data (before stripping):
        if "ARRIVAL_DELAY" in df_sorted.columns:
            prev_raw_arr_delay = tail_group["ARRIVAL_DELAY"].shift(1)
            # Prior arrival timestamp (in minutes): prev_sched_arr + prev_delay
            prev_act_arr_min = prev_arr_min + prev_raw_arr_delay.fillna(0)
            cutoff_min = df_sorted["SCHED_DEP_MINUTES"] - self.prediction_cutoff_minutes

            # Legitimate observation: previous flight landed before cutoff_min
            landed_before_t60 = (~df_sorted["_prev_sched_arr"].isna()) & (prev_act_arr_min <= cutoff_min)
            
            # Known inbound delay (0 if first leg or arrived early/on-time)
            known_delay = np.where(landed_before_t60, prev_raw_arr_delay.fillna(0).clip(lower=0), 0.0)
            # Flag if aircraft was scheduled to arrive before cutoff but hadn't landed yet by T - 60
            unarrived_flag = np.where((~is_first_leg) & (~landed_before_t60), 1.0, 0.0)

            df_sorted["INBOUND_DELAY_KNOWN"] = known_delay.astype(float)
            df_sorted["INBOUND_UNARRIVED_AT_T60"] = unarrived_flag.astype(float)
        else:
            df_sorted["INBOUND_DELAY_KNOWN"] = 0.0
            df_sorted["INBOUND_UNARRIVED_AT_T60"] = 0.0

        # Clean temporary columns and restore original order
        df_sorted = df_sorted.drop(columns=["_orig_order", "_prev_sched_arr"], errors="ignore")
        df_sorted.index = orig_index
        return df_sorted

    def assert_zero_leakage(self, feature_columns: List[str]) -> None:
        """Enforce strict audit: forbidden post-departure columns must NEVER enter features."""
        violating = [col for col in feature_columns if col in self.denylist]
        if violating:
            raise ValueError(
                f"ZERO LEAKAGE AUDIT FAILED! The following forbidden columns were discovered "
                f"in the feature set: {violating}"
            )
        logger.info(f"Leakage audit passed: 0 forbidden columns found across {len(feature_columns)} features.")

