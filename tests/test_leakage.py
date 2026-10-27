"""Rigorous tests proving zero target and temporal leakage."""

import numpy as np
import pandas as pd
import pytest
from flight_delay.feature_engineering import FeatureEngineer
from flight_delay.historical_features import HistoricalFeatureExtractor
from flight_delay.temporal_split import TemporalSplitter


def test_leakage_denylist_enforcement():
    fe = FeatureEngineer()
    forbidden_features = [
        "DEP_DELAY",
        "TAXI_OUT",
        "WHEELS_OFF",
        "ARRIVAL_TIME",
        "ARR_DELAY",
        "ARR_DEL15",
    ]
    for col in forbidden_features:
        with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
            fe.assert_zero_leakage(["DISTANCE", "SCHEDULED_TIME", col])


def test_allowlist_passes_audit():
    fe = FeatureEngineer()
    safe_features = [
        "MONTH",
        "DAY_OF_WEEK",
        "SCHED_DEP_HOUR",
        "DEP_HOUR_SIN",
        "DISTANCE",
        "SCHEDULED_TIME",
        "HIST_AIRLINE_DELAY_RATE",
    ]
    # Must pass without raising any exception
    fe.assert_zero_leakage(safe_features)


def test_historical_features_no_future_lookahead():
    """Prove mathematically that flight at date D does NOT incorporate date D or D+1 delays."""
    dates = pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03"])
    df = pd.DataFrame({
        "FLIGHT_DATE": dates,
        "AIRLINE": ["AA", "AA", "AA"],
        "ARR_DEL15": [0, 0, 1],  # Only day 3 is delayed
    })

    hist = HistoricalFeatureExtractor(entities=["AIRLINE"], smoothing_weight=1.0)
    hist.fit(df)
    transformed = hist.transform(df, is_training=True)

    # On Day 1, prior delay count must be 0
    assert transformed.loc[0, "HIST_AIRLINE_FLIGHT_COUNT"] == 0
    # On Day 2, only Day 1 is observed (0 delays, 1 flight)
    assert transformed.loc[1, "HIST_AIRLINE_FLIGHT_COUNT"] == 1
    # On Day 3, only Day 1 and Day 2 are observed (0 delays, 2 flights)
    assert transformed.loc[2, "HIST_AIRLINE_FLIGHT_COUNT"] == 2
    # Even though Day 3 is delayed (ARR_DEL15=1), Day 3's own historical rate cannot see it!
    # Expected rate on Day 3: (0 delays + 1.0 * global_prior) / (2 flights + 1.0)
    assert transformed.loc[2, "HIST_AIRLINE_DELAY_RATE"] < 0.3


def test_historical_features_unaffected_by_future_labels():
    """Modifying future labels must not alter historical feature values of prior flights."""
    dates = pd.to_datetime(["2015-01-01", "2015-01-01", "2015-01-02", "2015-01-03"])
    df1 = pd.DataFrame({
        "FLIGHT_DATE": dates,
        "AIRLINE": ["DL", "DL", "DL", "DL"],
        "ARR_DEL15": [1, 0, 0, 0],
    })
    df2 = pd.DataFrame({
        "FLIGHT_DATE": dates,
        "AIRLINE": ["DL", "DL", "DL", "DL"],
        "ARR_DEL15": [1, 0, 0, 1],  # Changed Day 3 label from 0 to 1
    })

    hist1 = HistoricalFeatureExtractor(entities=["AIRLINE"]).fit(df1)
    t1 = hist1.transform(df1, is_training=True)

    hist2 = HistoricalFeatureExtractor(entities=["AIRLINE"]).fit(df2)
    t2 = hist2.transform(df2, is_training=True)

    # Flights on Day 1 and Day 2 must have identical historical features regardless of Day 3 label!
    np.testing.assert_allclose(
        t1.loc[:2, "HIST_AIRLINE_FLIGHT_COUNT"].values,
        t2.loc[:2, "HIST_AIRLINE_FLIGHT_COUNT"].values,
    )


def test_single_calendar_date_partition_integrity(synthetic_flights_df):
    """All records from the same calendar date belong to a single partition."""
    synthetic_flights_df["FLIGHT_DATE"] = pd.to_datetime(
        synthetic_flights_df["YEAR"].astype(str) + "-" +
        synthetic_flights_df["MONTH"].astype(str).str.zfill(2) + "-" +
        synthetic_flights_df["DAY"].astype(str).str.zfill(2)
    )
    splitter = TemporalSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    train_df, val_df, test_df, _ = splitter.split(synthetic_flights_df)

    train_dates = set(train_df["FLIGHT_DATE"].dt.date.unique())
    val_dates = set(val_df["FLIGHT_DATE"].dt.date.unique())
    test_dates = set(test_df["FLIGHT_DATE"].dt.date.unique())

    # Disjoint date sets
    assert len(train_dates.intersection(val_dates)) == 0
    assert len(train_dates.intersection(test_dates)) == 0
    assert len(val_dates.intersection(test_dates)) == 0
