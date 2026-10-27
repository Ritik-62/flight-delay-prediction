"""Tests for chronological data partitioning and date consistency."""

import pandas as pd
import pytest
from flight_delay.temporal_split import TemporalSplitter


def test_temporal_split_order(synthetic_flights_df):
    synthetic_flights_df["FLIGHT_DATE"] = pd.to_datetime(
        synthetic_flights_df["YEAR"].astype(str) + "-" +
        synthetic_flights_df["MONTH"].astype(str).str.zfill(2) + "-" +
        synthetic_flights_df["DAY"].astype(str).str.zfill(2)
    )
    splitter = TemporalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df, manifest = splitter.split(synthetic_flights_df)

    max_train_date = train_df["FLIGHT_DATE"].max()
    min_val_date = val_df["FLIGHT_DATE"].min()
    max_val_date = val_df["FLIGHT_DATE"].max()
    min_test_date = test_df["FLIGHT_DATE"].min()

    # Chronological ordering strictly preserved
    assert max_train_date < min_val_date
    assert max_val_date < min_test_date

    assert manifest["train"]["record_count"] == len(train_df)
    assert manifest["validation"]["record_count"] == len(val_df)
    assert manifest["test"]["record_count"] == len(test_df)


def test_invalid_split_ratios():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        TemporalSplitter(train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)
