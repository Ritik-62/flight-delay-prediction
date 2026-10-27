"""Tests for data validation, schema enforcement, deduplication, and exclusion tracking."""

import pandas as pd
import pytest
from flight_delay.data_validation import DataValidator


def test_schema_validation_success(synthetic_flights_df):
    validator = DataValidator()
    is_valid, missing = validator.validate_schema(synthetic_flights_df)
    assert is_valid is True
    assert len(missing) == 0


def test_schema_validation_missing_required_column(synthetic_flights_df):
    df_missing = synthetic_flights_df.drop(columns=["ARRIVAL_DELAY"])
    validator = DataValidator()
    is_valid, missing = validator.validate_schema(df_missing)
    assert is_valid is False
    assert "ARRIVAL_DELAY" in missing


def test_missing_target_exclusions(synthetic_flights_df):
    validator = DataValidator(target_name="ARR_DEL15")
    clean_df, report = validator.clean_and_validate(synthetic_flights_df)
    
    # Cancellations (None in ARRIVAL_DELAY) must be excluded
    assert clean_df["ARR_DEL15"].isnull().sum() == 0
    assert report["exclusions_breakdown"]["missing_target_or_cancelled_diverted"] > 0
    assert len(clean_df) < len(synthetic_flights_df)


def test_target_binarization_exactness(synthetic_flights_df):
    validator = DataValidator(threshold_minutes=15)
    clean_df, _ = validator.clean_and_validate(synthetic_flights_df)
    
    # Target must be 1 if ARRIVAL_DELAY >= 15, else 0
    expected = (clean_df["ARRIVAL_DELAY"] >= 15).astype(int)
    pd.testing.assert_series_equal(clean_df["ARR_DEL15"], expected, check_names=False)


def test_deduplication_policy(synthetic_flights_df):
    # Duplicate a valid completed flight row (e.g. index 10)
    valid_idx = synthetic_flights_df[synthetic_flights_df["ARRIVAL_DELAY"].notnull()].index[0]
    dup_row = synthetic_flights_df.loc[[valid_idx]].copy()
    df_with_dup = pd.concat([synthetic_flights_df, dup_row], ignore_index=True)
    
    validator = DataValidator()
    clean_df, report = validator.clean_and_validate(df_with_dup)
    assert report["exclusions_breakdown"]["duplicate_records"] >= 1
