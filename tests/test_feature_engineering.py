"""Unit tests for feature engineering transformations, cyclical encodings, and edge cases."""

import numpy as np
import pandas as pd
import pytest
from flight_delay.feature_engineering import FeatureEngineer
from flight_delay.preprocessing import PreprocessingPipeline


def test_cyclical_encodings_bounded_range():
    df = pd.DataFrame({
        "YEAR": [2015] * 24,
        "MONTH": list(range(1, 13)) * 2,
        "DAY": [15] * 24,
        "DAY_OF_WEEK": ([1, 2, 3, 4, 5, 6, 7] * 4)[:24],
        "SCHEDULED_DEPARTURE": [f"{h:02d}00" for h in range(24)],
        "SCHEDULED_ARRIVAL": [f"{(h+2)%24:02d}00" for h in range(24)],
        "FLIGHT_DATE": pd.date_range("2015-01-01", periods=24, freq="D"),
        "ORIGIN_AIRPORT": ["JFK"] * 24,
        "DESTINATION_AIRPORT": ["LAX"] * 24,
    })

    fe = FeatureEngineer()
    df_feats = fe.extract_features(df)

    # Sine and cosine bounds
    for col in ["DEP_HOUR_SIN", "DEP_HOUR_COS", "DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS"]:
        assert df_feats[col].min() >= -1.0 - 1e-6
        assert df_feats[col].max() <= 1.0 + 1e-6

    # Circular identity: sin^2 + cos^2 == 1
    dep_radius = df_feats["DEP_HOUR_SIN"]**2 + df_feats["DEP_HOUR_COS"]**2
    np.testing.assert_allclose(dep_radius, 1.0, atol=1e-5)


def test_overnight_flight_detection():
    df = pd.DataFrame({
        "SCHEDULED_DEPARTURE": ["2300", "0800"],  # 11 PM and 8 AM
        "SCHEDULED_ARRIVAL": ["0600", "1200"],    # 6 AM (crosses midnight) and 12 PM (same day)
        "FLIGHT_DATE": pd.to_datetime(["2015-01-01", "2015-01-01"]),
        "MONTH": [1, 1],
        "DAY_OF_WEEK": [4, 4],
    })

    fe = FeatureEngineer()
    df_feats = fe.extract_features(df)

    assert df_feats.loc[0, "IS_OVERNIGHT"] == 1  # 2300 -> 0600 is overnight
    assert df_feats.loc[1, "IS_OVERNIGHT"] == 0  # 0800 -> 1200 is same-day


def test_preprocessing_unknown_categories():
    train_df = pd.DataFrame({
        "DISTANCE": [1000, 2000, 1500],
        "SCHEDULED_TIME": [120, 240, 180],
        "SCHED_DEP_HOUR": [8, 14, 18],
        "SCHED_DEP_MINUTE": [0, 30, 45],
        "DEP_HOUR_SIN": [0.5, 0.2, -0.3],
        "DEP_HOUR_COS": [0.8, -0.9, 0.9],
        "DOW_SIN": [0.0, 0.5, 0.8],
        "DOW_COS": [1.0, 0.8, 0.6],
        "MONTH_SIN": [0.0, 0.5, 0.8],
        "MONTH_COS": [1.0, 0.8, 0.6],
        "IS_WEEKEND": [0, 0, 1],
        "IS_OVERNIGHT": [0, 1, 0],
        "HIST_AIRLINE_DELAY_RATE": [0.18, 0.22, 0.15],
        "HIST_AIRLINE_FLIGHT_COUNT": [100, 200, 150],
        "HIST_ORIGIN_AIRPORT_DELAY_RATE": [0.20, 0.19, 0.17],
        "HIST_ORIGIN_AIRPORT_FLIGHT_COUNT": [500, 400, 300],
        "HIST_DESTINATION_AIRPORT_DELAY_RATE": [0.15, 0.25, 0.18],
        "HIST_DESTINATION_AIRPORT_FLIGHT_COUNT": [300, 600, 200],
        "HIST_ROUTE_DELAY_RATE": [0.18, 0.20, 0.16],
        "HIST_ROUTE_FLIGHT_COUNT": [50, 80, 40],
        "AIRLINE": ["AA", "DL", "UA"],
        "ORIGIN_AIRPORT": ["JFK", "LAX", "ORD"],
        "DESTINATION_AIRPORT": ["SFO", "MIA", "SEA"],
    })

    pipe = PreprocessingPipeline(top_airports_k=2)
    X_train = pipe.fit_transform(train_df)

    # Test with totally unseen airline and airport
    test_df = train_df.copy()
    test_df.loc[0, "AIRLINE"] = "NEW_AIRLINE_XYZ"
    test_df.loc[0, "ORIGIN_AIRPORT"] = "UNKNOWN_HUB_ABC"

    # Must transform without raising error
    X_test = pipe.transform(test_df)
    assert X_test.shape == X_train.shape
    assert not np.isnan(X_test).any()
