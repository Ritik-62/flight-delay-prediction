"""Tests for historical feature extraction and smoothed Bayesian estimation."""

import numpy as np
import pandas as pd
import pytest
from flight_delay.historical_features import HistoricalFeatureExtractor


def test_empirical_bayes_smoothing_behavior():
    """Verify that low-sample categories smoothly pull toward global prior."""
    df = pd.DataFrame({
        "FLIGHT_DATE": pd.to_datetime(["2015-01-01"] * 10),
        "AIRLINE": ["RARE_CARRIER"] * 2 + ["COMMON_CARRIER"] * 8,
        "ARR_DEL15": [1, 1] + [0] * 8,  # RARE has 100% delay rate (2 flights), COMMON has 0% (8 flights)
    })

    hist = HistoricalFeatureExtractor(entities=["AIRLINE"], smoothing_weight=10.0)
    hist.fit(df)

    lookup = hist.entity_lookups["AIRLINE"]
    rare_row = lookup[lookup["AIRLINE"] == "RARE_CARRIER"].iloc[0]
    
    # Global prior is 2 / 10 = 0.20
    # Smoothed rare rate: (2 delays + 10 * 0.20) / (2 + 10) = 4 / 12 = 0.3333
    # Without smoothing it would be 1.0; with smoothing it is shrunk toward 0.20!
    assert np.isclose(rare_row["hist_rate"], 4.0 / 12.0, atol=1e-3)
    assert rare_row["hist_rate"] < 0.50


def test_unseen_category_fallback():
    """Unseen categories in test partition must safely receive the global prior."""
    train_df = pd.DataFrame({
        "FLIGHT_DATE": pd.to_datetime(["2015-01-01"] * 5),
        "AIRLINE": ["AA"] * 5,
        "ARR_DEL15": [1, 0, 0, 0, 0],  # 20% prior
    })

    hist = HistoricalFeatureExtractor(entities=["AIRLINE"]).fit(train_df)

    test_df = pd.DataFrame({
        "FLIGHT_DATE": pd.to_datetime(["2015-01-05"]),
        "AIRLINE": ["UNSEEN_NEW_AIRLINE"],
    })

    transformed = hist.transform(test_df, is_training=False)
    assert transformed.loc[0, "HIST_AIRLINE_DELAY_RATE"] == hist.global_prior
    assert transformed.loc[0, "HIST_AIRLINE_FLIGHT_COUNT"] == 0
