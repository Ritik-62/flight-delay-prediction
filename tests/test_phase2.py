"""Unit and integration tests for Phase 2: Features, DL models, Rescheduling, and IEEE reports."""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
torch.set_num_threads(1)

from flight_delay.phase2_features import Phase2FeatureEngineer
from flight_delay.train_lightgbm import LightGBMTrainer, LIGHTGBM_AVAILABLE
from flight_delay.train_neural_net import (
    DeepLearningTrainer,
    FlightEmbeddingNet,
    TemporalFlightNet,
    TabularCategoryEncoder,
)
from flight_delay.rescheduling import ReschedulingPolicyEngine
from flight_delay.reporting_ieee import IEEEConferenceReportGenerator


@pytest.fixture
def sample_flight_data():
    """Generates synthetic flight records with turnaround links."""
    return pd.DataFrame({
        "YEAR": [2015] * 6,
        "MONTH": [3] * 6,
        "DAY": [15] * 6,
        "DAY_OF_WEEK": [7] * 6,
        "AIRLINE": ["AA", "AA", "DL", "DL", "UA", "UA"],
        "FLIGHT_NUMBER": [101, 102, 201, 202, 301, 302],
        "TAIL_NUMBER": ["N100AA", "N100AA", "N200DL", "N200DL", "N300UA", "N300UA"],
        "ORIGIN_AIRPORT": ["JFK", "LAX", "ATL", "ORD", "SFO", "DEN"],
        "DESTINATION_AIRPORT": ["LAX", "JFK", "ORD", "ATL", "DEN", "SFO"],
        "SCHEDULED_DEPARTURE": [600, 1200, 800, 1400, 900, 1500],
        "SCHEDULED_ARRIVAL": [900, 1500, 1030, 1630, 1130, 1730],
        "SCHEDULED_TIME": [300, 300, 150, 150, 150, 150],
        "DISTANCE": [2475, 2475, 606, 606, 967, 967],
        "ARRIVAL_DELAY": [10.0, -5.0, 45.0, 20.0, 0.0, 12.0],
        "ARR_DEL15": [0, 0, 1, 1, 0, 0],
    })


def test_phase2_feature_extraction(sample_flight_data):
    """Verify congestion and turnaround features are extracted without leakage."""
    fe = Phase2FeatureEngineer()
    df_feat = fe.extract_features(sample_flight_data)

    assert "ORIGIN_CONGESTION_1H" in df_feat.columns
    assert "DEST_CONGESTION_1H" in df_feat.columns
    assert "CARRIER_ORIGIN_SHARE" in df_feat.columns
    assert "INBOUND_SCHEDULED_BUFFER_MIN" in df_feat.columns
    assert "INBOUND_DELAY_KNOWN" in df_feat.columns
    assert "INBOUND_UNARRIVED_AT_T60" in df_feat.columns

    # Verify zero leakage audit
    fe.assert_zero_leakage(["ORIGIN_CONGESTION_1H", "DEST_CONGESTION_1H", "INBOUND_DELAY_KNOWN"])


def test_pytorch_embedding_net_shapes():
    """Verify forward pass output dimensions of Tabular Embedding MLP."""
    batch_size = 16
    num_cont = 10
    model = FlightEmbeddingNet(
        num_continuous_features=num_cont,
        num_airlines=10,
        airline_dim=8,
        num_origins=50,
        origin_dim=16,
        num_dests=50,
        dest_dim=16,
    )

    x_cont = torch.randn(batch_size, num_cont)
    x_air = torch.randint(0, 10, (batch_size,))
    x_orig = torch.randint(0, 50, (batch_size,))
    x_dest = torch.randint(0, 50, (batch_size,))

    logits = model(x_cont, x_air, x_orig, x_dest)
    assert logits.shape == (batch_size,)


def test_pytorch_temporal_net_shapes():
    """Verify forward pass output dimensions of Temporal Recurrent Net."""
    batch_size = 8
    seq_len = 4
    input_dim = 12

    model = TemporalFlightNet(input_dim=input_dim, hidden_dim=32, num_layers=1)
    x_seq = torch.randn(batch_size, seq_len, input_dim)
    logits = model(x_seq)
    assert logits.shape == (batch_size,)


def test_rescheduling_engine_policy(sample_flight_data):
    """Verify tactical rescheduling recommendation rules and counterfactual outcomes."""
    engine = ReschedulingPolicyEngine(threshold_risk=0.50, delay_buffer_minutes=30)
    probs = np.array([0.20, 0.40, 0.75, 0.65, 0.10, 0.30])

    rec_df = engine.recommend_actions(sample_flight_data, probs)
    assert "RECOMMENDED_ACTION" in rec_df.columns
    assert "POST_ACTION_ESTIMATED_RISK" in rec_df.columns
    assert "EXPECTED_DELAY_MINUTES_SAVED" in rec_df.columns

    # High-risk flights (indices 2 and 3) must be intervened
    assert rec_df.loc[2, "RECOMMENDED_ACTION"] != "MAINTAIN_SCHEDULE"
    assert rec_df.loc[3, "RECOMMENDED_ACTION"] != "MAINTAIN_SCHEDULE"

    # Low-risk flights must maintain schedule
    assert rec_df.loc[0, "RECOMMENDED_ACTION"] == "MAINTAIN_SCHEDULE"
    assert rec_df.loc[4, "RECOMMENDED_ACTION"] == "MAINTAIN_SCHEDULE"

    # Simulate policy sweep
    sweep_df = engine.simulate_policy_sweep(
        sample_flight_data, probs, sample_flight_data["ARR_DEL15"].values
    )
    assert len(sweep_df) > 0
    assert "Net Benefit (Min Saved - Friction)" in sweep_df.columns


def test_ieee_report_generator(tmp_path):
    """Verify IEEE two-column paper generation in both LaTeX and PDF."""
    generator = IEEEConferenceReportGenerator(reports_dir=tmp_path, figures_dir=tmp_path)

    metrics_df = pd.DataFrame([{
        "Model": "LightGBM",
        "ROC AUC": 0.612,
        "PR AUC": 0.228,
        "Precision": 0.320,
        "Recall": 0.250,
        "F1 Score": 0.280,
        "Brier Score": 0.142,
        "Latency (ms/1k)": 1.45,
    }])

    resched_df = pd.DataFrame([{
        "Risk Threshold (tau)": 0.45,
        "Intervention Rate (%)": 14.2,
        "Delay Coverage Rate (%)": 48.5,
        "Estimated Averted Delayed Flights": 3200,
        "Net Benefit (Min Saved - Friction)": 54200.0,
    }])

    tex_out = tmp_path / "test_paper.tex"
    pdf_out = tmp_path / "test_paper.pdf"

    generator.generate_latex_document(metrics_df, resched_df, tex_out)
    assert tex_out.exists()
    assert tex_out.stat().st_size > 0

    generator.generate_pdf_document(metrics_df, resched_df, {}, pdf_out)
    assert pdf_out.exists()
    assert pdf_out.stat().st_size > 0
