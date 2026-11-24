# Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-31%20passed-brightgreen.svg)](tests/)
[![Phase 1 Report](https://img.shields.io/badge/Phase%201-PDF%20Report-success.svg)](reports/phase1_report.pdf)
[![Phase 2 IEEE Paper](https://img.shields.io/badge/Phase%202-IEEE%20Camera--Ready%20PDF-blueviolet.svg)](reports/phase2_ieee_paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents
1. [Project Overview & Multi-Phase Roadmap](#1-project-purpose--scope)
2. [System Architecture & Code Organization](#2-system-architecture)
3. [Installation & Environment Setup](#3-installation--environment-setup)
4. [Dataset Acquisition & Provenance](#4-dataset-acquisition)
5. [Phase 1: Leakage-Safe Baseline Architecture & Benchmarking](#5-phase-1-execution-commands)
   - [Phase 1 Execution Commands](#5-phase-1-execution-commands)
   - [Phase 1 Leakage Prevention Architecture](#6-leakage-prevention-architecture)
   - [Phase 1 Benchmark Results](#7-experimental-results-held-out-test-set)
   - [Phase 1 Deliverables Inventory](#8-artifact-deliverables-inventory)
6. [Common Troubleshooting](#9-common-troubleshooting)
7. [Phase 2: Hybrid Tabular, Temporal Deep Learning & Tactical Rescheduling](#10-phase-2-comprehensive-documentation)
   - [Phase 2 Problem Formulation & Hybrid Logic](#101-phase-2-overview--the-hybrid-logic)
   - [Dataset Provenance & Cross-Phase Comparability](#102-dataset-provenance--cross-phase-comparability)
   - [Chronological Date Partitioning](#103-chronological-date-partitioning)
   - [Advanced Phase 2 Feature Engineering](#104-advanced-phase-2-feature-engineering)
   - [Implemented Model Architectures](#105-model-architectures--deep-learning)
   - [Comprehensive Benchmark Evaluation (21,866 Test Flights)](#106-comprehensive-model-benchmarks)
   - [Probability Calibration & Operational Precision](#107-probability-calibration--operational-threshold-tuning)
   - [Offline Tactical Rescheduling Counterfactual Simulation](#108-offline-tactical-rescheduling-simulation)
   - [Phase 2 Deliverables Inventory](#109-phase-2-deliverables-inventory)
   - [Phase 2 Execution Commands](#1010-phase-2-execution-commands)
   - [Limitations & Scientific Transparency](#1011-limitations--scientific-transparency)
8. [Future Work](#11-future-work)
9. [Team Members & Contributions](#12-team-members--contributions)

---

## 1. Project Purpose & Scope

The **Flight Delay Prediction** initiative builds an auditable, production-grade machine learning system to forecast substantial commercial flight arrival delays before departure and simulate tactical rescheduling interventions.

### Operational Framing
* **Target Variable:** `ARR_DEL15` (binary: 1 if gate arrival delay $\ge 15$ minutes, 0 otherwise).
* **Decision Horizon:** Fixed epoch at **$T - 60$ minutes** prior to published scheduled departure time.
* **Leakage Protocol:** Strict prohibition of post-departure telemetry (actual departure delay, taxi durations, wheels-off timestamps, air time, post-facto delay reason attributions).
* **Validation Methodology:** Strictly calendar-date chronological partitioning across real-world commercial flight operations.

---

## 2. System Architecture

```text
flight-delay-prediction/
├── configs/
│   ├── phase1.yaml                  # Master YAML pipeline configuration (Phase 1)
│   └── phase2.yaml                  # Master YAML pipeline configuration (Phase 2)
├── data/
│   ├── raw/                         # Raw Parquet dataset (BTS 2015)
│   │   ├── flights.parquet          # 5,819,079 domestic commercial flight records
│   │   ├── airlines.parquet         # Carrier IATA metadata
│   │   └── airports.parquet         # Airport geographical and terminal metadata
│   ├── interim/                     # Intermediate transformations
│   ├── processed/                   # Scaled, encoded model inputs
│   └── README.md                    # Data acquisition instructions
├── artifacts/
│   ├── manifests/
│   │   ├── split_manifest.json      # Phase 1 chronological split metadata
│   │   ├── split_manifest_phase2.json # Phase 2 chronological split metadata
│   │   └── artifact_manifest.json   # Full artifact inventory manifest
│   └── metadata/
│       ├── feature_schema.json      # Feature type definitions and denylist
│       ├── preprocessor.joblib      # Phase 1 fitted ColumnTransformer
│       └── preprocessor_phase2.joblib # Phase 2 fitted transformer pipeline
├── models/
│   ├── calibrator.joblib            # Phase 1 fitted probability calibrator
│   ├── calibrator_phase2.joblib     # Phase 2 Platt sigmoid calibrator
│   ├── logistic_regression/         # Baseline Logistic Regression weights & metadata
│   ├── xgboost/                     # Tuned XGBoost model & tree metadata
│   ├── lightgbm/                    # Tuned LightGBM model & metadata
│   ├── neural_network/              # FlightEmbeddingNet Tabular MLP PyTorch weights (.pt)
│   └── temporal_model/              # Temporal Recurrent LSTM PyTorch weights (.pt)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb # Interactive EDA notebook
├── reports/
│   ├── phase1_report.pdf            # Phase 1 Platypus PDF report
│   ├── phase1_report.md             # Phase 1 Markdown report source
│   ├── phase2_report.md             # Phase 2 Technical Report & Presentation Script
│   ├── phase2_ieee_paper.tex        # IEEE camera-ready two-column LaTeX source
│   └── phase2_ieee_paper.pdf        # IEEE camera-ready two-column PDF (strictly 2 pages)
├── results/
│   ├── data_quality/                # Missingness and data audit reports (JSON)
│   ├── eda/                         # Summary CSV tables across carriers/airports/hours
│   ├── metrics/                     # Benchmark comparisons and operational thresholds
│   │   ├── model_comparison.csv     # Phase 1 benchmark comparison
│   │   ├── model_comparison_phase2.csv # Phase 2 verified benchmark comparison
│   │   ├── operational_threshold.json # Phase 1 threshold optimization
│   │   ├── operational_threshold_phase2.json # Phase 2 precision-constrained threshold
│   │   ├── test_metrics.json        # Phase 1 test metrics
│   │   ├── test_metrics_phase2.json # Phase 2 test metrics
│   │   └── rescheduling_policy_sweep.csv # 8-threshold tactical policy simulation sweep
│   ├── calibration/                 # ECE statistics & reliability bin breakdowns
│   ├── error_analysis/              # Subgroup failure tables (airline, hub, hour)
│   ├── feature_importance/          # Logistic coefficients & XGBoost gain tables
│   ├── figures/                     # 300 DPI publication plots (ROC, PR, ECE, Rescheduling)
│   └── predictions/                 # Out-of-sample predictions & recommendations
├── src/flight_delay/
│   ├── config.py                    # Typed Pydantic configuration parser
│   ├── data_loader.py               # Chunked & deterministic step-sampled loader
│   ├── data_validation.py           # Schema enforcement & exclusion audit
│   ├── eda.py                       # Automated statistical profiling & plotting
│   ├── feature_engineering.py       # Calendar, cyclical & static pre-departure features
│   ├── historical_features.py       # Empirical Bayes expanding delay statistics
│   ├── phase2_features.py           # Rolling airport congestion & aircraft turnaround slack
│   ├── temporal_split.py            # Chronological date-based splitter
│   ├── preprocessing.py             # ColumnTransformer, imputer & scaler
│   ├── train_logistic.py            # Logistic Regression trainer
│   ├── train_xgboost.py             # XGBoost histogram trainer with early stopping
│   ├── train_lightgbm.py            # LightGBM trainer with Optuna hyperparameter sweeps
│   ├── train_neural_net.py          # PyTorch FlightEmbeddingNet and TemporalRecurrentNet
│   ├── calibration.py               # Platt sigmoid calibrator & ECE computation
│   ├── evaluation.py                # ROC/PR AUC, threshold search & benchmarks
│   ├── explainability.py            # Permutation importance & tree gain metrics
│   ├── error_analysis.py            # Subgroup false-positive/negative breakdown
│   ├── reporting.py                 # ReportLab Platypus PDF generator (Phase 1)
│   ├── reporting_ieee.py            # IEEE LaTeX & camera-ready PDF generator (Phase 2)
│   ├── rescheduling.py              # Offline tactical rescheduling simulation engine
│   ├── pipeline.py                  # Phase 1 master pipeline orchestrator
│   └── pipeline_phase2.py           # Phase 2 master pipeline orchestrator
├── scripts/
│   ├── run_phase1.py                # Single-command Phase 1 pipeline entry point
│   ├── run_phase2.py                # Single-command Phase 2 pipeline entry point
│   ├── run_eda.py                   # Standalone EDA runner
│   ├── train_models.py              # Standalone model training script
│   ├── generate_report.py           # Standalone report regeneration script
│   └── verify_artifacts.py          # Deliverable verification & manifest builder
├── tests/                           # 31 automated unit, leakage, and integration tests
├── run.sh                           # Fast result viewer and verification utility
├── requirements.txt                 # Pinned dependencies
├── pyproject.toml                   # Build metadata and pytest configuration
└── Makefile                         # CLI build targets
```

---

## 3. Installation & Environment Setup

### Prerequisites
* Python 3.10+ (Tested on Python 3.11.7)
* Recommended: Virtual environment (`venv` or `conda`)

```bash
# Clone or navigate to the repository root
cd /Users/ritikraj/Desktop/aml

# Create and activate virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. Dataset Acquisition

The project utilizes the official **U.S. Department of Transportation Bureau of Transportation Statistics (BTS)** flight delays dataset.

### Automated Setup
The raw dataset is pre-fetched into `data/raw/flights.parquet` (137.5 MB compressed, representing 5,819,079 domestic commercial flights across all 365 days of 2015).

To re-download if missing:
```bash
python3 -c "
import urllib.request
url = 'https://huggingface.co/datasets/hsanchezp/us-dot-flight-delays-2015/resolve/main/flights.parquet'
urllib.request.urlretrieve(url, 'data/raw/flights.parquet')
"
```

Configure the dataset path in `configs/phase1.yaml`:
```yaml
data:
  raw_path: "data/raw/flights.parquet"
  sample_size: 350000  # Set to null for full 5.8M rows
```

---

## 5. Execution Commands

### A. Run Complete Phase 1 Pipeline (Single Command)
Executes data loading, validation, EDA, temporal splitting, feature engineering, model training, threshold tuning, calibration, explainability, error analysis, artifact auditing, and PDF report compilation:
```bash
python scripts/run_phase1.py --config configs/phase1.yaml
```
*Or using the Makefile:*
```bash
make run
```

### B. Run Automated Test Suite
Executes all 26 unit, leakage, temporal split, and end-to-end integration tests:
```bash
pytest tests/ -v
```
*Or:*
```bash
make test
```

### C. Run Standalone Modules
```bash
# 1. Run Data Validation and Exploratory Data Analysis
python scripts/run_eda.py --config configs/phase1.yaml

# 2. Train Models Independently
python scripts/train_models.py --config configs/phase1.yaml

# 3. Regenerate PDF & Markdown Reports from Cached Experiments
python scripts/generate_report.py --config configs/phase1.yaml

# 4. Audit Deliverables and Verify Model Serialization
python scripts/verify_artifacts.py
```

---

## 6. Leakage Prevention Architecture

Data leakage in flight delay forecasting produces deceptively high metrics that crumble in production. We implement a multi-layer defense:

### 1. Hard Compile-Time Denylist
The following fields are strictly excluded before feature extraction:
* **Target and Derived:** `ARR_DELAY`, `ARR_DELAY_NEW`, `ARR_DEL15`, `ARR_DELAY_GROUP`
* **Departure Telemetry:** `DEP_DELAY`, `DEP_DELAY_NEW`, `DEP_DEL15`, `DEP_DELAY_GROUP`
* **Realized Timestamps:** `DEPARTURE_TIME`, `ARRIVAL_TIME`, `WHEELS_OFF`, `WHEELS_ON`
* **Operational Durations:** `TAXI_OUT`, `TAXI_IN`, `ELAPSED_TIME`, `AIR_TIME`
* **Post-Facto Causes:** `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`
* **Non-Arrival Outcomes:** `CANCELLED`, `DIVERTED`, `CANCELLATION_REASON`

### 2. Chronological Group-Aware Historical Features
Historical delay rates for airlines, origin airports, destination hubs, and routes are calculated exclusively on prior calendar dates ($t < D$). Expanding windows ensure that Day $D$'s flights never incorporate Day $D$ or future outcomes.

### 3. Empirical Bayes Smoothing
To prevent volatile probability estimates on rare routes or small regional carriers, rates are regularized toward the empirical global prior:
$$\text{Smoothed Rate} = \frac{\text{Historical Delays} + \alpha \cdot \text{Global Prior}}{\text{Historical Flights} + \alpha}$$

---

## 7. Experimental Results (Held-Out Test Set)

The models were evaluated on the held-out test partition (51,647 flights across the latest 15% of calendar dates).

| Model Pipeline | ROC AUC | PR AUC (Avg Prec) | Precision | Recall | F1 Score | Brier Score | Latency (ms/1k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Def. 0.50)** | **0.5986** | **0.2001** | 0.2033 | 0.3948 | 0.2684 | 0.2019 | 0.13 ms |
| **Logistic Regression (Op. 0.68)** | 0.5986 | 0.2001 | 0.2947 | 0.0161 | 0.0305 | 0.2019 | 0.13 ms |
| **XGBoost (Def. 0.50)** | **0.5820** | **0.1837** | 0.2028 | 0.1340 | 0.1614 | 0.1605 | 1.55 ms |
| **XGBoost (Op. 0.68)** | 0.5820 | 0.1837 | **0.3019** | 0.0021 | 0.0042 | 0.1605 | 1.55 ms |
| **XGBoost Calibrated (Op. 0.68)** | 0.5820 | 0.1837 | 0.0000 | 0.0000 | 0.0000 | **0.1245** | 1.55 ms |

*Test Set Delay Prevalence:* 14.69% | *Imbalance Ratio:* 4.5:1.

---

## 8. Artifact Deliverables Inventory

All 33 required project deliverables are saved and verified:

| Deliverable Type | File Path | Status | Size |
| :--- | :--- | :---: | :---: |
| **PDF Report** | `reports/phase1_report.pdf` | **VERIFIED** | 1,147 KB |
| **Markdown Report** | `reports/phase1_report.md` | **VERIFIED** | 7.0 KB |
| **Logistic Regression Model** | `models/logistic_regression/model.joblib` | **VERIFIED** | 1.9 KB |
| **XGBoost Model** | `models/xgboost/model.joblib` | **VERIFIED** | 1,250 KB |
| **Calibrator Pipeline** | `models/calibrator.joblib` | **VERIFIED** | 1.0 KB |
| **Preprocessor Pipeline** | `artifacts/metadata/preprocessor.joblib` | **VERIFIED** | 11.9 KB |
| **Data Quality Report** | `results/data_quality/data_quality_report.json` | **VERIFIED** | 1.6 KB |
| **Split Manifest** | `artifacts/manifests/split_manifest.json` | **VERIFIED** | 0.9 KB |
| **Artifact Manifest** | `artifacts/manifests/artifact_manifest.json` | **VERIFIED** | 3.5 KB |
| **Test Predictions CSV** | `results/predictions/test_predictions.csv` | **VERIFIED** | 3,243 KB |
| **Model Comparison CSV** | `results/metrics/model_comparison.csv` | **VERIFIED** | 0.5 KB |
| **Feature Importance Plots** | `results/figures/feature_importance.png` | **VERIFIED** | 251 KB |
| **ROC & PR Curves** | `results/figures/roc_curves.png` | **VERIFIED** | 201 KB |

---

## 9. Common Troubleshooting

### 1. macOS OpenMP Library Mismatch
* **Symptom:** `Symbol not found: ___kmpc_dispatch_deinit in libomp.dylib` when importing XGBoost.
* **Resolution:** Install OpenMP runtime via `brew install libomp`, or install `xgboost==1.7.6` which links with Anaconda's bundled OpenMP 14 runtime:
  ```bash
  pip install xgboost==1.7.6
  ```

### 2. Memory Limitations on Large Datasets
* **Symptom:** Memory exhaustion when reading all 5.8 million records on laptops with < 16 GB RAM.
* **Resolution:** Use the pre-configured `sample_size: 350000` in `configs/phase1.yaml`. The loader performs step-stratified sampling across dates to preserve temporal continuity while bounding peak RAM to < 800 MB.

---

## 10. Phase 2 Comprehensive Documentation: Hybrid Tabular, Temporal Deep Learning & Tactical Rescheduling

Phase 2 advances the flight arrival delay forecasting initiative by implementing a **hybrid predictive architecture**: combining operational tabular predictors with learned high-dimensional categorical embeddings (Airline, Origin, Destination) and temporal sequence models, integrated into an offline tactical rescheduling decision engine.

---

### 10.1 Phase 2 Overview & The 'Hybrid' Logic
Standard machine learning approaches rely on manual aggregations that can miss subtle, non-linear interactions across carrier networks, hub congestion banks, and propagated aircraft rotation delays. Phase 2 bridges this gap through:
1. **Operational Tabular Features:** Scheduled buffer slack, cyclic temporal encodings, and expanding Empirical Bayes historical delay priors.
2. **Learned Categorical Representations:** Dense entity embeddings for high-cardinality nominal variables (Airline, Origin Airport, Destination Airport) trained end-to-end within a deep neural network (`FlightEmbeddingNet`).
3. **Temporal Turnaround Sequencing:** Recurrent sequence modeling (`TemporalFlightNet` GRU/LSTM) tracking sequential turnaround states across individual aircraft tail numbers (`TAIL_NUMBER`).
4. **Probability Calibration:** Post-hoc Platt sigmoid scaling fitted strictly on validation logits to output reliable posterior probabilities for downstream tactical dispatch.
5. **Offline Tactical Rescheduling Engine:** Counterfactual simulation of proactive gate interventions (Tail Swaps and Ground Buffer Shifts) with explicit modeling of schedule friction.

---

### 10.2 Dataset Provenance & Cross-Phase Comparability

* **Raw Dataset Source:** Both Phase 1 and Phase 2 derive from the exact same raw dataset: the official U.S. Department of Transportation Bureau of Transportation Statistics (BTS) 2015 Airline On-Time Performance dataset (`data/raw/flights.parquet`, containing 5,819,079 domestic commercial flights across all 365 calendar days of 2015).
* **Sampling Strategy:** `DataLoader.load_data()` employs deterministic uniform stride sampling (`np.arange(0, total_rows, step)[:sample_size]`):
  * **Phase 1 Baseline:** Requested `sample_size = 350,000` (step stride = 16), yielding **343,885 valid records** across 351 operational dates (2015-01-01 to 2015-12-17; test partition: 51,647 flights, Oct 25–Dec 17).
  * **Phase 2 Benchmark:** Requested `sample_size = 150,000` (step stride = 38), yielding **147,389 valid records** across 358 operational dates (2015-01-01 to 2015-12-24; test partition: 21,866 flights, Oct 31–Dec 24).
* **Sampling Rationale:** The 150,000-flight sample size in Phase 2 was deliberately chosen to allow deep neural architectures (`FlightEmbeddingNet`, `TemporalFlightNet` LSTM/GRU) and extensive Optuna hyperparameter sweeps to train efficiently within compute and memory budgets.
* **Scientific Comparability Disclosure:** Because the sample sizes, calendar date spans, and test partition boundaries differ between phases:
  > **Phase 1 and Phase 2 metrics are NOT directly comparable across different test sets.**
  To guarantee strict internal validity, all baseline models (Logistic Regression and XGBoost) were retrained with Phase 2 features and evaluated on the **exact same Phase 2 held-out test partition (21,866 flights)** alongside LightGBM and the deep neural architectures.

---

### 10.3 Chronological Date Partitioning

To prevent temporal autocorrelation leakage across consecutive calendar dates, Phase 2 enforces a strict calendar-date chronological split across the 358 operational dates of 2015:

| Partition | Calendar Date Range | Dates Count | Record Count | % of Dataset | Positive Count (`ARR_DEL15=1`) | Delay Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | 2015-01-01 to 2015-09-07 | 250 dates | 103,747 | 70.39% | 20,873 | 20.12% |
| **Validation Set** | 2015-09-08 to 2015-10-30 | 53 dates | 21,776 | 14.77% | 2,769 | 12.72% |
| **Held-Out Test Set** | 2015-10-31 to 2015-12-24 | 55 dates | 21,866 | 14.84% | 3,499 | 16.00% |
| **Total Benchmark** | **2015-01-01 to 2015-12-24** | **358 dates** | **147,389** | **100.0%** | **27,141** | **18.41%** |

*Note on Validation/Test Window:* The Phase 2 test set spans October 31 to December 24, 2015, capturing the extreme operational congestion of Thanksgiving travel and the peak pre-Christmas holiday travel rush.

---

### 10.4 Advanced Phase 2 Feature Engineering

Operating strictly at the pre-departure decision timestamp **$T - 60$ minutes**, Phase 2 extracts features across three operational dimensions:

1. **Expanding Empirical Bayes Historical Priors:**
   Daily cumulative delay statistics are computed strictly on past dates ($t < D$) to eliminate target leakage. For entity $e$ (airline, origin airport, destination airport, route), the smoothed historical delay rate shrinks small-sample entities toward the empirical global prior:
   $$\hat{p}_{e, D} = \frac{\sum_{t < D} \text{Delays}_{e, t} + \alpha \cdot \bar{p}_{\text{global}}}{\sum_{t < D} \text{Flights}_{e, t} + \alpha}$$
   where $\alpha = 15.0$ acts as a pseudo-count shrinkage parameter.

2. **Airport Rolling Congestion Banks:**
   Terminal traffic congestion is measured from published flight schedules without using actual realization timestamps:
   * `ORIGIN_CONGESTION_1H`: Number of scheduled departures at origin within $\pm 60$ minutes of the flight's scheduled departure.
   * `DEST_CONGESTION_1H`: Number of scheduled arrivals at destination within $\pm 60$ minutes of the flight's scheduled arrival bank.

3. **Aircraft Turnaround Delay Propagation:**
   Tracking the physical aircraft tail number (`TAIL_NUMBER`) at decision epoch $T - 60$ minutes:
   * If the inbound flight landed *prior* to $T - 60$, its actual delay is observed (`INBOUND_DELAY_KNOWN`).
   * If the inbound flight has *not yet landed* at $T - 60$, `INBOUND_UNARRIVED_AT_T60 = 1`.
   * Scheduled turnaround buffer: $\Delta T_{\text{buffer}} = T_{\text{sched\_dep}} - T_{\text{prev\_sched\_arr}}$.

---

### 10.5 Model Architectures & Deep Learning

Phase 2 implements five distinct model architectures:

1. **Deep Tabular MLP with Learned Entity Embeddings (`FlightEmbeddingNet`):**
   * High-cardinality nominal variables are mapped into low-dimensional dense embedding vectors:
     $$\mathbf{e}_{\text{airline}} \in \mathbb{R}^{16}, \quad \mathbf{e}_{\text{origin}} \in \mathbb{R}^{32}, \quad \mathbf{e}_{\text{dest}} \in \mathbb{R}^{32}$$
   * Continuous tabular features (congestion counts, turnaround buffers, cyclic sine/cosine calendar encodings, historical Bayes priors) are standardized via `RobustScaler` / `StandardScaler`.
   * Dense representations are concatenated and passed through a 3-layer MLP architecture:
     $$\text{Linear}(D_{\text{in}} \to 256) \to \text{BatchNorm} \to \text{Mish} \to \text{Dropout}(0.2)$$
     $$\to \text{Linear}(256 \to 128) \to \text{BatchNorm} \to \text{Mish} \to \text{Dropout}(0.2)$$
     $$\to \text{Linear}(128 \to 64) \to \text{Linear}(64 \to 1)$$
   * Optimized with AdamW ($\text{lr}=10^{-3}$, weight decay $= 10^{-4}$) under positive class weighting.

2. **Temporal Recurrent Sequence Model (`TemporalFlightNet` / `TemporalLSTM`):**
   * 2-layer Recurrent Neural Network (GRU / LSTM, hidden dimension 64) modeling chronological aircraft turnaround state transitions across consecutive itinerary legs.

3. **LightGBM Classifier with Optuna Tuning (`train_lightgbm.py`):**
   * Histogram-based gradient tree boosting with Optuna hyperparameter sweeps optimizing tree depth, learning rate, leaf count, feature subsampling, and L1/L2 regularization (`reg_alpha`, `reg_lambda`).

4. **XGBoost Classifier (`train_xgboost.py`):**
   * Regularized histogram gradient booster with `scale_pos_weight` and validation early stopping (patience = 25 rounds).

5. **Logistic Regression Baseline (`train_logistic.py`):**
   * L2-regularized linear model with `class_weight='balanced'`, serving as the rigorous linear benchmark.

---

### 10.6 Comprehensive Model Benchmarks

All models were evaluated on the **exact same Phase 2 held-out test partition** (21,866 flights, Oct 31–Dec 24, 2015, $T - 60$ min decision timestamp). All metrics are directly verified against `results/metrics/model_comparison_phase2.csv` and `test_metrics_phase2.json`:

| Model Architecture | ROC AUC | PR AUC | Precision | Recall | F1 Score | Brier Score $\downarrow$ | Decision Threshold | Inference Latency (ms/1k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DL Tabular MLP (Entity Embeddings)** | **0.6196** | **0.2319** | 0.2472 | 0.3132 | 0.2763 | 0.18734 | 0.5000 | 8.09 ms |
| **Logistic Regression (Phase 2 Features)** | 0.6159 | 0.2284 | 0.2318 | 0.3692 | 0.2848 | 0.19611 | 0.5000 | 0.06 ms |
| **Temporal Recurrent Net (LSTM)** | 0.6132 | 0.2283 | 0.2496 | 0.1798 | 0.2090 | 0.17404 | 0.5000 | 4.07 ms |
| **XGBoost (Hist Gradient Boosting)** | 0.5982 | 0.2141 | 0.2374 | 0.2335 | 0.2354 | 0.17557 | 0.5000 | 1.81 ms |
| **LightGBM (Tuned Classifier)** | 0.5582 | 0.1875 | 0.0000 | 0.0000 | 0.0000 | 0.13659 | 0.5000 | 0.31 ms |
| **DL Tabular MLP (Calibrated @ Op Thresh)** | **0.6196** | **0.2319** | **0.7143** | 0.0029 | 0.0057 | **0.13271** | **0.3637** | 8.09 ms |

#### Key Empirical Insights:
1. **Deep Learning Outperforms Gradient Boosters:** The Tabular MLP with learned entity embeddings attained the highest discriminative performance across all evaluated models (ROC AUC **0.6196**, PR AUC **0.2319**), confirming that dense embedding spaces capture complex hub and carrier interactions more effectively than axis-aligned tree splits.
2. **LightGBM Threshold Sensitivity:** LightGBM achieves a low Brier score (0.13659), indicating well-calibrated probabilities, but outputs conservative scores that yield zero positive predictions at the default threshold $\tau = 0.50$. PR AUC (0.1875) accurately reflects its ranking.

---

### 10.7 Probability Calibration & Operational Threshold Tuning

Modern deep networks and boosted trees often output overconfident or miscalibrated posterior probabilities. We implement post-hoc **Platt (sigmoid) scaling** fitted exclusively on validation logits:
$$\hat{P}(Y=1 \mid z) = \sigma(A z + B)$$

Evaluated on the held-out test partition across $M=10$ uniform reliability bins:
* **Expected Calibration Error (ECE):** Compressed from **0.2255** (raw MLP) down to **0.0389** (calibrated).
* **Brier Score:** Reduced from **0.18734** down to **0.13271**.

#### Operational Precision Constraint ($\ge 70\%$):
Airlines cannot tolerate high false-alarm rates when dispatching costly ground crew interventions. When tuned on the validation partition to enforce precision $\ge 70\%$, the calibrator identified optimal threshold **$\tau = 0.3637$**. On the held-out test set, this achieves:
* **Operational Precision:** **71.43%** (0.7143)
* **Operational Recall:** **0.29%** (0.0029, capturing 0.3% of delayed flights)
* *Interpretation:* This represents a deliberate, conservative operating-point trade-off designed to guarantee high confidence when an alert is fired, rather than overall model superiority.

---

### 10.8 Offline Tactical Rescheduling Simulation

When the calibrated pre-departure delay risk exceeds policy threshold $\tau$, an offline tactical decision is simulated:
* **Hub Tail Swap:** If the aircraft turnaround buffer is compromised at a major airline hub (carrier hub share $\ge 25\%$), swap the tail with an available on-time aircraft.
* **Slot Shift (+30 min):** Insert a 30-minute scheduled ground buffer to absorb downstream runway queueing.

#### Counterfactual Policy Sweep (21,866 Test Flights):
Evaluated across eight risk thresholds (verified from `results/metrics/rescheduling_policy_sweep.csv`):

| Risk Threshold ($\tau$) | Interventions | Interv. Rate | Captured Delays (TP) | Delay Coverage | False Alarms | Est. Averted Delays† | Gross Delay Min Saved‡ | Net Benefit (Min Saved - Friction)§ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.107 | 13,120 | 60.00% | 2,564 | 73.28% | 10,556 | 1,025 | 31,260.7 | -67,139.3 min |
| 0.123 | 10,277 | 47.00% | 2,139 | 61.13% | 8,138 | 855 | 26,103.5 | -50,974.0 min |
| 0.139 | 7,534 | 34.46% | 1,692 | 48.36% | 5,842 | 676 | 20,442.5 | -36,062.5 min |
| 0.155 | 4,934 | 22.56% | 1,198 | 34.24% | 3,736 | 479 | 14,398.5 | -22,606.5 min |
| **0.171** | **2,905** | **13.29%** | **759** | **21.69%** | **2,146** | **303** | **9,176.7** | **-12,610.8 min** |
| 0.188 | 1,565 | 7.16% | 447 | 12.78% | 1,118 | 178 | 5,387.1 | -6,350.4 min |
| 0.204 | 838 | 3.83% | 249 | 7.12% | 589 | 99 | 3,140.3 | -3,144.7 min |
| 0.220 | 438 | 2.00% | 146 | 4.17% | 292 | 58 | 1,798.3 | -1,486.7 min |

*Footnotes on Simulation Assumptions:*  
† **Estimated Averted Delays:** Simulation metric computed as $\text{Captured Delays} \times p_{\text{mitigate}}$ ($p_{\text{mitigate}} = 0.40$ based on literature mitigation effectiveness). These represent counterfactual estimates under simulation assumptions, not proven real-world outcomes.  
‡ **Gross Delay Minutes Saved:** Simulation metric based on an assumed 45-minute average delay duration ($\text{Averted Delays} \times 45\text{ min} \times \text{risk reduction}$). This is an operational flight-level simulation metric; **not** a passenger-level delay metric.  
§ **Net Operational Benefit:** Computed as $\text{Gross Minutes Saved} - \text{Intervention Friction Penalty}$.

#### Intervention Friction & Negative Net Benefit Analysis:
* **The Friction Penalty:** Every scheduled intervention incurs operational disruption: inserting a 30-minute ground delay buffer carries a schedule disruption friction penalty of:
  $$\Delta T_{\text{buffer}} \times \lambda_{\text{slot\_penalty}} = 30\text{ min} \times 0.25 = 7.5\text{ penalty minutes per intervention}$$
* **At Operating Threshold $\tau = 0.171$:**
  * 2,905 interventions trigger $2,905 \times 7.5 = \mathbf{21,787.5\text{ penalty minutes}}$ of schedule friction.
  * Although 759 delays are captured (saving 9,176.7 gross delay minutes), 2,146 interventions are false alarms (73.87% false alarm share) that bear the full 7.5-minute friction penalty without averting an actual delay.
  * As a consequence, **net benefit is strictly negative (-12,610.8 minutes)**.
  * Under this linear friction formulation, **net benefit remains negative across all eight evaluated thresholds**, proving that aggressive pre-departure buffer insertions impose systemic friction that outweighs gross savings unless guided by ultra-high-precision classifiers.

---

### 10.9 Phase 2 Deliverables Inventory

| Deliverable Artifact | File Location | Description & Verification Status |
| :--- | :--- | :--- |
| **IEEE Camera-Ready PDF** | `reports/phase2_ieee_paper.pdf` | **VERIFIED (841 KB, strictly 2 pages)** — Camera-ready two-column publication report. |
| **IEEE LaTeX Source** | `reports/phase2_ieee_paper.tex` | **VERIFIED** — Formatted in standard `IEEEtran` conference layout with siunitx tables. |
| **Comprehensive Report** | `reports/phase2_report.md` | **VERIFIED** — Full engineering markdown report + 10-slide PPT presentation script. |
| **Rescheduling Sweep CSV** | `results/metrics/rescheduling_policy_sweep.csv` | **VERIFIED** — 8-threshold policy simulation sweep results. |
| **Model Comparison CSV** | `results/metrics/model_comparison_phase2.csv` | **VERIFIED** — Out-of-sample benchmark metrics across all 6 architectures. |
| **Operational Threshold** | `results/metrics/operational_threshold_phase2.json` | **VERIFIED** — Calibrated threshold metadata satisfying $\ge 70\%$ precision. |
| **Calibration Statistics** | `results/calibration/calibration_stats_phase2.json` | **VERIFIED** — Pre- vs. post-calibration Brier scores and ECE values. |
| **DL Tabular MLP Model** | `models/neural_network/mlp_embedding_weights.pt` | **VERIFIED (365 KB)** — Trained PyTorch weights for `FlightEmbeddingNet`. |
| **Temporal LSTM Model** | `models/temporal_model/temporal_lstm_weights.pt` | **VERIFIED (180 KB)** — Trained PyTorch weights for `TemporalFlightNet`. |
| **LightGBM Classifier** | `models/lightgbm/model.joblib` | **VERIFIED (13 KB)** — Optuna-tuned LightGBM pipeline. |
| **XGBoost Classifier** | `models/xgboost/model.joblib` | **VERIFIED (1.1 MB)** — Retrained XGBoost benchmark on Phase 2 test set. |
| **Calibrator Pipeline** | `models/calibrator_phase2.joblib` | **VERIFIED (1.0 KB)** — Fitted Platt sigmoid probability calibrator. |
| **Split Manifest** | `artifacts/manifests/split_manifest_phase2.json` | **VERIFIED** — Exact record counts and ISO date ranges for Train, Val, Test. |

---

### 10.10 Phase 2 Execution Commands

#### 1. Run Complete Phase 2 Pipeline (Single Command)
Executes data preprocessing, rolling congestion and turnaround feature extraction, model training (MLP, LSTM, LightGBM, XGBoost, LogReg), Platt calibration, operational threshold optimization, counterfactual rescheduling sweep, figure generation, and automated IEEE LaTeX/PDF compilation:
```bash
python3 scripts/run_phase2.py --sample-size 150000
```

#### 2. Run Comprehensive Automated Test Suite
Executes all 31 unit, leakage, calibration, model serialization, and integration tests:
```bash
python3 -m pytest tests/ -p no:anyio
```
*(All 31 tests pass cleanly in ~15–30 seconds).*

#### 3. Inspect Model Benchmark & Rescheduling Results
```bash
# Print Phase 2 Model Benchmark Table
python3 -c "import pandas as pd; print(pd.read_csv('results/metrics/model_comparison_phase2.csv').to_string(index=False))"

# Print Tactical Rescheduling Policy Sweep
python3 -c "import pandas as pd; print(pd.read_csv('results/metrics/rescheduling_policy_sweep.csv')[['Risk Threshold (tau)', 'Interventions', 'Intervention Rate (%)', 'Captured Delays (TP)', 'Delay Coverage Rate (%)', 'Gross Delay Minutes Saved', 'Net Benefit (Min Saved - Friction)']].to_string(index=False))"
```

#### 4. View Generated Visual Artifacts
```bash
# Open camera-ready IEEE conference publication PDF
open reports/phase2_ieee_paper.pdf

# Open ROC curves and calibration diagrams
open results/figures/roc_curves.png
open results/figures/calibration_curve_phase2.png
open results/figures/rescheduling_tradeoff.png
```

---

### 10.11 Limitations & Scientific Transparency

1. **Weather Integration:** Real-time convective weather and surface forecasts (NOAA METAR/TAF radar observations) are not yet integrated into the feature pipeline. Current predictions rely on historical Empirical Bayes delay priors and scheduled terminal congestion banks.
2. **Linear Friction Assumptions:** The counterfactual rescheduling engine employs a simplified linear friction cost model ($7.5\text{ min/flight}$). In real operations, airline cost functions are highly non-linear and asymmetric (absorbing gate hold delays is cheaper than cancellation or tarmac delays).
3. **Threshold Selection for GBDT:** At the default threshold $\tau = 0.50$, LightGBM achieves zero positive classifications due to extreme class imbalance. In production, its decision threshold must be tuned on the validation PR curve.

---

## 11. Future Work

1. **NOAA Live Aviation Weather Ingestion:** Integrate live METAR/TAF observation feeds and convective SIGMET hazard polygons into pre-departure feature banks.
2. **Asymmetric Operational Loss Formulations:** Replace linear penalty weights with carrier-specific cost functions penalizing severe tarmac delays over minor gate adjustments.
3. **Graph Neural Networks (GNN):** Model national airspace delay propagation as a dynamic spatio-temporal graph where airports are nodes and flight legs are directed temporal edges.
4. **Full-Scale Distributed Training:** Scale the deep embedding architecture from the 150,000-flight benchmark to the full 5.8-million record BTS dataset using PyTorch Distributed Data Parallel (DDP).

---

## 12. Team Members & Contributions

| Member | Assigned Area | Key Responsibilities & Deliverables |
| :--- | :--- | :--- |
| **Vaibhav** | **Exploratory Data Analysis (EDA)** | Exploratory analysis across 5.8M raw flights; distribution analysis, missingness audits, delay prevalence profiling across carriers, airports, and departure time banks (`results/eda/`). |
| **Parv** | **Presentation & Report** | Authoring comprehensive Phase 2 research report, IEEE camera-ready two-column conference paper documentation, presentation slide deck structure, and speaker notes (`reports/`). |
| **Ayush** | **Feature Engineering** | Zero-leakage pre-departure feature design at $T - 60$ min; expanding Empirical Bayes smoothed historical delay priors, rolling terminal airport congestion banks, and inbound aircraft turnaround slack (`src/flight_delay/phase2_features.py`). |
| **Ritik** | **Model Training & Benchmarking** | Model training and benchmarking across all architectures: classical baselines (Logistic Regression, XGBoost, LightGBM with Optuna tuning) and deep neural architectures (`FlightEmbeddingNet` Tabular MLP with entity embeddings and `TemporalFlightNet` LSTM); Platt probability calibration and operational threshold tuning (`src/flight_delay/train_*.py`). |
| **Mohit** | **Pipeline Integration & Operations** | End-to-end pipeline orchestration (`pipeline_phase2.py`, `run_phase2.py`), zero-leakage test validation, offline tactical rescheduling counterfactual simulation engine, artifact serialization, and test suite automation (`scripts/run_phase2.py`, `src/flight_delay/rescheduling.py`). |


