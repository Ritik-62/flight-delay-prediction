# Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning
## Phase 1: Leakage-Safe Baseline Architecture & Chronological Benchmarking

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)](tests/)
[![Report Validated](https://img.shields.io/badge/report-PDF%20Generated-success.svg)](reports/phase1_report.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Purpose & Scope

The **Flight Delay Prediction** initiative builds an auditable, production-grade machine learning system to forecast substantial flight arrival delays before flight departure. 

### Phase 1 Scope (This Deliverable)
Phase 1 focuses exclusively on establishing a mathematically sound, **leakage-safe tabular baseline**:
* **Target Variable:** `ARR_DEL15` (1 if gate arrival delay $\ge 15$ minutes, 0 otherwise).
* **Prediction Timestamp:** $T - 60$ minutes relative to scheduled departure.
* **Leakage Prevention:** Strict denylist forbidding post-departure telemetry (actual departure delays, taxi times, wheels-off timestamps, post-flight outcomes).
* **Evaluation Protocol:** Strictly chronological calendar-date partitioning (70% Train, 15% Validation, 15% Held-Out Test).
* **Trained Baselines:** Scikit-Learn Logistic Regression and XGBoost Gradient Boosted Trees with validation early stopping and Platt probability calibration.
* **Outputs:** Saved model pipelines, test predictions, metrics JSON/CSV, publication-quality figures, automated test suite, and a verified Platypus PDF research report.

### Explicitly Excluded from Phase 1 (Scheduled for Phase 2)
* Learned deep airline and airport embeddings.
* Recurrent or attention-based temporal neural architectures (LSTM, GRU, Transformers) tracking physical aircraft turnaround rotations.
* Production microservices, live streaming APIs, or automated flight rescheduling systems.

---

## 2. System Architecture

```
flight-delay-prediction/
├── configs/
│   └── phase1.yaml                  # Master YAML pipeline configuration
├── data/
│   ├── raw/                         # Raw Parquet/CSV dataset and metadata
│   ├── interim/                     # Intermediate transformations
│   ├── processed/                   # Preprocessed model inputs
│   └── README.md                    # Data acquisition instructions
├── models/
│   ├── logistic_regression/         # Serialized LR model & metadata
│   ├── xgboost/                     # Serialized XGBoost model & metadata
│   └── calibrator.joblib            # Fitted Platt probability calibrator
├── notebooks/
│   └── 01_exploratory_analysis.ipynb # Interactive EDA notebook
├── reports/
│   ├── phase1_report.pdf            # Publication-grade PDF report (Verified)
│   ├── phase1_report.md             # Markdown source of the project report
│   └── figures/                     # High-resolution report figures
├── results/
│   ├── data_quality/                # Data cleaning audit report (JSON)
│   ├── eda/                         # Aggregated summary CSV tables
│   ├── metrics/                     # Final evaluation metrics & comparisons
│   ├── predictions/                 # Out-of-sample predictions with probabilities
│   ├── feature_importance/          # Logistic coefficients & XGBoost gain tables
│   ├── calibration/                 # ECE statistics & bin breakdown
│   ├── error_analysis/              # Subgroup failure tables (airline, hub, hour)
│   └── figures/                     # 300 DPI publication PNG plots
├── src/flight_delay/
│   ├── config.py                    # Typed configuration parser
│   ├── data_loader.py               # Chunked & sampled Parquet/CSV loader
│   ├── data_validation.py           # Schema enforcement & exclusion audit
│   ├── eda.py                       # Automated statistical profiling & plotting
│   ├── feature_engineering.py       # Calendar, cyclical & static features
│   ├── historical_features.py       # Empirical Bayes expanding delay statistics
│   ├── temporal_split.py            # Chronological date-based splitter
│   ├── preprocessing.py             # ColumnTransformer, imputer & scaler
│   ├── train_logistic.py            # Logistic Regression trainer
│   ├── train_xgboost.py             # XGBoost trainer with early stopping
│   ├── calibration.py               # Platt calibrator & ECE computation
│   ├── evaluation.py                # ROC/PR AUC, threshold search & benchmarks
│   ├── explainability.py            # Permutation importance & coefficients
│   ├── error_analysis.py            # Subgroup false-positive/negative breakdown
│   ├── reporting.py                 # ReportLab Platypus PDF generator
│   └── pipeline.py                  # End-to-end master pipeline orchestrator
├── scripts/
│   ├── run_phase1.py                # Single-command pipeline entry point
│   ├── run_eda.py                   # Standalone EDA runner
│   ├── train_models.py              # Standalone model training script
│   ├── generate_report.py           # Standalone report regeneration script
│   └── verify_artifacts.py          # Deliverable verification & manifest builder
├── tests/                           # 26 automated unit and integration tests
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

## 10. Phase 2 Roadmap

With Phase 1 baselines established, Phase 2 will focus on:
1. **Learned Airport and Carrier Embeddings:** Replace sparse one-hot encoding with low-dimensional dense representation vectors.
2. **Temporal Aircraft Turnaround Networks (LSTM / Transformer):** Model inbound delay propagation across physical aircraft tail numbers.
3. **NOAA Aviation Weather Integration:** Ingest live METAR/TAF weather observation radar feeds.
