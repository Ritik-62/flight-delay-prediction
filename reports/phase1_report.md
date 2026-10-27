# Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning
## Phase 1 Research & Engineering Report: Leakage-Safe Baselines

**Author:** Antigravity ML Research Engineering  
**Version:** 1.0.0 — Phase 1 Deliverable  
**Date:** October 26, 2026  
**Target:** Arrival Delay $\ge 15$ Minutes (`ARR_DEL15`)  
**Prediction Horizon:** $T - 60$ Minutes Before Scheduled Departure

---

## 1. Executive Summary
This report presents the implementation and rigorous experimental validation of **Phase 1** of the Flight Delay Prediction initiative. Phase 1 establishes a reproducible, leakage-safe machine learning baseline targeting binary flight arrival delays of 15 minutes or greater (`ARR_DEL15`). 

All models operate under a strict operational constraint: **features are restricted entirely to information definitively available 60 minutes prior to scheduled departure (T - 60 minutes)**. Post-departure outcomes, ground taxi durations, and actual departure delays are strictly forbidden. Chronological data splitting (earliest 70% train, next 15% validation, final 15% held-out test) guarantees that no future knowledge contaminates training or threshold tuning.

### Key Benchmark Results (Held-Out Test Set)
| Model                           |   ROC AUC |   PR AUC |   Precision |   Recall |   F1 Score |   Brier Score |   Balanced Acc |   Threshold |   Latency (ms/1k) |
|:--------------------------------|----------:|---------:|------------:|---------:|-----------:|--------------:|---------------:|------------:|------------------:|
| Logistic Regression (Def. 0.50) |    0.5964 |   0.2411 |      0.2479 |   0.3546 |     0.2918 |       0.20198 |         0.5557 |    0.5      |              0.06 |
| Logistic Regression (Op. 0.28)  |    0.5964 |   0.2411 |      0.1999 |   0.8756 |     0.3255 |       0.20198 |         0.5417 |    0.277494 |              0.06 |
| XGBoost (Def. 0.50)             |    0.6022 |   0.2415 |      0.2552 |   0.3353 |     0.2898 |       0.19164 |         0.557  |    0.5      |              1.82 |
| XGBoost (Op. 0.28)              |    0.6022 |   0.2415 |      0.2172 |   0.7765 |     0.3395 |       0.19164 |         0.572  |    0.277494 |              1.82 |
| XGBoost Calibrated (Op. 0.28)   |    0.6022 |   0.2415 |      0      |   0      |     0      |       0.15006 |         0.4999 |    0.277494 |              1.82 |

---

## 2. Dataset Overview & Data Quality Assurance
The primary dataset utilized is the United States Department of Transportation (US DOT) Airline On-Time Performance dataset (`flights.parquet`), documenting commercial domestic passenger flights across all 365 calendar days of the calendar year.

* **Raw Total Records:** 50,000
* **Validated Usable Records:** 49,096
* **Excluded Non-Arrivals (Cancelled & Diverted):** 904
* **Arrival Delay Prevalence (Positive Class Rate):** 18.58%
* **Imbalance Ratio:** 4.38:1 (Negative:Positive)
* **Date Range:** 2015-01-01 to 2015-12-30

---

## 3. Data Leakage Prevention Architecture
To guarantee absolute operational fidelity, a centralized audit layer enforces a strict denylist preventing any target or post-decision column from entering feature computation:

### Explicitly Excluded Denylist
* `ARR_DELAY`, `ARR_DELAY_NEW`, `ARR_DEL15`, `ARR_DELAY_GROUP` (Direct target leakage)
* `DEP_DELAY`, `DEP_DELAY_NEW`, `DEP_DEL15`, `DEP_DELAY_GROUP` (Actual departure delay)
* `DEPARTURE_TIME`, `ARRIVAL_TIME`, `WHEELS_OFF`, `WHEELS_ON` (Actual event timestamps)
* `TAXI_OUT`, `TAXI_IN`, `ELAPSED_TIME`, `AIR_TIME` (Actual operational durations)
* `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY` (Post-facto delay causes)
* `CANCELLED`, `DIVERTED`, `CANCELLATION_REASON` (Operational non-completion outcomes)

### Permitted Feature Allowlist
1. **Calendar & Periodic:** Month, Day, Day of Week, Is Weekend, Day of Year, Cyclical sine/cosine encodings.
2. **Flight Schedule:** Scheduled departure hour/minute, scheduled arrival hour/minute, overnight crossing indicator, scheduled elapsed duration, scheduled distance.
3. **Route & Carrier:** Airline carrier, Origin airport, Destination airport, Origin-Destination route.
4. **Historical Smoothed Delay Statistics:** Expanding daily cumulative delay rate and flight volume for airline, origin airport, destination airport, and route, computed strictly on prior dates ($t < D$) using Empirical Bayes smoothing:
$$\text{Smoothed Rate} = \frac{\sum_{t < D} \text{Delays} + \alpha \cdot \text{Global Prior}}{\sum_{t < D} \text{Flights} + \alpha}$$

---

## 4. Chronological Splitting Protocol
Random data splits violate the arrow of time, inducing severe temporal leakage due to seasonal autocorrelation. We enforce strict calendar-date temporal partitioning:
* **Training Set:** 2015-01-01 to 2015-09-11 (34,520 flights, 254 dates)
* **Validation Set:** 2015-09-12 to 2015-11-04 (7,249 flights, 54 dates)
* **Test Set:** 2015-11-05 to 2015-12-30 (7,327 flights, 56 dates)

---

## 5. Model Architecture & Hyperparameters
1. **Logistic Regression Baseline:**
   * Scaled continuous features (`StandardScaler`), median imputation, One-Hot Encoding for top 50 origin and destination airports and airlines.
   * `class_weight='balanced'`, L2 regularization ($C=1.0$), L-BFGS solver ($max\_iter=1000$).
2. **XGBoost Classifier:**
   * Objective: `binary:logistic`, Evaluation Metric: `logloss`.
   * $n\_estimators=300$, $max\_depth=6$, $\eta=0.05$, $subsample=0.8$, $colsample=0.8$.
   * Class weighting: $scale\_pos\_weight$ derived strictly from training class ratio.
   * Validation early stopping with 25 rounds patience.

---

## 6. Probability Calibration & Threshold Optimization
* **Decision Threshold:** Optimized on validation set for target precision of 70%.
  * Selected Operating Threshold: **0.2775**
  * Decision Rule Applied: Target precision 70% unachievable; selected threshold maximizing validation F1
* **Probability Calibration:** Platt (sigmoid) scaling fitted exclusively on validation set probabilities.
  * Expected Calibration Error (ECE): Raw = 0.1791, Calibrated = 0.0486
  * Brier Score: Raw = 0.19164, Calibrated = 0.15006

---

## 7. Automated Testing Suite
A comprehensive test suite executed with `pytest` verifies structural integrity, zero leakage, mathematical boundary conditions, and model serialization consistency:
* **Total Tests Executed:** 26
* **Passed:** 26
* **Failed:** 0
* **Status:** 100% Passing

---

## 8. Limitations & Phase 2 Roadmap
### Current Limitations
1. **Static Pre-Departure Scope:** Weather forecasts at airports and en-route convective weather are currently reflected only through historical delay rates rather than real-time NOAA METAR/TAF radar feeds.
2. **Aircraft Inbound Rotation:** High-resolution tail number tracking for inbound turnaround delay propagation is deferred to Phase 2 temporal graph and sequence modeling.

### Phase 2 Implementation Roadmap
* Integration of learned airport and carrier dense embeddings.
* Temporal sequence modeling (LSTM / Transformer architectures) tracking aircraft tail turnaround trajectories.
* Ingestion of public NOAA METAR aviation weather features.
