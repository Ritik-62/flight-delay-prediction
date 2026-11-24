"""Phase 2 Master Pipeline: End-to-end execution of Gradient Boosting, Temporal DL, and Rescheduling."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from flight_delay.calibration import (
    ModelCalibrator,
    compute_expected_calibration_error,
    plot_reliability_diagram,
)
from flight_delay.config import Config
from flight_delay.data_loader import DataLoader
from flight_delay.data_validation import DataValidator
from flight_delay.evaluation import (
    benchmark_inference_speed,
    compute_comprehensive_metrics,
    find_operational_threshold,
    plot_confusion_matrices,
    plot_evaluation_curves,
)
from flight_delay.explainability import ModelExplainer
from flight_delay.historical_features import HistoricalFeatureExtractor
from flight_delay.phase2_features import Phase2FeatureEngineer
from flight_delay.preprocessing import PreprocessingPipeline
from flight_delay.reporting_ieee import IEEEConferenceReportGenerator
from flight_delay.rescheduling import ReschedulingPolicyEngine
from flight_delay.temporal_split import TemporalSplitter
from flight_delay.train_lightgbm import LightGBMTrainer, LIGHTGBM_AVAILABLE
from flight_delay.train_logistic import LogisticRegressionTrainer
from flight_delay.train_neural_net import DeepLearningTrainer
from flight_delay.train_xgboost import XGBoostTrainer, XGBOOST_AVAILABLE

logger = logging.getLogger(__name__)


class Phase2Pipeline:
    """Master orchestrator for Phase 2: Gradient Boosting, Temporal DL, and Tactical Rescheduling."""

    def __init__(self, config: Config):
        self.cfg = config
        self.cfg.ensure_directories()

    def run(self) -> Dict[str, Any]:
        """Execute full Phase 2 research pipeline end-to-end."""
        logger.info("=" * 75)
        logger.info("STARTING PHASE 2: GRADIENT BOOSTING, DEEP LEARNING & RESCHEDULING")
        logger.info("=" * 75)

        # 1. Load Data
        loader = DataLoader(
            raw_path=self.cfg.raw_path,
            airlines_path=self.cfg.airlines_path,
            airports_path=self.cfg.airports_path,
        )
        raw_df = loader.load_data(
            sample_size=self.cfg.sample_size,
            random_seed=self.cfg.seed,
        )
        airlines_df = loader.load_airlines()

        # 2. Clean & Validate
        validator = DataValidator(
            target_name=self.cfg.target_name,
            raw_target_col=self.cfg.raw_target_col,
            threshold_minutes=self.cfg.threshold_minutes,
        )
        dq_report_path = self.cfg.results_dir / "data_quality" / "data_quality_report.json"
        clean_df, dq_report = validator.clean_and_validate(raw_df, save_report_path=dq_report_path)

        # 3. Temporal Date Partitioning (Strict Chronological Split)
        splitter = TemporalSplitter(
            train_ratio=self.cfg.train_ratio,
            val_ratio=self.cfg.val_ratio,
            test_ratio=self.cfg.test_ratio,
            date_column=self.cfg.date_column,
            embargo_days=self.cfg.embargo_days,
        )
        split_manifest_path = self.cfg.artifacts_dir / "manifests" / "split_manifest_phase2.json"
        train_df, val_df, test_df, split_manifest = splitter.split(
            clean_df,
            target_col=self.cfg.target_name,
            save_manifest_path=split_manifest_path,
        )

        # 4. Phase 2 Feature Engineering (Airport Congestion & Tail Turnaround Propagation)
        fe = Phase2FeatureEngineer(denylist=self.cfg.denylist)
        train_feats = fe.extract_features(train_df)
        val_feats = fe.extract_features(val_df)
        test_feats = fe.extract_features(test_df)

        # Purge all forbidden post-departure denylist columns from datasets
        forbidden_cols = [c for c in self.cfg.denylist if c != self.cfg.target_name]
        train_feats = train_feats.drop(columns=[c for c in forbidden_cols if c in train_feats.columns])
        val_feats = val_feats.drop(columns=[c for c in forbidden_cols if c in val_feats.columns])
        test_feats = test_feats.drop(columns=[c for c in forbidden_cols if c in test_feats.columns])

        # 5. Expanding Historical Smoothed Aggregates (Strict Chronological Isolation)
        hist_extractor = HistoricalFeatureExtractor(
            entities=self.cfg.historical_entities,
            smoothing_weight=self.cfg.smoothing_weight,
            min_samples=self.cfg.min_samples,
            date_column=self.cfg.date_column,
            target_col=self.cfg.target_name,
        )
        hist_extractor.fit(train_feats)
        train_feats = hist_extractor.transform(train_feats, is_training=True)
        val_feats = hist_extractor.transform(val_feats, is_training=False)
        test_feats = hist_extractor.transform(test_feats, is_training=False)

        # Zero-Leakage Audit
        feature_cols = [c for c in train_feats.columns if c not in [self.cfg.target_name, self.cfg.date_column]]
        fe.assert_zero_leakage(feature_cols)

        # 6. Preprocessing for Continuous Features
        # Add new Phase 2 congestion and turnaround numerical columns
        phase2_numerics = [
            "DISTANCE", "SCHEDULED_TIME", "SCHED_DEP_HOUR", "SCHED_DEP_MINUTE",
            "DEP_HOUR_SIN", "DEP_HOUR_COS", "DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS",
            "IS_WEEKEND", "IS_OVERNIGHT", "HIST_AIRLINE_DELAY_RATE", "HIST_AIRLINE_FLIGHT_COUNT",
            "HIST_ORIGIN_AIRPORT_DELAY_RATE", "HIST_ORIGIN_AIRPORT_FLIGHT_COUNT",
            "HIST_DESTINATION_AIRPORT_DELAY_RATE", "HIST_DESTINATION_AIRPORT_FLIGHT_COUNT",
            "HIST_ROUTE_DELAY_RATE", "HIST_ROUTE_FLIGHT_COUNT",
            "ORIGIN_CONGESTION_1H", "DEST_CONGESTION_1H", "CARRIER_ORIGIN_SHARE",
            "INBOUND_SCHEDULED_BUFFER_MIN", "INBOUND_DELAY_KNOWN", "INBOUND_UNARRIVED_AT_T60",
        ]
        PreprocessingPipeline.NUMERIC_FEATURES = [c for c in phase2_numerics if c in train_feats.columns]

        preprocessor = PreprocessingPipeline(top_airports_k=50)
        X_train = preprocessor.fit_transform(train_feats)
        X_val = preprocessor.transform(val_feats)
        X_test = preprocessor.transform(test_feats)

        y_train = train_feats[self.cfg.target_name].values
        y_val = val_feats[self.cfg.target_name].values
        y_test = test_feats[self.cfg.target_name].values

        preprocessor.save(self.cfg.artifacts_dir / "metadata" / "preprocessor_phase2.joblib")

        # Track models, probabilities, and evaluation metrics
        val_probs_dict = {}
        test_probs_dict = {}
        benchmarks = {}

        # 7. Model 1: Baseline Logistic Regression
        lr_trainer = LogisticRegressionTrainer(**self.cfg.lr_params)
        lr_model, _ = lr_trainer.train(X_train, y_train)
        lr_trainer.save(self.cfg.models_dir / "logistic_regression")
        val_probs_dict["Logistic Regression"] = lr_trainer.predict_proba(X_val)
        test_probs_dict["Logistic Regression"] = lr_trainer.predict_proba(X_test)
        benchmarks["Logistic Regression"] = benchmark_inference_speed(lr_model, X_val[:2000])

        # 8. Model 2: Tuned XGBoost Classifier
        if XGBOOST_AVAILABLE:
            xgb_trainer = XGBoostTrainer(**self.cfg.xgb_params)
            xgb_model, _ = xgb_trainer.train(X_train, y_train, X_val, y_val)
            xgb_trainer.save(self.cfg.models_dir / "xgboost")
            val_probs_dict["XGBoost"] = xgb_trainer.predict_proba(X_val)
            test_probs_dict["XGBoost"] = xgb_trainer.predict_proba(X_test)
            benchmarks["XGBoost"] = benchmark_inference_speed(xgb_model, X_val[:2000])

        # 9. Model 3: Tuned LightGBM Classifier
        if LIGHTGBM_AVAILABLE:
            lgb_params = self.cfg.raw_dict.get("models", {}).get("lightgbm", {})
            lgb_trainer = LightGBMTrainer(**lgb_params)
            lgb_model, _ = lgb_trainer.train(X_train, y_train, X_val, y_val, tune=True, n_tuning_trials=5)
            lgb_trainer.save(self.cfg.models_dir / "lightgbm")
            val_probs_dict["LightGBM"] = lgb_trainer.predict_proba(X_val)
            test_probs_dict["LightGBM"] = lgb_trainer.predict_proba(X_test)
            benchmarks["LightGBM"] = benchmark_inference_speed(lgb_model, X_val[:2000])

        # 10. Model 4: Deep Learning Tabular MLP with Entity Embeddings
        # Extract continuous subset for neural network inputs
        cont_cols = PreprocessingPipeline.NUMERIC_FEATURES
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_c_tr = scaler.fit_transform(train_feats[cont_cols].fillna(0).values)
        X_c_val = scaler.transform(val_feats[cont_cols].fillna(0).values)
        X_c_te = scaler.transform(test_feats[cont_cols].fillna(0).values)

        dl_mlp = DeepLearningTrainer(
            model_type="mlp_embedding",
            airline_dim=16,
            origin_dim=32,
            dest_dim=32,
            hidden_dims=[256, 128, 64],
            epochs=10,
            batch_size=512,
        )
        dl_mlp.train(X_c_tr, y_train, train_feats, X_c_val, y_val, val_feats)
        dl_mlp.save(self.cfg.models_dir / "neural_network")
        val_probs_dict["DL Tabular MLP"] = dl_mlp.predict_proba(X_c_val, val_feats)
        test_probs_dict["DL Tabular MLP"] = dl_mlp.predict_proba(X_c_te, test_feats)
        t0 = time.perf_counter()
        _ = dl_mlp.predict_proba(X_c_val[:1000], val_feats.iloc[:1000])
        dl_lat = (time.perf_counter() - t0) * 1000
        benchmarks["DL Tabular MLP"] = {"inference_ms_per_1000_samples": round(dl_lat, 2)}

        # 11. Model 5: Temporal Sequence Recurrent Net (LSTM/GRU)
        dl_temp = DeepLearningTrainer(
            model_type="temporal_lstm",
            epochs=8,
            batch_size=512,
        )
        dl_temp.train(X_c_tr, y_train, train_feats, X_c_val, y_val, val_feats)
        dl_temp.save(self.cfg.models_dir / "temporal_model")
        val_probs_dict["Temporal Recurrent Net"] = dl_temp.predict_proba(X_c_val, val_feats)
        test_probs_dict["Temporal Recurrent Net"] = dl_temp.predict_proba(X_c_te, test_feats)
        t0 = time.perf_counter()
        _ = dl_temp.predict_proba(X_c_val[:1000], val_feats.iloc[:1000])
        temp_lat = (time.perf_counter() - t0) * 1000
        benchmarks["Temporal Recurrent Net"] = {"inference_ms_per_1000_samples": round(temp_lat, 2)}

        # 12. Model Selection & Probability Calibration (Validation Data ONLY)
        from sklearn.metrics import average_precision_score
        # Automatically select the highest PR AUC model on the validation set as Primary Engine
        primary_name = max(val_probs_dict.keys(), key=lambda k: average_precision_score(y_val, val_probs_dict[k]))
        logger.info(f"Selected '{primary_name}' as Primary Operational Engine based on Validation PR AUC.")
        primary_val_probs = val_probs_dict[primary_name]
        primary_test_probs = test_probs_dict[primary_name]

        calibrator = ModelCalibrator(method="sigmoid")
        calibrator.fit(primary_val_probs, y_val)
        calibrator.save(self.cfg.models_dir / "calibrator_phase2.joblib")

        primary_val_calibrated = calibrator.calibrate(primary_val_probs)
        primary_test_calibrated = calibrator.calibrate(primary_test_probs)
        test_probs_dict[f"{primary_name} Calibrated"] = primary_test_calibrated

        # Operational Threshold Optimization (Target Precision on Calibrated Validation Set)
        operational_thresh, thresh_info = find_operational_threshold(
            y_true=y_val,
            y_prob=primary_val_calibrated,
            target_precision=self.cfg.target_precision,
        )
        with open(self.cfg.results_dir / "metrics" / "operational_threshold_phase2.json", "w") as f:
            json.dump(thresh_info, f, indent=2)

        # Calibration Curves & Reliability Diagram
        calib_stats = plot_reliability_diagram(
            y_true=y_test,
            raw_probs=primary_test_probs,
            calibrated_probs=primary_test_calibrated,
            save_path=self.cfg.results_dir / "figures" / "calibration_curve_phase2.png",
            model_name=primary_name,
            n_bins=10,
        )
        with open(self.cfg.results_dir / "calibration" / "calibration_stats_phase2.json", "w") as f:
            json.dump(calib_stats, f, indent=2)

        # 13. Comprehensive Benchmark Evaluation Matrix
        test_metric_records = []
        for name, probs in test_probs_dict.items():
            thresh = 0.50 if "Calibrated" not in name else operational_thresh
            metrics = compute_comprehensive_metrics(y_test, probs, threshold=thresh)
            lat = benchmarks.get(name.replace(" Calibrated", ""), {}).get("inference_ms_per_1000_samples", 1.0)
            test_metric_records.append({
                "Model": name,
                "ROC AUC": metrics["roc_auc"],
                "PR AUC": metrics["pr_auc_average_precision"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1"],
                "Brier Score": metrics["brier_score"],
                "Threshold": thresh,
                "Latency (ms/1k)": lat,
            })

        metrics_df = pd.DataFrame(test_metric_records).sort_values("PR AUC", ascending=False)
        metrics_df.to_csv(self.cfg.results_dir / "metrics" / "model_comparison_phase2.csv", index=False)
        with open(self.cfg.results_dir / "metrics" / "test_metrics_phase2.json", "w") as f:
            json.dump(test_metric_records, f, indent=2)

        # 14. Offline What-If Tactical Rescheduling Simulation
        rescheduler = ReschedulingPolicyEngine(
            threshold_risk=self.cfg.raw_dict.get("rescheduling", {}).get("threshold_risk", 0.45),
            delay_buffer_minutes=self.cfg.raw_dict.get("rescheduling", {}).get("delay_buffer_minutes", 30),
        )
        rec_audit_df = rescheduler.recommend_actions(test_feats, primary_test_calibrated)
        rec_audit_path = self.cfg.results_dir / "predictions" / "rescheduling_recommendations.csv"
        rec_audit_df[[
            "FLIGHT_DATE", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
            "SCHEDULED_DEPARTURE", "ARR_DEL15", "CALIBRATED_DELAY_RISK",
            "RECOMMENDED_ACTION", "POST_ACTION_ESTIMATED_RISK", "EXPECTED_DELAY_MINUTES_SAVED"
        ]].to_csv(rec_audit_path, index=False)

        policy_sweep_df = rescheduler.simulate_policy_sweep(test_feats, primary_test_calibrated, y_test)
        policy_sweep_df.to_csv(self.cfg.results_dir / "metrics" / "rescheduling_policy_sweep.csv", index=False)

        tradeoff_fig_path = self.cfg.results_dir / "figures" / "rescheduling_tradeoff.png"
        rescheduler.plot_rescheduling_tradeoff(policy_sweep_df, tradeoff_fig_path)

        # 15. Plots: Evaluation Curves & Feature Importance
        plot_evaluation_curves(y_test, test_probs_dict, self.cfg.results_dir / "figures", title_suffix="Phase 2 Test")
        preds_dict = {primary_name: (primary_test_probs >= operational_thresh).astype(int)}
        plot_confusion_matrices(y_test, preds_dict, self.cfg.results_dir / "figures")

        # 16. IEEE Conference Paper Generation (.tex and .pdf)
        figures_paths = {
            "roc_curves": self.cfg.results_dir / "figures" / "roc_curves.png",
            "calibration_curve": self.cfg.results_dir / "figures" / "calibration_curve_phase2.png",
            "rescheduling_tradeoff": tradeoff_fig_path,
        }
        ieee_reporter = IEEEConferenceReportGenerator(
            reports_dir=self.cfg.reports_dir,
            figures_dir=self.cfg.results_dir / "figures",
        )
        tex_path = self.cfg.reports_dir / "phase2_ieee_paper.tex"
        pdf_path = self.cfg.reports_dir / "phase2_ieee_paper.pdf"

        ieee_reporter.generate_latex_document(metrics_df, policy_sweep_df, tex_path)
        ieee_reporter.generate_pdf_document(metrics_df, policy_sweep_df, figures_paths, pdf_path)

        logger.info("=" * 75)
        logger.info("PHASE 2 COMPLETED SUCCESSFULLY!")
        logger.info(f"LaTeX Paper: {tex_path}")
        logger.info(f"PDF Paper:   {pdf_path}")
        logger.info("=" * 75)

        return {
            "status": "SUCCESS",
            "metrics": test_metric_records,
            "rescheduling_summary": policy_sweep_df.to_dict(orient="records"),
            "tex_path": str(tex_path),
            "pdf_path": str(pdf_path),
        }
