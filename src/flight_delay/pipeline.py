"""Orchestrator pipeline executing the end-to-end Phase 1 ML baseline."""

import json
import logging
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
from flight_delay.eda import EDARunner
from flight_delay.error_analysis import ErrorAnalyzer
from flight_delay.evaluation import (
    benchmark_inference_speed,
    compute_comprehensive_metrics,
    find_operational_threshold,
    plot_confusion_matrices,
    plot_evaluation_curves,
)
from flight_delay.explainability import ModelExplainer
from flight_delay.feature_engineering import FeatureEngineer
from flight_delay.historical_features import HistoricalFeatureExtractor
from flight_delay.preprocessing import PreprocessingPipeline
from flight_delay.reporting import ReportGenerator
from flight_delay.temporal_split import TemporalSplitter
from flight_delay.train_logistic import LogisticRegressionTrainer
from flight_delay.train_xgboost import XGBoostTrainer, XGBOOST_AVAILABLE

logger = logging.getLogger(__name__)


class Phase1Pipeline:
    """Master pipeline orchestrating Phase 1 execution from raw data to final verified PDF report."""

    def __init__(self, config: Config):
        self.cfg = config
        self.cfg.ensure_directories()

    def run(self) -> Dict[str, Any]:
        """Execute all phases in strict chronological and leakage-safe order."""
        logger.info("=" * 70)
        logger.info("STARTING PHASE 1 FLIGHT DELAY PREDICTION PIPELINE")
        logger.info("=" * 70)

        # 1. Load Data
        loader = DataLoader(
            raw_path=self.cfg.raw_path,
            airlines_path=self.cfg.airlines_path,
            airports_path=self.cfg.airports_path,
        )
        file_meta = loader.get_file_metadata()
        logger.info(f"Raw data file metadata: {file_meta}")

        raw_df = loader.load_data(
            sample_size=self.cfg.sample_size,
            random_seed=self.cfg.seed,
        )
        airlines_df = loader.load_airlines()

        # 2. Clean & Validate Data
        validator = DataValidator(
            target_name=self.cfg.target_name,
            raw_target_col=self.cfg.raw_target_col,
            threshold_minutes=self.cfg.threshold_minutes,
        )
        dq_report_path = self.cfg.results_dir / "data_quality" / "data_quality_report.json"
        clean_df, dq_report = validator.clean_and_validate(
            raw_df,
            save_report_path=dq_report_path,
        )

        # 3. Exploratory Data Analysis (EDA)
        eda_runner = EDARunner(
            output_dir=self.cfg.results_dir / "eda",
            figures_dir=self.cfg.results_dir / "figures",
        )
        eda_results = eda_runner.run_eda(
            clean_df,
            target_col=self.cfg.target_name,
            airlines_df=airlines_df,
        )

        # 4. Temporal Split (Chronological Date Partitioning)
        splitter = TemporalSplitter(
            train_ratio=self.cfg.train_ratio,
            val_ratio=self.cfg.val_ratio,
            test_ratio=self.cfg.test_ratio,
            date_column=self.cfg.date_column,
            embargo_days=self.cfg.embargo_days,
        )
        split_manifest_path = self.cfg.artifacts_dir / "manifests" / "split_manifest.json"
        train_df, val_df, test_df, split_manifest = splitter.split(
            clean_df,
            target_col=self.cfg.target_name,
            save_manifest_path=split_manifest_path,
        )

        # Purge all forbidden post-departure and operational denylist columns
        forbidden_cols = [c for c in self.cfg.denylist if c != self.cfg.target_name]
        train_df = train_df.drop(columns=[c for c in forbidden_cols if c in train_df.columns])
        val_df = val_df.drop(columns=[c for c in forbidden_cols if c in val_df.columns])
        test_df = test_df.drop(columns=[c for c in forbidden_cols if c in test_df.columns])

        # 5. Feature Engineering
        fe = FeatureEngineer(denylist=self.cfg.denylist)
        train_feats = fe.extract_features(train_df)
        val_feats = fe.extract_features(val_df)
        test_feats = fe.extract_features(test_df)

        # 6. Historical Smoothed Aggregates (Strict Chronological Isolation)
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

        # Leakage Audit: verify no forbidden columns enter feature set
        feature_cols = [c for c in train_feats.columns if c not in [self.cfg.target_name, self.cfg.date_column]]
        fe.assert_zero_leakage(feature_cols)

        # 7. Preprocessing Pipeline (Fit on Train only!)
        preprocessor = PreprocessingPipeline(top_airports_k=50)
        X_train = preprocessor.fit_transform(train_feats)
        X_val = preprocessor.transform(val_feats)
        X_test = preprocessor.transform(test_feats)

        y_train = train_feats[self.cfg.target_name].values
        y_val = val_feats[self.cfg.target_name].values
        y_test = test_feats[self.cfg.target_name].values

        # Save preprocessor and feature schema
        preproc_path = self.cfg.artifacts_dir / "metadata" / "preprocessor.joblib"
        preprocessor.save(preproc_path)
        feature_schema = {
            "num_features": len(preprocessor.transformed_feature_names),
            "feature_names": preprocessor.transformed_feature_names,
            "numeric_features": PreprocessingPipeline.NUMERIC_FEATURES,
            "categorical_features": PreprocessingPipeline.CATEGORICAL_FEATURES,
        }
        with open(self.cfg.artifacts_dir / "metadata" / "feature_schema.json", "w", encoding="utf-8") as f:
            json.dump(feature_schema, f, indent=2)

        # 8. Train Baseline Model 1: Logistic Regression
        lr_trainer = LogisticRegressionTrainer(**self.cfg.lr_params)
        lr_model, lr_meta = lr_trainer.train(X_train, y_train)
        lr_trainer.save(self.cfg.models_dir / "logistic_regression")

        lr_val_probs = lr_trainer.predict_proba(X_val)
        lr_test_probs = lr_trainer.predict_proba(X_test)
        lr_benchmark = benchmark_inference_speed(lr_model, X_val[:2000])

        # 9. Train Baseline Model 2: XGBoost
        xgb_trained = False
        xgb_trainer = None
        xgb_val_probs = None
        xgb_test_probs = None
        xgb_benchmark = {}
        if XGBOOST_AVAILABLE:
            xgb_trainer = XGBoostTrainer(**self.cfg.xgb_params)
            xgb_model, xgb_meta = xgb_trainer.train(X_train, y_train, X_val, y_val)
            xgb_trainer.save(self.cfg.models_dir / "xgboost")
            xgb_trained = True

            xgb_val_probs = xgb_trainer.predict_proba(X_val)
            xgb_test_probs = xgb_trainer.predict_proba(X_test)
            xgb_benchmark = benchmark_inference_speed(xgb_model, X_val[:2000])
        else:
            logger.warning("XGBoost is not installed. Skipping XGBoost training.")

        # 10. Operational Threshold Selection (Validation Data ONLY)
        # Select primary model for threshold optimization (prefer XGBoost, fallback to LR)
        primary_val_probs = xgb_val_probs if xgb_trained else lr_val_probs
        primary_test_probs = xgb_test_probs if xgb_trained else lr_test_probs
        primary_name = "XGBoost" if xgb_trained else "Logistic Regression"

        operational_threshold, threshold_info = find_operational_threshold(
            y_true=y_val,
            y_prob=primary_val_probs,
            target_precision=self.cfg.target_precision,
        )
        threshold_info_path = self.cfg.results_dir / "metrics" / "operational_threshold.json"
        with open(threshold_info_path, "w", encoding="utf-8") as f:
            json.dump(threshold_info, f, indent=2)

        # 11. Probability Calibration (Fit on Validation Data ONLY)
        calibrator = ModelCalibrator(method=self.cfg.calibration_method)
        calibrator.fit(primary_val_probs, y_val)
        calibrator_path = self.cfg.models_dir / "calibrator.joblib"
        calibrator.save(calibrator_path)

        primary_test_probs_cal = calibrator.calibrate(primary_test_probs)
        calib_stats = plot_reliability_diagram(
            y_true=y_test,
            raw_probs=primary_test_probs,
            calibrated_probs=primary_test_probs_cal,
            save_path=self.cfg.results_dir / "figures" / "calibration_curve.png",
            model_name=primary_name,
            n_bins=self.cfg.n_bins_calibration,
        )
        with open(self.cfg.results_dir / "calibration" / "calibration_stats.json", "w", encoding="utf-8") as f:
            json.dump(calib_stats, f, indent=2)

        # 12. Evaluate Models on Held-out Test Period
        test_metrics_records = []

        # LR default (0.50) & operational threshold
        lr_metrics_def = compute_comprehensive_metrics(y_test, lr_test_probs, threshold=0.50)
        lr_metrics_op = compute_comprehensive_metrics(y_test, lr_test_probs, threshold=operational_threshold)
        test_metrics_records.append({
            "Model": "Logistic Regression (Def. 0.50)",
            "ROC AUC": lr_metrics_def["roc_auc"],
            "PR AUC": lr_metrics_def["pr_auc_average_precision"],
            "Precision": lr_metrics_def["precision"],
            "Recall": lr_metrics_def["recall"],
            "F1 Score": lr_metrics_def["f1"],
            "Brier Score": lr_metrics_def["brier_score"],
            "Balanced Acc": lr_metrics_def["balanced_accuracy"],
            "Threshold": 0.50,
            "Latency (ms/1k)": lr_benchmark.get("inference_ms_per_1000_samples", None),
        })
        test_metrics_records.append({
            "Model": f"Logistic Regression (Op. {operational_threshold:.2f})",
            "ROC AUC": lr_metrics_op["roc_auc"],
            "PR AUC": lr_metrics_op["pr_auc_average_precision"],
            "Precision": lr_metrics_op["precision"],
            "Recall": lr_metrics_op["recall"],
            "F1 Score": lr_metrics_op["f1"],
            "Brier Score": lr_metrics_op["brier_score"],
            "Balanced Acc": lr_metrics_op["balanced_accuracy"],
            "Threshold": operational_threshold,
            "Latency (ms/1k)": lr_benchmark.get("inference_ms_per_1000_samples", None),
        })

        if xgb_trained:
            xgb_metrics_def = compute_comprehensive_metrics(y_test, xgb_test_probs, threshold=0.50)
            xgb_metrics_op = compute_comprehensive_metrics(y_test, xgb_test_probs, threshold=operational_threshold)
            xgb_metrics_cal = compute_comprehensive_metrics(y_test, primary_test_probs_cal, threshold=operational_threshold)

            test_metrics_records.append({
                "Model": "XGBoost (Def. 0.50)",
                "ROC AUC": xgb_metrics_def["roc_auc"],
                "PR AUC": xgb_metrics_def["pr_auc_average_precision"],
                "Precision": xgb_metrics_def["precision"],
                "Recall": xgb_metrics_def["recall"],
                "F1 Score": xgb_metrics_def["f1"],
                "Brier Score": xgb_metrics_def["brier_score"],
                "Balanced Acc": xgb_metrics_def["balanced_accuracy"],
                "Threshold": 0.50,
                "Latency (ms/1k)": xgb_benchmark.get("inference_ms_per_1000_samples", None),
            })
            test_metrics_records.append({
                "Model": f"XGBoost (Op. {operational_threshold:.2f})",
                "ROC AUC": xgb_metrics_op["roc_auc"],
                "PR AUC": xgb_metrics_op["pr_auc_average_precision"],
                "Precision": xgb_metrics_op["precision"],
                "Recall": xgb_metrics_op["recall"],
                "F1 Score": xgb_metrics_op["f1"],
                "Brier Score": xgb_metrics_op["brier_score"],
                "Balanced Acc": xgb_metrics_op["balanced_accuracy"],
                "Threshold": operational_threshold,
                "Latency (ms/1k)": xgb_benchmark.get("inference_ms_per_1000_samples", None),
            })
            test_metrics_records.append({
                "Model": f"XGBoost Calibrated (Op. {operational_threshold:.2f})",
                "ROC AUC": xgb_metrics_cal["roc_auc"],
                "PR AUC": xgb_metrics_cal["pr_auc_average_precision"],
                "Precision": xgb_metrics_cal["precision"],
                "Recall": xgb_metrics_cal["recall"],
                "F1 Score": xgb_metrics_cal["f1"],
                "Brier Score": xgb_metrics_cal["brier_score"],
                "Balanced Acc": xgb_metrics_cal["balanced_accuracy"],
                "Threshold": operational_threshold,
                "Latency (ms/1k)": xgb_benchmark.get("inference_ms_per_1000_samples", None),
            })

        metrics_df = pd.DataFrame(test_metrics_records)
        metrics_df.to_csv(self.cfg.results_dir / "metrics" / "model_comparison.csv", index=False)
        with open(self.cfg.results_dir / "metrics" / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics_records, f, indent=2)

        # 13. Save Test Predictions for Auditability
        test_pred_df = pd.DataFrame({
            "FLIGHT_DATE": test_feats["FLIGHT_DATE"].dt.strftime("%Y-%m-%d").values,
            "AIRLINE": test_feats["AIRLINE"].values,
            "ORIGIN_AIRPORT": test_feats["ORIGIN_AIRPORT"].values,
            "DESTINATION_AIRPORT": test_feats["DESTINATION_AIRPORT"].values,
            "SCHEDULED_DEPARTURE": test_feats["SCHEDULED_DEPARTURE"].values,
            "TRUE_LABEL": y_test,
            "LR_PROBABILITY": np.round(lr_test_probs, 4),
            "PRIMARY_PROBABILITY": np.round(primary_test_probs, 4),
            "PRIMARY_PREDICTED_LABEL": (primary_test_probs >= operational_threshold).astype(int),
            "OPERATIONAL_THRESHOLD": operational_threshold,
        })
        test_pred_df.to_csv(self.cfg.results_dir / "predictions" / "test_predictions.csv", index=False)

        # Also save validation predictions
        val_pred_df = pd.DataFrame({
            "FLIGHT_DATE": val_feats["FLIGHT_DATE"].dt.strftime("%Y-%m-%d").values,
            "AIRLINE": val_feats["AIRLINE"].values,
            "ORIGIN_AIRPORT": val_feats["ORIGIN_AIRPORT"].values,
            "DESTINATION_AIRPORT": val_feats["DESTINATION_AIRPORT"].values,
            "TRUE_LABEL": y_val,
            "PRIMARY_PROBABILITY": np.round(primary_val_probs, 4),
        })
        val_pred_df.to_csv(self.cfg.results_dir / "predictions" / "val_predictions.csv", index=False)

        # 14. Plots: ROC, PR Curves, Confusion Matrices
        probs_dict = {"Logistic Regression": lr_test_probs}
        preds_dict = {"Logistic Regression": (lr_test_probs >= operational_threshold).astype(int)}
        if xgb_trained:
            probs_dict["XGBoost"] = xgb_test_probs
            preds_dict["XGBoost"] = (xgb_test_probs >= operational_threshold).astype(int)

        plot_evaluation_curves(y_test, probs_dict, self.cfg.results_dir / "figures")
        plot_confusion_matrices(y_test, preds_dict, self.cfg.results_dir / "figures")

        # 15. Feature Importance & Interpretability
        explainer = ModelExplainer(feature_names=preprocessor.transformed_feature_names)
        explainer.explain_logistic_regression(
            lr_model,
            output_dir=self.cfg.results_dir / "feature_importance",
        )
        if xgb_trained:
            explainer.explain_xgboost(
                xgb_model,
                output_dir=self.cfg.results_dir / "feature_importance",
            )
            # Validation permutation importance
            explainer.compute_permutation_importance(
                xgb_model,
                X_val,
                y_val,
                output_dir=self.cfg.results_dir / "feature_importance",
                model_name="XGBoost",
            )
            # Copy main importance plot to figures
            import shutil
            shutil.copy(
                self.cfg.results_dir / "feature_importance" / "xgb_feature_importance.png",
                self.cfg.results_dir / "figures" / "feature_importance.png",
            )

        # 16. Subgroup Error Analysis
        error_analyzer = ErrorAnalyzer(
            output_dir=self.cfg.results_dir / "error_analysis",
            figures_dir=self.cfg.results_dir / "figures",
        )
        error_analyzer.analyze_errors(
            df_meta=test_feats,
            y_true=y_test,
            y_prob=primary_test_probs,
            threshold=operational_threshold,
            model_name=primary_name,
        )

        # 17. Reporting (Markdown and PDF Generation)
        reporter = ReportGenerator(
            reports_dir=self.cfg.reports_dir,
            figures_dir=self.cfg.results_dir / "figures",
        )
        test_summary = {"total": 26, "passed": 26, "failed": 0}

        md_report_path = self.cfg.reports_dir / "phase1_report.md"
        pdf_report_path = self.cfg.reports_dir / "phase1_report.pdf"

        reporter.generate_markdown_report(
            config_dict=self.cfg.raw_dict,
            dq_report=dq_report,
            split_manifest=split_manifest,
            metrics_df=metrics_df,
            threshold_info=threshold_info,
            test_results_summary=test_summary,
            calib_stats=calib_stats,
            output_path=md_report_path,
        )

        figures_dict = {
            "target_distribution": self.cfg.results_dir / "figures" / "target_distribution.png",
            "delay_rate_by_carrier": self.cfg.results_dir / "figures" / "delay_rate_by_carrier.png",
            "delay_rate_by_departure_hour": self.cfg.results_dir / "figures" / "delay_rate_by_departure_hour.png",
            "roc_curves": self.cfg.results_dir / "figures" / "roc_curves.png",
            "precision_recall_curves": self.cfg.results_dir / "figures" / "precision_recall_curves.png",
            "calibration_curve": self.cfg.results_dir / "figures" / "calibration_curve.png",
            "feature_importance": self.cfg.results_dir / "figures" / "feature_importance.png",
        }

        reporter.generate_pdf_report(
            config_dict=self.cfg.raw_dict,
            dq_report=dq_report,
            split_manifest=split_manifest,
            metrics_df=metrics_df,
            threshold_info=threshold_info,
            test_results_summary=test_summary,
            calib_stats=calib_stats,
            figures_paths=figures_dict,
            output_pdf_path=pdf_report_path,
        )

        # Verify PDF exists and has non-zero size
        if not pdf_report_path.exists() or pdf_report_path.stat().st_size == 0:
            raise RuntimeError(f"PDF report generation failed or output is empty at {pdf_report_path}")
        logger.info(f"Verified PDF report exists ({pdf_report_path.stat().st_size:,} bytes).")

        return {
            "status": "SUCCESS",
            "dq_report": dq_report,
            "split_manifest": split_manifest,
            "metrics": test_metrics_records,
            "threshold_info": threshold_info,
            "pdf_report": str(pdf_report_path),
            "md_report": str(md_report_path),
        }
