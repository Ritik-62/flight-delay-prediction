#!/usr/bin/env python3
"""Audits and verifies all required Phase 1 output deliverables and builds the artifact manifest."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_file(path: Path) -> dict:
    """Check existence, size, and status of a deliverable file."""
    if path.exists() and path.is_file():
        size = path.stat().st_size
        return {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "status": "GENERATED" if size > 0 else "EMPTY_FILE",
            "size_bytes": size,
            "size_kb": round(size / 1024, 2),
            "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "status": "MISSING",
        "size_bytes": 0,
        "size_kb": 0.0,
        "last_modified": None,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Auditing Phase 1 output artifacts...")

    required_deliverables = [
        # Models
        PROJECT_ROOT / "models" / "logistic_regression" / "model.joblib",
        PROJECT_ROOT / "models" / "logistic_regression" / "metadata.json",
        PROJECT_ROOT / "models" / "xgboost" / "model.joblib",
        PROJECT_ROOT / "models" / "xgboost" / "metadata.json",
        PROJECT_ROOT / "models" / "calibrator.joblib",
        PROJECT_ROOT / "artifacts" / "metadata" / "preprocessor.joblib",
        PROJECT_ROOT / "artifacts" / "metadata" / "feature_schema.json",

        # Data & Manifests
        PROJECT_ROOT / "results" / "data_quality" / "data_quality_report.json",
        PROJECT_ROOT / "artifacts" / "manifests" / "split_manifest.json",

        # Metrics & Predictions
        PROJECT_ROOT / "results" / "metrics" / "model_comparison.csv",
        PROJECT_ROOT / "results" / "metrics" / "test_metrics.json",
        PROJECT_ROOT / "results" / "metrics" / "operational_threshold.json",
        PROJECT_ROOT / "results" / "predictions" / "test_predictions.csv",
        PROJECT_ROOT / "results" / "predictions" / "val_predictions.csv",
        PROJECT_ROOT / "results" / "calibration" / "calibration_stats.json",

        # Feature Importance & Error Analysis
        PROJECT_ROOT / "results" / "feature_importance" / "logistic_regression_coefficients.csv",
        PROJECT_ROOT / "results" / "feature_importance" / "xgboost_feature_importance.csv",
        PROJECT_ROOT / "results" / "error_analysis" / "error_analysis_airline_xgboost.csv",
        PROJECT_ROOT / "results" / "error_analysis" / "error_analysis_departure_hour_xgboost.csv",

        # Figures
        PROJECT_ROOT / "results" / "figures" / "target_distribution.png",
        PROJECT_ROOT / "results" / "figures" / "delay_rate_by_carrier.png",
        PROJECT_ROOT / "results" / "figures" / "delay_rate_by_departure_hour.png",
        PROJECT_ROOT / "results" / "figures" / "delay_rate_by_weekday_and_month.png",
        PROJECT_ROOT / "results" / "figures" / "delay_rate_by_airport.png",
        PROJECT_ROOT / "results" / "figures" / "delay_prevalence_over_time.png",
        PROJECT_ROOT / "results" / "figures" / "roc_curves.png",
        PROJECT_ROOT / "results" / "figures" / "precision_recall_curves.png",
        PROJECT_ROOT / "results" / "figures" / "confusion_matrices.png",
        PROJECT_ROOT / "results" / "figures" / "calibration_curve.png",
        PROJECT_ROOT / "results" / "figures" / "feature_importance.png",
        PROJECT_ROOT / "results" / "figures" / "error_analysis.png",

        # Reports
        PROJECT_ROOT / "reports" / "phase1_report.md",
        PROJECT_ROOT / "reports" / "phase1_report.pdf",
    ]

    manifest_entries = []
    generated_count = 0
    missing_count = 0

    for path in required_deliverables:
        entry = check_file(path)
        manifest_entries.append(entry)
        if entry["status"] == "GENERATED":
            generated_count += 1
            logging.info(f" [PASS] {entry['path']} ({entry['size_kb']} KB)")
        else:
            missing_count += 1
            logging.warning(f" [FAIL] {entry['path']} - {entry['status']}")

    # Model deserialization verification test
    lr_path = PROJECT_ROOT / "models" / "logistic_regression" / "model.joblib"
    xgb_path = PROJECT_ROOT / "models" / "xgboost" / "model.joblib"

    lr_loaded = False
    xgb_loaded = False
    try:
        if lr_path.exists():
            _ = joblib.load(lr_path)
            lr_loaded = True
        if xgb_path.exists():
            _ = joblib.load(xgb_path)
            xgb_loaded = True
    except Exception as e:
        logging.error(f"Model deserialization error: {e}")

    # Verify PDF opens and has pages
    pdf_path = PROJECT_ROOT / "reports" / "phase1_report.pdf"
    pdf_valid = False
    if pdf_path.exists() and pdf_path.stat().st_size > 5000:
        try:
            with open(pdf_path, "rb") as f:
                header = f.read(5)
                if header == b"%PDF-":
                    pdf_valid = True
        except Exception:
            pdf_valid = False

    manifest = {
        "manifest_version": "1.0.0",
        "generated_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_deliverables": len(required_deliverables),
            "generated": generated_count,
            "missing": missing_count,
            "completion_pct": round(generated_count / len(required_deliverables) * 100, 1),
            "models_verified_loadable": lr_loaded and xgb_loaded,
            "pdf_report_valid": pdf_valid,
        },
        "artifacts": manifest_entries,
    }

    manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "artifact_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logging.info("=" * 60)
    logging.info(f"Artifact Manifest Written: {manifest_path}")
    logging.info(f"Completion: {generated_count}/{len(required_deliverables)} ({manifest['summary']['completion_pct']}%)")
    logging.info(f"Models Loadable: {lr_loaded and xgb_loaded} | PDF Valid: {pdf_valid}")
    logging.info("=" * 60)

    if missing_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
