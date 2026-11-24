#!/usr/bin/env bash
# ==============================================================================
# Flight Delay Prediction — Phase 1  ·  Result Summary Viewer
# ==============================================================================
# Loads the already-trained models and prints a rich terminal summary.
# Does NOT re-train, re-process data, or re-run the full pipeline.
#
# Usage:
#   ./run.sh            # print result summary to terminal
#   ./run.sh -o         # also open the PDF report
#   ./run.sh -h         # show this help
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# ── ANSI colours ──────────────────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
RESET="\033[0m"

OPEN_REPORT=false

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "========================================================================"
    echo "       FLIGHT DELAY PREDICTION — PHASE 1  ·  RESULT SUMMARY"
    echo "========================================================================"
    echo -e "${RESET}"
}

print_usage() {
    echo -e "${BOLD}Usage:${RESET} ./run.sh [-o] [-h]"
    echo ""
    echo -e "${BOLD}Options:${RESET}"
    echo "  -o, --open   Open the PDF report in the default viewer after summary"
    echo "  -h, --help   Show this help message"
    echo ""
    echo "This script reads the already-trained model artefacts and metrics"
    echo "produced by the pipeline.  It does NOT re-train any model."
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--open)  OPEN_REPORT=true; shift ;;
        -h|--help)  print_banner; print_usage; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${RESET}"; print_usage; exit 1 ;;
    esac
done

print_banner

PYTHON_CMD="python3"

# ── 1. Quick environment check ────────────────────────────────────────────────
echo -e "${BLUE}${BOLD}[1/4] Environment${RESET}"
if ! command -v "${PYTHON_CMD}" &>/dev/null; then
    echo -e "${RED}✗ python3 not found in PATH.${RESET}"; exit 1
fi
PY_VER=$(${PYTHON_CMD} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo -e "${GREEN}✓ Python ${PY_VER}${RESET}  ($(which ${PYTHON_CMD}))"

# ── 2. Verify artefact files exist ───────────────────────────────────────────
echo -e "\n${BLUE}${BOLD}[2/4] Verifying saved artefacts${RESET}"

REQUIRED_FILES=(
    "models/logistic_regression/model.joblib"
    "models/logistic_regression/metadata.json"
    "models/xgboost/model.joblib"
    "models/xgboost/metadata.json"
    "models/calibrator.joblib"
    "artifacts/metadata/preprocessor.joblib"
    "results/metrics/model_comparison.csv"
    "results/metrics/test_metrics.json"
    "results/metrics/operational_threshold.json"
    "reports/phase1_report.pdf"
)

ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [[ -f "${f}" ]]; then
        SIZE=$(${PYTHON_CMD} -c "import os; s=os.path.getsize('${f}'); print(f'{s/1024:.1f} KB' if s<1048576 else f'{s/1048576:.2f} MB')")
        echo -e "  ${GREEN}✓${RESET} ${f}  ${YELLOW}(${SIZE})${RESET}"
    else
        echo -e "  ${RED}✗ MISSING: ${f}${RESET}"
        ALL_OK=false
    fi
done

if [[ "${ALL_OK}" == false ]]; then
    echo -e "\n${RED}${BOLD}Some artefacts are missing. Run the full pipeline first:${RESET}"
    echo -e "  ${CYAN}./run_full_pipeline.sh --skip-tests${RESET}"
    exit 1
fi

# ── 3. Load models & print summary ───────────────────────────────────────────
echo -e "\n${BLUE}${BOLD}[3/4] Loading models & printing result summary${RESET}"

PYTHONPATH="${PROJECT_DIR}/src" ${PYTHON_CMD} - <<'PYEOF'
import json, os, sys
from pathlib import Path

# ── colour helpers (ANSI) ─────────────────────────────────────────────────────
BOLD  = "\033[1m"
GREEN = "\033[0;32m"
CYAN  = "\033[0;36m"
YELLOW= "\033[0;33m"
RESET = "\033[0m"

def hdr(title):
    bar = "─" * 68
    print(f"\n{BOLD}{CYAN}{bar}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{bar}{RESET}")

def ok(label, value):
    print(f"  {GREEN}●{RESET} {BOLD}{label:<30}{RESET} {value}")

# ── Read all data from JSON / CSV files only (no model imports needed) ─────────
with open("models/logistic_regression/metadata.json") as f: lr_meta  = json.load(f)
with open("models/xgboost/metadata.json")             as f: xgb_meta = json.load(f)
with open("results/metrics/test_metrics.json")        as f: metrics  = json.load(f)
with open("results/metrics/operational_threshold.json")as f: thresh  = json.load(f)
with open("artifacts/metadata/feature_schema.json")   as f: schema   = json.load(f)

# Preprocessor info from feature schema (no joblib load needed)
num_features = schema.get("num_features", "N/A")

# ── Project overview ──────────────────────────────────────────────────────────
hdr("PROJECT OVERVIEW")
ok("Task",           "Binary classification — flight arrival delayed ≥ 15 min")
ok("Dataset",        "US DOT 2015 Airline On-Time Performance (5.8 M flights)")
ok("Temporal split", "Train → Validation → Test  (chronological, no leakage)")
ok("Feature count",  f"{num_features} engineered features")

# ── Model info from metadata JSON ────────────────────────────────────────────
hdr("TRAINED MODELS")
ok("Logistic Regression",
   f"scikit-learn  |  {lr_meta.get('n_samples', 'N/A'):,} training rows  |  "
   f"{lr_meta.get('training_duration_seconds', 'N/A')} s fit time")
ok("XGBoost",
   f"xgboost  |  {xgb_meta.get('n_samples', 'N/A'):,} training rows  |  "
   f"{xgb_meta.get('training_duration_seconds', 'N/A')} s fit time  |  "
   f"best iter {xgb_meta.get('best_iteration', 'N/A')}")
ok("Calibrator",  "Sigmoid / Platt scaling — fitted on validation set only")
ok("Preprocessor", f"{num_features} final feature columns after encoding")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
hdr("HYPER-PARAMETERS")
print(f"\n  {BOLD}Logistic Regression:{RESET}")
for k, v in lr_meta.get("hyperparameters", {}).items():
    print(f"    {k:<30} {v}")
print(f"\n  {BOLD}XGBoost:{RESET}")
for k, v in xgb_meta.get("hyperparameters", {}).items():
    print(f"    {k:<30} {v}")

# ── Test-set metrics ──────────────────────────────────────────────────────────
hdr("TEST-SET METRICS")
col_w = 18
header = (f"  {'Model':<38} {'ROC AUC':>{col_w}} {'PR AUC':>{col_w}}"
          f" {'Precision':>{col_w}} {'Recall':>{col_w}} {'F1':>{col_w}} {'Brier':>{col_w}}")
print(f"\n{BOLD}{header}{RESET}")
print("  " + "─" * (38 + col_w * 6))
for row in metrics:
    name   = row["Model"][:37]
    roc    = f"{row.get('ROC AUC', 0):.4f}"
    prauc  = f"{row.get('PR AUC', 0):.4f}"
    prec   = f"{row.get('Precision', 0):.4f}"
    rec    = f"{row.get('Recall', 0):.4f}"
    f1     = f"{row.get('F1 Score', 0):.4f}"
    brier  = f"{row.get('Brier Score', 0):.5f}"
    print(f"  {name:<38} {roc:>{col_w}} {prauc:>{col_w}} {prec:>{col_w}} {rec:>{col_w}} {f1:>{col_w}} {brier:>{col_w}}")

# ── Operational threshold ─────────────────────────────────────────────────────
hdr("OPERATIONAL THRESHOLD")
ok("Decision threshold",        f"{thresh.get('selected_threshold', 'N/A')}")
ok("Rule applied",              thresh.get('rule_applied', 'N/A'))
ok("Target precision (val)",    f"{thresh.get('target_precision_requested', 'N/A')}")
ok("Achieved target",           str(thresh.get('achieved_target_on_validation', 'N/A')))
ok("Precision @ threshold",     f"{thresh.get('validation_precision_at_threshold', 'N/A')}")
ok("Recall @ threshold",        f"{thresh.get('validation_recall_at_threshold', 'N/A')}")

# ── Output artefacts ──────────────────────────────────────────────────────────
hdr("OUTPUT ARTEFACTS")
import os
artefacts = {
    "PDF Report":         "reports/phase1_report.pdf",
    "Markdown Report":    "reports/phase1_report.md",
    "LR model":           "models/logistic_regression/model.joblib",
    "XGBoost model":      "models/xgboost/model.joblib",
    "Calibrator":         "models/calibrator.joblib",
    "Preprocessor":       "artifacts/metadata/preprocessor.joblib",
    "Model comparison":   "results/metrics/model_comparison.csv",
    "Test predictions":   "results/predictions/test_predictions.csv",
    "Feature importance": "results/feature_importance/xgboost_feature_importance.csv",
}
for label, path in artefacts.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        sz_str = f"{size/1024:.1f} KB" if size < 1048576 else f"{size/1048576:.2f} MB"
        ok(label, f"{path}  {YELLOW}({sz_str}){RESET}")
    else:
        print(f"  \033[0;31m✗{RESET} {label:<30} not found")

print()
PYEOF

# ── 4. Final banner ───────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}========================================================================"
echo "  ✅  SUMMARY COMPLETE  —  All models loaded successfully"
echo -e "========================================================================${RESET}"
echo ""
echo -e "  📄 ${BOLD}PDF Report:${RESET}  ${PROJECT_DIR}/reports/phase1_report.pdf"
echo -e "  📝 ${BOLD}MD  Report:${RESET}  ${PROJECT_DIR}/reports/phase1_report.md"
echo -e "  📊 ${BOLD}Metrics CSV:${RESET} ${PROJECT_DIR}/results/metrics/model_comparison.csv"
echo ""

# ── Optionally open PDF ───────────────────────────────────────────────────────
PDF_REPORT="${PROJECT_DIR}/reports/phase1_report.pdf"
if [[ "${OPEN_REPORT}" == true && -f "${PDF_REPORT}" ]]; then
    echo -e "${CYAN}Opening PDF report...${RESET}"
    if   command -v open     &>/dev/null; then open     "${PDF_REPORT}"
    elif command -v xdg-open &>/dev/null; then xdg-open "${PDF_REPORT}" &>/dev/null &
    fi
fi

echo -e "${CYAN}Done.${RESET}"
