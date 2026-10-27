#!/usr/bin/env bash
# ==============================================================================
# Flight Delay Prediction — Phase 1 Direct Execution Script
# ==============================================================================
# This script executes the complete machine learning pipeline:
# 1. Environment & dependency verification
# 2. Raw dataset check (with automated download if missing)
# 3. Automated unit and integration test suite (pytest)
# 4. End-to-end Phase 1 ML pipeline (cleaning, EDA, training, calibration, reports)
# 5. Output artifact audit & deliverable verification
# ==============================================================================

set -euo pipefail

# Determine repository root
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# ANSI Colors for terminal output
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
RESET="\033[0m"

# Default configuration
CONFIG_PATH="configs/phase1.yaml"
SAMPLE_SIZE=""
SKIP_TESTS=false
OPEN_REPORT=false

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "========================================================================"
    echo "       FLIGHT DELAY PREDICTION — PHASE 1 MASTER RUNNER"
    echo "========================================================================"
    echo -e "${RESET}"
}

print_usage() {
    echo -e "${BOLD}Usage:${RESET} ./run.sh [OPTIONS]"
    echo ""
    echo -e "${BOLD}Options:${RESET}"
    echo "  -c, --config <path>      Path to YAML configuration (default: configs/phase1.yaml)"
    echo "  -s, --sample-size <N>    Override dataset sample size (e.g., 50000, 350000)"
    echo "  -q, --quick              Fast run mode with 50,000 samples"
    echo "      --skip-tests         Skip running automated unit and integration tests"
    echo "  -o, --open               Open the generated PDF report automatically upon completion"
    echo "  -h, --help               Display this help message and exit"
    echo ""
    echo -e "${BOLD}Examples:${RESET}"
    echo "  ./run.sh                 # Full default execution (350k samples + tests + report)"
    echo "  ./run.sh --quick -o      # Quick run (50k samples) and open PDF report"
    echo "  ./run.sh --skip-tests    # Run pipeline directly without executing pytest"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -s|--sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        -q|--quick)
            SAMPLE_SIZE="50000"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -o|--open)
            OPEN_REPORT=true
            shift
            ;;
        -h|--help)
            print_banner
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${RESET}"
            print_usage
            exit 1
            ;;
    esac
done

print_banner

# ------------------------------------------------------------------------------
# 1. Environment & Python Verification
# ------------------------------------------------------------------------------
echo -e "${BLUE}${BOLD}[Step 1/5] Checking Python Environment...${RESET}"
PYTHON_CMD="python3"
if ! command -v "${PYTHON_CMD}" &>/dev/null; then
    echo -e "${RED}Error: python3 is not installed or not in PATH.${RESET}"
    exit 1
fi

PY_VER=$(${PYTHON_CMD} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo -e "${GREEN}✓ Python executable:${RESET} $(which ${PYTHON_CMD}) (${PY_VER})"

# Verify critical packages
${PYTHON_CMD} -c "
import numpy, pandas, sklearn, xgboost, reportlab, pytest, yaml
print('✓ All critical dependencies verified (numpy, pandas, sklearn, xgboost, reportlab, pytest, pyyaml)')
" || {
    echo -e "${YELLOW}Installing missing dependencies from requirements.txt...${RESET}"
    ${PYTHON_CMD} -m pip install -r requirements.txt
}

# ------------------------------------------------------------------------------
# 2. Dataset Verification
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[Step 2/5] Checking Dataset Availability...${RESET}"
RAW_DATA_PATH="data/raw/flights.parquet"

if [[ ! -f "${RAW_DATA_PATH}" ]]; then
    echo -e "${YELLOW}Raw dataset not found at ${RAW_DATA_PATH}.${RESET}"
    echo -e "${CYAN}Attempting automated download of US DOT flight delays dataset (137 MB)...${RESET}"
    mkdir -p data/raw
    ${PYTHON_CMD} -c "
import urllib.request
url = 'https://huggingface.co/datasets/hsanchezp/us-dot-flight-delays-2015/resolve/main/flights.parquet'
target = 'data/raw/flights.parquet'
print('Downloading flights.parquet...')
urllib.request.urlretrieve(url, target)
print('✓ Download complete!')
"
else
    DATA_SIZE_MB=$(${PYTHON_CMD} -c "import os; print(f'{os.path.getsize(\"${RAW_DATA_PATH}\") / (1024*1024):.1f}')")
    echo -e "${GREEN}✓ Dataset present:${RESET} ${RAW_DATA_PATH} (${DATA_SIZE_MB} MB)"
fi

# ------------------------------------------------------------------------------
# 3. Automated Test Suite (Pytest)
# ------------------------------------------------------------------------------
if [[ "${SKIP_TESTS}" == false ]]; then
    echo -e "\n${BLUE}${BOLD}[Step 3/5] Running Automated Test Suite (Pytest)...${RESET}"
    ${PYTHON_CMD} -m pytest tests/ -v --tb=short
    echo -e "${GREEN}✓ All unit, leakage, and integration tests passed successfully!${RESET}"
else
    echo -e "\n${YELLOW}${BOLD}[Step 3/5] Skipping tests as requested (--skip-tests).${RESET}"
fi

# ------------------------------------------------------------------------------
# 4. Execute Master Phase 1 Pipeline
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[Step 4/5] Executing Complete Phase 1 Pipeline...${RESET}"
RUN_ARGS=(--config "${CONFIG_PATH}")
if [[ -n "${SAMPLE_SIZE}" ]]; then
    RUN_ARGS+=(--sample-size "${SAMPLE_SIZE}")
    echo -e "${CYAN}Using custom sample size: ${SAMPLE_SIZE} flights${RESET}"
fi

${PYTHON_CMD} scripts/run_phase1.py "${RUN_ARGS[@]}"

# ------------------------------------------------------------------------------
# 5. Verify Deliverables & Artifact Manifest
# ------------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[Step 5/5] Auditing Output Deliverables & Manifest...${RESET}"
${PYTHON_CMD} scripts/verify_artifacts.py

# ------------------------------------------------------------------------------
# Completion Summary
# ------------------------------------------------------------------------------
PDF_REPORT="${PROJECT_DIR}/reports/phase1_report.pdf"
MD_REPORT="${PROJECT_DIR}/reports/phase1_report.md"
METRICS_CSV="${PROJECT_DIR}/results/metrics/model_comparison.csv"

echo -e "\n${GREEN}${BOLD}========================================================================${RESET}"
echo -e "${GREEN}${BOLD}             PHASE 1 EXECUTION COMPLETED SUCCESSFULLY!${RESET}"
echo -e "${GREEN}${BOLD}========================================================================${RESET}"
echo ""
echo -e "${BOLD}Generated Core Deliverables:${RESET}"
echo -e "  📄 ${BOLD}PDF Project Report:${RESET}    ${PDF_REPORT}"
echo -e "  📝 ${BOLD}Markdown Report:${RESET}       ${MD_REPORT}"
echo -e "  📊 ${BOLD}Model Comparison:${RESET}      ${METRICS_CSV}"
echo -e "  🤖 ${BOLD}Logistic Model:${RESET}        ${PROJECT_DIR}/models/logistic_regression/model.joblib"
echo -e "  🚀 ${BOLD}XGBoost Model:${RESET}         ${PROJECT_DIR}/models/xgboost/model.joblib"
echo -e "  📈 ${BOLD}Figures Directory:${RESET}     ${PROJECT_DIR}/results/figures/"
echo -e "  📋 ${BOLD}Artifact Manifest:${RESET}     ${PROJECT_DIR}/artifacts/manifests/artifact_manifest.json"
echo ""

# Optional: Open PDF report in viewer
if [[ "${OPEN_REPORT}" == true && -f "${PDF_REPORT}" ]]; then
    echo -e "${CYAN}Opening PDF report in default viewer...${RESET}"
    if command -v open &>/dev/null; then
        open "${PDF_REPORT}"
    elif command -v xdg-open &>/dev/null; then
        xdg-open "${PDF_REPORT}" &>/dev/null &
    fi
fi

echo -e "${CYAN}Run complete. Ready for operational review!${RESET}"
