.PHONY: help install test eda train evaluate report verify run all clean

PYTHON := python3

help:
	@echo "Flight Delay Prediction - Phase 1 Automation"
	@echo "Available commands:"
	@echo "  make install    : Install Python dependencies"
	@echo "  make test       : Run full automated pytest test suite"
	@echo "  make eda        : Run data cleaning and exploratory data analysis"
	@echo "  make train      : Train Logistic Regression and XGBoost baseline models"
	@echo "  make report     : Regenerate PDF and Markdown reports"
	@echo "  make verify     : Audit all required project deliverable artifacts"
	@echo "  make run        : Execute complete end-to-end Phase 1 pipeline"
	@echo "  make clean      : Remove caches and temporary test directories"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

eda:
	$(PYTHON) scripts/run_eda.py --config configs/phase1.yaml

train:
	$(PYTHON) scripts/train_models.py --config configs/phase1.yaml

report:
	$(PYTHON) scripts/generate_report.py --config configs/phase1.yaml

verify:
	$(PYTHON) scripts/verify_artifacts.py

run:
	$(PYTHON) scripts/run_phase1.py --config configs/phase1.yaml

all: run verify test

clean:
	rm -rf .pytest_cache `find . -name __pycache__`
