# Tests

This directory contains the test suite for the **Flight Delay Prediction** project.

## Test Files

| File | Description |
|------|-------------|
| `conftest.py` | Shared fixtures and test configuration |
| `test_data_validation.py` | Data validation and schema checks |
| `test_feature_engineering.py` | Feature engineering pipeline tests |
| `test_historical_features.py` | Historical feature computation tests |
| `test_leakage.py` | Data leakage detection tests |
| `test_metrics.py` | Model evaluation metric tests |
| `test_models.py` | Model training and inference tests |
| `test_pipeline_integration.py` | End-to-end pipeline integration tests |
| `test_serialization.py` | Model serialization/deserialization tests |
| `test_temporal_split.py` | Temporal train/test split tests |

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_models.py
```
