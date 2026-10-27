# Data Directory Documentation

This directory manages all raw, interim, and processed data assets for the **Flight Delay Prediction** initiative.

## Directory Structure
* `raw/`: Stores pristine, immutable raw flight records downloaded from the official sources.
  * `flights.parquet`: 5,819,079 domestic commercial flight records from the US DOT Bureau of Transportation Statistics (BTS).
  * `airlines.parquet`: Carrier code lookup metadata (e.g. `AA` -> American Airlines Inc.).
  * `airports.parquet`: Airport code, name, city, and geographic coordinates.
* `interim/`: Intermediate transformations prior to modeling.
* `processed/`: Final modeling-ready dataset exports.

## Dataset Acquisition Instructions
If `flights.parquet` is missing, you can obtain it via:

### Option 1: Direct Download via Hugging Face Mirror
```bash
python3 -c "
import urllib.request
url = 'https://huggingface.co/datasets/hsanchezp/us-dot-flight-delays-2015/resolve/main/flights.parquet'
urllib.request.urlretrieve(url, 'data/raw/flights.parquet')
"
```

### Option 2: Kaggle Dataset
Download the official 2015 U.S. DOT Flight Delays and Cancellations dataset from Kaggle:
[https://www.kaggle.com/datasets/usdot/flight-delays](https://www.kaggle.com/datasets/usdot/flight-delays)
Unzip the files and place `flights.csv` (or convert to `flights.parquet`) into `data/raw/`.

Configure the path in `configs/phase1.yaml` under `data.raw_path`.
