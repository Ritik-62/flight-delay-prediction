"""Data loading module supporting CSV, compressed CSV, and Parquet formats."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DataLoader:
    """Robust data loader for airline on-time performance records."""

    def __init__(self, raw_path: str | Path, airlines_path: Optional[str | Path] = None, airports_path: Optional[str | Path] = None):
        self.raw_path = Path(raw_path)
        self.airlines_path = Path(airlines_path) if airlines_path else None
        self.airports_path = Path(airports_path) if airports_path else None

    def get_file_metadata(self) -> Dict[str, Any]:
        """Inspect the raw file and return metadata."""
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data file does not exist at: {self.raw_path}")

        file_size_bytes = self.raw_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        ext = self.raw_path.suffix.lower()

        if ext == ".parquet":
            schema = pq.read_schema(self.raw_path)
            # Fast row count from parquet metadata
            parquet_file = pq.ParquetFile(self.raw_path)
            num_rows = parquet_file.metadata.num_rows
            num_cols = len(schema.names)
            column_names = schema.names
        else:
            # For CSV, count headers
            with open(self.raw_path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.readline().strip().split(",")
                num_cols = len(header)
                column_names = header
            num_rows = sum(1 for _ in open(self.raw_path, "rb")) - 1

        return {
            "path": str(self.raw_path),
            "format": ext,
            "size_bytes": file_size_bytes,
            "size_mb": round(file_size_mb, 2),
            "num_rows": num_rows,
            "num_cols": num_cols,
            "columns": column_names,
        }

    def load_data(
        self,
        sample_size: Optional[int] = None,
        random_seed: int = 42,
        columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """Load flights data into a pandas DataFrame.
        
        If sample_size is provided, performs a stratified/uniform sample across dates
        to preserve full 365-day chronological coverage.
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Raw dataset not found at {self.raw_path}. "
                "Please place the Kaggle flights.parquet or flights.csv in data/raw/."
            )

        ext = self.raw_path.suffix.lower()
        logger.info(f"Loading data from {self.raw_path} (format: {ext})...")

        if ext == ".parquet":
            if sample_size is None:
                df = pd.read_parquet(self.raw_path, columns=columns)
            else:
                # To maintain seasonal and chronological integrity across all 365 days of the year,
                # we sample uniformly across the entire dataset.
                parquet_file = pq.ParquetFile(self.raw_path)
                total_rows = parquet_file.metadata.num_rows
                if total_rows <= sample_size:
                    df = pd.read_parquet(self.raw_path, columns=columns)
                else:
                    # Deterministic step sampling to guarantee uniform temporal representation
                    # without loading all columns into memory at once
                    step = max(1, total_rows // sample_size)
                    indices = np.arange(0, total_rows, step)[:sample_size]
                    
                    # Read table and take indices
                    table = pq.read_table(self.raw_path, columns=columns)
                    table_sampled = table.take(indices)
                    df = table_sampled.to_pandas()
                    logger.info(f"Sampled {len(df):,} rows from {total_rows:,} total rows preserving temporal distribution.")
        elif ext in [".csv", ".gz", ".zip"]:
            if sample_size is None:
                df = pd.read_csv(self.raw_path, usecols=columns, low_memory=False)
            else:
                # Reservoir or step sample
                df_chunks = []
                total_loaded = 0
                chunksize = 100000
                rng = np.random.RandomState(random_seed)
                # First estimate total rows
                total_rows = sum(1 for _ in open(self.raw_path, "rb")) - 1
                sample_fraction = min(1.0, sample_size / total_rows)
                for chunk in pd.read_csv(self.raw_path, usecols=columns, chunksize=chunksize, low_memory=False):
                    sampled_chunk = chunk.sample(frac=sample_fraction, random_state=rng)
                    df_chunks.append(sampled_chunk)
                    total_loaded += len(sampled_chunk)
                    if total_loaded >= sample_size * 1.05:
                        break
                df = pd.concat(df_chunks, ignore_index=True)
                if len(df) > sample_size:
                    df = df.iloc[:sample_size]
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        logger.info(f"Successfully loaded DataFrame with shape {df.shape}.")
        return df

    def load_airlines(self) -> Optional[pd.DataFrame]:
        """Load airlines lookup table if available."""
        if self.airlines_path and self.airlines_path.exists():
            if self.airlines_path.suffix == ".parquet":
                return pd.read_parquet(self.airlines_path)
            return pd.read_csv(self.airlines_path)
        return None

    def load_airports(self) -> Optional[pd.DataFrame]:
        """Load airports lookup table if available."""
        if self.airports_path and self.airports_path.exists():
            if self.airports_path.suffix == ".parquet":
                return pd.read_parquet(self.airports_path)
            return pd.read_csv(self.airports_path)
        return None
