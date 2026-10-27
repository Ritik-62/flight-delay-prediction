"""Preprocessing pipeline with scikit-learn ColumnTransformer, imputers, scalers, and encoders."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Manages feature transformations, column selection, scaling, and categorical encoding."""

    NUMERIC_FEATURES = [
        "DISTANCE",
        "SCHEDULED_TIME",
        "SCHED_DEP_HOUR",
        "SCHED_DEP_MINUTE",
        "DEP_HOUR_SIN",
        "DEP_HOUR_COS",
        "DOW_SIN",
        "DOW_COS",
        "MONTH_SIN",
        "MONTH_COS",
        "IS_WEEKEND",
        "IS_OVERNIGHT",
        "HIST_AIRLINE_DELAY_RATE",
        "HIST_AIRLINE_FLIGHT_COUNT",
        "HIST_ORIGIN_AIRPORT_DELAY_RATE",
        "HIST_ORIGIN_AIRPORT_FLIGHT_COUNT",
        "HIST_DESTINATION_AIRPORT_DELAY_RATE",
        "HIST_DESTINATION_AIRPORT_FLIGHT_COUNT",
        "HIST_ROUTE_DELAY_RATE",
        "HIST_ROUTE_FLIGHT_COUNT",
    ]

    CATEGORICAL_FEATURES = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
    ]

    def __init__(self, top_airports_k: int = 50):
        self.top_airports_k = top_airports_k
        self.top_origin_airports: List[str] = []
        self.top_dest_airports: List[str] = []
        self.fitted: bool = False
        self.column_transformer: Optional[ColumnTransformer] = None
        self.transformed_feature_names: List[str] = []

    def _prepare_categories(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Handle high-cardinality airports and ensure schema completeness."""
        df_cat = df.copy()

        # Ensure all expected numeric columns exist with default 0.0
        for num_col in self.NUMERIC_FEATURES:
            if num_col not in df_cat.columns:
                df_cat[num_col] = 0.0

        # Ensure all expected categorical columns exist with default UNKNOWN
        for cat_col in self.CATEGORICAL_FEATURES:
            if cat_col not in df_cat.columns:
                df_cat[cat_col] = "UNKNOWN"

        if is_train:
            self.top_origin_airports = df_cat["ORIGIN_AIRPORT"].value_counts().head(self.top_airports_k).index.tolist()
            self.top_dest_airports = df_cat["DESTINATION_AIRPORT"].value_counts().head(self.top_airports_k).index.tolist()

        # Mask non-top airports with 'OTHER'
        df_cat["ORIGIN_AIRPORT"] = df_cat["ORIGIN_AIRPORT"].where(
            df_cat["ORIGIN_AIRPORT"].isin(self.top_origin_airports), "OTHER"
        )
        df_cat["DESTINATION_AIRPORT"] = df_cat["DESTINATION_AIRPORT"].where(
            df_cat["DESTINATION_AIRPORT"].isin(self.top_dest_airports), "OTHER"
        )
        return df_cat

    def build_transformer(self) -> ColumnTransformer:
        """Construct scikit-learn ColumnTransformer."""
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.NUMERIC_FEATURES),
                ("cat", cat_pipeline, self.CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )
        return transformer

    def fit(self, train_df: pd.DataFrame) -> "PreprocessingPipeline":
        """Fit preprocessor strictly on training data."""
        logger.info("Fitting PreprocessingPipeline on training data...")
        df_prep = self._prepare_categories(train_df, is_train=True)
        self.column_transformer = self.build_transformer()
        self.column_transformer.fit(df_prep)

        # Extract generated feature names
        try:
            self.transformed_feature_names = self.column_transformer.get_feature_names_out().tolist()
        except Exception:
            self.transformed_feature_names = [f"feature_{i}" for i in range(100)]

        self.fitted = True
        logger.info(f"Preprocessing fitted. Output dimensionality: {len(self.transformed_feature_names)} features.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform input DataFrame into numeric feature array."""
        if not self.fitted or self.column_transformer is None:
            raise RuntimeError("PreprocessingPipeline must be fitted before transforming data.")
        df_prep = self._prepare_categories(df, is_train=False)
        return self.column_transformer.transform(df_prep)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fit and transform training DataFrame."""
        return self.fit(train_df).transform(train_df)

    def save(self, filepath: Path) -> None:
        """Serialize fitted pipeline to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved PreprocessingPipeline to {filepath}.")

    @classmethod
    def load(cls, filepath: Path) -> "PreprocessingPipeline":
        """Load serialized pipeline from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Serialized pipeline not found at {filepath}")
        return joblib.load(filepath)
