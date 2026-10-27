"""Historical smoothed statistics calculation with strict chronological isolation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalFeatureExtractor:
    """Computes leakage-safe expanding historical delay statistics using empirical Bayes smoothing.
    
    Guarantees:
    1. A flight at date D NEVER incorporates observations from date D or later.
    2. Test labels NEVER enter historical feature lookup tables.
    3. Low-volume categories smoothly shrink toward the global empirical prior.
    """

    def __init__(
        self,
        entities: Optional[List[str]] = None,
        smoothing_weight: float = 15.0,
        min_samples: int = 5,
        date_column: str = "FLIGHT_DATE",
        target_col: str = "ARR_DEL15",
    ):
        self.entities = entities or ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE"]
        self.smoothing_weight = smoothing_weight
        self.min_samples = min_samples
        self.date_column = date_column
        self.target_col = target_col

        # State saved during fit
        self.global_prior: float = 0.18
        self.entity_lookups: Dict[str, pd.DataFrame] = {}
        self.fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "HistoricalFeatureExtractor":
        """Compute historical statistics exclusively on the training partition."""
        logger.info("Fitting HistoricalFeatureExtractor on training partition...")
        if self.target_col not in train_df.columns:
            raise ValueError(f"Target column '{self.target_col}' missing from training dataframe.")

        self.global_prior = float(train_df[self.target_col].mean())
        logger.info(f"Global training delay prior: {self.global_prior:.4f}")

        # Compute cumulative entity statistics across training days
        for entity in self.entities:
            if entity not in train_df.columns:
                logger.warning(f"Entity '{entity}' not present in training dataframe. Skipping.")
                continue

            agg = train_df.groupby(entity)[self.target_col].agg(["count", "sum"]).reset_index()
            agg.columns = [entity, "hist_count", "hist_delays"]

            # Empirical Bayes smoothing:
            # smoothed_rate = (hist_delays + alpha * global_prior) / (hist_count + alpha)
            agg["hist_rate"] = (
                agg["hist_delays"] + self.smoothing_weight * self.global_prior
            ) / (agg["hist_count"] + self.smoothing_weight)

            self.entity_lookups[entity] = agg
            logger.info(f"Constructed historical lookup for '{entity}' with {len(agg):,} unique categories.")

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Attach historical features to DataFrame.
        
        If is_training is True, computes chronological day-by-day expanding history
        strictly using prior days (day < current_day) so earlier training rows do NOT
        see later training rows!
        
        If is_training is False (Validation or Test), uses the fitted training lookups.
        """
        if not self.fitted and not is_training:
            raise RuntimeError("HistoricalFeatureExtractor must be fitted on training data before transforming.")

        df_out = df.copy()

        if is_training:
            # Chronological expanding window across training dates
            # For each date D, compute cumulative stats up to D - 1
            unique_dates = np.sort(df_out[self.date_column].dt.date.unique())
            
            # Daily aggregates per entity
            for entity in self.entities:
                if entity not in df_out.columns:
                    continue

                col_rate = f"HIST_{entity}_DELAY_RATE"
                col_count = f"HIST_{entity}_FLIGHT_COUNT"

                # Precompute daily counts and delays
                daily_agg = df_out.groupby([df_out[self.date_column].dt.date, entity])[self.target_col].agg(["count", "sum"]).reset_index()
                daily_agg.columns = ["date", entity, "day_count", "day_delays"]

                # Pivot to compute cumulative sum strictly up to previous date
                # We shift by 1 date so date D only gets sum of dates < D
                cum_records = []
                for ent, ent_group in daily_agg.groupby(entity):
                    ent_group = ent_group.sort_values("date").reset_index(drop=True)
                    ent_group["cum_count"] = ent_group["day_count"].cumsum().shift(1).fillna(0)
                    ent_group["cum_delays"] = ent_group["day_delays"].cumsum().shift(1).fillna(0)
                    ent_group["hist_rate"] = (
                        ent_group["cum_delays"] + self.smoothing_weight * self.global_prior
                    ) / (ent_group["cum_count"] + self.smoothing_weight)
                    cum_records.append(ent_group[["date", entity, "cum_count", "hist_rate"]])

                if cum_records:
                    cum_df = pd.concat(cum_records, ignore_index=True)
                    # Merge back onto df_out by date and entity
                    df_out["temp_date"] = df_out[self.date_column].dt.date
                    merged = pd.merge(
                        df_out[[entity, "temp_date"]].reset_index(),
                        cum_df,
                        left_on=[entity, "temp_date"],
                        right_on=[entity, "date"],
                        how="left",
                    ).sort_values("index").set_index("index")

                    df_out[col_rate] = merged["hist_rate"].fillna(self.global_prior)
                    df_out[col_count] = merged["cum_count"].fillna(0)
                    df_out.drop(columns=["temp_date"], inplace=True)
                else:
                    df_out[col_rate] = self.global_prior
                    df_out[col_count] = 0.0

        else:
            # Transform validation or test set using frozen training lookups
            for entity in self.entities:
                if entity not in df_out.columns or entity not in self.entity_lookups:
                    continue

                col_rate = f"HIST_{entity}_DELAY_RATE"
                col_count = f"HIST_{entity}_FLIGHT_COUNT"

                lookup = self.entity_lookups[entity]
                merged = pd.merge(df_out[[entity]], lookup, on=entity, how="left")

                # Fallback to smoothed global prior for unseen categories
                df_out[col_rate] = merged["hist_rate"].fillna(self.global_prior).values
                df_out[col_count] = merged["hist_count"].fillna(0).values

        return df_out
