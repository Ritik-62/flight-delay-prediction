"""Offline What-If Rescheduling Recommendation Engine and Policy Simulator."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReschedulingPolicyEngine:
    """Simulates an offline tactical rescheduling recommendation rule based on calibrated delay risk.
    
    Operational Framework:
    - At decision timestamp T - 60 minutes:
      * Flight i has calibrated delay probability p_i = P(ARR_DEL15=1 | x_i).
      * If p_i >= threshold_risk, flight is flagged for operational intervention.
    - Candidate Interventions:
      1. Action 1: Tail Swap (reassign to on-time aircraft if hub share is high and inbound is delayed).
      2. Action 2: Departure Slot Shift (+30 min buffer to absorb ground congestion / turnaround delay).
      3. Action 0: Maintain schedule (no intervention).
    """

    def __init__(
        self,
        threshold_risk: float = 0.45,
        delay_buffer_minutes: int = 30,
        high_risk_cutoff: float = 0.65,
        slot_penalty_weight: float = 0.25,
        avg_delayed_flight_minutes: float = 45.0,
    ):
        self.threshold_risk = threshold_risk
        self.delay_buffer_minutes = delay_buffer_minutes
        self.high_risk_cutoff = high_risk_cutoff
        self.slot_penalty_weight = slot_penalty_weight
        self.avg_delayed_flight_minutes = avg_delayed_flight_minutes

    def recommend_actions(
        self,
        df_meta: pd.DataFrame,
        calibrated_probs: np.ndarray,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Apply operational decision policy to flight candidate pool."""
        thresh = threshold if threshold is not None else self.threshold_risk
        df_rec = df_meta.copy()
        df_rec["CALIBRATED_DELAY_RISK"] = np.round(calibrated_probs, 4)
        df_rec["RECOMMENDED_ACTION"] = "MAINTAIN_SCHEDULE"
        df_rec["POST_ACTION_ESTIMATED_RISK"] = df_rec["CALIBRATED_DELAY_RISK"]
        df_rec["EXPECTED_DELAY_MINUTES_SAVED"] = 0.0

        # Identification of high-risk flights
        flagged_mask = df_rec["CALIBRATED_DELAY_RISK"] >= thresh

        # Turnaround / inbound delay indicator
        has_inbound_delay = (
            (df_rec.get("INBOUND_DELAY_KNOWN", 0) > 15) |
            (df_rec.get("INBOUND_UNARRIVED_AT_T60", 0) == 1) |
            (df_rec.get("INBOUND_SCHEDULED_BUFFER_MIN", 180) < 45)
        )
        # Hub station check (carrier operates > 25% of flights at origin)
        is_hub_station = df_rec.get("CARRIER_ORIGIN_SHARE", 0) >= 0.25

        # Decision Rule: Tail Swap if inbound turnaround is compromised at a major carrier hub
        tail_swap_mask = flagged_mask & has_inbound_delay & is_hub_station
        # Decision Rule: Slot Shift (+30m) for congested flights or outstation turnarounds
        slot_shift_mask = flagged_mask & (~tail_swap_mask)

        # Action 1: Tail Swap
        # Counterfactual effect: Eliminates propagated tail delay; reduces risk to route baseline
        df_rec.loc[tail_swap_mask, "RECOMMENDED_ACTION"] = "TAIL_SWAP"
        # Estimated post-action risk reduces by 40% (relative)
        df_rec.loc[tail_swap_mask, "POST_ACTION_ESTIMATED_RISK"] = np.round(
            df_rec.loc[tail_swap_mask, "CALIBRATED_DELAY_RISK"] * 0.55, 4
        )

        # Action 2: Slot Shift (+30m buffer)
        # Counterfactual effect: Absorbs turnaround gap and shifts out of peak departure bank
        df_rec.loc[slot_shift_mask, "RECOMMENDED_ACTION"] = f"SLOT_SHIFT_+{self.delay_buffer_minutes}M"
        df_rec.loc[slot_shift_mask, "POST_ACTION_ESTIMATED_RISK"] = np.round(
            df_rec.loc[slot_shift_mask, "CALIBRATED_DELAY_RISK"] * 0.65, 4
        )

        # Expected delay minutes saved = (P_initial - P_post) * Avg Delay Minutes
        risk_reduction = df_rec["CALIBRATED_DELAY_RISK"] - df_rec["POST_ACTION_ESTIMATED_RISK"]
        df_rec["EXPECTED_DELAY_MINUTES_SAVED"] = np.round(risk_reduction * self.avg_delayed_flight_minutes, 2)

        return df_rec

    def simulate_policy_sweep(
        self,
        df_meta: pd.DataFrame,
        calibrated_probs: np.ndarray,
        y_true: np.ndarray,
        threshold_range: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Simulate policy outcomes across varying operational risk tolerance thresholds."""
        if threshold_range is None:
            p_min = float(np.percentile(calibrated_probs, 40))
            p_max = float(np.percentile(calibrated_probs, 98))
            threshold_range = np.linspace(p_min, p_max, 8)

        records = []
        n_total = len(df_meta)
        total_actual_delays = int(np.sum(y_true == 1))

        for tau in threshold_range:
            rec_df = self.recommend_actions(df_meta, calibrated_probs, threshold=tau)
            intervened = rec_df["RECOMMENDED_ACTION"] != "MAINTAIN_SCHEDULE"
            n_interventions = int(intervened.sum())

            # Actual delays captured (True Positives among intervened)
            captured_delays = int(np.sum((intervened) & (y_true == 1)))
            false_alarms = int(np.sum((intervened) & (y_true == 0)))

            # Estimated averted delay flights: captured delays * average risk reduction (35%)
            est_averted_delayed_flights = int(captured_delays * 0.40)
            total_net_minutes_saved = float(rec_df["EXPECTED_DELAY_MINUTES_SAVED"].sum())

            # Rescheduling overhead penalty
            buffer_cost_minutes = n_interventions * (self.delay_buffer_minutes * self.slot_penalty_weight)
            net_benefit_minutes = total_net_minutes_saved - buffer_cost_minutes

            records.append({
                "Risk Threshold (tau)": round(float(tau), 3),
                "Total Flights": n_total,
                "Interventions": n_interventions,
                "Intervention Rate (%)": round(float(n_interventions / n_total * 100), 2),
                "Captured Delays (TP)": captured_delays,
                "Delay Coverage Rate (%)": round(float(captured_delays / max(1, total_actual_delays) * 100), 2),
                "False Alarm Interventions": false_alarms,
                "Estimated Averted Delayed Flights": est_averted_delayed_flights,
                "Gross Delay Minutes Saved": round(total_net_minutes_saved, 1),
                "Net Benefit (Min Saved - Friction)": round(net_benefit_minutes, 1),
            })

        return pd.DataFrame(records)

    def plot_rescheduling_tradeoff(
        self,
        sweep_df: pd.DataFrame,
        save_path: Path,
    ) -> None:
        """Plot the operational Pareto Frontier of Delay Reduction vs Interventions."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

        # Plot 1: Delay Coverage vs Interventions
        ax1.plot(
            sweep_df["Intervention Rate (%)"],
            sweep_df["Delay Coverage Rate (%)"],
            marker="o",
            color="#1f77b4",
            linewidth=2.5,
            label="Empirical Policy Curve",
        )
        ax1.plot([0, 100], [0, 100], "k--", alpha=0.5, label="Random Policy Baseline")
        ax1.set_xlabel("Operational Intervention Rate (%)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Arrival Delay Coverage (%)", fontsize=11, fontweight="bold")
        ax1.set_title("Rescheduling Operating Characteristic (ROC-Like)", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="lower right")

        # Plot 2: Net Benefit Minutes vs Threshold
        ax2.plot(
            sweep_df["Risk Threshold (tau)"],
            sweep_df["Net Benefit (Min Saved - Friction)"] / 60.0,
            marker="s",
            color="#2ca02c",
            linewidth=2.5,
            label="Net Delay Hours Saved",
        )
        # Highlight optimal peak
        best_row = sweep_df.loc[sweep_df["Net Benefit (Min Saved - Friction)"].idxmax()]
        ax2.axvline(best_row["Risk Threshold (tau)"], color="#d62728", linestyle=":", label=f"Optimal tau = {best_row['Risk Threshold (tau)']:.2f}")

        ax2.set_xlabel("Delay Risk Decision Threshold (tau)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Net Passenger Delay Hours Saved", fontsize=11, fontweight="bold")
        ax2.set_title("Tactical Rescheduling Net Operational Utility", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved rescheduling trade-off curve to {save_path}.")
