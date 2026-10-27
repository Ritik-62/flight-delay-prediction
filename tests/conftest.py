"""Pytest fixtures providing synthetic and realistic flight datasets for testing."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def synthetic_flights_df() -> pd.DataFrame:
    """Generate deterministic synthetic flight DataFrame covering multiple calendar dates."""
    rng = np.random.RandomState(42)
    n = 200

    dates = pd.date_range("2015-01-01", periods=10, freq="D")
    assigned_dates = pd.DatetimeIndex(rng.choice(dates, size=n))

    airlines = ["AA", "DL", "UA", "WN", "B6"]
    airports = ["JFK", "LAX", "ORD", "ATL", "DFW", "SFO"]

    origins = rng.choice(airports, size=n)
    destinations = []
    for orig in origins:
        avail = [a for a in airports if a != orig]
        destinations.append(rng.choice(avail))

    sched_dep_hours = rng.randint(5, 23, size=n)
    sched_dep_mins = rng.choice([0, 15, 30, 45], size=n)
    sched_deps = [f"{h:02d}{m:02d}" for h, m in zip(sched_dep_hours, sched_dep_mins)]

    durations = rng.randint(60, 360, size=n)
    distances = (durations * 7.5).astype(int)

    sched_arrs = []
    for dep_h, dep_m, dur in zip(sched_dep_hours, sched_dep_mins, durations):
        total_mins = (dep_h * 60 + dep_m + dur) % (24 * 60)
        arr_h = total_mins // 60
        arr_m = total_mins % 60
        sched_arrs.append(f"{arr_h:02d}{arr_m:02d}")

    # Arrival delay: ~20% delayed by >= 15 min
    arrival_delays = rng.normal(loc=2, scale=25, size=n).astype(int)
    # Add a few cancellations (None)
    arrival_delays = [d if i > 5 else None for i, d in enumerate(arrival_delays)]

    df = pd.DataFrame({
        "YEAR": assigned_dates.year,
        "MONTH": assigned_dates.month,
        "DAY": assigned_dates.day,
        "DAY_OF_WEEK": assigned_dates.dayofweek + 1,
        "AIRLINE": rng.choice(airlines, size=n),
        "FLIGHT_NUMBER": rng.randint(100, 9000, size=n),
        "ORIGIN_AIRPORT": origins,
        "DESTINATION_AIRPORT": destinations,
        "SCHEDULED_DEPARTURE": sched_deps,
        "SCHEDULED_ARRIVAL": sched_arrs,
        "SCHEDULED_TIME": durations,
        "DISTANCE": distances,
        "ARRIVAL_DELAY": arrival_delays,
        "CANCELLED": [1 if d is None else 0 for d in arrival_delays],
        "DIVERTED": [0] * n,
    })
    return df
