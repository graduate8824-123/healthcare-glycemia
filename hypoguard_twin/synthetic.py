"""Deterministic, explicitly labelled demonstration data; never clinical evidence."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_data(output_dir: str | Path, seed: int = 2026, patients: int = 6,
                            points: int = 210) -> list[Path]:
    rng = np.random.default_rng(seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output = []
    start = pd.Timestamp("2026-01-05 06:00:00")
    for index in range(patients):
        patient_id = f"SYN-{index + 1:03d}"
        timestamps = pd.date_range(start + pd.Timedelta(days=index), periods=points, freq="5min")
        drift = np.cumsum(rng.normal(0, 1.5, points))
        rhythm = 18 * np.sin(np.linspace(0, 4 * np.pi, points) + index / 2)
        glucose = 122 + rhythm + drift + rng.normal(0, 3.5, points)
        # Distinct downward excursions create synthetic event examples.
        for event_start in (55 + index % 4, 150 - index % 5):
            # Truncate the pulse gracefully for small test/demo series.
            down_end = min(event_start + 10, points)
            if event_start < points:
                glucose[event_start:down_end] -= np.linspace(0, 65, down_end - event_start)
            up_start = event_start + 10
            up_end = min(event_start + 21, points)
            if up_start < points:
                glucose[up_start:up_end] -= np.linspace(65, 0, up_end - up_start)
        steps = np.maximum(0, rng.poisson(7, points) + (rhythm > 0) * rng.integers(0, 12, points))
        frame = pd.DataFrame({
            "patient_id": patient_id,
            "timestamp": timestamps,
            "glucose": np.round(np.clip(glucose, 42, 280), 1),
            "heart_rate": np.round(70 + steps * .8 + rng.normal(0, 4, points), 1),
            "steps": steps,
            "carbs": np.where(np.arange(points) % 60 == 0, rng.integers(10, 45, points), 0),
            "is_synthetic": True,
        })
        # Intentional sparse missing wearable fields exercise missing-value handling.
        frame.loc[frame.index % 37 == 0, "heart_rate"] = np.nan
        path = destination / f"{patient_id}.csv"
        frame.to_csv(path, index=False)
        output.append(path)
    return output
