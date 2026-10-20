"""Leakage-safe temporal feature and target construction."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import HORIZON_MINUTES

FEATURE_COLUMNS = [
    "glucose_current", "glucose_change_5m", "glucose_change_10m", "glucose_change_15m",
    "glucose_mean_30m", "glucose_min_30m", "glucose_max_30m", "glucose_std_30m",
    "glucose_rate_mgdl_per_min", "minutes_since_previous", "heart_rate_current",
    "heart_rate_change_5m", "steps_30m", "carbs_30m", "missing_heart_rate",
    "irregular_previous_interval",
]


@dataclass
class FeatureResult:
    all_rows: pd.DataFrame
    training_rows: pd.DataFrame
    feature_columns: list[str]


def _past_value(history: pd.DataFrame, now: pd.Timestamp, column: str, minutes: int,
                tolerance_minutes: int = 3) -> float:
    eligible = history.loc[history.timestamp <= now - pd.Timedelta(minutes=minutes)]
    if eligible.empty or column not in eligible:
        return np.nan
    candidate = eligible.iloc[-1]
    age = (now - candidate.timestamp).total_seconds() / 60
    return candidate[column] if age <= minutes + tolerance_minutes else np.nan


def build_features_and_labels(observations: pd.DataFrame, horizon_minutes: int = HORIZON_MINUTES) -> FeatureResult:
    """Build only current/past features; labels use a separate future time window.

    Invalid glucose observations cannot become model samples. Missing wearables remain NaN
    and are imputed only inside the model pipeline fitted on the train split.
    """
    if observations.empty:
        return FeatureResult(pd.DataFrame(), pd.DataFrame(), FEATURE_COLUMNS)
    rows = []
    valid = observations.loc[observations.valid_timestamp & observations.valid_glucose].copy()
    for patient_id, group in valid.groupby("patient_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for idx, current in group.iterrows():
            history = group.iloc[:idx + 1]
            previous = group.iloc[:idx]
            now = current.timestamp
            previous_time = previous.timestamp.iloc[-1] if not previous.empty else pd.NaT
            elapsed = ((now - previous_time).total_seconds() / 60) if pd.notna(previous_time) else np.nan
            glucose_now = current.glucose
            g5 = _past_value(previous, now, "glucose", 5)
            g10 = _past_value(previous, now, "glucose", 10)
            g15 = _past_value(previous, now, "glucose", 15)
            trailing = history.loc[history.timestamp >= now - pd.Timedelta(minutes=30), "glucose"]
            prior_hr = _past_value(previous, now, "heart_rate", 5) if "heart_rate" in group else np.nan
            future = group.loc[(group.timestamp > now) &
                               (group.timestamp <= now + pd.Timedelta(minutes=horizon_minutes)), "glucose"]
            # A target is valid only where there is a genuine recorded measurement in-window.
            target = int((future < 70).any()) if not future.empty else np.nan
            rows.append({
                "patient_id": str(patient_id), "timestamp": now, "is_synthetic": bool(current.is_synthetic),
                "glucose_current": glucose_now,
                "glucose_change_5m": glucose_now - g5 if pd.notna(g5) else np.nan,
                "glucose_change_10m": glucose_now - g10 if pd.notna(g10) else np.nan,
                "glucose_change_15m": glucose_now - g15 if pd.notna(g15) else np.nan,
                "glucose_mean_30m": trailing.mean(), "glucose_min_30m": trailing.min(),
                "glucose_max_30m": trailing.max(), "glucose_std_30m": trailing.std(ddof=0),
                "glucose_rate_mgdl_per_min": ((glucose_now - previous.glucose.iloc[-1]) / elapsed
                                                if pd.notna(elapsed) and elapsed > 0 else np.nan),
                "minutes_since_previous": elapsed,
                "heart_rate_current": current.get("heart_rate", np.nan),
                "heart_rate_change_5m": (current.get("heart_rate", np.nan) - prior_hr
                                           if pd.notna(prior_hr) else np.nan),
                "steps_30m": history.loc[history.timestamp >= now - pd.Timedelta(minutes=30),
                                           "steps"].sum(min_count=1) if "steps" in group else np.nan,
                "carbs_30m": history.loc[history.timestamp >= now - pd.Timedelta(minutes=30),
                                           "carbs"].sum(min_count=1) if "carbs" in group else np.nan,
                "missing_heart_rate": int(pd.isna(current.get("heart_rate", np.nan))),
                "irregular_previous_interval": int(pd.notna(elapsed) and not 4 <= elapsed <= 6),
                "target_low_30m": target,
            })
    all_rows = pd.DataFrame(rows)
    training_rows = all_rows.loc[all_rows.target_low_30m.notna()].copy()
    if not training_rows.empty:
        training_rows["target_low_30m"] = training_rows.target_low_30m.astype(int)
    return FeatureResult(all_rows, training_rows, FEATURE_COLUMNS)


def latest_feature_row(observations: pd.DataFrame, patient_id: str) -> pd.Series | None:
    result = build_features_and_labels(observations.loc[observations.patient_id == patient_id])
    if result.all_rows.empty:
        return None
    return result.all_rows.sort_values("timestamp").iloc[-1]
