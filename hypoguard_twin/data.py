"""CSV ingestion and quality checks. Raw data are never silently repaired."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from .config import MAX_GLUCOSE_MGDL, MIN_GLUCOSE_MGDL

CANONICAL_ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date_time", "date"),
    "glucose": ("glucose", "cgm", "sensor_glucose", "sgv", "blood_glucose"),
    "patient_id": ("patient_id", "patient", "subject_id", "subject", "participant_id", "id"),
    "heart_rate": ("heart_rate", "heartrate", "hr"),
    "steps": ("steps", "step_count", "stepcount"),
    "carbs": ("carbs", "carbohydrates", "carb_intake", "cho"),
    "basal_insulin": ("basal_insulin", "basal"),
    "bolus_insulin": ("bolus_insulin", "bolus"),
    "sleep": ("sleep", "sleep_minutes", "sleep_duration"),
}


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="replace")[:8192]
    candidates = [",", ";", "\t", "|"]
    return max(candidates, key=lambda value: sample.count(value))


def _canonicalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: normalize_name(column) for column in frame.columns}
    frame = frame.rename(columns=renamed)
    for canonical, aliases in CANONICAL_ALIASES.items():
        if canonical in frame.columns:
            continue
        found = next((name for name in aliases if name in frame.columns), None)
        if found:
            frame = frame.rename(columns={found: canonical})
    return frame


@dataclass
class IngestResult:
    observations: pd.DataFrame
    quality: dict


def inspect_csv(path: str | Path, patient_id: str | None = None) -> IngestResult:
    """Load one patient file; invalid glucose records are retained but flagged then excluded.

    Temporal policy: timestamps are parsed to UTC-naive timestamps and sorted. A valid
    target needs at least one valid future reading strictly after now and at most 30 min.
    """
    csv_path = Path(path)
    frame = pd.read_csv(csv_path, sep=detect_delimiter(csv_path))
    source_columns = list(frame.columns)
    frame = _canonicalise_columns(frame)
    required = {"timestamp", "glucose"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required column(s): {sorted(missing)}")
    if "patient_id" not in frame:
        frame["patient_id"] = patient_id or csv_path.stem
    frame["patient_id"] = frame["patient_id"].fillna(patient_id or csv_path.stem).astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    frame["glucose"] = pd.to_numeric(frame["glucose"], errors="coerce")
    frame["valid_timestamp"] = frame["timestamp"].notna()
    frame["valid_glucose"] = frame["glucose"].between(MIN_GLUCOSE_MGDL, MAX_GLUCOSE_MGDL, inclusive="both")
    if "is_synthetic" in frame.columns:
        frame["is_synthetic"] = frame["is_synthetic"].fillna(False).astype(bool)
    else:
        frame["is_synthetic"] = False
    frame = frame.sort_values(["patient_id", "timestamp"], na_position="last").reset_index(drop=True)
    duplicate_count = int(frame.duplicated(["patient_id", "timestamp"], keep=False).sum())
    valid = frame.loc[frame.valid_timestamp & frame.valid_glucose].copy()
    intervals = valid.groupby("patient_id")["timestamp"].diff().dt.total_seconds().div(60)
    quality = {
        "source_columns": source_columns,
        "records": len(frame),
        "invalid_timestamps": int((~frame.valid_timestamp).sum()),
        "invalid_glucose": int((~frame.valid_glucose).sum()),
        "missing_glucose": int(frame.glucose.isna().sum()),
        "duplicate_timestamps": duplicate_count,
        "irregular_intervals": int(((intervals.notna()) & ~intervals.between(4, 6)).sum()),
    }
    return IngestResult(frame, quality)


def load_directory(directory: str | Path, synthetic: bool = False) -> tuple[pd.DataFrame, dict]:
    paths = sorted(Path(directory).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {directory}")
    loaded = [inspect_csv(path, path.stem) for path in paths]
    observations = pd.concat([item.observations for item in loaded], ignore_index=True)
    observations["is_synthetic"] = synthetic
    return observations, {path.name: item.quality for path, item in zip(paths, loaded)}


def quality_by_patient(observations: pd.DataFrame) -> list[dict]:
    report = []
    for patient_id, group in observations.groupby("patient_id"):
        valid = group.loc[group.valid_timestamp & group.valid_glucose]
        report.append({
            "patient_id": patient_id,
            "records": len(group),
            "valid_glucose_records": len(valid),
            "invalid_records": int((~(group.valid_timestamp & group.valid_glucose)).sum()),
            "missing_glucose": int(group.glucose.isna().sum()),
            "start": valid.timestamp.min().isoformat() if not valid.empty else None,
            "end": valid.timestamp.max().isoformat() if not valid.empty else None,
        })
    return report
