"""Patient state engine: state is derived from recorded observations, never invented."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .features import latest_feature_row
from .modeling import predict_features


class DigitalTwinEngine:
    def __init__(self, observations: pd.DataFrame, model: dict | None = None):
        self.observations = observations.copy()
        self.model = model
        self.states: dict[str, dict[str, Any]] = {}

    def patients(self) -> list[str]:
        return sorted(self.observations.patient_id.astype(str).unique().tolist())

    def _validate_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        required = {"patient_id", "timestamp", "glucose"}
        absent = required - set(observation)
        if absent:
            raise ValueError(f"Missing observation fields: {sorted(absent)}")
        timestamp = pd.to_datetime(observation["timestamp"], errors="coerce", utc=True)
        glucose = pd.to_numeric(observation["glucose"], errors="coerce")
        if pd.isna(timestamp) or pd.isna(glucose) or not 20 <= glucose <= 600:
            raise ValueError("Observation requires a valid timestamp and glucose from 20 to 600 mg/dL")
        result = dict(observation)
        result.update({"patient_id": str(observation["patient_id"]), "timestamp": timestamp.tz_localize(None),
                       "glucose": float(glucose), "valid_timestamp": True, "valid_glucose": True,
                       "is_synthetic": bool(observation.get("is_synthetic", False))})
        return result

    def update(self, observation: dict[str, Any]) -> dict[str, Any]:
        clean = self._validate_observation(observation)
        self.observations = pd.concat([self.observations, pd.DataFrame([clean])], ignore_index=True, sort=False)
        self.observations = self.observations.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
        return self.derive_state(clean["patient_id"])

    def derive_state(self, patient_id: str) -> dict[str, Any]:
        patient = self.observations.loc[(self.observations.patient_id == patient_id) & self.observations.valid_timestamp].copy()
        if patient.empty:
            raise ValueError("Patient has no valid timestamped observations for a twin state")
        feature = latest_feature_row(patient, patient_id)
        latest = patient.loc[patient.valid_glucose].sort_values("timestamp").iloc[-1] if patient.valid_glucose.any() else None
        output = predict_features(self.model, feature) if feature is not None else {"available": False, "message": "Insufficient history"}
        state = {
            "patient_id": patient_id, "latest_valid_glucose": float(latest.glucose) if latest is not None else None,
            "latest_observation_timestamp": latest.timestamp.isoformat() if latest is not None else None,
            "recent_glucose_history": [{"timestamp": row.timestamp.isoformat(), "glucose": float(row.glucose)} for _, row in patient.tail(12).iterrows() if row.valid_glucose],
            "recent_glucose_trend_mgdl_per_min": float(feature.glucose_rate_mgdl_per_min) if feature is not None and pd.notna(feature.glucose_rate_mgdl_per_min) else None,
            "data_quality": {"valid_glucose_observations": int(patient.valid_glucose.sum()), "invalid_observations": int((~patient.valid_glucose).sum())},
            "historical_low_glucose_event_count": int((patient.loc[patient.valid_glucose, "glucose"] < 70).sum()),
            "available_features": [key for key in (feature.index if feature is not None else []) if pd.notna(feature[key])],
            "latest_model_output": output, "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.states[patient_id] = state
        return state

    def replay_next(self, patient_id: str, position: int) -> tuple[dict[str, Any], int]:
        source = self.observations.loc[
            (self.observations.patient_id == patient_id)
            & self.observations.valid_timestamp
            & self.observations.valid_glucose
        ].sort_values("timestamp").reset_index(drop=True)
        if position >= len(source):
            raise IndexError("Replay is complete")
        if position < 0:
            raise IndexError("Replay position cannot be negative")
        # The temporary engine only sees the selected prefix, preserving the live twin state.
        replay_engine = DigitalTwinEngine(source.iloc[:position + 1].copy(), self.model)
        state = replay_engine.derive_state(patient_id)
        return state, position + 1
