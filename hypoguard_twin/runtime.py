"""Application composition and deterministic demo bootstrap."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from .config import METADATA_PATH, MODEL_PATH, RAW_DIR, SYNTHETIC_DIR
from .data import load_directory, quality_by_patient
from .features import build_features_and_labels
from .modeling import load_model, train_model
from .storage import Storage
from .synthetic import generate_synthetic_data
from .twin import DigitalTwinEngine


# These are recorded sensor/context measurements. Altering model-derived glucose
# features directly would yield an internally inconsistent hypothetical feature vector.
SCENARIO_FEATURE_BOUNDS = {
    "steps_30m": (0.0, 100_000.0),
    "carbs_30m": (0.0, 1_000.0),
    "heart_rate_current": (20.0, 260.0),
}


class Runtime:
    def __init__(self, mode: str = "auto"):
        raw_files = list(RAW_DIR.glob("*.csv"))
        if mode == "real" and not raw_files:
            raise FileNotFoundError("No real CSV dataset files are present in data/raw")
        self.mode = "REAL DATASET" if raw_files and mode != "synthetic" else "SYNTHETIC DEMONSTRATION"
        if self.mode == "SYNTHETIC DEMONSTRATION":
            if not list(SYNTHETIC_DIR.glob("*.csv")):
                generate_synthetic_data(SYNTHETIC_DIR)
            self.observations, self.quality = load_directory(SYNTHETIC_DIR, synthetic=True)
        else:
            self.observations, self.quality = load_directory(RAW_DIR, synthetic=False)
        self.features = build_features_and_labels(self.observations)
        self.storage = Storage()
        self.model_bootstrap_error: str | None = None
        model = load_model()
        existing_metadata = self.metadata()
        # Never silently apply an artifact trained on synthetic data to a real dataset (or reverse).
        if existing_metadata.get("training_data_mode") != self.mode:
            model = None
        if model is None and not self.features.training_rows.empty:
            try:
                metadata = train_model(self.features.training_rows)
                self.storage.save_model_metadata(metadata)
                model = load_model()
            except ValueError as exc:
                # Serving remains available for data inspection even when training is indefensible.
                self.model_bootstrap_error = str(exc)
        self.model = model
        self.engine = DigitalTwinEngine(self.observations, self.model)
        for patient_id in self.engine.patients():
            self.storage.upsert_patient(patient_id, self.mode)

    def metadata(self) -> dict:
        return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}

    def patient_history(self, patient_id: str) -> list[dict]:
        frame = self.observations.loc[self.observations.patient_id == patient_id].sort_values("timestamp")
        if frame.empty:
            raise KeyError(patient_id)
        allowed = [column for column in ("timestamp", "glucose", "heart_rate", "steps", "carbs", "is_synthetic", "valid_glucose") if column in frame]
        records = frame[allowed].copy()
        records["timestamp"] = records.timestamp.astype(str)
        return records.where(pd.notna(records), None).to_dict("records")

    def predict(self, patient_id: str) -> dict:
        if patient_id not in self.engine.patients():
            raise KeyError(patient_id)
        state = self.engine.derive_state(patient_id)
        output = dict(state["latest_model_output"])
        output["patient_id"] = patient_id
        output["prediction_timestamp"] = state["latest_observation_timestamp"]
        output["input_data_quality"] = state["data_quality"]
        if output.get("available"):
            self.storage.save_prediction(patient_id, output)
        self.storage.save_state(patient_id, state)
        return output

    def historical_replay(self, patient_id: str, position: int) -> dict:
        """Derive an isolated historical state from valid observations up to position."""
        if patient_id not in self.engine.patients():
            raise KeyError(patient_id)
        source = self.observations.loc[
            (self.observations.patient_id == patient_id)
            & self.observations.valid_timestamp
            & self.observations.valid_glucose
        ].sort_values("timestamp").reset_index(drop=True)
        if position < 0 or position >= len(source):
            raise IndexError("Replay position is outside available valid historical observations")
        replay_engine = DigitalTwinEngine(source.iloc[:position + 1].copy(), self.model)
        return {
            "historical_replay": True,
            "position": position,
            "total": len(source),
            "state": replay_engine.derive_state(patient_id),
        }

    def scenario(self, patient_id: str, overrides: dict) -> dict:
        if patient_id not in self.engine.patients():
            raise KeyError(patient_id)
        if not overrides:
            return {"available": False, "message": "Provide at least one hypothetical input for comparison."}
        from .features import latest_feature_row
        from .modeling import predict_features
        feature = latest_feature_row(self.observations, patient_id).copy()
        supported = set(self.model["feature_columns"]) if self.model else set()
        unknown = set(overrides) - supported
        if unknown:
            return {"available": False, "message": f"Unsupported scenario feature(s): {sorted(unknown)}"}
        unsupported = set(overrides) - set(SCENARIO_FEATURE_BOUNDS)
        if unsupported:
            return {"available": False, "message": "Scenario comparison supports only recorded activity, carbohydrate, or heart-rate inputs; model-derived glucose features cannot be changed directly."}
        for name, value in overrides.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return {"available": False, "message": f"Scenario value for {name} must be numeric."}
            lower, upper = SCENARIO_FEATURE_BOUNDS[name]
            if not lower <= numeric_value <= upper:
                return {"available": False, "message": f"Scenario value for {name} must be between {lower:g} and {upper:g}."}
            feature[name] = numeric_value
        comparison = predict_features(self.model, feature)
        # Do not write a prediction or mutate twin state merely because a user explores a scenario.
        patient_observations = self.observations.loc[self.observations.patient_id == patient_id]
        valid_glucose = patient_observations.loc[patient_observations.valid_glucose].sort_values("timestamp")
        baseline = predict_features(self.model, latest_feature_row(self.observations, patient_id))
        baseline.update({
            "patient_id": patient_id,
            "prediction_timestamp": valid_glucose.timestamp.iloc[-1].isoformat() if not valid_glucose.empty else None,
            "input_data_quality": {
                "valid_glucose_observations": int(patient_observations.valid_glucose.sum()),
                "invalid_observations": int((~patient_observations.valid_glucose).sum()),
            },
        })
        return {"available": comparison.get("available", False), "hypothetical_inputs": overrides,
                "baseline": baseline, "hypothetical_output": comparison,
                "note": "Hypothetical model comparison only; it is not a causal or treatment recommendation."}

    def quality_summary(self) -> list[dict]:
        return quality_by_patient(self.observations)
