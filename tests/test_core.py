from pathlib import Path

import joblib
import pandas as pd
import pytest

from hypoguard_twin.data import inspect_csv
from hypoguard_twin.features import FEATURE_COLUMNS, build_features_and_labels, latest_feature_row
from hypoguard_twin.modeling import load_model, patient_split, predict_features, train_model
from hypoguard_twin.runtime import Runtime
from hypoguard_twin.synthetic import generate_synthetic_data
from hypoguard_twin.twin import DigitalTwinEngine


def observations() -> pd.DataFrame:
    times = pd.date_range("2026-01-01 08:00", periods=12, freq="5min")
    return pd.DataFrame({"patient_id": ["P1"] * 12, "timestamp": times,
                         "glucose": [130, 124, 118, 110, 101, 88, 75, 65, 66, 80, 95, 110],
                         "heart_rate": [70] * 12, "steps": [0, 2] * 6, "carbs": [0] * 12,
                         "valid_timestamp": True, "valid_glucose": True, "is_synthetic": False})


def test_timestamp_parsing_sorting_and_invalid_glucose(tmp_path: Path):
    path = tmp_path / "a.csv"
    path.write_text("time;blood_glucose;subject\n2026-01-01 08:05;120;A\n2026-01-01 08:00;0;A\nbad;90;A\n")
    result = inspect_csv(path)
    assert list(result.observations.timestamp.dropna()) == sorted(result.observations.timestamp.dropna())
    assert result.quality["invalid_glucose"] == 1
    assert result.quality["invalid_timestamps"] == 1


def test_missing_glucose_remains_missing(tmp_path: Path):
    path = tmp_path / "a.csv"
    path.write_text("timestamp,glucose\n2026-01-01 08:00,\n")
    result = inspect_csv(path)
    assert pd.isna(result.observations.glucose.iloc[0])
    assert result.quality["missing_glucose"] == 1


def test_labels_use_future_window_and_boundary():
    result = build_features_and_labels(observations(), horizon_minutes=30)
    row = result.training_rows.loc[result.training_rows.timestamp == pd.Timestamp("2026-01-01 08:25")].iloc[0]
    assert row.target_low_30m == 1  # 65 appears 10 minutes later.
    final = result.all_rows.iloc[-1]
    assert pd.isna(final.target_low_30m)  # no valid future observation.


def test_feature_does_not_see_future_glucose():
    data = observations()
    original = latest_feature_row(data.iloc[:7], "P1")
    changed = data.copy()
    changed.loc[7:, "glucose"] = 599
    later = latest_feature_row(changed.iloc[:7], "P1")
    assert original.glucose_current == later.glucose_current
    assert original.glucose_mean_30m == later.glucose_mean_30m


def test_insufficient_history_has_nan_lags_but_safe_row():
    row = build_features_and_labels(observations().iloc[:1]).all_rows.iloc[0]
    assert pd.isna(row.glucose_change_5m)
    assert pd.isna(row.target_low_30m)


def test_patient_split_has_no_overlap():
    split = patient_split(["A", "B", "C", "D", "E", "F"])
    assert not (set(split["train"]) & set(split["test"]))
    assert set().union(*map(set, split.values())) == {"A", "B", "C", "D", "E", "F"}


def test_patient_split_requires_three_patients():
    with pytest.raises(ValueError, match="At least 3"):
        patient_split(["A", "B"])


def test_state_update_validates_and_updates():
    engine = DigitalTwinEngine(observations().iloc[:2])
    state = engine.update({"patient_id": "P1", "timestamp": "2026-01-01T08:10:00", "glucose": 110})
    assert state["latest_valid_glucose"] == 110
    with pytest.raises(ValueError):
        engine.update({"patient_id": "P1", "timestamp": "not-time", "glucose": 110})


def test_prediction_without_model_is_informative():
    row = build_features_and_labels(observations()).all_rows.iloc[0]
    assert not predict_features(None, row)["available"]


def test_replay_uses_only_prefix_and_preserves_live_state():
    engine = DigitalTwinEngine(observations())
    state, next_position = engine.replay_next("P1", 2)
    assert state["latest_valid_glucose"] == 118
    assert next_position == 3
    assert engine.states == {}


def test_scenario_isolated_and_validates_inputs():
    runtime = Runtime(mode="synthetic")
    patient_id = runtime.engine.patients()[0]
    with runtime.storage.connect() as con:
        before = con.execute("SELECT COUNT(*) FROM predictions WHERE patient_id=?", (patient_id,)).fetchone()[0]
    before_state = dict(runtime.engine.states)
    result = runtime.scenario(patient_id, {"steps_30m": 12})
    assert result["available"]
    assert runtime.engine.states == before_state
    with runtime.storage.connect() as con:
        after = con.execute("SELECT COUNT(*) FROM predictions WHERE patient_id=?", (patient_id,)).fetchone()[0]
    assert after == before
    assert not runtime.scenario(patient_id, {"glucose_current": 80})["available"]
    assert not runtime.scenario(patient_id, {"steps_30m": -1})["available"]


def test_synthetic_is_explicit_and_reproducible(tmp_path: Path):
    generate_synthetic_data(tmp_path, seed=11, patients=3, points=30)
    first = (tmp_path / "SYN-001.csv").read_text()
    generate_synthetic_data(tmp_path, seed=11, patients=3, points=30)
    assert first == (tmp_path / "SYN-001.csv").read_text()
    assert "is_synthetic" in first


def test_persistence_reload_and_schema(tmp_path: Path):
    generate_synthetic_data(tmp_path / "synth", patients=4, points=90)
    records = []
    for path in (tmp_path / "synth").glob("*.csv"):
        item = inspect_csv(path)
        records.append(item.observations)
    training = build_features_and_labels(pd.concat(records, ignore_index=True)).training_rows
    metadata = train_model(training, tmp_path / "model.joblib", tmp_path / "metadata.json")
    loaded = load_model(tmp_path / "model.joblib")
    assert loaded and loaded["feature_columns"] == metadata["feature_columns"]
    assert set(metadata["feature_columns"]).issubset(FEATURE_COLUMNS)
