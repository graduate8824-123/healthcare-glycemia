"""Small CPU-only training and honest held-out patient evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline

from .config import METADATA_PATH, MODEL_PATH, MODEL_VERSION
from .features import FEATURE_COLUMNS


def patient_split(patient_ids: list[str], seed: int = 2026) -> dict[str, list[str]]:
    """Deterministic patient-level split; no patient's adjacent samples cross partitions."""
    ids = sorted(set(map(str, patient_ids)))
    if len(ids) < 3:
        raise ValueError("At least 3 distinct patients are required for patient-level train/test evaluation")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(ids))
    test_count = max(1, round(len(ids) * .2))
    validation_count = 1 if len(ids) >= 5 else 0
    test = shuffled[:test_count]
    validation = shuffled[test_count:test_count + validation_count]
    train = shuffled[test_count + validation_count:]
    if not train:
        raise ValueError("Patient split left no training patients")
    return {"train": train, "validation": validation, "test": test}


def _metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= .5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1]).tolist()
    result: dict[str, Any] = {
        "test_observations": int(len(y_true)), "positive_prevalence": float(y_true.mean()),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "confusion_matrix_labels": [0, 1], "confusion_matrix": cm,
        "false_positives": int(cm[0][1]), "false_negatives": int(cm[1][0]),
        "roc_auc": None, "pr_auc": None,
    }
    if y_true.nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        result["pr_auc"] = float(average_precision_score(y_true, probabilities))
    return result


def train_model(training_rows: pd.DataFrame, model_path: str | Path = MODEL_PATH,
                metadata_path: str | Path = METADATA_PATH, seed: int = 2026) -> dict[str, Any]:
    if training_rows.empty:
        raise ValueError("No samples have sufficient future observations for training")
    available = [name for name in FEATURE_COLUMNS if name in training_rows]
    if not available:
        raise ValueError("No supported feature columns available")
    split = patient_split(training_rows.patient_id.tolist(), seed)
    train_rows = training_rows.loc[training_rows.patient_id.isin(split["train"])]
    test_rows = training_rows.loc[training_rows.patient_id.isin(split["test"])]
    if train_rows.target_low_30m.nunique() < 2:
        raise ValueError("Training patients contain only one target class; cannot fit classifier")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=3,
                                                 class_weight="balanced", random_state=seed, n_jobs=1)),
    ])
    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", DummyClassifier(strategy="prior", random_state=seed)),
    ])
    x_train, y_train = train_rows[available], train_rows.target_low_30m
    pipeline.fit(x_train, y_train)
    baseline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(test_rows[available])[:, 1]
    baseline_probabilities = baseline.predict_proba(test_rows[available])[:, 1]
    evaluation = _metrics(test_rows.target_low_30m, probabilities)
    evaluation["baseline_prior"] = _metrics(test_rows.target_low_30m, baseline_probabilities)
    evaluation["test_patients"] = len(split["test"])
    evaluation["evaluation_note"] = "Synthetic demonstration evaluation only; not clinical evidence." if bool(training_rows.is_synthetic.all()) else "Held-out patient evaluation."
    importances = pipeline.named_steps["classifier"].feature_importances_[:len(available)]
    payload = {"pipeline": pipeline, "feature_columns": available, "model_version": MODEL_VERSION}
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)
    metadata = {
        "model_version": MODEL_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed, "feature_columns": available, "split": split,
        "package_versions": {"scikit_learn": sklearn.__version__, "joblib": joblib.__version__, "numpy": np.__version__},
        "training_patients": len(split["train"]), "training_observations": len(train_rows),
        "training_data_mode": "SYNTHETIC DEMONSTRATION" if bool(training_rows.is_synthetic.all()) else "REAL DATASET",
        "label_prevalence_train": float(y_train.mean()), "calibration": "not calibrated; uncalibrated model estimate",
        "feature_importance": dict(sorted(zip(available, map(float, importances)), key=lambda pair: pair[1], reverse=True)),
        "evaluation": evaluation,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_model(model_path: str | Path = MODEL_PATH) -> dict[str, Any] | None:
    path = Path(model_path)
    return joblib.load(path) if path.exists() else None


def predict_features(model: dict[str, Any] | None, row: pd.Series) -> dict[str, Any]:
    if model is None:
        return {"available": False, "message": "No trained model is available."}
    columns = model["feature_columns"]
    missing = [name for name in columns if name not in row.index]
    if missing:
        return {"available": False, "message": f"Feature schema mismatch: {missing}"}
    frame = pd.DataFrame([{name: row[name] for name in columns}])
    probability = float(model["pipeline"].predict_proba(frame)[0, 1])
    return {"available": True, "probability_low_30m": probability,
            "model_version": model["model_version"], "calibration": "not calibrated; uncalibrated model estimate"}
