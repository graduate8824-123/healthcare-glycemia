"""Small SQLite metadata store; observations remain in source CSV files."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import DATABASE_PATH


class Storage:
    def __init__(self, path: str | Path = DATABASE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS patients (patient_id TEXT PRIMARY KEY, source_mode TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS twin_states (patient_id TEXT PRIMARY KEY, updated_at TEXT, state_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL, prediction_at TEXT NOT NULL,
                probability REAL, model_version TEXT, output_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS model_metadata (model_version TEXT PRIMARY KEY, created_at TEXT, metadata_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS replay_sessions (session_id TEXT PRIMARY KEY, patient_id TEXT, position INTEGER, updated_at TEXT);
            """)

    def upsert_patient(self, patient_id: str, source_mode: str) -> None:
        with self.connect() as con:
            con.execute("INSERT INTO patients(patient_id, source_mode) VALUES (?, ?) ON CONFLICT(patient_id) DO UPDATE SET source_mode=excluded.source_mode", (patient_id, source_mode))

    def save_state(self, patient_id: str, state: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute("INSERT INTO twin_states(patient_id, updated_at, state_json) VALUES (?, ?, ?) ON CONFLICT(patient_id) DO UPDATE SET updated_at=excluded.updated_at, state_json=excluded.state_json", (patient_id, state.get("updated_at"), json.dumps(state, default=str)))

    def save_prediction(self, patient_id: str, output: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute("INSERT INTO predictions(patient_id, prediction_at, probability, model_version, output_json) VALUES (?, ?, ?, ?, ?)",
                        (patient_id, output.get("prediction_timestamp"), output.get("probability_low_30m"), output.get("model_version"), json.dumps(output, default=str)))

    def latest_prediction(self, patient_id: str) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute("SELECT output_json FROM predictions WHERE patient_id=? ORDER BY id DESC LIMIT 1", (patient_id,)).fetchone()
        return json.loads(row["output_json"]) if row else None

    def save_model_metadata(self, metadata: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute("INSERT INTO model_metadata(model_version, created_at, metadata_json) VALUES (?, ?, ?) ON CONFLICT(model_version) DO UPDATE SET created_at=excluded.created_at, metadata_json=excluded.metadata_json", (metadata["model_version"], metadata["created_at"], json.dumps(metadata)))
