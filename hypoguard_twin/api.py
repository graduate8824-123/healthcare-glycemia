"""Local FastAPI service. Bind with 127.0.0.1 only (see README)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .runtime import Runtime


class PredictionRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=128)


class ScenarioRequest(PredictionRequest):
    overrides: dict[str, float] = Field(default_factory=dict)


@lru_cache
def get_runtime() -> Runtime:
    return Runtime()


app = FastAPI(title="HypoGuard Twin API", version="0.1.0", description="Local research-prototype API; not a medical device.")


def _patient_or_404(patient_id: str) -> Runtime:
    runtime = get_runtime()
    if patient_id not in runtime.engine.patients():
        raise HTTPException(status_code=404, detail="Unknown patient identifier")
    return runtime


@app.get("/health")
def health() -> dict[str, Any]:
    runtime = get_runtime()
    return {"status": "ok", "data_mode": runtime.mode, "model_available": runtime.model is not None,
            "patients": len(runtime.engine.patients()), "clinical_disclaimer": "Research prototype; not clinically validated."}


@app.get("/patients")
def patients() -> dict[str, Any]:
    runtime = get_runtime()
    return {"data_mode": runtime.mode, "patient_ids": runtime.engine.patients()}


@app.get("/patients/{patient_id}/history")
def history(patient_id: str, limit: int = Query(default=500, ge=1, le=5000)) -> dict[str, Any]:
    runtime = _patient_or_404(patient_id)
    return {"patient_id": patient_id, "observations": runtime.patient_history(patient_id)[-limit:]}


@app.get("/patients/{patient_id}/state")
def state(patient_id: str) -> dict[str, Any]:
    runtime = _patient_or_404(patient_id)
    try:
        output = runtime.engine.derive_state(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    runtime.storage.save_state(patient_id, output)
    return output


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    runtime = _patient_or_404(request.patient_id)
    try:
        return runtime.predict(request.patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/patients/{patient_id}/latest-prediction")
def latest_prediction(patient_id: str) -> dict[str, Any]:
    runtime = _patient_or_404(patient_id)
    output = runtime.storage.latest_prediction(patient_id)
    if output is None:
        raise HTTPException(status_code=404, detail="No saved prediction for this patient")
    return output


@app.get("/patients/{patient_id}/replay/{position}")
def replay(patient_id: str, position: int = 0) -> dict[str, Any]:
    runtime = _patient_or_404(patient_id)
    try:
        return runtime.historical_replay(patient_id, position)
    except IndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/scenario")
def scenario(request: ScenarioRequest) -> dict[str, Any]:
    runtime = _patient_or_404(request.patient_id)
    return runtime.scenario(request.patient_id, request.overrides)


@app.get("/quality")
def quality() -> dict[str, Any]:
    runtime = get_runtime()
    return {"data_mode": runtime.mode, "by_patient": runtime.quality_summary(), "by_file": runtime.quality}
