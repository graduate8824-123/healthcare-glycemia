from fastapi.testclient import TestClient

from hypoguard_twin import api
from hypoguard_twin.runtime import Runtime


def client(monkeypatch):
    runtime = Runtime(mode="synthetic")
    api.get_runtime.cache_clear()
    monkeypatch.setattr(api, "get_runtime", lambda: runtime)
    return TestClient(api.app), runtime


def test_health_endpoint(monkeypatch):
    test_client, _ = client(monkeypatch)
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_patients_and_unknown_patient(monkeypatch):
    test_client, runtime = client(monkeypatch)
    assert test_client.get("/patients").status_code == 200
    assert test_client.get("/patients/not-a-patient/history").status_code == 404
    assert test_client.post("/predict", json={"patient_id": "not-a-patient"}).status_code == 404


def test_prediction_request_validation(monkeypatch):
    test_client, runtime = client(monkeypatch)
    assert test_client.post("/predict", json={}).status_code == 422
    response = test_client.post("/predict", json={"patient_id": runtime.engine.patients()[0]})
    assert response.status_code == 200
    assert "prediction_timestamp" in response.json()


def test_replay_endpoint(monkeypatch):
    test_client, runtime = client(monkeypatch)
    patient = runtime.engine.patients()[0]
    response = test_client.get(f"/patients/{patient}/replay/2")
    assert response.status_code == 200
    assert response.json()["historical_replay"] is True


def test_scenario_rejects_unsupported_model_derived_input(monkeypatch):
    test_client, runtime = client(monkeypatch)
    patient = runtime.engine.patients()[0]
    response = test_client.post("/scenario", json={"patient_id": patient, "overrides": {"glucose_current": 80}})
    assert response.status_code == 200
    assert response.json()["available"] is False
