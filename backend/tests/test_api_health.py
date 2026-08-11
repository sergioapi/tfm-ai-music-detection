from __future__ import annotations

from fastapi.testclient import TestClient

from app.inference.errors import ModelArtifactError
from app.main import create_app


class FakeService:
    def predict_file(self, path):  # pragma: no cover - must not be called by /health.
        raise AssertionError("/health must not run inference")


def test_health_reports_ok_when_service_loads() -> None:
    calls = {"count": 0}

    def service_factory() -> FakeService:
        calls["count"] += 1
        return FakeService()

    application = create_app(service_factory=service_factory)

    with TestClient(application) as client:
        response = client.get("/health")

    assert calls["count"] == 1
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_ready": True}


def test_health_reports_degraded_when_model_artifact_load_fails() -> None:
    def service_factory() -> FakeService:
        raise ModelArtifactError("Could not load C:/private/model.joblib")

    application = create_app(service_factory=service_factory)

    with TestClient(application) as client:
        response = client.get("/health")

    payload = response.json()
    assert response.status_code == 503
    assert payload == {"status": "degraded", "model_ready": False}
    assert "Could not load" not in response.text
    assert "C:/private/model.joblib" not in response.text


def test_health_does_not_call_predict_file() -> None:
    application = create_app(service_factory=FakeService)

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_ready"] is True
