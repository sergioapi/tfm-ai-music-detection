from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from app.config import ApiSettings
from app.inference.config import InferenceConfig
from app.inference.errors import ModelArtifactError
from app.inference.schemas import StartupWarmupResult, WarmupResult
from app.main import create_app


class FakeService:
    config = InferenceConfig()

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


def test_startup_warmups_are_not_scheduled_when_disabled(monkeypatch) -> None:
    calls = {"count": 0}

    def warmup(*args) -> StartupWarmupResult:
        calls["count"] += 1
        raise AssertionError("Warm-ups must be disabled")

    monkeypatch.setattr("app.main.run_startup_warmups", warmup)
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(resample_warmup_enabled=False),
    )

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200

    assert calls["count"] == 0


def test_startup_warmups_run_once_in_background_when_enabled(monkeypatch) -> None:
    calls = {"count": 0}
    completed = threading.Event()

    result = StartupWarmupResult(
        resampling=WarmupResult("resampling", True, 0.1),
        mfcc=WarmupResult("mfcc", True, 0.2),
    )

    def warmup(*args) -> StartupWarmupResult:
        calls["count"] += 1
        completed.set()
        return result

    monkeypatch.setattr("app.main.run_startup_warmups", warmup)
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(resample_warmup_enabled=True),
    )

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert completed.wait(timeout=1)

    assert calls["count"] == 1
    assert application.state.startup_warmup_result is result


def test_startup_warmup_failure_does_not_degrade_health(monkeypatch) -> None:
    completed = threading.Event()
    result = StartupWarmupResult(
        resampling=WarmupResult("resampling", False, 0.1, "RuntimeError"),
        mfcc=WarmupResult("mfcc", False, 0.2, "RuntimeError"),
    )

    def warmup(*args) -> StartupWarmupResult:
        completed.set()
        return result

    monkeypatch.setattr("app.main.run_startup_warmups", warmup)
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(resample_warmup_enabled=True),
    )

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert completed.wait(timeout=1)

    assert application.state.startup_warmup_result is result
