from __future__ import annotations

import logging
import threading

from fastapi.testclient import TestClient

from app.config import ApiSettings
from app.inference.audio import warm_up_resampling
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


def test_resample_warmup_is_not_scheduled_when_disabled(monkeypatch) -> None:
    calls = {"count": 0}

    def warmup(*args) -> None:
        calls["count"] += 1

    monkeypatch.setattr("app.main.warm_up_resampling", warmup)
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(resample_warmup_enabled=False),
    )

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200

    assert calls["count"] == 0


def test_resample_warmup_runs_once_in_background_when_enabled(monkeypatch) -> None:
    calls = {"count": 0}
    completed = threading.Event()

    def warmup(*args) -> None:
        calls["count"] += 1
        completed.set()

    monkeypatch.setattr("app.main.warm_up_resampling", warmup)
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(resample_warmup_enabled=True),
    )

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert completed.wait(timeout=1)

    assert calls["count"] == 1


def test_resample_warmup_failure_does_not_degrade_health(monkeypatch, caplog) -> None:
    completed = threading.Event()

    def fail_resample(*args, **kwargs):
        raise RuntimeError("warm-up failed")

    def warmup(profiler) -> None:
        warm_up_resampling(profiler)
        completed.set()

    monkeypatch.setattr("app.inference.audio.librosa.resample", fail_resample)
    monkeypatch.setattr("app.main.warm_up_resampling", warmup)
    application = create_app(
        service_factory=FakeService,
        settings=ApiSettings(resample_warmup_enabled=True),
    )

    with caplog.at_level(logging.WARNING):
        with TestClient(application) as client:
            assert client.get("/health").status_code == 200
            assert completed.wait(timeout=1)

    assert "resample_warmup status=failed error_type=RuntimeError" in caplog.text
