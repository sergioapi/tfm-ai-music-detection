from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from app.inference.memory import MemoryProfiler
from app.inference.service import AudioInferenceService


def sine(seconds: float, sample_rate: int = 16_000) -> np.ndarray:
    samples = int(round(seconds * sample_rate))
    timeline = np.arange(samples, dtype=np.float32) / sample_rate
    return np.sin(2 * np.pi * 440.0 * timeline).astype(np.float32)


class FakeProcess:
    def __init__(self, rss_bytes: int = 32 * 1024 * 1024) -> None:
        self.rss_bytes = rss_bytes

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self.rss_bytes)


def test_disabled_memory_profiler_does_not_query_or_log(caplog) -> None:
    def fail_process_factory() -> FakeProcess:
        raise AssertionError("The process must not be queried when profiling is disabled")

    profiler = MemoryProfiler(enabled=False, process_factory=fail_process_factory)
    with caplog.at_level(logging.INFO):
        measurement = profiler.measure("A1B2C3D4", "start")

    assert measurement is None
    assert "memory_profile" not in caplog.text


def test_enabled_memory_profiler_logs_phase_and_rss(caplog) -> None:
    profiler = MemoryProfiler(enabled=True, process_factory=FakeProcess)
    with caplog.at_level(logging.INFO):
        measurement = profiler.measure("A1B2C3D4", "after_decode", n_fragments=3)

    assert measurement is not None
    assert measurement.phase == "after_decode"
    assert measurement.rss_mib == 32.0
    assert "memory_profile request=A1B2C3D4 phase=after_decode" in caplog.text
    assert "rss_mib=32.0" in caplog.text


def test_memory_profiler_failure_does_not_raise(caplog) -> None:
    def fail_process_factory() -> FakeProcess:
        raise RuntimeError("psutil unavailable")

    profiler = MemoryProfiler(enabled=True, process_factory=fail_process_factory)
    with caplog.at_level(logging.WARNING):
        measurement = profiler.measure("A1B2C3D4", "start")

    assert measurement is None
    assert "memory_profile_failed request=A1B2C3D4 phase=start" in caplog.text


def test_disabled_profiling_preserves_service_result(artifact_factory, wav_factory) -> None:
    profiler = MemoryProfiler(enabled=False)
    service = AudioInferenceService(
        model_path=artifact_factory(),
        memory_profiler=profiler,
    )

    result = service.predict_file(wav_factory(sine(3.0), 16_000))

    assert result.n_fragments == 1
    assert result.decision_threshold == 0.0


def test_enabled_profiling_records_inference_phases(
    artifact_factory,
    wav_factory,
    caplog,
) -> None:
    service = AudioInferenceService(
        model_path=artifact_factory(),
        memory_profiler=MemoryProfiler(enabled=True, process_factory=FakeProcess),
    )

    with caplog.at_level(logging.INFO):
        result = service.predict_file(wav_factory(sine(3.0), 16_000))

    assert result.n_fragments == 1
    for phase in (
        "start",
        "after_audio_info",
        "after_decode",
        "after_validate_audio",
        "after_segment",
        "after_preprocess",
        "after_mfcc_fragment",
        "after_feature_matrix",
        "after_decision_function",
        "after_aggregation",
        "before_return",
    ):
        assert f"phase={phase}" in caplog.text
