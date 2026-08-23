from __future__ import annotations

import logging
from types import SimpleNamespace

from app.inference.memory import MemoryProfiler


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
