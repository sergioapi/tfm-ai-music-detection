from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from typing import Callable

import psutil


logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class MemoryMeasurement:
    request_id: str
    phase: str
    pid: int
    rss_mib: float


class MemoryProfiler:
    """Temporary, opt-in RSS logging for inference diagnostics."""

    def __init__(
        self,
        enabled: bool,
        process_factory: Callable[[], psutil.Process] = psutil.Process,
    ) -> None:
        self.enabled = enabled
        self._process_factory = process_factory
        self._process: psutil.Process | None = None

    def new_request_id(self) -> str | None:
        if not self.enabled:
            return None
        return secrets.token_hex(4).upper()

    def measure(
        self,
        request_id: str | None,
        phase: str,
        **context: float | int,
    ) -> MemoryMeasurement | None:
        if not self.enabled or request_id is None:
            return None

        try:
            if self._process is None:
                self._process = self._process_factory()
            rss_bytes = int(self._process.memory_info().rss)
        except Exception:  # noqa: BLE001 - diagnostic logging must not break inference.
            logger.warning(
                "memory_profile_failed request=%s phase=%s",
                request_id,
                phase,
            )
            return None

        measurement = MemoryMeasurement(
            request_id=request_id,
            phase=phase,
            pid=os.getpid(),
            rss_mib=rss_bytes / (1024 * 1024),
        )
        fields = " ".join(f"{name}={value}" for name, value in context.items())
        logger.info(
            "memory_profile request=%s phase=%s pid=%s rss_mib=%.1f%s",
            measurement.request_id,
            measurement.phase,
            measurement.pid,
            measurement.rss_mib,
            f" {fields}" if fields else "",
        )
        return measurement
