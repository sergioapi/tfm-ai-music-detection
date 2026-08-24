from __future__ import annotations

import logging
import os
import secrets
import sys
from threading import Lock
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
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
        self._preprocess_profile_claimed = False
        self._preprocess_profile_lock = Lock()

    def new_request_id(self) -> str | None:
        if not self.enabled:
            return None
        return secrets.token_hex(4).upper()

    def log_runtime_versions(self) -> None:
        """Log effective runtime versions once when diagnostic profiling is enabled."""
        if not self.enabled:
            return
        package_versions = {
            package: _package_version(package)
            for package in ("librosa", "numpy", "scipy", "soxr", "numba", "llvmlite")
        }
        logger.info(
            "runtime_profile python=%s librosa=%s numpy=%s scipy=%s soxr=%s "
            "numba=%s llvmlite=%s",
            sys.version.split()[0],
            package_versions["librosa"],
            package_versions["numpy"],
            package_versions["scipy"],
            package_versions["soxr"],
            package_versions["numba"],
            package_versions["llvmlite"],
        )

    def claim_preprocess_profile(self, request_id: str | None) -> bool:
        """Allow the first profiled resample in this process to emit phase timings."""
        if not self.enabled or request_id is None:
            return False
        with self._preprocess_profile_lock:
            if self._preprocess_profile_claimed:
                return False
            self._preprocess_profile_claimed = True
            return True

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

    def log_duration(
        self,
        profile: str,
        request_id: str | None,
        fragment_index: int,
        phase: str,
        seconds: float,
    ) -> None:
        if not self.enabled or request_id is None:
            return
        try:
            logger.info(
                "%s request=%s fragment_index=%s phase=%s seconds=%.4f",
                profile,
                request_id,
                fragment_index,
                phase,
                seconds,
            )
        except Exception:  # noqa: BLE001 - diagnostic logging must not break inference.
            return


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"
    except Exception:  # noqa: BLE001 - diagnostics must not affect startup.
        return "unavailable"
