from __future__ import annotations

import os
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_AUDIO_DURATION_SECONDS = 300.0
DEFAULT_ALLOWED_AUDIO_EXTENSIONS = (".wav",)
DEFAULT_ALLOWED_AUDIO_MIME_TYPES = ("audio/wav", "audio/wave", "audio/x-wav")


@dataclass(frozen=True)
class ApiSettings:
    max_upload_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES
    max_audio_duration_seconds: float = DEFAULT_MAX_AUDIO_DURATION_SECONDS
    allowed_audio_extensions: tuple[str, ...] = DEFAULT_ALLOWED_AUDIO_EXTENSIONS
    allowed_audio_mime_types: tuple[str, ...] = DEFAULT_ALLOWED_AUDIO_MIME_TYPES
    temp_dir: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_upload_size_bytes, int):
            raise ValueError("max_upload_size_bytes must be an integer")
        if self.max_upload_size_bytes <= 0:
            raise ValueError("max_upload_size_bytes must be greater than zero")
        if not _is_positive_finite(self.max_audio_duration_seconds):
            raise ValueError("max_audio_duration_seconds must be a finite value greater than zero")

        object.__setattr__(
            self,
            "allowed_audio_extensions",
            _normalize_extensions(self.allowed_audio_extensions, "allowed_audio_extensions"),
        )
        object.__setattr__(
            self,
            "allowed_audio_mime_types",
            _normalize_values(self.allowed_audio_mime_types, "allowed_audio_mime_types"),
        )
        object.__setattr__(self, "temp_dir", _validate_temp_dir(self.temp_dir))

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            max_upload_size_bytes=_read_int(
                "MAX_UPLOAD_SIZE_BYTES",
                DEFAULT_MAX_UPLOAD_SIZE_BYTES,
            ),
            max_audio_duration_seconds=_read_float(
                "MAX_AUDIO_DURATION_SECONDS",
                DEFAULT_MAX_AUDIO_DURATION_SECONDS,
            ),
            allowed_audio_extensions=_read_extensions(
                "ALLOWED_AUDIO_EXTENSIONS",
                DEFAULT_ALLOWED_AUDIO_EXTENSIONS,
            ),
            allowed_audio_mime_types=_read_values(
                "ALLOWED_AUDIO_MIME_TYPES",
                DEFAULT_ALLOWED_AUDIO_MIME_TYPES,
            ),
            temp_dir=_read_optional_path("TEMP_DIR"),
        )


def _read_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _read_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not _is_positive_finite(value):
        raise ValueError(f"{name} must be greater than zero")
    return value


def _read_extensions(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = _read_values(name, default)
    return _normalize_extensions(values, name)


def _read_values(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    return tuple(
        item.strip().lower()
        for item in raw_value.split(",")
        if item.strip()
    )


def _read_optional_path(name: str) -> Path | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return Path(raw_value).expanduser()


def _is_positive_finite(value: float) -> bool:
    numeric_value = float(value)
    return numeric_value > 0.0 and math.isfinite(numeric_value)


def _normalize_extensions(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    normalized = tuple(
        value.strip().lower() if value.strip().startswith(".") else f".{value.strip().lower()}"
        for value in values
        if value.strip()
    )
    if not normalized:
        raise ValueError(f"{name} must contain at least one value")
    if any(value == "." or "/" in value or "\\" in value for value in normalized):
        raise ValueError(f"{name} contains an invalid extension")
    return normalized


def _normalize_values(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    normalized = tuple(value.strip().lower() for value in values if value.strip())
    if not normalized:
        raise ValueError(f"{name} must contain at least one value")
    return normalized


def _validate_temp_dir(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError("temp_dir must exist")
    if not path.is_dir():
        raise ValueError("temp_dir must be a directory")
    try:
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            pass
    except OSError as exc:
        raise ValueError("temp_dir must be writable") from exc
    return path
