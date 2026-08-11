from __future__ import annotations

import os
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
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _read_extensions(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = _read_values(name, default)
    normalized = tuple(
        value if value.startswith(".") else f".{value}"
        for value in values
    )
    if any(value == "." for value in normalized):
        raise ValueError(f"{name} contains an invalid extension")
    return normalized


def _read_values(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    values = tuple(
        item.strip().lower()
        for item in raw_value.split(",")
        if item.strip()
    )
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _read_optional_path(name: str) -> Path | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return Path(raw_value).expanduser()
