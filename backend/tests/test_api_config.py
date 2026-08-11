from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    DEFAULT_ALLOWED_AUDIO_EXTENSIONS,
    DEFAULT_ALLOWED_AUDIO_MIME_TYPES,
    DEFAULT_MAX_AUDIO_DURATION_SECONDS,
    DEFAULT_MAX_UPLOAD_SIZE_BYTES,
    ApiSettings,
)


def test_api_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_api_env(monkeypatch)

    settings = ApiSettings.from_env()

    assert settings.max_upload_size_bytes == DEFAULT_MAX_UPLOAD_SIZE_BYTES
    assert settings.max_audio_duration_seconds == DEFAULT_MAX_AUDIO_DURATION_SECONDS
    assert settings.allowed_audio_extensions == DEFAULT_ALLOWED_AUDIO_EXTENSIONS
    assert settings.allowed_audio_mime_types == DEFAULT_ALLOWED_AUDIO_MIME_TYPES
    assert settings.temp_dir is None


def test_api_settings_reads_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", "12345")
    monkeypatch.setenv("MAX_AUDIO_DURATION_SECONDS", "12.5")
    monkeypatch.setenv("ALLOWED_AUDIO_EXTENSIONS", "wav, .aiff")
    monkeypatch.setenv("ALLOWED_AUDIO_MIME_TYPES", "audio/wav, audio/x-aiff")
    monkeypatch.setenv("TEMP_DIR", "tmp/api")

    settings = ApiSettings.from_env()

    assert settings.max_upload_size_bytes == 12345
    assert settings.max_audio_duration_seconds == pytest.approx(12.5)
    assert settings.allowed_audio_extensions == (".wav", ".aiff")
    assert settings.allowed_audio_mime_types == ("audio/wav", "audio/x-aiff")
    assert settings.temp_dir == Path("tmp/api")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MAX_UPLOAD_SIZE_BYTES", "abc", "integer"),
        ("MAX_UPLOAD_SIZE_BYTES", "0", "greater than zero"),
        ("MAX_AUDIO_DURATION_SECONDS", "abc", "number"),
        ("MAX_AUDIO_DURATION_SECONDS", "-1", "greater than zero"),
    ],
)
def test_api_settings_rejects_invalid_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    _clear_api_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        ApiSettings.from_env()


def test_api_settings_rejects_empty_extension_after_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_api_env(monkeypatch)
    monkeypatch.setenv("ALLOWED_AUDIO_EXTENSIONS", ".")

    with pytest.raises(ValueError, match="invalid extension"):
        ApiSettings.from_env()


def _clear_api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MAX_UPLOAD_SIZE_BYTES",
        "MAX_AUDIO_DURATION_SECONDS",
        "ALLOWED_AUDIO_EXTENSIONS",
        "ALLOWED_AUDIO_MIME_TYPES",
        "TEMP_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
