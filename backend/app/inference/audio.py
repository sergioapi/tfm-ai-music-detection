from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from app.inference.config import InferenceConfig
from app.inference.errors import AudioDecodingError, AudioValidationError
from app.inference.schemas import AudioFragment


def decode_audio_file(path: Path) -> tuple[np.ndarray, int]:
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.exists():
        raise AudioDecodingError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioDecodingError(f"Audio path is not a file: {audio_path}")

    try:
        audio, sample_rate = sf.read(audio_path, always_2d=False)
    except Exception as exc:  # noqa: BLE001 - expose a domain-specific error.
        raise AudioDecodingError(f"Could not decode audio file {audio_path}: {exc}") from exc

    return validate_decoded_audio(audio, int(sample_rate))


def validate_decoded_audio(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    if sample_rate <= 0:
        raise AudioValidationError(f"Invalid sample rate: {sample_rate}")

    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim not in (1, 2):
        raise AudioValidationError(f"Expected mono or stereo audio, found shape {signal.shape}")
    if signal.shape[0] == 0:
        raise AudioValidationError("Audio signal is empty")
    _validate_finite(signal, "decoded audio")
    return signal, sample_rate


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    fragment_duration_seconds: float,
) -> tuple[AudioFragment, ...]:
    signal, sample_rate = validate_decoded_audio(audio, sample_rate)
    if fragment_duration_seconds <= 0:
        raise AudioValidationError(
            f"Invalid fragment duration: {fragment_duration_seconds}"
        )

    fragment_samples = int(round(sample_rate * fragment_duration_seconds))
    if fragment_samples <= 0:
        raise AudioValidationError(f"Invalid fragment length: {fragment_samples}")

    fragments: list[AudioFragment] = []
    total_samples = signal.shape[0]
    for index, start in enumerate(range(0, total_samples, fragment_samples)):
        end = min(start + fragment_samples, total_samples)
        if end <= start:
            continue
        fragment_signal = signal[start:end].copy()
        duration = (end - start) / sample_rate
        fragments.append(
            AudioFragment(
                index=index,
                start_seconds=start / sample_rate,
                end_seconds=end / sample_rate,
                duration_seconds=duration,
                signal=fragment_signal,
                sample_rate=sample_rate,
                is_incomplete=(end - start) < fragment_samples,
            )
        )

    if not fragments:
        raise AudioValidationError("Audio segmentation produced no fragments")
    return tuple(fragments)


def preprocess_fragment(
    audio: np.ndarray,
    sample_rate: int,
    config: InferenceConfig,
) -> np.ndarray:
    if sample_rate <= 0:
        raise AudioValidationError(f"Invalid sample rate: {sample_rate}")

    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim == 2:
        signal = signal.mean(axis=1, dtype=np.float32)
    elif signal.ndim != 1:
        raise AudioValidationError(f"Expected mono or stereo fragment, found shape {signal.shape}")

    if signal.size == 0:
        raise AudioValidationError("Audio fragment is empty")
    _validate_finite(signal, "audio fragment")

    source_window_samples = int(round(sample_rate * config.fragment_duration_seconds))
    signal = _select_or_pad_window(signal, source_window_samples)

    if sample_rate != config.target_sample_rate:
        signal = librosa.resample(
            signal,
            orig_sr=sample_rate,
            target_sr=config.target_sample_rate,
        ).astype(np.float32, copy=False)

    signal = _fix_length(signal, config.target_samples)
    _validate_finite(signal, "preprocessed audio")
    return signal.astype(np.float32, copy=False)


def _select_or_pad_window(signal: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 0:
        raise AudioValidationError(f"Invalid window length: {window_samples}")
    if signal.size <= window_samples:
        return _fix_length(signal, window_samples)

    squared = np.square(signal, dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(squared)])
    energies = cumulative[window_samples:] - cumulative[:-window_samples]
    best_start = int(np.argmax(energies))
    return signal[best_start : best_start + window_samples].astype(np.float32, copy=False)


def _fix_length(signal: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= 0:
        raise AudioValidationError(f"Invalid target length: {target_samples}")
    if signal.size == target_samples:
        return signal.astype(np.float32, copy=False)
    if signal.size > target_samples:
        return signal[:target_samples].astype(np.float32, copy=False)
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: signal.size] = signal.astype(np.float32, copy=False)
    return padded


def _validate_finite(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise AudioValidationError(f"{name} contains NaN or infinite values")
