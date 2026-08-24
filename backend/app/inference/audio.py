from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

from app.inference.config import InferenceConfig
from app.inference.errors import AudioDecodingError, AudioValidationError
from app.inference.memory import MemoryProfiler
from app.inference.schemas import AudioFragment


@dataclass(frozen=True)
class AudioFileInfo:
    sample_rate: int
    frames: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


def get_audio_duration_seconds(path: Path) -> float:
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.exists():
        raise AudioDecodingError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioDecodingError(f"Audio path is not a file: {audio_path}")

    try:
        info = sf.info(audio_path)
    except Exception as exc:  # noqa: BLE001 - expose a domain-specific error.
        raise AudioDecodingError(f"Could not inspect audio file {audio_path}: {exc}") from exc

    audio_info = _validate_audio_info(int(info.samplerate), int(info.frames))
    duration_seconds = audio_info.duration_seconds
    if duration_seconds <= 0.0 or not math.isfinite(duration_seconds):
        raise AudioValidationError(f"Invalid audio duration: {duration_seconds!r}")
    return float(duration_seconds)


@contextmanager
def open_audio_fragments(
    path: Path,
    fragment_duration_seconds: float,
) -> Iterator[tuple[AudioFileInfo, Iterator[AudioFragment]]]:
    """Open an audio file once and expose consecutive decoded fragments."""
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.exists():
        raise AudioDecodingError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioDecodingError(f"Audio path is not a file: {audio_path}")
    if fragment_duration_seconds <= 0:
        raise AudioValidationError(
            f"Invalid fragment duration: {fragment_duration_seconds}"
        )

    try:
        with sf.SoundFile(audio_path) as source:
            audio_info = _validate_audio_info(int(source.samplerate), int(source.frames))
            fragment_samples = int(round(audio_info.sample_rate * fragment_duration_seconds))
            if fragment_samples <= 0:
                raise AudioValidationError(f"Invalid fragment length: {fragment_samples}")
            yield audio_info, _read_audio_fragments(source, audio_info, fragment_samples)
    except (AudioDecodingError, AudioValidationError):
        raise
    except Exception as exc:  # noqa: BLE001 - expose a domain-specific error.
        raise AudioDecodingError(f"Could not decode audio file {audio_path}: {exc}") from exc


def decode_audio_file(
    path: Path,
    memory_profiler: MemoryProfiler | None = None,
    profiling_request_id: str | None = None,
) -> tuple[np.ndarray, int]:
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.exists():
        raise AudioDecodingError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioDecodingError(f"Audio path is not a file: {audio_path}")

    try:
        audio, sample_rate = sf.read(audio_path, always_2d=False)
    except Exception as exc:  # noqa: BLE001 - expose a domain-specific error.
        raise AudioDecodingError(f"Could not decode audio file {audio_path}: {exc}") from exc

    if memory_profiler is not None:
        memory_profiler.measure(profiling_request_id, "after_decode")
    decoded = validate_decoded_audio(audio, int(sample_rate))
    if memory_profiler is not None:
        memory_profiler.measure(profiling_request_id, "after_validate_audio")
    return decoded


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


def _read_audio_fragments(
    source: sf.SoundFile,
    audio_info: AudioFileInfo,
    fragment_samples: int,
) -> Iterator[AudioFragment]:
    index = 0
    decoded_frames = 0
    while True:
        try:
            audio = source.read(
                frames=fragment_samples,
                dtype="float64",
                always_2d=False,
            )
        except Exception as exc:  # noqa: BLE001 - expose a domain-specific error.
            raise AudioDecodingError(f"Could not decode audio fragment: {exc}") from exc

        if np.asarray(audio).shape[0] == 0:
            if decoded_frames == 0:
                raise AudioValidationError("Audio signal is empty")
            break
        signal, sample_rate = validate_decoded_audio(audio, audio_info.sample_rate)
        frames_read = signal.shape[0]
        start = decoded_frames
        end = decoded_frames + frames_read
        yield AudioFragment(
            index=index,
            start_seconds=start / sample_rate,
            end_seconds=end / sample_rate,
            duration_seconds=frames_read / sample_rate,
            signal=signal,
            sample_rate=sample_rate,
            is_incomplete=frames_read < fragment_samples,
        )
        index += 1
        decoded_frames = end


def _validate_audio_info(sample_rate: int, frames: int) -> AudioFileInfo:
    if sample_rate <= 0:
        raise AudioValidationError(f"Invalid sample rate: {sample_rate}")
    if frames <= 0:
        raise AudioValidationError("Audio signal is empty")
    duration_seconds = frames / sample_rate
    if duration_seconds <= 0.0 or not math.isfinite(duration_seconds):
        raise AudioValidationError(f"Invalid audio duration: {duration_seconds!r}")
    return AudioFileInfo(sample_rate=sample_rate, frames=frames)


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
    *,
    timing_callback: Callable[[str, float], None] | None = None,
) -> np.ndarray:
    total_start = _timing_start(timing_callback)
    if sample_rate <= 0:
        raise AudioValidationError(f"Invalid sample rate: {sample_rate}")

    phase_start = _timing_start(timing_callback)
    signal = np.asarray(audio, dtype=np.float32)
    _record_timing(timing_callback, "float32", phase_start)
    if signal.ndim == 2:
        phase_start = _timing_start(timing_callback)
        signal = signal.mean(axis=1, dtype=np.float32)
        _record_timing(timing_callback, "mono", phase_start)
    elif signal.ndim != 1:
        raise AudioValidationError(f"Expected mono or stereo fragment, found shape {signal.shape}")

    if signal.size == 0:
        raise AudioValidationError("Audio fragment is empty")
    _validate_finite(signal, "audio fragment")

    source_window_samples = int(round(sample_rate * config.fragment_duration_seconds))
    phase_start = _timing_start(timing_callback)
    signal = _select_or_pad_window(signal, source_window_samples)
    _record_timing(timing_callback, "select_or_pad", phase_start)

    if sample_rate != config.target_sample_rate:
        phase_start = _timing_start(timing_callback)
        resample_fn = librosa.resample
        _record_timing(timing_callback, "resample_resolve", phase_start)
        phase_start = _timing_start(timing_callback)
        signal = resample_fn(
            signal,
            orig_sr=sample_rate,
            target_sr=config.target_sample_rate,
        ).astype(np.float32, copy=False)
        _record_timing(timing_callback, "resample_execute", phase_start)

    phase_start = _timing_start(timing_callback)
    signal = _fix_length(signal, config.target_samples)
    _record_timing(timing_callback, "fix_length", phase_start)
    phase_start = _timing_start(timing_callback)
    _validate_finite(signal, "preprocessed audio")
    result = signal.astype(np.float32, copy=False)
    _record_timing(timing_callback, "finalize", phase_start)
    _record_timing(timing_callback, "total", total_start)
    return result


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


def _record_timing(
    timing_callback: Callable[[str, float], None] | None,
    phase: str,
    start: float | None,
) -> None:
    if timing_callback is None or start is None:
        return
    try:
        timing_callback(phase, max(0.0, float(time.perf_counter() - start)))
    except Exception:  # noqa: BLE001 - diagnostics must not break preprocessing.
        return


def _timing_start(timing_callback: Callable[[str, float], None] | None) -> float | None:
    if timing_callback is None:
        return None
    return time.perf_counter()
