from __future__ import annotations

import librosa
import numpy as np

from app.inference.config import InferenceConfig
from app.inference.errors import PredictionError


def extract_mfcc_features(
    signal: np.ndarray,
    sample_rate: int,
    config: InferenceConfig,
) -> np.ndarray:
    if sample_rate != config.target_sample_rate:
        raise PredictionError(
            f"MFCC extraction expects {config.target_sample_rate} Hz, found {sample_rate}"
        )
    audio = np.asarray(signal, dtype=np.float32)
    if audio.ndim != 1:
        raise PredictionError(f"Expected preprocessed mono audio, found shape {audio.shape}")
    if audio.size == 0:
        raise PredictionError("Preprocessed audio is empty")
    _validate_finite(audio, "preprocessed audio")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=config.target_sample_rate,
        n_mfcc=config.n_mfcc,
    )
    features = np.concatenate(
        [
            mfcc.mean(axis=1, dtype=np.float64),
            mfcc.std(axis=1, dtype=np.float64),
        ]
    ).astype(np.float32)
    expected_shape = (config.n_mfcc * 2,)
    if features.shape != expected_shape:
        raise PredictionError(
            f"Expected {expected_shape[0]} MFCC features, found shape {features.shape}"
        )
    if features.dtype != np.float32:
        raise PredictionError(f"Expected float32 MFCC features, found {features.dtype}")
    _validate_finite(features, "MFCC features")
    return features


def feature_columns(config: InferenceConfig | None = None) -> tuple[str, ...]:
    config = config or InferenceConfig()
    return tuple(f"mfcc_mean_{index:02d}" for index in range(config.n_mfcc)) + tuple(
        f"mfcc_std_{index:02d}" for index in range(config.n_mfcc)
    )


def _validate_finite(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise PredictionError(f"{name} contains NaN or infinite values")
