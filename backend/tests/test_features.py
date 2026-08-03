from __future__ import annotations

import numpy as np
import pytest

from app.inference.config import InferenceConfig
from app.inference.errors import PredictionError
from app.inference.features import extract_mfcc_features, feature_columns


def test_extract_mfcc_features_shape_dtype_and_finiteness(config: InferenceConfig) -> None:
    signal = np.sin(
        2 * np.pi * 440 * np.arange(config.target_samples, dtype=np.float32) / config.target_sample_rate
    ).astype(np.float32)

    features = extract_mfcc_features(signal, config.target_sample_rate, config)

    assert features.shape == (40,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_feature_columns_has_exact_order(config: InferenceConfig) -> None:
    columns = feature_columns(config)

    assert len(columns) == 40
    assert columns[:3] == ("mfcc_mean_00", "mfcc_mean_01", "mfcc_mean_02")
    assert columns[19] == "mfcc_mean_19"
    assert columns[20] == "mfcc_std_00"
    assert columns[-1] == "mfcc_std_19"


def test_extract_mfcc_rejects_wrong_shape(config: InferenceConfig) -> None:
    signal = np.zeros((config.target_samples, 2), dtype=np.float32)

    with pytest.raises(PredictionError, match="Expected preprocessed mono"):
        extract_mfcc_features(signal, config.target_sample_rate, config)


def test_extract_mfcc_rejects_non_finite_signal(config: InferenceConfig) -> None:
    signal = np.zeros(config.target_samples, dtype=np.float32)
    signal[0] = np.nan

    with pytest.raises(PredictionError, match="NaN or infinite"):
        extract_mfcc_features(signal, config.target_sample_rate, config)


def test_extract_mfcc_rejects_unexpected_sample_rate(config: InferenceConfig) -> None:
    signal = np.zeros(config.target_samples, dtype=np.float32)

    with pytest.raises(PredictionError, match="expects 16000 Hz"):
        extract_mfcc_features(signal, 8_000, config)
