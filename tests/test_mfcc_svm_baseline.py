from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mfcc_svm_baseline import (  # noqa: E402
    PreprocessConfig,
    build_svm_pipeline,
    extract_mfcc_features,
    feature_columns,
    load_manifest,
    preprocess_audio_array,
    validate_description_splits,
)


def test_preprocessing_converts_stereo_to_mono() -> None:
    sample_rate = 16_000
    left = np.ones(sample_rate, dtype=np.float32)
    right = np.zeros(sample_rate, dtype=np.float32)
    stereo = np.column_stack([left, right])

    processed = preprocess_audio_array(stereo, sample_rate)

    assert processed.shape == (160_000,)
    assert np.allclose(processed[:sample_rate], 0.5)


def test_preprocessing_returns_exactly_160000_samples() -> None:
    signal = np.ones(200_000, dtype=np.float32)

    processed = preprocess_audio_array(signal, 16_000)

    assert processed.dtype == np.float32
    assert processed.shape == (160_000,)


def test_maximum_energy_window_selection_is_deterministic() -> None:
    sample_rate = 10
    config = PreprocessConfig(target_sample_rate=10, duration_seconds=2.0)
    signal = np.zeros(70, dtype=np.float32)
    signal[20:40] = 0.2
    signal[40:60] = 1.0

    first = preprocess_audio_array(signal, sample_rate, config=config)
    second = preprocess_audio_array(signal, sample_rate, config=config)

    assert np.array_equal(first, second)
    assert np.allclose(first, 1.0)


def test_short_signals_are_zero_padded() -> None:
    signal = np.arange(5, dtype=np.float32)
    config = PreprocessConfig(target_sample_rate=10, duration_seconds=1.0)

    processed = preprocess_audio_array(signal, 10, config=config)

    assert processed.shape == (10,)
    assert np.array_equal(processed[:5], signal)
    assert np.array_equal(processed[5:], np.zeros(5, dtype=np.float32))


def test_mfcc_extraction_returns_40_finite_values() -> None:
    sample_rate = 16_000
    t = np.arange(160_000, dtype=np.float32) / sample_rate
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    features = extract_mfcc_features(signal, sample_rate)

    assert features.shape == (40,)
    assert np.isfinite(features).all()


def test_svm_pipeline_fits_predicts_and_scores_synthetic_data() -> None:
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=-1.0, scale=0.2, size=(12, 40))
    X1 = rng.normal(loc=1.0, scale=0.2, size=(12, 40))
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0] * 12 + [1] * 12)

    pipeline = build_svm_pipeline(C=1, gamma="scale")
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    scores = pipeline.decision_function(X)

    assert predictions.shape == (24,)
    assert scores.shape == (24,)
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_saved_model_artifact_can_be_loaded_and_predict_again(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 40)).astype(np.float32)
    y = np.array([0, 1] * 10)
    pipeline = build_svm_pipeline(C=1, gamma="scale").fit(X, y)
    artifact_path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipeline, "feature_columns": feature_columns()}, artifact_path)

    loaded = joblib.load(artifact_path)

    assert loaded["pipeline"].predict(X).shape == (20,)
    assert loaded["pipeline"].decision_function(X).shape == (20,)


def test_real_manifest_has_no_description_overlap_between_splits() -> None:
    manifest = load_manifest(Path("data/aime_splits.csv"))

    validate_description_splits(manifest)

    split_sets = {
        split: set(group["description"].tolist())
        for split, group in manifest.groupby("split", observed=False)
    }
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])


def test_feature_columns_are_stable() -> None:
    frame = pd.DataFrame(columns=feature_columns())

    assert len(frame.columns) == 40
    assert frame.columns[0] == "mfcc_mean_00"
    assert frame.columns[-1] == "mfcc_std_19"
