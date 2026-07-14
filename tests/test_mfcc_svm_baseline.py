from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.mfcc_svm_baseline as baseline  # noqa: E402
from scripts.mfcc_svm_baseline import (  # noqa: E402
    BaselineError,
    PreprocessConfig,
    build_svm_pipeline,
    extract_mfcc_features,
    feature_columns,
    load_manifest,
    preprocess_audio_array,
    validate_description_splits,
)


class LazyAudioRow:
    def __init__(self, audio_id: str) -> None:
        self.audio_id = audio_id
        self.audio_accesses = 0

    def __getitem__(self, key: str) -> object:
        if key == "id":
            return self.audio_id
        if key == "audio":
            self.audio_accesses += 1
            return {"bytes": b"fake-wav"}
        raise KeyError(key)


class CountingIterable:
    def __init__(self, rows: list[LazyAudioRow]) -> None:
        self.rows = rows
        self.iterated = 0

    def __iter__(self):
        for row in self.rows:
            self.iterated += 1
            yield row


def _metadata(ids: list[str]) -> dict[str, dict[str, object]]:
    return {
        audio_id: {
            "id": audio_id,
            "description": f"description-{audio_id}",
            "model": "synthetic",
            "label": 1,
            "split": "train",
        }
        for audio_id in ids
    }


def _patch_fast_feature_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        baseline,
        "_decode_remote_audio_payload",
        lambda audio, preprocess_config: np.zeros(preprocess_config.target_samples, dtype=np.float32),
    )
    monkeypatch.setattr(
        baseline,
        "extract_mfcc_features",
        lambda signal, sample_rate, config=None: np.arange(40, dtype=np.float32),
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


def test_streaming_skips_unselected_rows_without_audio_access(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fast_feature_extraction(monkeypatch)
    skipped = LazyAudioRow("skip")
    selected = LazyAudioRow("keep")

    rows, failures, stats = baseline._extract_remote_streaming_rows_from_iterable(
        stream=CountingIterable([skipped, selected]),
        pending_ids={"keep"},
        metadata_by_id=_metadata(["keep"]),
        columns=feature_columns(),
        preprocess_config=PreprocessConfig(),
        mfcc_config=baseline.MfccConfig(),
    )

    assert len(rows) == 1
    assert failures == []
    assert stats["stream_rows_scanned"] == 2
    assert skipped.audio_accesses == 0
    assert selected.audio_accesses == 1


def test_streaming_processes_only_required_ids_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fast_feature_extraction(monkeypatch)
    first = LazyAudioRow("a")
    ignored = LazyAudioRow("ignored")
    second = LazyAudioRow("b")
    not_reached = LazyAudioRow("not-reached")
    stream = CountingIterable([first, ignored, second, not_reached])

    rows, failures, stats = baseline._extract_remote_streaming_rows_from_iterable(
        stream=stream,
        pending_ids={"a", "b"},
        metadata_by_id=_metadata(["a", "b"]),
        columns=feature_columns(),
        preprocess_config=PreprocessConfig(),
        mfcc_config=baseline.MfccConfig(),
    )

    assert [row["id"] for row in rows] == ["a", "b"]
    assert failures == []
    assert stats["stream_ids_found"] == 2
    assert stream.iterated == 3
    assert ignored.audio_accesses == 0
    assert not_reached.audio_accesses == 0


def test_streaming_reports_missing_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fast_feature_extraction(monkeypatch)

    rows, failures, stats = baseline._extract_remote_streaming_rows_from_iterable(
        stream=CountingIterable([LazyAudioRow("present")]),
        pending_ids={"present", "missing"},
        metadata_by_id=_metadata(["present", "missing"]),
        columns=feature_columns(),
        preprocess_config=PreprocessConfig(),
        mfcc_config=baseline.MfccConfig(),
    )

    assert [row["id"] for row in rows] == ["present"]
    assert stats["stream_ids_missing"] == ["missing"]
    assert len(failures) == 1
    assert failures[0]["id"] == "missing"
    assert failures[0]["phase"] == "dataset_streaming"


def test_streaming_does_not_accumulate_all_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fast_feature_extraction(monkeypatch)
    rows_before_match = [LazyAudioRow(f"skip-{index}") for index in range(100)]
    selected = LazyAudioRow("target")
    stream = CountingIterable([*rows_before_match, selected, LazyAudioRow("after-target")])

    rows, failures, stats = baseline._extract_remote_streaming_rows_from_iterable(
        stream=stream,
        pending_ids={"target"},
        metadata_by_id=_metadata(["target"]),
        columns=feature_columns(),
        preprocess_config=PreprocessConfig(),
        mfcc_config=baseline.MfccConfig(),
    )

    assert len(rows) == 1
    assert failures == []
    assert stats["stream_rows_scanned"] == 101
    assert stream.iterated == 101
    assert all(row.audio_accesses == 0 for row in rows_before_match)


def test_load_hf_audio_stream_uses_streaming_and_disables_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeStream:
        def decode(self, enable: bool):
            calls["decode"] = enable
            return self

    class FakeDatasets:
        def load_dataset(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return FakeStream()

    monkeypatch.setattr(baseline, "datasets", FakeDatasets())

    stream = baseline._load_hf_audio_stream("dataset/name", "revision-sha")

    assert isinstance(stream, FakeStream)
    assert calls["args"] == ("dataset/name",)
    assert calls["kwargs"]["split"] == "train"
    assert calls["kwargs"]["revision"] == "revision-sha"
    assert calls["kwargs"]["streaming"] is True
    assert calls["decode"] is False


def test_existing_features_detect_duplicate_ids() -> None:
    columns = feature_columns()
    frame = pd.DataFrame(
        [
            {"id": "a", "description": "d", "model": "m", "label": 1, "split": "train", **{col: 0.0 for col in columns}},
            {"id": "a", "description": "d", "model": "m", "label": 1, "split": "train", **{col: 0.0 for col in columns}},
        ]
    )

    with pytest.raises(BaselineError, match="duplicate ids"):
        baseline._validate_existing_features(frame, columns, _metadata(["a"]))


def test_local_extraction_mode_still_loads_from_audio_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        baseline,
        "preprocess_audio_file",
        lambda path, config=None: np.zeros((config or PreprocessConfig()).target_samples, dtype=np.float32),
    )
    monkeypatch.setattr(
        baseline,
        "extract_mfcc_features",
        lambda signal, sample_rate, config=None: np.arange(40, dtype=np.float32),
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_path = audio_dir / "sample_keep.wav"
    audio_path.write_bytes(b"placeholder")
    manifest = pd.DataFrame(
        [{"id": "keep", "description": "d", "model": "m", "label": 1, "split": "train"}]
    )

    rows, failures, stats = baseline._extract_local_rows(
        manifest=manifest,
        processed_ids=set(),
        audio_dir=audio_dir,
        columns=feature_columns(),
        preprocess_config=PreprocessConfig(),
        mfcc_config=baseline.MfccConfig(),
    )

    assert failures == []
    assert rows[0]["id"] == "keep"
    assert stats["stream_rows_scanned"] is None
