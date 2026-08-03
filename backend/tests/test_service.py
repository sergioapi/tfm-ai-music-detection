from __future__ import annotations

import numpy as np
import pytest

from app.inference.aggregation import AGGREGATION_STRATEGY
from app.inference.errors import ModelArtifactError
from app.inference.service import AudioInferenceService


def sine(seconds: float, sample_rate: int = 16_000) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate)), dtype=np.float32) / sample_rate
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


def test_constructor_accepts_explicit_path(artifact_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())

    assert service.model.metadata.model_id == "mfcc-svm-baseline"


def test_constructor_reads_model_path_environment(monkeypatch: pytest.MonkeyPatch, artifact_factory) -> None:
    path = artifact_factory()
    monkeypatch.setenv("MODEL_PATH", str(path))

    service = AudioInferenceService()

    assert service.model.path == path.resolve()


def test_explicit_path_has_priority_over_environment(
    monkeypatch: pytest.MonkeyPatch,
    artifact_factory,
) -> None:
    env_path = artifact_factory(positive_label=0)
    explicit_path = artifact_factory()
    monkeypatch.setenv("MODEL_PATH", str(env_path))

    service = AudioInferenceService(model_path=explicit_path)

    assert service.model.path == explicit_path.resolve()


def test_constructor_errors_if_model_path_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)

    with pytest.raises(ModelArtifactError, match="MODEL_PATH"):
        AudioInferenceService()


def test_model_is_loaded_once(monkeypatch: pytest.MonkeyPatch, artifact_factory) -> None:
    calls = {"count": 0}
    original_load = AudioInferenceService.__init__.__globals__["MfccSvmModel"].load

    def counting_load(path, config):
        calls["count"] += 1
        return original_load(path, config)

    monkeypatch.setattr(
        "app.inference.service.MfccSvmModel.load",
        staticmethod(counting_load),
    )
    service = AudioInferenceService(model_path=artifact_factory())

    assert calls["count"] == 1
    assert service.model is not None


def test_predict_short_audio_returns_one_fragment(artifact_factory, wav_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(3.0), 16_000)

    result = service.predict_file(path)

    assert result.n_fragments == 1
    assert result.fragments[0].was_padded is True
    assert result.audio_duration_seconds == pytest.approx(3.0, abs=1e-3)


def test_predict_exactly_ten_seconds_returns_one_fragment(artifact_factory, wav_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(10.0), 16_000)

    result = service.predict_file(path)

    assert result.n_fragments == 1
    assert result.fragments[0].was_padded is False


def test_predict_multiple_fragments_returns_ordered_results(artifact_factory, wav_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(21.0), 16_000)

    result = service.predict_file(path)

    assert result.n_fragments == 3
    starts = [fragment.start_seconds for fragment in result.fragments]
    assert starts == sorted(starts)
    assert result.fragments[-1].was_padded is True


def test_global_score_is_duration_weighted_mean_of_fragment_scores(
    artifact_factory,
    wav_factory,
) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(21.0), 16_000)

    result = service.predict_file(path)

    scores = np.array([fragment.ai_score for fragment in result.fragments])
    durations = np.array([fragment.duration_seconds for fragment in result.fragments])
    expected = np.sum(scores * durations) / np.sum(durations)
    assert result.ai_score == pytest.approx(expected)


def test_global_label_and_class_follow_threshold(artifact_factory, wav_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(10.0), 16_000)

    result = service.predict_file(path)

    expected_label = 1 if result.ai_score >= 0.0 else 0
    assert result.predicted_label == expected_label
    expected_class = "ai_generated" if expected_label == 1 else "human"
    assert result.predicted_class == expected_class


def test_result_contains_metadata_warning_timings_and_original_audio_info(
    artifact_factory,
    wav_factory,
) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(1.0, 8_000), 8_000)

    result = service.predict_file(path)

    assert result.model.model_id == "mfcc-svm-baseline"
    assert result.model.aggregation_strategy == AGGREGATION_STRATEGY
    assert result.model.score_is_calibrated_probability is False
    assert "no es una probabilidad calibrada" in result.usage_warning
    assert "no constituye un veredicto forense" in result.usage_warning
    assert result.original_sample_rate == 8_000
    assert result.audio_duration_seconds == pytest.approx(1.0, abs=1e-3)
    assert all(value >= 0.0 for value in result.timings.__dict__.values())


def test_fragment_results_are_chronological(artifact_factory, wav_factory) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(35.0), 16_000)

    result = service.predict_file(path)

    assert result.n_fragments == 4
    assert [fragment.index for fragment in result.fragments] == [0, 1, 2, 3]
    assert [fragment.start_seconds for fragment in result.fragments] == pytest.approx(
        [0.0, 10.0, 20.0, 30.0]
    )


def test_weighted_aggregation_does_not_change_individual_scores(
    artifact_factory,
    wav_factory,
) -> None:
    service = AudioInferenceService(model_path=artifact_factory())
    path = wav_factory(sine(21.0), 16_000)

    result = service.predict_file(path)
    fragment_scores = np.array([fragment.ai_score for fragment in result.fragments])
    labels_from_scores = np.where(fragment_scores >= result.decision_threshold, 1, 0)

    assert [fragment.predicted_label for fragment in result.fragments] == labels_from_scores.tolist()
    assert result.fragments[-1].duration_seconds == pytest.approx(1.0)
