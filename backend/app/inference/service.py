from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from app.inference.aggregation import aggregate_duration_weighted_scores
from app.inference.audio import decode_audio_file, preprocess_fragment, segment_audio
from app.inference.config import InferenceConfig, class_name_for_label
from app.inference.errors import ModelArtifactError, PredictionError
from app.inference.features import extract_mfcc_features
from app.inference.model import MfccSvmModel, predict_labels_from_scores
from app.inference.schemas import FragmentPrediction, InferenceTimings, PredictionResult


class AudioInferenceService:
    def __init__(
        self,
        model_path: Path | None = None,
        config: InferenceConfig | None = None,
    ) -> None:
        self.config = config or InferenceConfig()
        resolved_model_path = self._resolve_model_path(model_path)
        self.model = MfccSvmModel.load(resolved_model_path, self.config)

    def predict_file(self, path: str | Path) -> PredictionResult:
        total_start = time.perf_counter()

        decode_start = time.perf_counter()
        audio, sample_rate = decode_audio_file(Path(path))
        decode_seconds = _elapsed(decode_start)
        audio_duration_seconds = audio.shape[0] / sample_rate

        segmentation_start = time.perf_counter()
        fragments = segment_audio(
            audio,
            sample_rate,
            self.config.fragment_duration_seconds,
        )
        segmentation_seconds = _elapsed(segmentation_start)

        preprocessing_start = time.perf_counter()
        preprocessed = tuple(
            preprocess_fragment(fragment.signal, fragment.sample_rate, self.config)
            for fragment in fragments
        )
        preprocessing_seconds = _elapsed(preprocessing_start)

        mfcc_start = time.perf_counter()
        feature_rows = tuple(
            extract_mfcc_features(
                signal,
                self.config.target_sample_rate,
                self.config,
            )
            for signal in preprocessed
        )
        features = np.vstack(feature_rows).astype(np.float32, copy=False)
        mfcc_seconds = _elapsed(mfcc_start)

        prediction_start = time.perf_counter()
        fragment_scores = self.model.predict_scores(features)
        fragment_labels = predict_labels_from_scores(
            fragment_scores,
            threshold=self.config.decision_threshold,
        )
        prediction_seconds = _elapsed(prediction_start)

        aggregation_start = time.perf_counter()
        fragment_durations = np.asarray(
            [fragment.duration_seconds for fragment in fragments],
            dtype=np.float64,
        )
        global_score = aggregate_duration_weighted_scores(
            fragment_scores,
            fragment_durations,
        )
        global_label = int(
            predict_labels_from_scores(
                np.asarray([global_score], dtype=np.float64),
                threshold=self.config.decision_threshold,
            )[0]
        )
        aggregation_seconds = _elapsed(aggregation_start)

        fragment_predictions = tuple(
            FragmentPrediction(
                index=fragment.index,
                start_seconds=fragment.start_seconds,
                end_seconds=fragment.end_seconds,
                duration_seconds=fragment.duration_seconds,
                ai_score=float(score),
                predicted_label=int(label),
                predicted_class=class_name_for_label(int(label)),
                was_padded=fragment.is_incomplete,
            )
            for fragment, score, label in zip(fragments, fragment_scores, fragment_labels)
        )

        total_seconds = _elapsed(total_start)
        timings = InferenceTimings(
            decode_seconds=decode_seconds,
            segmentation_seconds=segmentation_seconds,
            preprocessing_seconds=preprocessing_seconds,
            mfcc_seconds=mfcc_seconds,
            prediction_seconds=prediction_seconds,
            aggregation_seconds=aggregation_seconds,
            total_seconds=total_seconds,
        )
        _validate_timings(timings)

        return PredictionResult(
            predicted_label=global_label,
            predicted_class=class_name_for_label(global_label),
            ai_score=global_score,
            decision_threshold=self.config.decision_threshold,
            audio_duration_seconds=audio_duration_seconds,
            original_sample_rate=sample_rate,
            n_fragments=len(fragment_predictions),
            fragments=fragment_predictions,
            timings=timings,
            model=self.model.metadata,
            usage_warning=self.config.usage_warning,
        )

    @staticmethod
    def _resolve_model_path(model_path: Path | None) -> Path:
        candidate = model_path
        if candidate is None:
            env_path = os.environ.get("MODEL_PATH")
            if not env_path:
                raise ModelArtifactError(
                    "Model path was not provided and MODEL_PATH is not set"
                )
            candidate = Path(env_path)
        return Path(candidate).expanduser().resolve()


def _elapsed(start: float) -> float:
    return max(0.0, float(time.perf_counter() - start))


def _validate_timings(timings: InferenceTimings) -> None:
    for value in timings.__dict__.values():
        if not isinstance(value, float) or value < 0.0:
            raise PredictionError(f"Invalid inference timing value: {value!r}")
