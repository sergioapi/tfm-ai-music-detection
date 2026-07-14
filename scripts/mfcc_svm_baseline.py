from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    plt = None
    ConfusionMatrixDisplay = None

try:
    import sklearn
except ImportError:  # pragma: no cover
    sklearn = None

try:
    import datasets
except ImportError:  # pragma: no cover
    datasets = None


DATASET_NAME = "disco-eth/AIME"
DATASET_REVISION = "b84d4be5eda830b6eb714998569dba73530f2601"
DEFAULT_MANIFEST = Path("data/aime_splits.csv")
DEFAULT_FEATURES = Path("data/processed/aime_mfcc_features.parquet")
DEFAULT_FAILURES = Path("data/processed/aime_mfcc_failures.csv")
DEFAULT_EXTRACTION_SUMMARY = Path("data/processed/aime_mfcc_extraction_summary.json")
DEFAULT_MODEL = Path("data/models/mfcc_svm_baseline.joblib")
DEFAULT_METRICS = Path("data/models/mfcc_svm_metrics.json")
DEFAULT_PREDICTIONS = Path("data/models/mfcc_svm_predictions.csv")
DEFAULT_CONFUSION = Path("data/models/mfcc_svm_confusion_matrix.png")
DEFAULT_REPORT = Path("docs/mfcc_svm_baseline_summary.md")
DEFAULT_SEED = 42
SPLIT_ORDER = ("train", "val", "test")
METADATA_COLUMNS = ["id", "description", "model", "label", "split"]


@dataclass(frozen=True)
class PreprocessConfig:
    target_sample_rate: int = 16_000
    duration_seconds: float = 10.0

    @property
    def target_samples(self) -> int:
        return int(self.target_sample_rate * self.duration_seconds)


@dataclass(frozen=True)
class MfccConfig:
    n_mfcc: int = 20


class BaselineError(RuntimeError):
    """Raised when a baseline invariant is violated."""


def preprocess_audio_array(
    audio: np.ndarray,
    sample_rate: int,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    config = config or PreprocessConfig()
    if sample_rate <= 0:
        raise BaselineError(f"Invalid sample rate: {sample_rate}")

    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim == 2:
        signal = signal.mean(axis=1, dtype=np.float32)
    elif signal.ndim != 1:
        raise BaselineError(f"Expected mono or stereo audio, found shape {signal.shape}")

    if signal.size == 0:
        raise BaselineError("Audio signal is empty")
    _validate_finite(signal, "decoded audio")

    window_samples = int(round(sample_rate * config.duration_seconds))
    signal = _select_or_pad_window(signal, window_samples)

    if sample_rate != config.target_sample_rate:
        signal = librosa.resample(
            signal,
            orig_sr=sample_rate,
            target_sr=config.target_sample_rate,
        ).astype(np.float32, copy=False)

    signal = _fix_length(signal, config.target_samples)
    _validate_finite(signal, "preprocessed audio")
    return signal.astype(np.float32, copy=False)


def preprocess_audio_file(
    path: Path, config: PreprocessConfig | None = None
) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=False)
    return preprocess_audio_array(audio, sample_rate, config=config)


def _select_or_pad_window(signal: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 0:
        raise BaselineError(f"Invalid window length: {window_samples}")
    if signal.size <= window_samples:
        return _fix_length(signal, window_samples)

    squared = np.square(signal, dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(squared)])
    energies = cumulative[window_samples:] - cumulative[:-window_samples]
    best_start = int(np.argmax(energies))
    return signal[best_start : best_start + window_samples]


def _fix_length(signal: np.ndarray, target_samples: int) -> np.ndarray:
    if signal.size == target_samples:
        return signal.astype(np.float32, copy=False)
    if signal.size > target_samples:
        return signal[:target_samples].astype(np.float32, copy=False)
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: signal.size] = signal.astype(np.float32, copy=False)
    return padded


def extract_mfcc_features(
    signal: np.ndarray,
    sample_rate: int = 16_000,
    config: MfccConfig | None = None,
) -> np.ndarray:
    config = config or MfccConfig()
    mfcc = librosa.feature.mfcc(
        y=np.asarray(signal, dtype=np.float32),
        sr=sample_rate,
        n_mfcc=config.n_mfcc,
    )
    features = np.concatenate(
        [
            mfcc.mean(axis=1, dtype=np.float64),
            mfcc.std(axis=1, dtype=np.float64),
        ]
    ).astype(np.float32)
    if features.shape != (config.n_mfcc * 2,):
        raise BaselineError(f"Expected 40 MFCC features, found {features.shape}")
    _validate_finite(features, "MFCC features")
    return features


def feature_columns(config: MfccConfig | None = None) -> list[str]:
    config = config or MfccConfig()
    return [f"mfcc_mean_{i:02d}" for i in range(config.n_mfcc)] + [
        f"mfcc_std_{i:02d}" for i in range(config.n_mfcc)
    ]


def build_svm_pipeline(C: float = 1.0, gamma: str | float = "scale") -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=C, gamma=gamma, probability=False)),
        ]
    )


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype={"id": "string"})
    missing = set(METADATA_COLUMNS).difference(manifest.columns)
    if missing:
        raise BaselineError(f"Manifest missing required columns: {sorted(missing)}")
    manifest = manifest.sort_values(["split", "description", "label", "id"])
    manifest = manifest.reset_index(drop=True)
    return manifest


def validate_description_splits(frame: pd.DataFrame) -> None:
    split_counts = frame.groupby("description", observed=False)["split"].nunique()
    overlapping = split_counts[split_counts > 1]
    if not overlapping.empty:
        examples = overlapping.head(5).index.tolist()
        raise BaselineError(f"Descriptions found in multiple splits: {examples}")


def extract_features_from_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_path: Path = DEFAULT_FEATURES,
    failures_path: Path = DEFAULT_FAILURES,
    summary_path: Path = DEFAULT_EXTRACTION_SUMMARY,
    audio_dir: Path | None = None,
    dataset_name: str = DATASET_NAME,
    dataset_revision: str = DATASET_REVISION,
    overwrite: bool = False,
) -> dict[str, Any]:
    preprocess_config = PreprocessConfig()
    mfcc_config = MfccConfig()
    manifest = load_manifest(manifest_path)
    validate_description_splits(manifest)
    metadata_by_id = _metadata_by_id(manifest)

    columns = feature_columns(mfcc_config)
    existing = _load_existing_features(output_path, overwrite, columns, metadata_by_id)
    processed_ids = set(existing["id"].astype(str).tolist()) if existing is not None else set()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    start = time.perf_counter()
    pending_ids = set(metadata_by_id).difference(processed_ids)
    if audio_dir is None:
        rows, failures, stream_stats = _extract_remote_streaming_rows(
            pending_ids=pending_ids,
            metadata_by_id=metadata_by_id,
            columns=columns,
            preprocess_config=preprocess_config,
            mfcc_config=mfcc_config,
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
        )
    else:
        rows, failures, stream_stats = _extract_local_rows(
            manifest=manifest,
            processed_ids=processed_ids,
            audio_dir=audio_dir,
            columns=columns,
            preprocess_config=preprocess_config,
            mfcc_config=mfcc_config,
        )

    elapsed = time.perf_counter() - start
    new_features = pd.DataFrame(rows)
    combined = _combine_features(existing, new_features, columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    failures_frame = pd.DataFrame(
        failures,
        columns=["id", "description", "model", "split", "phase", "reason"],
    )
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_frame.to_csv(failures_path, index=False, encoding="utf-8", lineterminator="\n")

    summary = {
        "manifest_path": str(manifest_path),
        "features_path": str(output_path),
        "failures_path": str(failures_path),
        "preprocessing": _dataclass_dict(preprocess_config),
        "mfcc": _dataclass_dict(mfcc_config),
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "n_manifest_rows": int(len(manifest)),
        "n_processed_total": int(len(combined)),
        "n_processed_this_run": int(len(new_features)),
        "n_failed_this_run": int(len(failures_frame)),
        "feature_extraction_seconds": elapsed,
        **stream_stats,
    }
    _write_json(summary, summary_path)
    if failures:
        raise BaselineError(
            f"Feature extraction finished with {len(failures)} failures. "
            f"See {failures_path}."
        )
    if len(combined) != len(manifest):
        missing = len(manifest) - len(combined)
        raise BaselineError(
            f"Feature extraction incomplete: {len(combined)} processed, {missing} missing."
        )
    return summary


def train_evaluate_baseline(
    features_path: Path = DEFAULT_FEATURES,
    model_path: Path = DEFAULT_MODEL,
    metrics_path: Path = DEFAULT_METRICS,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    confusion_path: Path = DEFAULT_CONFUSION,
    report_path: Path = DEFAULT_REPORT,
    extraction_summary_path: Path = DEFAULT_EXTRACTION_SUMMARY,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    features = pd.read_parquet(features_path)
    validate_description_splits(features)
    columns = feature_columns()
    missing = set(METADATA_COLUMNS + columns).difference(features.columns)
    if missing:
        raise BaselineError(f"Features file missing columns: {sorted(missing)}")

    features = features.sort_values(["split", "description", "label", "id"]).reset_index(drop=True)
    split_frames = {split: features[features["split"].eq(split)] for split in SPLIT_ORDER}
    if any(frame.empty for frame in split_frames.values()):
        sizes = {split: int(len(frame)) for split, frame in split_frames.items()}
        raise BaselineError(f"All splits must be present in features: {sizes}")

    X_train = split_frames["train"][columns].to_numpy(dtype=np.float32)
    y_train = split_frames["train"]["label"].to_numpy(dtype=int)
    grid = [{"C": C, "gamma": gamma} for C in [0.1, 1, 10, 100] for gamma in ["scale", 0.001, 0.01, 0.1]]

    training_start = time.perf_counter()
    evaluated: list[dict[str, Any]] = []
    best_pipeline: Pipeline | None = None
    best_key: tuple[float, float] | None = None
    best_params: dict[str, Any] | None = None
    for params in grid:
        pipeline = build_svm_pipeline(**params)
        pipeline.fit(X_train, y_train)
        val_predictions, val_scores, val_seconds = _predict_split(pipeline, split_frames["val"], columns)
        val_metrics = classification_metrics(
            split_frames["val"]["label"].to_numpy(dtype=int),
            val_predictions,
            val_scores,
        )
        item = {"params": params, "validation_metrics": val_metrics, "validation_inference_seconds": val_seconds}
        evaluated.append(item)
        key = (val_metrics["balanced_accuracy"], val_metrics["f1_ai"])
        if best_key is None or key > best_key:
            best_key = key
            best_params = params
            best_pipeline = pipeline

    training_seconds = time.perf_counter() - training_start
    if best_pipeline is None or best_params is None:
        raise BaselineError("No SVM configuration was evaluated")

    val_predictions, val_scores, val_inference_seconds = _predict_split(best_pipeline, split_frames["val"], columns)
    test_predictions, test_scores, test_inference_seconds = _predict_split(best_pipeline, split_frames["test"], columns)
    val_metrics = classification_metrics(split_frames["val"]["label"].to_numpy(dtype=int), val_predictions, val_scores)
    test_metrics = classification_metrics(split_frames["test"]["label"].to_numpy(dtype=int), test_predictions, test_scores)
    generator_breakdown = ai_generator_breakdown(split_frames["test"], test_predictions, test_scores)

    model_artifact = {
        "pipeline": best_pipeline,
        "feature_columns": columns,
        "preprocessing": _dataclass_dict(PreprocessConfig()),
        "mfcc": _dataclass_dict(MfccConfig()),
        "selected_params": best_params,
        "positive_label": 1,
        "seed": seed,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, model_path)
    model_size = model_path.stat().st_size

    predictions = pd.concat(
        [
            prediction_frame(split_frames["val"], val_predictions, val_scores),
            prediction_frame(split_frames["test"], test_predictions, test_scores),
        ],
        ignore_index=True,
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False, encoding="utf-8", lineterminator="\n")
    save_confusion_matrix(split_frames["test"]["label"].to_numpy(dtype=int), test_predictions, confusion_path)

    extraction_summary = _read_json_if_exists(extraction_summary_path)
    operational_metrics = {
        "feature_extraction_seconds": extraction_summary.get("feature_extraction_seconds"),
        "training_seconds": training_seconds,
        "validation_inference_seconds": val_inference_seconds,
        "test_inference_seconds": test_inference_seconds,
        "test_latency_seconds_per_fragment": test_inference_seconds / max(1, len(split_frames["test"])),
        "model_size_bytes": int(model_size),
        "n_processed_examples": int(len(features)),
        "n_failed_examples": int(extraction_summary.get("n_failed_this_run", 0)),
        "memory_rss_bytes": None,
        "memory_note": "No registrada: no se anade dependencia nueva para medir memoria.",
    }
    results = {
        "seed": seed,
        "versions": version_info(),
        "preprocessing": _dataclass_dict(PreprocessConfig()),
        "mfcc": _dataclass_dict(MfccConfig()),
        "svm_grid": grid,
        "evaluated_hyperparameters": evaluated,
        "selected_hyperparameters": best_params,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_ai_generator_breakdown": generator_breakdown,
        "operational_metrics": operational_metrics,
        "paths": {
            "features": str(features_path),
            "model": str(model_path),
            "predictions": str(predictions_path),
            "confusion_matrix": str(confusion_path),
            "report": str(report_path),
        },
    }
    _write_json(results, metrics_path)
    write_markdown_report(results, report_path)
    return results


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = float("nan")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_ai": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_ai": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_ai": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def prediction_frame(frame: pd.DataFrame, predictions: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    output = frame[["id", "description", "model", "split", "label"]].copy()
    output["predicted_label"] = predictions.astype(int)
    output["decision_score"] = scores.astype(float)
    return output


def ai_generator_breakdown(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    frame = prediction_frame(test_frame, predictions, scores)
    ai_rows = frame[frame["label"].eq(1)]
    breakdown = []
    for model, group in ai_rows.groupby("model", observed=False):
        correct = int(group["predicted_label"].eq(1).sum())
        n = int(len(group))
        breakdown.append(
            {
                "model": model,
                "n_examples": n,
                "n_correct": correct,
                "recall_ai": float(correct / n) if n else float("nan"),
                "mean_decision_score": float(group["decision_score"].mean()) if n else float("nan"),
            }
        )
    return sorted(breakdown, key=lambda item: item["model"])


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    if plt is None or ConfusionMatrixDisplay is None:
        raise BaselineError("matplotlib is required to save the confusion matrix image")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["human", "AI"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("MFCC + SVM test confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=160)
    plt.close(display.figure_)


def write_markdown_report(results: dict[str, Any], output_path: Path) -> None:
    val = results["validation_metrics"]
    test = results["test_metrics"]
    op = results["operational_metrics"]
    selected = results["selected_hyperparameters"]
    lines = [
        "# Baseline MFCC + SVM",
        "",
        "## Que se implemento",
        "",
        "Baseline clasico binario a nivel de fragmento con preprocesamiento comun, MFCC y SVM RBF. El pipeline entrenado incluye `StandardScaler` y `SVC(kernel=\"rbf\")`.",
        "",
        "## Datos utilizados",
        "",
        "- Manifiesto: `data/aime_splits.csv`.",
        f"- Dataset: `{DATASET_NAME}`.",
        f"- Revision AIME: `{DATASET_REVISION}`.",
        "- Clases: `0` musica humana de MTG-Jamendo; `1` musica generada por IA.",
        "",
        "## Control de fuga de informacion",
        "",
        "Las particiones existentes se respetan sin regenerar el manifiesto. La seleccion de hiperparametros usa solo validacion; test se usa una unica vez para evaluacion final. No se reentrena con `train + val`.",
        "",
        "## Configuracion seleccionada",
        "",
        f"- `C`: `{selected['C']}`.",
        f"- `gamma`: `{selected['gamma']}`.",
        "",
        "## Metricas",
        "",
        "| split | balanced accuracy | precision IA | recall IA | F1 IA | ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| val | {val['balanced_accuracy']:.4f} | {val['precision_ai']:.4f} | {val['recall_ai']:.4f} | {val['f1_ai']:.4f} | {_format_float(val['roc_auc'])} |",
        f"| test | {test['balanced_accuracy']:.4f} | {test['precision_ai']:.4f} | {test['recall_ai']:.4f} | {test['f1_ai']:.4f} | {_format_float(test['roc_auc'])} |",
        "",
        "## Tiempos y artefactos",
        "",
        f"- Extraccion de caracteristicas: `{_format_optional_seconds(op['feature_extraction_seconds'])}`.",
        f"- Entrenamiento y busqueda: `{op['training_seconds']:.4f} s`.",
        f"- Inferencia validacion: `{op['validation_inference_seconds']:.4f} s`.",
        f"- Inferencia test: `{op['test_inference_seconds']:.4f} s`.",
        f"- Latencia media test: `{op['test_latency_seconds_per_fragment']:.6f} s/fragmento`.",
        f"- Tamano del modelo: `{op['model_size_bytes']}` bytes.",
        f"- Memoria RSS: `{op['memory_note']}`",
        "",
        "## Incidencias",
        "",
        f"- Ejemplos procesados: `{op['n_processed_examples']}`.",
        f"- Ejemplos fallidos registrados: `{op['n_failed_examples']}`.",
        "",
        "## Limitaciones",
        "",
        "- AIME se evalua a nivel de fragmento.",
        "- `description` es una agrupacion semantica, no un identificador verificable de cancion.",
        "- La clase humana procede unicamente de MTG-Jamendo.",
        "- La clase IA combina 12 generadores.",
        "- El score SVM no es una probabilidad.",
        "- Un buen resultado dentro de AIME no demuestra generalizacion a otros datasets o generadores.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def version_info() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "librosa": librosa.__version__,
        "soundfile": sf.__version__,
        "scikit_learn": sklearn.__version__ if sklearn else None,
        "datasets": datasets.__version__ if datasets else None,
    }


def _predict_split(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    X = frame[columns].to_numpy(dtype=np.float32)
    start = time.perf_counter()
    predictions = pipeline.predict(X)
    scores = pipeline.decision_function(X)
    elapsed = time.perf_counter() - start
    scores = np.asarray(scores, dtype=np.float64)
    if pipeline.classes_.tolist() != [0, 1]:
        positive_index = int(np.where(pipeline.classes_ == 1)[0][0])
        if positive_index == 0:
            scores = -scores
    return predictions.astype(int), scores, elapsed


def _metadata_by_id(manifest: pd.DataFrame) -> dict[str, dict[str, Any]]:
    duplicate_ids = manifest["id"][manifest["id"].astype(str).duplicated()].astype(str).tolist()
    if duplicate_ids:
        examples = sorted(set(duplicate_ids))[:10]
        raise BaselineError(f"Manifest contains duplicate ids: {examples}")
    return {
        str(record["id"]): {column: record[column] for column in METADATA_COLUMNS}
        for _, record in manifest.iterrows()
    }


def _extract_local_rows(
    manifest: pd.DataFrame,
    processed_ids: set[str],
    audio_dir: Path,
    columns: list[str],
    preprocess_config: PreprocessConfig,
    mfcc_config: MfccConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped = 0
    for index, record in manifest.iterrows():
        audio_id = str(record["id"])
        if audio_id in processed_ids:
            skipped += 1
            continue
        metadata = {column: record[column] for column in METADATA_COLUMNS}
        try:
            signal = _load_audio_for_record(record, audio_dir=audio_dir, preprocess_config=preprocess_config)
            rows.append(_feature_row(metadata, signal, columns, preprocess_config, mfcc_config))
        except Exception as exc:  # noqa: BLE001 - every failed id must be reported.
            failures.append(_failure_row(metadata, "feature_extraction", exc))
        if (index + 1) % 50 == 0:
            _print_extraction_progress(
                rows_scanned=index + 1,
                ids_found=len(rows) + len(failures),
                ids_remaining=max(0, len(manifest) - len(processed_ids) - len(rows) - len(failures)),
                examples_processed=len(rows),
                failures=len(failures),
            )
    return rows, failures, {"stream_rows_scanned": None, "n_skipped_existing": int(skipped)}


def _extract_remote_streaming_rows(
    pending_ids: set[str],
    metadata_by_id: dict[str, dict[str, Any]],
    columns: list[str],
    preprocess_config: PreprocessConfig,
    mfcc_config: MfccConfig,
    dataset_name: str,
    dataset_revision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not pending_ids:
        return [], [], {"stream_rows_scanned": 0, "stream_ids_found": 0, "stream_ids_missing": []}
    stream = _load_hf_audio_stream(dataset_name, dataset_revision)
    return _extract_remote_streaming_rows_from_iterable(
        stream=stream,
        pending_ids=pending_ids,
        metadata_by_id=metadata_by_id,
        columns=columns,
        preprocess_config=preprocess_config,
        mfcc_config=mfcc_config,
    )


def _load_hf_audio_stream(dataset_name: str, dataset_revision: str) -> Any:
    if datasets is None:
        raise BaselineError("Install `datasets` or pass --audio-dir with local audio files")
    stream = datasets.load_dataset(
        dataset_name,
        split="train",
        revision=dataset_revision,
        streaming=True,
    )
    if not hasattr(stream, "decode"):
        raise BaselineError("The installed datasets version does not expose IterableDataset.decode")
    return stream.decode(False)


def _extract_remote_streaming_rows_from_iterable(
    stream: Any,
    pending_ids: set[str],
    metadata_by_id: dict[str, dict[str, Any]],
    columns: list[str],
    preprocess_config: PreprocessConfig,
    mfcc_config: MfccConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    rows_scanned = 0
    ids_found = 0
    pending_ids = set(pending_ids)
    for row in stream:
        rows_scanned += 1
        audio_id = str(row["id"])
        if audio_id not in pending_ids:
            if rows_scanned % 250 == 0:
                _print_extraction_progress(rows_scanned, ids_found, len(pending_ids), len(rows), len(failures))
            continue

        metadata = metadata_by_id[audio_id]
        ids_found += 1
        try:
            signal = _decode_remote_audio_payload(row["audio"], preprocess_config)
            rows.append(_feature_row(metadata, signal, columns, preprocess_config, mfcc_config))
        except Exception as exc:  # noqa: BLE001 - every failed id must be reported.
            failures.append(_failure_row(metadata, "feature_extraction", exc))
        finally:
            pending_ids.remove(audio_id)
            del row

        _print_extraction_progress(rows_scanned, ids_found, len(pending_ids), len(rows), len(failures))
        if not pending_ids:
            break

    if pending_ids:
        for audio_id in sorted(pending_ids):
            failures.append(
                {
                    **metadata_by_id[audio_id],
                    "phase": "dataset_streaming",
                    "reason": f"Audio id {audio_id!r} was not found in streamed dataset",
                }
            )
    stats = {
        "stream_rows_scanned": int(rows_scanned),
        "stream_ids_found": int(ids_found),
        "stream_ids_missing": sorted(pending_ids),
    }
    return rows, failures, stats


def _feature_row(
    metadata: dict[str, Any],
    signal: np.ndarray,
    columns: list[str],
    preprocess_config: PreprocessConfig,
    mfcc_config: MfccConfig,
) -> dict[str, Any]:
    features = extract_mfcc_features(signal, preprocess_config.target_sample_rate, config=mfcc_config)
    row = {column: float(value) for column, value in zip(columns, features)}
    row.update(metadata)
    return row


def _failure_row(metadata: dict[str, Any], phase: str, exc: Exception) -> dict[str, Any]:
    return {
        "id": metadata.get("id", ""),
        "description": metadata.get("description", ""),
        "model": metadata.get("model", ""),
        "split": metadata.get("split", ""),
        "phase": phase,
        "reason": f"{type(exc).__name__}: {exc}",
    }


def _print_extraction_progress(
    rows_scanned: int,
    ids_found: int,
    ids_remaining: int,
    examples_processed: int,
    failures: int,
) -> None:
    print(
        "rows_scanned="
        f"{rows_scanned} ids_found={ids_found} ids_remaining={ids_remaining} "
        f"examples_processed={examples_processed} failures={failures}"
    )


def _load_audio_for_record(
    record: pd.Series,
    audio_dir: Path,
    preprocess_config: PreprocessConfig,
) -> np.ndarray:
    audio_id = str(record["id"])
    path = find_local_audio(audio_dir, audio_id)
    return preprocess_audio_file(path, config=preprocess_config)


def _decode_remote_audio_payload(audio: Any, preprocess_config: PreprocessConfig) -> np.ndarray:
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        return preprocess_audio_array(audio["array"], int(audio["sampling_rate"]), preprocess_config)
    if isinstance(audio, dict) and "bytes" in audio and audio["bytes"] is not None:
        import io

        decoded, sample_rate = sf.read(io.BytesIO(audio["bytes"]), always_2d=False)
        return preprocess_audio_array(decoded, sample_rate, preprocess_config)
    if isinstance(audio, dict) and "path" in audio and audio["path"]:
        try:
            path = Path(str(audio["path"]))
        except ValueError:
            path = None
        if path is not None and path.exists():
            return preprocess_audio_file(path, preprocess_config)
        try:
            from datasets.utils.file_utils import xopen

            with xopen(str(audio["path"]), "rb") as handle:
                decoded, sample_rate = sf.read(handle, always_2d=False)
            return preprocess_audio_array(decoded, sample_rate, preprocess_config)
        except Exception as exc:  # noqa: BLE001
            raise BaselineError(f"Could not read remote audio path {audio['path']!r}: {exc}") from exc

    raise BaselineError("Unsupported remote audio payload")


def find_local_audio(audio_dir: Path, audio_id: str) -> Path:
    matches = sorted(path for path in audio_dir.rglob("*") if path.is_file() and path.stem.endswith(audio_id))
    if not matches:
        matches = sorted(path for path in audio_dir.rglob(f"{audio_id}.*") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"No local audio found for id {audio_id!r} under {audio_dir}")
    return matches[0]


def _load_existing_features(
    output_path: Path,
    overwrite: bool,
    columns: list[str],
    metadata_by_id: dict[str, dict[str, Any]],
) -> pd.DataFrame | None:
    if overwrite or not output_path.exists():
        return None
    existing = pd.read_parquet(output_path)
    _validate_existing_features(existing, columns, metadata_by_id)
    return existing


def _validate_existing_features(
    existing: pd.DataFrame,
    columns: list[str],
    metadata_by_id: dict[str, dict[str, Any]],
) -> None:
    missing = set(METADATA_COLUMNS + columns).difference(existing.columns)
    if missing:
        raise BaselineError(f"Existing features file missing columns: {sorted(missing)}")

    ids = existing["id"].astype(str)
    duplicate_ids = ids[ids.duplicated()].tolist()
    if duplicate_ids:
        examples = sorted(set(duplicate_ids))[:10]
        raise BaselineError(f"Existing features contain duplicate ids: {examples}")

    unknown_ids = sorted(set(ids).difference(metadata_by_id))
    if unknown_ids:
        raise BaselineError(f"Existing features contain ids not present in manifest: {unknown_ids[:10]}")

    for _, row in existing.iterrows():
        audio_id = str(row["id"])
        expected = metadata_by_id[audio_id]
        mismatched = [
            column
            for column in METADATA_COLUMNS
            if str(row[column]) != str(expected[column])
        ]
        if mismatched:
            raise BaselineError(
                f"Existing features metadata mismatch for id {audio_id!r}: {mismatched}"
            )


def _combine_features(
    existing: pd.DataFrame | None,
    new_features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    frames = [frame for frame in [existing, new_features] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=METADATA_COLUMNS + columns)
    combined = pd.concat(frames, ignore_index=True)
    duplicate_ids = combined["id"][combined["id"].astype(str).duplicated()].astype(str).tolist()
    if duplicate_ids:
        examples = sorted(set(duplicate_ids))[:10]
        raise BaselineError(f"Combined features contain duplicate ids: {examples}")
    combined = combined.sort_values(["split", "description", "label", "id"]).reset_index(drop=True)
    return combined[METADATA_COLUMNS + columns]


def _validate_finite(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise BaselineError(f"{name} contains NaN or infinite values")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _dataclass_dict(value: Any) -> dict[str, Any]:
    return dict(value.__dict__)


def _format_float(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def _format_optional_seconds(value: float | None) -> str:
    if value is None:
        return "no disponible"
    return f"{float(value):.4f} s"


def extract_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MFCC features for the AIME manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_EXTRACTION_SUMMARY)
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the MFCC + SVM baseline.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--confusion-matrix", type=Path, default=DEFAULT_CONFUSION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--extraction-summary", type=Path, default=DEFAULT_EXTRACTION_SUMMARY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def run_extract_cli(argv: list[str] | None = None) -> None:
    args = extract_arg_parser().parse_args(argv)
    summary = extract_features_from_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        failures_path=args.failures,
        summary_path=args.summary,
        audio_dir=args.audio_dir,
        dataset_name=args.dataset_name,
        dataset_revision=args.dataset_revision,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, default=_json_default))


def run_train_cli(argv: list[str] | None = None) -> None:
    args = train_arg_parser().parse_args(argv)
    results = train_evaluate_baseline(
        features_path=args.features,
        model_path=args.model,
        metrics_path=args.metrics,
        predictions_path=args.predictions,
        confusion_path=args.confusion_matrix,
        report_path=args.report,
        extraction_summary_path=args.extraction_summary,
        seed=args.seed,
    )
    print(json.dumps(results["test_metrics"], indent=2, default=_json_default))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "extract":
        run_extract_cli(sys.argv[2:])
    elif command == "train":
        run_train_cli(sys.argv[2:])
    else:
        raise SystemExit("Usage: python scripts/mfcc_svm_baseline.py [extract|train] ...")
