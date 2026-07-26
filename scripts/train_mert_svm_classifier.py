from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
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


DEFAULT_CONFIG = Path("configs/mert_svm_classifier.yaml")
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
LATENCY_SELECTION_DECIMALS = 3


class MertSvmClassifierError(RuntimeError):
    """Raised when MERT SVM classifier selection invariants are violated."""


def load_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise MertSvmClassifierError("PyYAML is required to read the config") from exc

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise MertSvmClassifierError(f"Invalid config file: {config_path}")
    for section in ["experiment", "input", "classifier", "selection_rule", "outputs"]:
        if section not in raw:
            raise MertSvmClassifierError(f"Missing config section: {section}")
    if bool(raw["classifier"].get("svc_probability", False)):
        raise MertSvmClassifierError("SVC probability must remain false in this phase")
    raw["config_path"] = str(config_path)
    return raw


def embedding_columns(expected_dim: int, prefix: str = "mert_") -> list[str]:
    return [f"{prefix}{index:03d}" for index in range(expected_dim)]


def candidate_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for candidate in config["classifier"]["candidates"]:
        name = str(candidate["name"])
        kernel = str(candidate["kernel"])
        for c_value in candidate["C"]:
            if kernel == "linear":
                candidates.append({"name": name, "kernel": kernel, "C": float(c_value), "gamma": None})
            elif kernel == "rbf":
                for gamma in candidate["gamma"]:
                    candidates.append({"name": name, "kernel": kernel, "C": float(c_value), "gamma": gamma})
            else:
                raise MertSvmClassifierError(f"Unsupported SVM kernel: {kernel}")
    return candidates


def validate_input_artifact(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_config = config["input"]
    path = Path(input_config["embeddings_path"])
    if not path.exists():
        raise MertSvmClassifierError(f"Embedding Parquet does not exist: {path}")

    expected_dim = int(input_config["expected_embedding_dim"])
    expected_dtype = str(input_config["expected_embedding_dtype"])
    columns = embedding_columns(expected_dim, str(input_config.get("embedding_prefix", "mert_")))
    schema = pq.read_schema(path)
    schema_names = schema.names
    actual_embedding_columns = [column for column in schema_names if column.startswith(str(input_config["embedding_prefix"]))]
    if actual_embedding_columns != columns:
        raise MertSvmClassifierError(
            f"Expected exactly {expected_dim} embedding columns named {columns[0]}..{columns[-1]}"
        )
    parquet_types = sorted({str(schema.field(column).type) for column in columns})
    if expected_dtype != "float32" or parquet_types != ["float"]:
        raise MertSvmClassifierError(
            f"Expected Parquet embedding dtype float32, found {parquet_types}"
        )

    frame = pd.read_parquet(path)
    metadata_columns = list(input_config["metadata_columns"])
    missing = set([*metadata_columns, *columns]).difference(frame.columns)
    if missing:
        raise MertSvmClassifierError(f"Embedding artifact missing columns: {sorted(missing)[:10]}")
    expected_rows = int(input_config["expected_rows"])
    if len(frame) != expected_rows:
        raise MertSvmClassifierError(f"Expected {expected_rows} rows, found {len(frame)}")

    id_column = str(input_config["id_column"])
    duplicate_ids = frame[id_column].astype(str)[frame[id_column].astype(str).duplicated()].tolist()
    if duplicate_ids:
        raise MertSvmClassifierError(f"Embedding artifact contains duplicate ids: {sorted(set(duplicate_ids))[:10]}")

    embeddings = frame[columns].to_numpy(dtype=np.float32)
    if embeddings.shape != (expected_rows, expected_dim):
        raise MertSvmClassifierError(f"Expected embedding matrix {(expected_rows, expected_dim)}, found {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise MertSvmClassifierError("Embedding matrix contains NaN or infinite values")

    split_column = str(input_config["split_column"])
    label_column = str(input_config["label_column"])
    split_distribution = frame[split_column].astype(str).value_counts().sort_index().to_dict()
    expected_split_distribution = {str(key): int(value) for key, value in input_config["expected_split_distribution"].items()}
    if split_distribution != expected_split_distribution:
        raise MertSvmClassifierError(
            f"Split distribution mismatch: {split_distribution} != {expected_split_distribution}"
        )
    label_distribution = frame[label_column].astype(int).astype(str).value_counts().sort_index().to_dict()
    expected_label_distribution = {str(key): int(value) for key, value in input_config["expected_label_distribution"].items()}
    if label_distribution != expected_label_distribution:
        raise MertSvmClassifierError(
            f"Label distribution mismatch: {label_distribution} != {expected_label_distribution}"
        )

    manifest_path = input_config.get("manifest_path")
    metadata_match_manifest = None
    if manifest_path:
        manifest = pd.read_csv(Path(manifest_path), dtype={id_column: "string"})
        metadata_match_manifest = metadata_matches_manifest(frame, manifest, metadata_columns, id_column)
        if not metadata_match_manifest:
            raise MertSvmClassifierError("Embedding metadata does not match manifest")

    frame = frame.sort_values([split_column, "description", label_column, id_column]).reset_index(drop=True)
    return frame, {
        "path": format_path(path),
        "n_rows": int(len(frame)),
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "embedding_columns": int(len(columns)),
        "embedding_dtype": expected_dtype,
        "parquet_embedding_dtypes": parquet_types,
        "all_finite": True,
        "duplicate_ids": 0,
        "split_distribution": split_distribution,
        "label_distribution": label_distribution,
        "metadata_match_manifest": metadata_match_manifest,
    }


def metadata_matches_manifest(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    metadata_columns: list[str],
    id_column: str,
) -> bool:
    if len(frame) != len(manifest):
        return False
    frame_ids = set(frame[id_column].astype(str))
    manifest_ids = set(manifest[id_column].astype(str))
    if frame_ids != manifest_ids:
        return False
    frame_by_id = frame.assign(**{id_column: frame[id_column].astype(str)}).set_index(id_column)
    manifest_by_id = manifest.assign(**{id_column: manifest[id_column].astype(str)}).set_index(id_column)
    ordered_ids = sorted(manifest_ids)
    for column in metadata_columns:
        if column == id_column:
            continue
        actual = frame_by_id.loc[ordered_ids, column].astype("string").fillna("<NA>")
        expected = manifest_by_id.loc[ordered_ids, column].astype("string").fillna("<NA>")
        if not actual.equals(expected):
            return False
    return True


def split_train_val_test(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    split_column = str(config["input"]["split_column"])
    split_names = config["input"]["splits"]
    split_frames = {
        "train": frame[frame[split_column].eq(split_names["train"])].copy(),
        "val": frame[frame[split_column].eq(split_names["validation"])].copy(),
        "test": frame[frame[split_column].eq(split_names["test"])].copy(),
    }
    if any(split_frame.empty for split_frame in split_frames.values()):
        sizes = {name: int(len(split_frame)) for name, split_frame in split_frames.items()}
        raise MertSvmClassifierError(f"All train, val and test splits must be present: {sizes}")
    return split_frames


def build_pipeline(candidate: dict[str, Any], seed: int) -> Pipeline:
    svm_kwargs: dict[str, Any] = {
        "kernel": candidate["kernel"],
        "C": candidate["C"],
        "random_state": seed,
    }
    if candidate["kernel"] == "rbf":
        svm_kwargs["gamma"] = candidate["gamma"]
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC(**svm_kwargs))])


def evaluate_candidate(
    candidate: dict[str, Any],
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    seed: int,
) -> dict[str, Any]:
    pipeline = build_pipeline(candidate, seed)
    X_train = train_frame[feature_columns].to_numpy(dtype=np.float32)
    y_train = train_frame[label_column].to_numpy(dtype=int)
    X_val = val_frame[feature_columns].to_numpy(dtype=np.float32)
    y_val = val_frame[label_column].to_numpy(dtype=int)

    train_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_seconds = time.perf_counter() - train_start

    prediction_start = time.perf_counter()
    predictions = pipeline.predict(X_val).astype(int)
    decision_scores = positive_decision_scores(pipeline, X_val)
    prediction_seconds = time.perf_counter() - prediction_start

    metrics = classification_metrics(y_val, predictions, decision_scores)
    model_size_bytes = approximate_serialized_size_bytes(pipeline)
    return {
        "candidate": candidate,
        "pipeline": pipeline,
        "validation_predictions": predictions,
        "validation_decision_scores": decision_scores,
        "validation_metrics": metrics,
        "training_seconds": float(training_seconds),
        "prediction_seconds_total": float(prediction_seconds),
        "prediction_latency_seconds_per_example": float(prediction_seconds / max(1, len(val_frame))),
        "model_size_bytes": int(model_size_bytes),
        "fit_split": "train",
        "selection_split": "val",
        "probability": False,
    }


def positive_decision_scores(pipeline: Pipeline, X: np.ndarray, positive_label: int = POSITIVE_LABEL) -> np.ndarray:
    svm = pipeline.named_steps["svm"]
    scores = np.asarray(pipeline.decision_function(X), dtype=np.float64)
    classes = svm.classes_.tolist()
    if len(classes) != 2 or positive_label not in classes:
        raise MertSvmClassifierError(f"Expected binary classes including {positive_label}, found {classes}")
    positive_index = classes.index(positive_label)
    if positive_index == 0:
        scores = -scores
    return scores


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_scores: np.ndarray,
    positive_label: int = POSITIVE_LABEL,
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[NEGATIVE_LABEL, positive_label])
    try:
        roc_auc = float(roc_auc_score(y_true, decision_scores))
    except ValueError:
        roc_auc = float("nan")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_ai": float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "recall_ai": float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "f1_ai": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def select_best_candidate(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluated:
        raise MertSvmClassifierError("No candidates were evaluated")

    def score(item: dict[str, Any]) -> tuple[float, float, float, float, int, int]:
        metrics = item["validation_metrics"]
        roc_auc = metrics["roc_auc"]
        roc_auc_key = float(roc_auc) if np.isfinite(roc_auc) else float("-inf")
        prefer_linear = 1 if item["candidate"]["kernel"] == "linear" else 0
        return (
            float(metrics["balanced_accuracy"]),
            roc_auc_key,
            float(metrics["f1_ai"]),
            -round(float(item["prediction_latency_seconds_per_example"]), LATENCY_SELECTION_DECIMALS),
            prefer_linear,
            -int(item["candidate_index"]),
        )

    return max(evaluated, key=score)


def prediction_frame(
    val_frame: pd.DataFrame,
    predictions: np.ndarray,
    decision_scores: np.ndarray,
) -> pd.DataFrame:
    output = val_frame[["id", "description", "model", "label", "split"]].copy()
    output["predicted_label"] = predictions.astype(int)
    output["decision_score"] = decision_scores.astype(float)
    output["is_correct"] = output["label"].astype(int).eq(output["predicted_label"].astype(int))
    if not output["split"].eq("val").all():
        raise MertSvmClassifierError("Validation prediction artifact must contain only split=val")
    return output


def run_selection(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config["experiment"].get("seed", 42))
    feature_columns = embedding_columns(
        int(config["input"]["expected_embedding_dim"]),
        str(config["input"].get("embedding_prefix", "mert_")),
    )
    label_column = str(config["input"]["label_column"])

    frame, input_validation = validate_input_artifact(config)
    split_frames = split_train_val_test(frame, config)
    data_usage = {
        "train_examples_used_for_fit": int(len(split_frames["train"])),
        "validation_examples_used_for_selection": int(len(split_frames["val"])),
        "test_examples_loaded_for_structural_validation_only": int(len(split_frames["test"])),
        "test_examples_used_for_fit": 0,
        "test_examples_used_for_selection": 0,
        "test_predictive_metrics_computed": False,
    }

    candidates = candidate_grid(config)
    evaluated: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for index, candidate in enumerate(candidates):
        item = evaluate_candidate(
            candidate,
            split_frames["train"],
            split_frames["val"],
            feature_columns,
            label_column,
            seed,
        )
        item["candidate_index"] = index
        evaluated.append(item)
    total_seconds = time.perf_counter() - total_start
    selected = select_best_candidate(evaluated)

    outputs = config["outputs"]
    model_path = Path(outputs["model_path"])
    results_path = Path(outputs["results_path"])
    predictions_path = Path(outputs["validation_predictions_path"])
    confusion_path = Path(outputs["validation_confusion_matrix_path"])
    report_path = Path(outputs["report_path"])

    model_artifact = {
        "pipeline": selected["pipeline"],
        "feature_columns": feature_columns,
        "metadata_columns": list(config["input"]["metadata_columns"]),
        "selected_candidate": selected["candidate"],
        "positive_label": int(config["input"]["positive_label"]),
        "seed": seed,
        "fit_split": "train",
        "selection_split": "val",
        "test_locked": True,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, model_path)
    selected["model_size_bytes"] = int(model_path.stat().st_size)

    val_predictions = prediction_frame(
        split_frames["val"],
        selected["validation_predictions"],
        selected["validation_decision_scores"],
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    val_predictions.to_csv(predictions_path, index=False, encoding="utf-8", lineterminator="\n")
    if len(val_predictions) != len(split_frames["val"]):
        raise MertSvmClassifierError("Validation prediction row count mismatch")
    save_confusion_matrix(
        split_frames["val"][label_column].to_numpy(dtype=int),
        selected["validation_predictions"],
        confusion_path,
    )

    serializable_evaluated = [serializable_candidate_result(item) for item in evaluated]
    result = {
        "status": "satisfactory",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": format_path(config_path),
        "input_validation": input_validation,
        "data_usage": data_usage,
        "candidate_grid": candidates,
        "evaluated_candidates": serializable_evaluated,
        "selection_rule": config["selection_rule"],
        "selected_candidate": serializable_candidate_result(selected),
        "validation_metrics": selected["validation_metrics"],
        "timings": {
            "candidate_search_seconds_total": float(total_seconds),
            "selected_training_seconds": float(selected["training_seconds"]),
            "selected_prediction_seconds_total": float(selected["prediction_seconds_total"]),
            "selected_prediction_latency_seconds_per_example": float(
                selected["prediction_latency_seconds_per_example"]
            ),
        },
        "test_lock": {
            "test_split_present": True,
            "test_predictive_metrics_computed": False,
            "test_predictions_written": False,
            "test_confusion_matrix_written": False,
            "selection_used_test": False,
        },
        "outputs": {
            "model": format_path(model_path),
            "results": format_path(results_path),
            "validation_predictions": format_path(predictions_path),
            "validation_confusion_matrix": format_path(confusion_path),
            "report": format_path(report_path),
        },
        "versions": version_info(),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_markdown_report(result, report_path)
    return result


def serializable_candidate_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_index": int(item["candidate_index"]),
        "candidate": item["candidate"],
        "validation_metrics": item["validation_metrics"],
        "training_seconds": float(item["training_seconds"]),
        "prediction_seconds_total": float(item["prediction_seconds_total"]),
        "prediction_latency_seconds_per_example": float(item["prediction_latency_seconds_per_example"]),
        "model_size_bytes": int(item["model_size_bytes"]),
        "fit_split": item["fit_split"],
        "selection_split": item["selection_split"],
        "probability": bool(item["probability"]),
    }


def approximate_serialized_size_bytes(pipeline: Pipeline) -> int:
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    return len(buffer.getvalue())


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    *,
    title: str = "MERT + SVM validation confusion matrix",
) -> None:
    if plt is None or ConfusionMatrixDisplay is None:
        raise MertSvmClassifierError("matplotlib is required to save the confusion matrix image")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["human", "AI"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=160)
    plt.close(display.figure_)


def write_markdown_report(result: dict[str, Any], output_path: Path) -> None:
    selected = result["selected_candidate"]
    selected_metrics = selected["validation_metrics"]
    input_validation = result["input_validation"]
    data_usage = result["data_usage"]
    lines = [
        "# Seleccion MERT congelado + SVM",
        "",
        "## Objetivo",
        "",
        "Entrenar y seleccionar un clasificador supervisado sencillo sobre embeddings MERT ya extraidos, manteniendo el encoder congelado y sin recalcular embeddings.",
        "",
        "## Artefacto de entrada",
        "",
        f"- Parquet: `{input_validation['path']}`.",
        f"- Filas: `{input_validation['n_rows']}`.",
        f"- Forma de embeddings: `({input_validation['embedding_shape'][0]}, {input_validation['embedding_shape'][1]})`.",
        f"- Dtype Parquet de embeddings: `{', '.join(input_validation['parquet_embedding_dtypes'])}`.",
        f"- Valores finitos: `{str(input_validation['all_finite']).lower()}`.",
        "",
        "## Separacion train/val/test",
        "",
        f"- Train usado para ajuste: `{data_usage['train_examples_used_for_fit']}` ejemplos.",
        f"- Validacion usada para seleccion: `{data_usage['validation_examples_used_for_selection']}` ejemplos.",
        f"- Test cargado solo para validacion estructural: `{data_usage['test_examples_loaded_for_structural_validation_only']}` ejemplos.",
        "- No se calcularon metricas predictivas de test.",
        "- No se generaron predicciones ni matriz de confusion de test.",
        "",
        "## Candidatos evaluados",
        "",
        "Se evaluaron unicamente pipelines `StandardScaler` + `SVC(probability=False)` con kernel lineal o RBF.",
        "",
        "| candidato | kernel | C | gamma | balanced accuracy val | precision IA val | recall IA val | F1 IA val | ROC-AUC val | train s | pred val s | latencia val s/ej | tamano bytes |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["evaluated_candidates"]:
        candidate = item["candidate"]
        metrics = item["validation_metrics"]
        lines.append(
            "| "
            f"{candidate['name']} | {candidate['kernel']} | {candidate['C']:g} | "
            f"{_format_gamma(candidate.get('gamma'))} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['precision_ai']:.4f} | {metrics['recall_ai']:.4f} | {metrics['f1_ai']:.4f} | "
            f"{_format_float(metrics['roc_auc'])} | {item['training_seconds']:.4f} | "
            f"{item['prediction_seconds_total']:.4f} | {item['prediction_latency_seconds_per_example']:.8f} | "
            f"{item['model_size_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Regla de seleccion",
            "",
            "La configuracion se selecciono maximizando `balanced_accuracy` en validacion; en empate se uso `ROC-AUC`, despues `F1 IA`, despues menor latencia media de prediccion redondeada de forma determinista y, si persistia el empate, preferencia por el kernel lineal.",
            "",
            "## Configuracion seleccionada",
            "",
            f"- Candidato: `{selected['candidate']['name']}`.",
            f"- Kernel: `{selected['candidate']['kernel']}`.",
            f"- `C`: `{selected['candidate']['C']:g}`.",
            f"- `gamma`: `{_format_gamma(selected['candidate'].get('gamma'))}`.",
            "",
            "## Metricas de validacion",
            "",
            f"- Balanced accuracy: `{selected_metrics['balanced_accuracy']:.4f}`.",
            f"- Precision IA: `{selected_metrics['precision_ai']:.4f}`.",
            f"- Recall IA: `{selected_metrics['recall_ai']:.4f}`.",
            f"- F1 IA: `{selected_metrics['f1_ai']:.4f}`.",
            f"- ROC-AUC: `{_format_float(selected_metrics['roc_auc'])}`.",
            f"- Matriz de confusion val `[label 0, label 1]`: `{selected_metrics['confusion_matrix']}`.",
            "",
            "## Tiempos y tamano",
            "",
            f"- Tiempo total de busqueda: `{result['timings']['candidate_search_seconds_total']:.4f} s`.",
            f"- Tiempo de entrenamiento seleccionado: `{result['timings']['selected_training_seconds']:.4f} s`.",
            f"- Tiempo total de prediccion val seleccionado: `{result['timings']['selected_prediction_seconds_total']:.4f} s`.",
            f"- Latencia media val seleccionada: `{result['timings']['selected_prediction_latency_seconds_per_example']:.8f} s/ejemplo`.",
            f"- Tamano del modelo seleccionado: `{selected['model_size_bytes']}` bytes.",
            "",
            "## Artefactos",
            "",
            f"- Modelo: `{result['outputs']['model']}`.",
            f"- Resultados JSON: `{result['outputs']['results']}`.",
            f"- Predicciones de validacion: `{result['outputs']['validation_predictions']}`.",
            f"- Matriz de confusion de validacion: `{result['outputs']['validation_confusion_matrix']}`.",
            "",
            "## Limitaciones",
            "",
            "- La seleccion se basa solo en validacion y no mide todavia rendimiento final.",
            "- No se reentreno con `train + val` en esta fase.",
            "- El score SVM procede de `decision_function` y no es una probabilidad calibrada.",
            "",
            "## Siguiente paso",
            "",
            "La evaluacion final sobre test se realizo en la fase posterior y queda documentada en `docs/mert_svm_test_summary.md`. El siguiente paso es realizar una comparacion formal con MFCC + SVM en una issue distinta.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def version_info() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit_learn": sklearn.__version__ if sklearn else None,
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def format_path(path: Path) -> str:
    return path.as_posix()


def _format_float(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def _format_gamma(value: Any) -> str:
    return "-" if value is None else str(value)


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and select an SVM classifier on frozen MERT embeddings.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = arg_parser().parse_args(argv)
    try:
        result = run_selection(args.config)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "input_validation": result["input_validation"],
                    "data_usage": result["data_usage"],
                    "selected_candidate": result["selected_candidate"],
                    "timings": result["timings"],
                    "test_lock": result["test_lock"],
                },
                indent=2,
                default=json_default,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should fail loudly for automation.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
