from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from scripts.train_mert_svm_classifier import (  # type: ignore
        DEFAULT_CONFIG,
        MertSvmClassifierError,
        classification_metrics,
        embedding_columns,
        format_path,
        json_default,
        load_config,
        positive_decision_scores,
        save_confusion_matrix,
        split_train_val_test,
        validate_input_artifact,
        version_info,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from scripts/.
    from train_mert_svm_classifier import (  # type: ignore
        DEFAULT_CONFIG,
        MertSvmClassifierError,
        classification_metrics,
        embedding_columns,
        format_path,
        json_default,
        load_config,
        positive_decision_scores,
        save_confusion_matrix,
        split_train_val_test,
        validate_input_artifact,
        version_info,
    )


class MertSvmTestEvaluationError(MertSvmClassifierError):
    """Raised when the final MERT SVM test evaluation protocol is violated."""


def final_evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    if "final_evaluation" not in config:
        raise MertSvmTestEvaluationError("Missing final_evaluation config section")
    final_config = config["final_evaluation"]
    for section in ["protocol", "selected_candidate", "expected_validation_metrics", "outputs"]:
        if section not in final_config:
            raise MertSvmTestEvaluationError(f"Missing final_evaluation.{section} config section")
    protocol = final_config["protocol"]
    if protocol.get("retrain_with_train_val") is not False:
        raise MertSvmTestEvaluationError("final_evaluation.protocol.retrain_with_train_val must be false")
    if protocol.get("evaluation_split") != "test":
        raise MertSvmTestEvaluationError("final_evaluation.protocol.evaluation_split must be test")
    if float(protocol.get("decision_threshold", 0.0)) != 0.0:
        raise MertSvmTestEvaluationError("final_evaluation.protocol.decision_threshold must be 0.0")
    if protocol.get("evaluate_once") is not True:
        raise MertSvmTestEvaluationError("final_evaluation.protocol.evaluate_once must be true")
    return final_config


def refuse_overwrite(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(format_path(path) for path in existing)
        raise MertSvmTestEvaluationError(
            f"Final test artifacts already exist and --overwrite was not set: {formatted}"
        )


def load_selection_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MertSvmTestEvaluationError(f"Selection results JSON does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_and_validate_selected_model(
    model_path: Path,
    config: dict[str, Any],
    selection_results: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], float]:
    if not model_path.exists():
        raise MertSvmTestEvaluationError(f"Selected model artifact does not exist: {model_path}")

    load_start = time.perf_counter()
    artifact = joblib.load(model_path)
    load_seconds = time.perf_counter() - load_start
    validation = validate_selected_model_artifact(artifact, config, selection_results, model_path)
    return artifact, validation, float(load_seconds)


def validate_selected_model_artifact(
    artifact: dict[str, Any],
    config: dict[str, Any],
    selection_results: dict[str, Any],
    model_path: Path,
) -> dict[str, Any]:
    final_config = final_evaluation_config(config)
    protocol = final_config["protocol"]
    expected_candidate = final_config["selected_candidate"]
    expected_validation_metrics = final_config["expected_validation_metrics"]

    for key in ["pipeline", "feature_columns", "selected_candidate", "positive_label", "fit_split", "selection_split", "test_locked"]:
        if key not in artifact:
            raise MertSvmTestEvaluationError(f"Selected model artifact missing key: {key}")

    if artifact["fit_split"] != protocol["fit_split"]:
        raise MertSvmTestEvaluationError("Selected model must be fitted only with train")
    if artifact["selection_split"] != protocol["selection_split"]:
        raise MertSvmTestEvaluationError("Selected model must have been selected only with val")
    if artifact["test_locked"] is not True:
        raise MertSvmTestEvaluationError("Selected model artifact must keep test_locked=true")
    if int(artifact["positive_label"]) != int(protocol["positive_label"]):
        raise MertSvmTestEvaluationError("Positive label mismatch")

    feature_columns = list(artifact["feature_columns"])
    expected_feature_columns = embedding_columns(
        int(config["input"]["expected_embedding_dim"]),
        str(config["input"].get("embedding_prefix", "mert_")),
    )
    if feature_columns != expected_feature_columns:
        raise MertSvmTestEvaluationError("Selected model feature columns do not match MERT embeddings")

    pipeline = artifact["pipeline"]
    if not isinstance(pipeline, Pipeline):
        raise MertSvmTestEvaluationError("Selected model artifact pipeline is not a sklearn Pipeline")
    step_names = [name for name, _ in pipeline.steps]
    if step_names != ["scaler", "svm"]:
        raise MertSvmTestEvaluationError(f"Pipeline must contain exactly scaler and svm steps, found {step_names}")
    if not isinstance(pipeline.named_steps["scaler"], StandardScaler):
        raise MertSvmTestEvaluationError("Pipeline scaler step must be StandardScaler")
    svm = pipeline.named_steps["svm"]
    if not isinstance(svm, SVC):
        raise MertSvmTestEvaluationError("Pipeline svm step must be SVC")

    if svm.kernel != expected_candidate["kernel"]:
        raise MertSvmTestEvaluationError(f"Expected SVM kernel {expected_candidate['kernel']}, found {svm.kernel}")
    if not np.isclose(float(svm.C), float(expected_candidate["C"])):
        raise MertSvmTestEvaluationError(f"Expected SVM C={expected_candidate['C']}, found {svm.C}")
    probability_enabled = svm_probability_enabled(svm)
    if bool(expected_candidate["probability"]) or probability_enabled:
        raise MertSvmTestEvaluationError("SVC probability must remain false")
    if expected_candidate.get("gamma") not in (None, "") and svm.gamma != expected_candidate["gamma"]:
        raise MertSvmTestEvaluationError("SVM gamma mismatch")
    if getattr(svm, "classes_", None) is None or svm.classes_.tolist() != [0, 1]:
        raise MertSvmTestEvaluationError(f"Expected fitted binary SVC classes [0, 1], found {getattr(svm, 'classes_', None)}")
    if not hasattr(pipeline.named_steps["scaler"], "mean_"):
        raise MertSvmTestEvaluationError("StandardScaler does not appear to be fitted")

    if artifact["selected_candidate"] != {
        "name": expected_candidate["name"],
        "kernel": expected_candidate["kernel"],
        "C": float(expected_candidate["C"]),
        "gamma": expected_candidate.get("gamma"),
    }:
        raise MertSvmTestEvaluationError("Selected model candidate does not match final config")

    selected_from_results = selection_results.get("selected_candidate", {})
    if selected_from_results.get("candidate") != artifact["selected_candidate"]:
        raise MertSvmTestEvaluationError("Selected model candidate does not match selection results")
    if selected_from_results.get("fit_split") != protocol["fit_split"]:
        raise MertSvmTestEvaluationError("Selection results fit split must be train")
    if selected_from_results.get("selection_split") != protocol["selection_split"]:
        raise MertSvmTestEvaluationError("Selection results selection split must be val")
    if selected_from_results.get("probability") is not False:
        raise MertSvmTestEvaluationError("Selection results must record probability=false")

    for metric, expected_value in expected_validation_metrics.items():
        actual_value = selection_results.get("validation_metrics", {}).get(metric)
        if isinstance(expected_value, list):
            if actual_value != expected_value:
                raise MertSvmTestEvaluationError(f"Validation metric {metric} does not match config")
        elif not np.isclose(float(actual_value), float(expected_value), atol=1e-12):
            raise MertSvmTestEvaluationError(f"Validation metric {metric} does not match config")

    return {
        "model_path": format_path(model_path),
        "fit_split": artifact["fit_split"],
        "selection_split": artifact["selection_split"],
        "test_locked": bool(artifact["test_locked"]),
        "feature_columns": len(feature_columns),
        "positive_label": int(artifact["positive_label"]),
        "pipeline_steps": step_names,
        "svm": {
            "kernel": svm.kernel,
            "C": float(svm.C),
            "gamma": None if expected_candidate.get("gamma") in (None, "") else svm.gamma,
            "probability": False,
            "classes": svm.classes_.astype(int).tolist(),
        },
        "validation_metrics_match_selection": True,
        "model_size_bytes": int(model_path.stat().st_size),
    }


def svm_probability_enabled(svm: SVC) -> bool:
    raw_probability = getattr(svm, "probability", False)
    if raw_probability is True:
        return True
    if isinstance(raw_probability, str) and raw_probability not in {"deprecated", "False", "false"}:
        return True
    return hasattr(svm, "predict_proba")


def build_test_prediction_frame(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    decision_scores: np.ndarray,
) -> pd.DataFrame:
    output = test_frame[["id", "description", "model", "label", "split"]].copy()
    output["predicted_label"] = predictions.astype(int)
    output["decision_score"] = decision_scores.astype(float)
    output["is_correct"] = output["label"].astype(int).eq(output["predicted_label"].astype(int))
    validate_test_predictions(output, test_frame)
    return output


def validate_test_predictions(predictions: pd.DataFrame, test_frame: pd.DataFrame) -> None:
    if len(predictions) != len(test_frame):
        raise MertSvmTestEvaluationError("Test prediction row count mismatch")
    if len(predictions) != 150:
        raise MertSvmTestEvaluationError(f"Expected exactly 150 test predictions, found {len(predictions)}")
    if not predictions["split"].eq("test").all():
        raise MertSvmTestEvaluationError("Test prediction artifact must contain only split=test")
    if predictions["id"].astype(str).duplicated().any():
        raise MertSvmTestEvaluationError("Test prediction artifact contains duplicate ids")
    if set(predictions["id"].astype(str)) != set(test_frame["id"].astype(str)):
        raise MertSvmTestEvaluationError("Test prediction IDs do not match test split IDs")
    metadata_columns = ["description", "model", "label", "split"]
    pred_by_id = predictions.assign(id=predictions["id"].astype(str)).set_index("id")
    test_by_id = test_frame.assign(id=test_frame["id"].astype(str)).set_index("id")
    ordered_ids = sorted(set(test_by_id.index))
    for column in metadata_columns:
        actual = pred_by_id.loc[ordered_ids, column].astype("string").fillna("<NA>")
        expected = test_by_id.loc[ordered_ids, column].astype("string").fillna("<NA>")
        if not actual.equals(expected):
            raise MertSvmTestEvaluationError(f"Test prediction metadata mismatch for column {column}")


def validate_prediction_confusion_consistency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, Any],
) -> dict[str, int]:
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    if cm.shape != (2, 2):
        raise MertSvmTestEvaluationError(f"Expected 2x2 confusion matrix, found {cm.shape}")
    tn, fp, fn, tp = cm.ravel()
    correct = int((y_true == y_pred).sum())
    errors = int((y_true != y_pred).sum())
    if correct != int(tn + tp) or errors != int(fp + fn):
        raise MertSvmTestEvaluationError("Predictions are not coherent with confusion matrix")
    return {
        "n_correct": correct,
        "n_errors": errors,
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def ai_generator_breakdown(
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    decision_scores: np.ndarray,
) -> list[dict[str, Any]]:
    frame = build_test_prediction_frame(test_frame, predictions, decision_scores)
    ai_rows = frame[frame["label"].astype(int).eq(1)]
    breakdown = []
    for model, group in ai_rows.groupby("model", observed=False):
        n_examples = int(len(group))
        n_correct = int(group["predicted_label"].astype(int).eq(1).sum())
        breakdown.append(
            {
                "model": str(model),
                "n_examples": n_examples,
                "n_correct": n_correct,
                "recall_ai": float(n_correct / n_examples) if n_examples else float("nan"),
                "mean_decision_score": float(group["decision_score"].mean()) if n_examples else float("nan"),
            }
        )
    expected_ai = int(test_frame["label"].astype(int).eq(1).sum())
    if sum(item["n_examples"] for item in breakdown) != expected_ai:
        raise MertSvmTestEvaluationError("AI generator breakdown count does not match test AI examples")
    return sorted(breakdown, key=lambda item: item["model"])


def evaluate_test(config_path: Path = DEFAULT_CONFIG, *, overwrite: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    final_config = final_evaluation_config(config)
    output_paths = {
        "metrics": Path(final_config["outputs"]["metrics_path"]),
        "predictions": Path(final_config["outputs"]["predictions_path"]),
        "confusion_matrix": Path(final_config["outputs"]["confusion_matrix_path"]),
        "report": Path(final_config["outputs"]["report_path"]),
    }
    refuse_overwrite(list(output_paths.values()), overwrite=overwrite)

    frame, input_validation = validate_input_artifact(config)
    split_frames = split_train_val_test(frame, config)
    test_frame = split_frames["test"]
    if len(test_frame) != 150:
        raise MertSvmTestEvaluationError(f"Expected 150 test rows, found {len(test_frame)}")
    if not test_frame[str(config["input"]["split_column"])].eq(final_config["protocol"]["evaluation_split"]).all():
        raise MertSvmTestEvaluationError("Evaluation split must contain only test rows")

    outputs = config["outputs"]
    model_path = Path(outputs["model_path"])
    selection_results_path = Path(outputs["results_path"])
    selection_results = load_selection_results(selection_results_path)
    artifact, model_validation, load_seconds = load_and_validate_selected_model(
        model_path,
        config,
        selection_results,
    )

    pipeline = artifact["pipeline"]
    feature_columns = list(artifact["feature_columns"])
    X_test = test_frame[feature_columns].to_numpy(dtype=np.float32)
    y_test = test_frame[str(config["input"]["label_column"])].to_numpy(dtype=int)

    prediction_start = time.perf_counter()
    predictions = pipeline.predict(X_test).astype(int)
    decision_scores = positive_decision_scores(pipeline, X_test, int(final_config["protocol"]["positive_label"]))
    prediction_seconds = time.perf_counter() - prediction_start

    metrics = classification_metrics(y_test, predictions, decision_scores, int(final_config["protocol"]["positive_label"]))
    prediction_counts = validate_prediction_confusion_consistency(y_test, predictions, metrics)
    prediction_rows = build_test_prediction_frame(test_frame, predictions, decision_scores)
    breakdown = ai_generator_breakdown(test_frame, predictions, decision_scores)

    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    prediction_rows.to_csv(output_paths["predictions"], index=False, encoding="utf-8", lineterminator="\n")
    save_confusion_matrix(
        y_test,
        predictions,
        output_paths["confusion_matrix"],
        title="MERT + SVM test confusion matrix",
    )

    result = {
        "status": "satisfactory",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": format_path(config_path),
        "model": {
            "mert_identifier": "m-a-p/MERT-v1-95M",
            "mert_revision": "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
            "svm": model_validation["svm"],
            "decision_threshold": float(final_config["protocol"]["decision_threshold"]),
            "score": "decision_function",
        },
        "protocol": {
            "fit_split": final_config["protocol"]["fit_split"],
            "selection_split": final_config["protocol"]["selection_split"],
            "retrain_with_train_val": bool(final_config["protocol"]["retrain_with_train_val"]),
            "evaluation_split": final_config["protocol"]["evaluation_split"],
            "single_test_evaluation": bool(final_config["protocol"]["evaluate_once"]),
            "threshold_optimized_on_test": False,
            "model_modified_after_test": False,
        },
        "input_validation": input_validation,
        "model_validation": model_validation,
        "validation_context": selection_results["validation_metrics"],
        "test": {
            "n_examples": int(len(test_frame)),
            **prediction_counts,
            "metrics": metrics,
            "ai_generator_breakdown": breakdown,
        },
        "operational_metrics": {
            "classifier_load_seconds": float(load_seconds),
            "classifier_prediction_seconds_total": float(prediction_seconds),
            "classifier_latency_seconds_per_example": float(prediction_seconds / max(1, len(test_frame))),
            "model_size_bytes": int(model_validation["model_size_bytes"]),
            "note": "These metrics cover only loading and running the SVM classifier on precomputed MERT embeddings.",
        },
        "outputs": {
            "metrics": format_path(output_paths["metrics"]),
            "predictions": format_path(output_paths["predictions"]),
            "confusion_matrix": format_path(output_paths["confusion_matrix"]),
            "report": format_path(output_paths["report"]),
        },
        "selection_artifacts": {
            "model": format_path(model_path),
            "results": format_path(selection_results_path),
        },
        "versions": version_info(),
    }
    output_paths["metrics"].write_text(
        json.dumps(result, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    write_markdown_report(result, output_paths["report"])
    return result


def write_markdown_report(result: dict[str, Any], output_path: Path) -> None:
    test = result["test"]
    metrics = test["metrics"]
    op = result["operational_metrics"]
    val = result["validation_context"]
    svm = result["model"]["svm"]
    lines = [
        "# Evaluacion final MERT congelado + SVM",
        "",
        "## Objetivo",
        "",
        "Evaluar una unica vez en test la configuracion cerrada del enfoque profundo con embeddings MERT congelados y clasificador SVM lineal.",
        "",
        "## Configuracion cerrada antes de test",
        "",
        "- Pipeline: `StandardScaler` + `SVC`.",
        f"- Kernel: `{svm['kernel']}`.",
        f"- `C`: `{svm['C']:g}`.",
        f"- `probability`: `{str(svm['probability']).lower()}`.",
        "- Score: `decision_function`.",
        f"- Umbral de decision: `{result['model']['decision_threshold']}`.",
        "",
        "## Protocolo",
        "",
        "- El pipeline evaluado permanecio ajustado unicamente con `train`.",
        "- La seleccion de hiperparametros se realizo unicamente con `val`.",
        "- No se reentreno con `train + val` para mantener el mismo protocolo que el baseline clasico.",
        "- No se reajusto el escalador, no se recalibro el score y no se modifico el umbral.",
        "- No se modifico el modelo despues de observar test.",
        "",
        "## Contexto de validacion",
        "",
        f"- Balanced accuracy val: `{val['balanced_accuracy']:.4f}`.",
        f"- Precision IA val: `{val['precision_ai']:.4f}`.",
        f"- Recall IA val: `{val['recall_ai']:.4f}`.",
        f"- F1 IA val: `{val['f1_ai']:.4f}`.",
        f"- ROC-AUC val: `{val['roc_auc']:.4f}`.",
        f"- Matriz de confusion val: `{val['confusion_matrix']}`.",
        "",
        "## Resultados de test",
        "",
        f"- Ejemplos de test: `{test['n_examples']}`.",
        f"- Aciertos: `{test['n_correct']}`.",
        f"- Errores: `{test['n_errors']}`.",
        f"- Falsos positivos: `{test['false_positives']}`.",
        f"- Falsos negativos: `{test['false_negatives']}`.",
        f"- Balanced accuracy: `{metrics['balanced_accuracy']:.4f}`.",
        f"- Precision IA: `{metrics['precision_ai']:.4f}`.",
        f"- Recall IA: `{metrics['recall_ai']:.4f}`.",
        f"- F1 IA: `{metrics['f1_ai']:.4f}`.",
        f"- ROC-AUC: `{metrics['roc_auc']:.4f}`.",
        f"- Matriz de confusion test `[label 0, label 1]`: `{metrics['confusion_matrix']}`.",
        "",
        "## Desglose por generador IA",
        "",
        "| generador | ejemplos IA test | correctos como IA | recall IA | score medio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in test["ai_generator_breakdown"]:
        lines.append(
            f"| {item['model']} | {item['n_examples']} | {item['n_correct']} | "
            f"{item['recall_ai']:.4f} | {item['mean_decision_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Metricas operacionales del clasificador",
            "",
            f"- Carga del artefacto joblib: `{op['classifier_load_seconds']:.6f} s`.",
            f"- Prediccion total sobre test: `{op['classifier_prediction_seconds_total']:.6f} s`.",
            f"- Latencia media: `{op['classifier_latency_seconds_per_example']:.8f} s/ejemplo`.",
            f"- Tamano del modelo: `{op['model_size_bytes']}` bytes.",
            "",
            "Estas metricas cubren solo la carga y ejecucion del clasificador SVM sobre embeddings MERT ya calculados. No representan latencia extremo a extremo de MERT + SVM; la extraccion MERT tiene su propio resumen en `docs/mert_embedding_extraction_summary.md`.",
            "",
            "## Artefactos",
            "",
            f"- Metricas JSON: `{result['outputs']['metrics']}`.",
            f"- Predicciones test: `{result['outputs']['predictions']}`.",
            f"- Matriz de confusion test: `{result['outputs']['confusion_matrix']}`.",
            "",
            "## Limitaciones",
            "",
            "- La evaluacion es a nivel de fragmento en AIME.",
            "- El score SVM no es una probabilidad calibrada.",
            "- Este resultado no demuestra generalizacion fuera de AIME.",
            "- El desglose por generador es diagnostico y no se uso para modificar modelo, umbral ni hiperparametros.",
            "",
            "## Veredicto",
            "",
            "La evaluacion individual del enfoque profundo queda cerrada en test con la configuracion previamente fijada. El siguiente paso es realizar una comparacion formal con MFCC + SVM en una issue distinta, sin decidir todavia el modelo de despliegue.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the closed MERT SVM classifier once on AIME test.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = arg_parser().parse_args(argv)
    try:
        result = evaluate_test(args.config, overwrite=args.overwrite)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "protocol": result["protocol"],
                    "model_validation": result["model_validation"],
                    "test": result["test"],
                    "operational_metrics": result["operational_metrics"],
                    "outputs": result["outputs"],
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
