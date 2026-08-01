from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = ROOT / "data" / "aime_splits.csv"
MFCC_METRICS_PATH = ROOT / "data" / "models" / "mfcc_svm_metrics.json"
MFCC_PREDICTIONS_PATH = ROOT / "data" / "models" / "mfcc_svm_predictions.csv"
MFCC_EXTRACTION_PATH = ROOT / "data" / "processed" / "aime_mfcc_extraction_summary.json"
MFCC_MODEL_PATH = ROOT / "data" / "models" / "mfcc_svm_baseline.joblib"
MERT_SELECTION_PATH = ROOT / "data" / "models" / "mert_svm_selection_results.json"
MERT_TEST_PATH = ROOT / "data" / "models" / "mert_svm_test_metrics.json"
MERT_TEST_PREDICTIONS_PATH = ROOT / "data" / "models" / "mert_svm_test_predictions.csv"
MERT_EXTRACTION_PATH = ROOT / "data" / "processed" / "aime_mert_embedding_extraction_summary.json"
MERT_SMOKE_PATH = ROOT / "data" / "processed" / "mert_smoke_test_result.json"
MERT_EMBEDDING_CONFIG_PATH = ROOT / "configs" / "mert_frozen_embeddings.yaml"
MERT_SVM_CONFIG_PATH = ROOT / "configs" / "mert_svm_classifier.yaml"
MERT_MODEL_PATH = ROOT / "data" / "models" / "mert_svm_selection_model.joblib"

OUTPUT_JSON_PATH = ROOT / "docs" / "model_comparison.json"
OUTPUT_SUMMARY_PATH = ROOT / "docs" / "model_comparison_summary.md"
OUTPUT_DECISION_PATH = ROOT / "docs" / "decisions" / "seleccion-modelo-despliegue.md"

METRIC_KEYS = ["balanced_accuracy", "precision_ai", "recall_ai", "f1_ai", "roc_auc"]
TOLERANCE = 1e-9


class ModelComparisonError(RuntimeError):
    """Raised when existing artifacts are inconsistent."""


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_no_duplicate_ids(rows: list[dict[str, str]], artifact: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        identifier = str(row["id"])
        if identifier in seen:
            duplicates.add(identifier)
        seen.add(identifier)
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        raise ModelComparisonError(f"{artifact} contains duplicate ids: {sample}")


def rows_by_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("split") == split]


def confusion_matrix(rows: list[dict[str, str]]) -> list[list[int]]:
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        true_label = int(row["label"])
        predicted_label = int(row["predicted_label"])
        if true_label not in (0, 1) or predicted_label not in (0, 1):
            raise ModelComparisonError("Only binary labels 0/1 are supported")
        matrix[true_label][predicted_label] += 1
    return matrix


def counts_from_matrix(matrix: list[list[int]]) -> dict[str, int]:
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    return {
        "n_examples": tn + fp + fn + tp,
        "n_correct": tn + tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "true_positives": tp,
    }


def metrics_from_predictions(rows: list[dict[str, str]]) -> dict[str, Any]:
    matrix = confusion_matrix(rows)
    counts = counts_from_matrix(matrix)
    tp = counts["true_positives"]
    tn = counts["true_negatives"]
    fp = counts["false_positives"]
    fn = counts["false_negatives"]
    recall_ai = tp / (tp + fn)
    recall_human = tn / (tn + fp)
    precision_ai = tp / (tp + fp)
    f1_ai = 2 * precision_ai * recall_ai / (precision_ai + recall_ai)
    return {
        "balanced_accuracy": (recall_ai + recall_human) / 2,
        "precision_ai": precision_ai,
        "recall_ai": recall_ai,
        "f1_ai": f1_ai,
        "confusion_matrix": matrix,
        **counts,
    }


def roc_auc_from_decision_scores(rows: list[dict[str, str]]) -> float:
    pairs = sorted((float(row["decision_score"]), int(row["label"])) for row in rows)
    n_pos = sum(label for _, label in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ModelComparisonError("ROC-AUC requires both classes")
    rank_sum_pos = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum_pos += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def split_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row["split"])
        counts[split] = counts.get(split, 0) + 1
    return dict(sorted(counts.items()))


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def assert_close(actual: float, expected: float, artifact: str, field: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=TOLERANCE, abs_tol=TOLERANCE):
        raise ModelComparisonError(
            f"{artifact} mismatch for {field}: found {actual!r}, expected {expected!r}"
        )


def assert_metric_block(
    reconstructed: dict[str, Any],
    expected: dict[str, Any],
    artifact: str,
    include_counts: bool = False,
) -> None:
    for key in METRIC_KEYS:
        if key in expected:
            assert_close(reconstructed[key], expected[key], artifact, key)
    if reconstructed["confusion_matrix"] != expected["confusion_matrix"]:
        raise ModelComparisonError(
            f"{artifact} confusion matrix mismatch: "
            f"found {reconstructed['confusion_matrix']}, expected {expected['confusion_matrix']}"
        )
    if include_counts:
        for key in ["n_correct", "false_positives", "false_negatives", "n_examples"]:
            if key in expected and reconstructed[key] != int(expected[key]):
                raise ModelComparisonError(
                    f"{artifact} count mismatch for {key}: "
                    f"found {reconstructed[key]}, expected {expected[key]}"
                )


def validate_prediction_columns(rows: list[dict[str, str]], artifact: str) -> None:
    required = {"id", "description", "model", "label", "split", "predicted_label", "decision_score"}
    if not rows:
        raise ModelComparisonError(f"{artifact} is empty")
    missing = required - set(rows[0])
    if missing:
        raise ModelComparisonError(f"{artifact} missing columns: {sorted(missing)}")
    probability_columns = [column for column in rows[0] if "probab" in column.lower()]
    if probability_columns:
        raise ModelComparisonError(
            f"{artifact} contains probability-like columns: {probability_columns}"
        )


def validate_manifest_alignment(
    manifest_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    artifact: str,
) -> None:
    manifest_by_id = {str(row["id"]): row for row in manifest_rows}
    for row in prediction_rows:
        identifier = str(row["id"])
        if identifier not in manifest_by_id:
            raise ModelComparisonError(f"{artifact} id {identifier} is not present in manifest")
        manifest_row = manifest_by_id[identifier]
        for column in ["label", "split", "description"]:
            if str(row[column]) != str(manifest_row[column]):
                raise ModelComparisonError(
                    f"{artifact} {identifier} {column} mismatch: "
                    f"found {row[column]!r}, expected {manifest_row[column]!r}"
                )


def metric_view(block: dict[str, Any]) -> dict[str, float]:
    return {key: float(block[key]) for key in METRIC_KEYS}


def round4(value: float) -> str:
    return f"{value:.4f}"


def bytes_to_mb(value: int | float | None) -> str:
    if value is None:
        return "no registrado"
    return f"{float(value) / 1_000_000:.2f} MB"


def build_comparison() -> dict[str, Any]:
    manifest = read_csv_rows(MANIFEST_PATH)
    mfcc_metrics = read_json(MFCC_METRICS_PATH)
    mfcc_predictions = read_csv_rows(MFCC_PREDICTIONS_PATH)
    mfcc_extraction = read_json(MFCC_EXTRACTION_PATH)
    mert_selection = read_json(MERT_SELECTION_PATH)
    mert_test = read_json(MERT_TEST_PATH)
    mert_test_predictions = read_csv_rows(MERT_TEST_PREDICTIONS_PATH)
    mert_extraction = read_json(MERT_EXTRACTION_PATH)
    mert_smoke = read_json(MERT_SMOKE_PATH)

    ensure_no_duplicate_ids(manifest, rel(MANIFEST_PATH))
    validate_prediction_columns(mfcc_predictions, rel(MFCC_PREDICTIONS_PATH))
    validate_prediction_columns(mert_test_predictions, rel(MERT_TEST_PREDICTIONS_PATH))
    ensure_no_duplicate_ids(mfcc_predictions, rel(MFCC_PREDICTIONS_PATH))
    ensure_no_duplicate_ids(mert_test_predictions, rel(MERT_TEST_PREDICTIONS_PATH))

    mfcc_val_rows = rows_by_split(mfcc_predictions, "val")
    mfcc_test_rows = rows_by_split(mfcc_predictions, "test")
    mert_test_rows = rows_by_split(mert_test_predictions, "test")
    if len(mfcc_val_rows) != 150 or len(mfcc_test_rows) != 150 or len(mert_test_rows) != 150:
        raise ModelComparisonError("Expected 150 rows for each val/test prediction split")
    if set(row["id"] for row in mfcc_test_rows) != set(row["id"] for row in mert_test_rows):
        raise ModelComparisonError("MFCC and MERT test prediction IDs do not match")

    validate_manifest_alignment(manifest, mfcc_val_rows, rel(MFCC_PREDICTIONS_PATH))
    validate_manifest_alignment(manifest, mfcc_test_rows, rel(MFCC_PREDICTIONS_PATH))
    validate_manifest_alignment(manifest, mert_test_rows, rel(MERT_TEST_PREDICTIONS_PATH))

    mfcc_by_id = {row["id"]: row for row in mfcc_test_rows}
    for row in mert_test_rows:
        other = mfcc_by_id[row["id"]]
        if row["label"] != other["label"] or row["description"] != other["description"]:
            raise ModelComparisonError("MFCC and MERT test labels/descriptions differ")

    mfcc_val_reconstructed = metrics_from_predictions(mfcc_val_rows)
    mfcc_val_reconstructed["roc_auc"] = roc_auc_from_decision_scores(mfcc_val_rows)
    mfcc_test_reconstructed = metrics_from_predictions(mfcc_test_rows)
    mfcc_test_reconstructed["roc_auc"] = roc_auc_from_decision_scores(mfcc_test_rows)
    mert_test_reconstructed = metrics_from_predictions(mert_test_rows)
    mert_test_reconstructed["roc_auc"] = roc_auc_from_decision_scores(mert_test_rows)

    assert_metric_block(
        mfcc_val_reconstructed,
        mfcc_metrics["validation_metrics"],
        rel(MFCC_METRICS_PATH) + ":validation_metrics",
    )
    assert_metric_block(
        mfcc_test_reconstructed,
        mfcc_metrics["test_metrics"],
        rel(MFCC_METRICS_PATH) + ":test_metrics",
    )
    assert_metric_block(
        mert_test_reconstructed,
        mert_test["test"]["metrics"],
        rel(MERT_TEST_PATH) + ":test.metrics",
    )

    mert_validation_metrics = mert_selection["validation_metrics"]
    if mert_validation_metrics != mert_test["validation_context"]:
        raise ModelComparisonError("MERT validation metrics differ between selection and test artifacts")

    expected_mert_counts = {
        "n_examples": mert_test["test"]["n_examples"],
        "n_correct": mert_test["test"]["n_correct"],
        "false_positives": mert_test["test"]["false_positives"],
        "false_negatives": mert_test["test"]["false_negatives"],
        **mert_test["test"]["metrics"],
    }
    assert_metric_block(
        mert_test_reconstructed,
        expected_mert_counts,
        rel(MERT_TEST_PATH) + ":test",
        include_counts=True,
    )

    mfcc_validation = metric_view(mfcc_metrics["validation_metrics"])
    mert_validation = metric_view(mert_validation_metrics)
    mfcc_test_metrics = metric_view(mfcc_metrics["test_metrics"])
    mert_test_metrics = metric_view(mert_test["test"]["metrics"])
    validation_differences = {
        key: mert_validation[key] - mfcc_validation[key] for key in METRIC_KEYS
    }
    test_differences = {key: mert_test_metrics[key] - mfcc_test_metrics[key] for key in METRIC_KEYS}

    mfcc_test_counts = counts_from_matrix(mfcc_metrics["test_metrics"]["confusion_matrix"])
    mert_test_counts = counts_from_matrix(mert_test["test"]["metrics"]["confusion_matrix"])
    if mfcc_test_counts != {
        key: mfcc_test_reconstructed[key]
        for key in ["n_examples", "n_correct", "false_positives", "false_negatives", "true_negatives", "true_positives"]
    }:
        raise ModelComparisonError("MFCC reconstructed counts do not match matrix counts")

    comparison = {
        "schema_version": "1.0",
        "sources": {
            "primary": [
                rel(MANIFEST_PATH),
                rel(MFCC_METRICS_PATH),
                rel(MFCC_PREDICTIONS_PATH),
                rel(MFCC_EXTRACTION_PATH),
                rel(MFCC_MODEL_PATH),
                rel(MERT_SELECTION_PATH),
                rel(MERT_TEST_PATH),
                rel(MERT_TEST_PREDICTIONS_PATH),
                rel(MERT_EXTRACTION_PATH),
                rel(MERT_SMOKE_PATH),
                rel(MERT_EMBEDDING_CONFIG_PATH),
                rel(MERT_SVM_CONFIG_PATH),
                rel(MERT_MODEL_PATH),
            ],
            "secondary": [
                "docs/mfcc_svm_baseline_summary.md",
                "docs/mert_embedding_extraction_summary.md",
                "docs/mert_svm_selection_summary.md",
                "docs/mert_svm_test_summary.md",
                "docs/decisions/seleccion-modelo-profundo-mert.md",
            ],
        },
        "protocol": {
            "dataset": "AIME",
            "manifest": rel(MANIFEST_PATH),
            "n_examples": len(manifest),
            "split_distribution": split_counts(manifest),
            "label_distribution": label_counts(manifest),
            "positive_label": 1,
            "test_ids_match": True,
            "test_label_alignment": True,
            "duplicates": {
                "manifest": 0,
                "mfcc_predictions": 0,
                "mert_test_predictions": 0,
            },
            "score_source": "decision_function",
            "scores_are_calibrated_probabilities": False,
            "test_used_for_selection": False,
            "retrained_with_train_val": False,
        },
        "final_configurations": {
            "mfcc_svm": {
                "name": "MFCC + StandardScaler + SVM RBF",
                "features": "MFCC mean and standard deviation",
                "n_mfcc": mfcc_metrics["mfcc"]["n_mfcc"],
                "preprocessing": mfcc_metrics["preprocessing"],
                "pipeline": ["StandardScaler", "SVC"],
                "svm": {
                    "kernel": "rbf",
                    "C": mfcc_metrics["selected_hyperparameters"]["C"],
                    "gamma": mfcc_metrics["selected_hyperparameters"]["gamma"],
                    "probability": False,
                },
            },
            "mert_svm": {
                "name": "MERT frozen embeddings + StandardScaler + SVM linear",
                "encoder": mert_test["model"]["mert_identifier"],
                "encoder_revision": mert_test["model"]["mert_revision"],
                "embedding_dim": mert_extraction["model"]["expected_embedding_dim"],
                "audio": mert_extraction["audio"],
                "pooling": {
                    "hidden_state": "last_hidden_state",
                    "window_pooling": "temporal_mean",
                    "window_aggregation": "mean",
                },
                "pipeline": ["StandardScaler", "SVC"],
                "svm": mert_test["model"]["svm"],
                "decision_threshold": mert_test["model"]["decision_threshold"],
            },
        },
        "validation_metrics": {
            "mfcc_svm": {
                **mfcc_validation,
                "confusion_matrix": mfcc_metrics["validation_metrics"]["confusion_matrix"],
            },
            "mert_svm": {
                **mert_validation,
                "confusion_matrix": mert_validation_metrics["confusion_matrix"],
            },
        },
        "test_metrics": {
            "mfcc_svm": {
                **mfcc_test_metrics,
                "confusion_matrix": mfcc_metrics["test_metrics"]["confusion_matrix"],
            },
            "mert_svm": {
                **mert_test_metrics,
                "confusion_matrix": mert_test["test"]["metrics"]["confusion_matrix"],
            },
        },
        "confusion_matrices": {
            "mfcc_svm_test": mfcc_metrics["test_metrics"]["confusion_matrix"],
            "mert_svm_validation": mert_validation_metrics["confusion_matrix"],
            "mert_svm_test": mert_test["test"]["metrics"]["confusion_matrix"],
        },
        "error_counts": {
            "mfcc_svm_test": mfcc_test_counts,
            "mert_svm_test": mert_test_counts,
        },
        "differences_mert_minus_mfcc": {
            "validation": validation_differences,
            "test": test_differences,
            "test_counts": {
                "n_correct": mert_test_counts["n_correct"] - mfcc_test_counts["n_correct"],
                "false_positives": mert_test_counts["false_positives"] - mfcc_test_counts["false_positives"],
                "false_negatives": mert_test_counts["false_negatives"] - mfcc_test_counts["false_negatives"],
            },
        },
        "operational_metrics": {
            "mfcc_svm": {
                "feature_extraction_seconds_total": mfcc_extraction["feature_extraction_seconds"],
                "training_and_search_seconds": mfcc_metrics["operational_metrics"]["training_seconds"],
                "classifier_latency_seconds_per_fragment_test": mfcc_metrics["operational_metrics"][
                    "test_latency_seconds_per_fragment"
                ],
                "classifier_prediction_seconds_test": mfcc_metrics["operational_metrics"][
                    "test_inference_seconds"
                ],
                "pipeline_joblib_size_bytes": mfcc_metrics["operational_metrics"]["model_size_bytes"],
                "memory_rss_bytes": mfcc_metrics["operational_metrics"]["memory_rss_bytes"],
                "memory_note": mfcc_metrics["operational_metrics"]["memory_note"],
                "pipeline_complexity": "low",
                "latency_scope": "SVM classifier on precomputed MFCC features; MFCC extraction not included",
            },
            "mert_svm": {
                "preprocessing_seconds_mean_per_clip": mert_extraction["timings"][
                    "preprocessing_seconds_mean"
                ],
                "encoder_inference_seconds_mean_per_10s_clip": mert_extraction["timings"][
                    "inference_clip_seconds_mean"
                ],
                "encoder_inference_seconds_mean_per_5s_window": mert_extraction["timings"][
                    "inference_window_seconds_mean"
                ],
                "processor_load_seconds": mert_extraction["timings"]["processor_load_seconds"],
                "encoder_model_load_seconds": mert_extraction["timings"]["model_load_seconds"],
                "svm_classifier_load_seconds": mert_test["operational_metrics"]["classifier_load_seconds"],
                "svm_classifier_latency_seconds_per_embedding": mert_test["operational_metrics"][
                    "classifier_latency_seconds_per_example"
                ],
                "svm_classifier_prediction_seconds_test": mert_test["operational_metrics"][
                    "classifier_prediction_seconds_total"
                ],
                "svm_joblib_size_bytes": mert_test["operational_metrics"]["model_size_bytes"],
                "svm_joblib_size_is_full_pipeline": False,
                "encoder_snapshot_size_bytes": mert_extraction["memory"]["local_snapshot_size_bytes"],
                "rss_after_encoder_load_bytes": mert_extraction["memory"]["rss_after_load_bytes"],
                "rss_peak_extraction_bytes": mert_extraction["memory"]["rss_peak_approx_bytes"],
                "rss_peak_scope": "bulk embedding extraction, streaming, decoding, preprocessing, inference and writing",
                "smoke_test_encoder_inference_seconds_per_clip": mert_smoke["timings"][
                    "inference_clip_seconds_mean"
                ],
                "pipeline_complexity": "high",
                "latency_scope": "SVM classifier latency excludes MERT encoder; encoder timings are recorded separately",
            },
        },
        "comparability_warnings": [
            "The 1693580 bytes recorded for MERT correspond only to StandardScaler + SVM, not to the full deep pipeline.",
            "The approximately 10.9 GB peak RSS belongs to bulk embedding extraction and is not the memory required for one MVP request.",
            "Remote streaming times are not comparable with local inference latency.",
            "MERT SVM latency excludes the encoder.",
            "MFCC SVM latency excludes MFCC feature extraction.",
            "There is no comparable end-to-end latency measurement per complete song.",
            "Decision scores come from decision_function and are not calibrated probabilities.",
        ],
        "data_not_available": [
            "Comparable end-to-end latency per complete song",
            "Final song-level aggregation strategy",
            "MVP maximum duration limit",
            "MVP maximum upload size",
            "MVP timeout",
            "Calibrated confidence percentage",
            "External-dataset generalization evidence",
        ],
        "selected_model": {
            "id": "mfcc_svm",
            "name": "MFCC + StandardScaler + SVM RBF",
            "selected_for": "MVP web",
        },
        "selection_reasons": [
            "Better final test balanced accuracy, precision AI, F1 AI and ROC-AUC.",
            "Two more correct predictions on AIME test.",
            "Fewer false positives on AIME test.",
            "Much smaller local joblib artifact.",
            "Fewer runtime dependencies.",
            "No deep encoder has to be loaded at API startup.",
            "Lower cold-start, memory and deployment risk.",
            "Simpler initial FastAPI and Hugging Face Spaces integration.",
            "MERT does not provide a global predictive improvement that compensates its operational cost.",
        ],
        "mert_status": {
            "failed_experiment": False,
            "cpu_viable": True,
            "better_validation": True,
            "close_test_result": True,
            "higher_test_recall_ai": True,
            "kept_as_experimental_comparison": True,
        },
        "mvp_implications": {
            "backend_preprocessing": "Reproduce the MFCC preprocessing used in the experiments.",
            "model_loading": "Load the MFCC joblib pipeline once during FastAPI startup.",
            "initial_flow": "Synchronous flow by default for simplicity, to be confirmed during MVP integration.",
            "redis": "Not included in this phase.",
            "open_limits": ["upload size", "audio duration", "timeout", "number of fragments"],
            "open_aggregation": True,
            "output_calibrated_probability": False,
            "forensic_detector_claim": False,
        },
    }

    return comparison


def metric_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_summary(comparison: dict[str, Any]) -> str:
    val = comparison["validation_metrics"]
    test = comparison["test_metrics"]
    diffs = comparison["differences_mert_minus_mfcc"]
    counts = comparison["error_counts"]
    op = comparison["operational_metrics"]

    lines: list[str] = [
        "# Comparacion de enfoques y seleccion del modelo de despliegue",
        "",
        "## 1. Objetivo y alcance",
        "",
        "Este documento compara formalmente los dos enfoques ya implementados para la deteccion binaria de musica humana frente a musica generada por IA en AIME: el baseline MFCC + StandardScaler + SVM RBF y MERT congelado + StandardScaler + SVM lineal sobre embeddings. La comparacion utiliza exclusivamente los artefactos estructurados existentes; no se reentrena, no se recalculan embeddings, no se modifican umbrales y no se ejecutan nuevas predicciones.",
        "",
        "## 2. Protocolo comun",
        "",
        f"- Manifiesto: `{comparison['protocol']['manifest']}`.",
        f"- Ejemplos: `{comparison['protocol']['n_examples']}`.",
        "- Particiones: train `700`, validacion `150`, test `150`.",
        "- Clase positiva: IA, etiqueta `1`.",
        "- Ambos enfoques usan los mismos `150` IDs de test y las etiquetas reales coinciden con el manifiesto.",
        "- Los scores proceden de `decision_function`; no son probabilidades calibradas.",
        "",
        "## 3. Configuracion final de cada enfoque",
        "",
        "- MFCC + SVM: `StandardScaler` + `SVC(kernel=\"rbf\", C=10, gamma=0.01, probability=False)`, sobre estadisticos de MFCC.",
        "- MERT + SVM: encoder `m-a-p/MERT-v1-95M` congelado, dos ventanas de `5 s`, pooling temporal del ultimo hidden state, media entre ventanas, embedding de `768` dimensiones y `StandardScaler` + `SVC(kernel=\"linear\", C=0.1, probability=False)`.",
        "",
        "## 4. Metricas de validacion",
        "",
    ]
    lines.extend(
        metric_table(
            ["Modelo", "Balanced accuracy", "Precision IA", "Recall IA", "F1 IA", "ROC-AUC"],
            [
                [
                    "MFCC + SVM",
                    round4(val["mfcc_svm"]["balanced_accuracy"]),
                    round4(val["mfcc_svm"]["precision_ai"]),
                    round4(val["mfcc_svm"]["recall_ai"]),
                    round4(val["mfcc_svm"]["f1_ai"]),
                    round4(val["mfcc_svm"]["roc_auc"]),
                ],
                [
                    "MERT + SVM",
                    round4(val["mert_svm"]["balanced_accuracy"]),
                    round4(val["mert_svm"]["precision_ai"]),
                    round4(val["mert_svm"]["recall_ai"]),
                    round4(val["mert_svm"]["f1_ai"]),
                    round4(val["mert_svm"]["roc_auc"]),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 5. Metricas de test",
            "",
        ]
    )
    lines.extend(
        metric_table(
            ["Modelo", "Balanced accuracy", "Precision IA", "Recall IA", "F1 IA", "ROC-AUC"],
            [
                [
                    "MFCC + SVM",
                    round4(test["mfcc_svm"]["balanced_accuracy"]),
                    round4(test["mfcc_svm"]["precision_ai"]),
                    round4(test["mfcc_svm"]["recall_ai"]),
                    round4(test["mfcc_svm"]["f1_ai"]),
                    round4(test["mfcc_svm"]["roc_auc"]),
                ],
                [
                    "MERT + SVM",
                    round4(test["mert_svm"]["balanced_accuracy"]),
                    round4(test["mert_svm"]["precision_ai"]),
                    round4(test["mert_svm"]["recall_ai"]),
                    round4(test["mert_svm"]["f1_ai"]),
                    round4(test["mert_svm"]["roc_auc"]),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 6. Diferencias MERT menos MFCC",
            "",
        ]
    )
    lines.extend(
        metric_table(
            ["Split", "Balanced accuracy", "Precision IA", "Recall IA", "F1 IA", "ROC-AUC"],
            [
                [
                    "Validacion",
                    round4(diffs["validation"]["balanced_accuracy"]),
                    round4(diffs["validation"]["precision_ai"]),
                    round4(diffs["validation"]["recall_ai"]),
                    round4(diffs["validation"]["f1_ai"]),
                    round4(diffs["validation"]["roc_auc"]),
                ],
                [
                    "Test",
                    round4(diffs["test"]["balanced_accuracy"]),
                    round4(diffs["test"]["precision_ai"]),
                    round4(diffs["test"]["recall_ai"]),
                    round4(diffs["test"]["f1_ai"]),
                    round4(diffs["test"]["roc_auc"]),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 7. Matrices de confusion",
            "",
            f"- MFCC + SVM test: `{test['mfcc_svm']['confusion_matrix']}`.",
            f"- MERT + SVM validacion: `{val['mert_svm']['confusion_matrix']}`.",
            f"- MERT + SVM test: `{test['mert_svm']['confusion_matrix']}`.",
            "",
            "## 8. Aciertos, falsos positivos y falsos negativos",
            "",
        ]
    )
    lines.extend(
        metric_table(
            ["Modelo", "Split", "Aciertos", "Falsos positivos", "Falsos negativos"],
            [
                [
                    "MFCC + SVM",
                    "test",
                    str(counts["mfcc_svm_test"]["n_correct"]),
                    str(counts["mfcc_svm_test"]["false_positives"]),
                    str(counts["mfcc_svm_test"]["false_negatives"]),
                ],
                [
                    "MERT + SVM",
                    "test",
                    str(counts["mert_svm_test"]["n_correct"]),
                    str(counts["mert_svm_test"]["false_positives"]),
                    str(counts["mert_svm_test"]["false_negatives"]),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 9. Comparacion operacional",
            "",
            f"- MFCC registra ` {op['mfcc_svm']['feature_extraction_seconds_total']:.4f} s` de extraccion total de caracteristicas, ` {op['mfcc_svm']['training_and_search_seconds']:.4f} s` de entrenamiento y busqueda, y ` {op['mfcc_svm']['classifier_latency_seconds_per_fragment_test']:.8f} s/fragmento` para la SVM sobre MFCC ya calculados.",
            f"- El pipeline joblib de MFCC ocupa `{op['mfcc_svm']['pipeline_joblib_size_bytes']}` bytes ({bytes_to_mb(op['mfcc_svm']['pipeline_joblib_size_bytes'])}). La memoria RSS del baseline no esta registrada.",
            f"- MERT registra ` {op['mert_svm']['preprocessing_seconds_mean_per_clip']:.4f} s/clip` de preprocesamiento medio y ` {op['mert_svm']['encoder_inference_seconds_mean_per_10s_clip']:.4f} s/clip` de inferencia aproximada del encoder para clips de 10 segundos.",
            f"- La carga registrada de MERT es ` {op['mert_svm']['processor_load_seconds']:.4f} s` para el procesador y ` {op['mert_svm']['encoder_model_load_seconds']:.4f} s` para el encoder.",
            f"- La SVM de MERT sobre embeddings ya calculados registra ` {op['mert_svm']['svm_classifier_latency_seconds_per_embedding']:.8f} s/ejemplo` y su joblib ocupa `{op['mert_svm']['svm_joblib_size_bytes']}` bytes ({bytes_to_mb(op['mert_svm']['svm_joblib_size_bytes'])}).",
            f"- El snapshot local aproximado del encoder MERT ocupa `{op['mert_svm']['encoder_snapshot_size_bytes']}` bytes ({bytes_to_mb(op['mert_svm']['encoder_snapshot_size_bytes'])}) y el RSS despues de cargar el encoder fue `{op['mert_svm']['rss_after_encoder_load_bytes']}` bytes.",
            "",
            "Advertencias de comparabilidad:",
            "",
            "- Los `1693580` bytes de MERT corresponden solo al `StandardScaler` + SVM, no al pipeline profundo completo.",
            "- El pico RSS aproximado de `10.9 GB` corresponde a la extraccion masiva y no representa la memoria necesaria para una peticion del MVP.",
            "- Los tiempos con streaming remoto no son comparables con inferencia local.",
            "- La latencia de la SVM de MERT no incluye el encoder.",
            "- La latencia de la SVM del baseline no incluye la extraccion de MFCC.",
            "- No existe todavia una medicion extremo a extremo comparable por cancion.",
            "",
            "## 10. Interpretacion",
            "",
            "MERT fue mejor en validacion: mejora la balanced accuracy, el recall IA, el F1 IA y el ROC-AUC frente al baseline. Sin embargo, esa ventaja no se mantuvo en test. En la particion final, MFCC + SVM obtiene mejor balanced accuracy, precision IA, F1 IA y ROC-AUC. MERT obtiene un recall IA ligeramente superior, pero tambien produce mas falsos positivos.",
            "",
            "MFCC + SVM alcanza `125` aciertos sobre `150`, mientras que MERT + SVM alcanza `123`. Las diferencias predictivas son pequenas y no se ha realizado una prueba inferencial, por lo que no se afirma significacion estadistica. La lectura prudente es que MFCC rindio ligeramente mejor dentro del test de AIME, no que generalice mejor fuera de ese contexto.",
            "",
            "Operacionalmente, MERT introduce mayor tamano, mas dependencias y mayor complejidad de despliegue por la necesidad de cargar y ejecutar un encoder profundo. Esta complejidad no queda compensada por una mejora predictiva global en test.",
            "",
            "## 11. Limitaciones y amenazas a la validez",
            "",
            "- Los resultados corresponden a clips AIME de 10 segundos.",
            "- La generalizacion a canciones completas, otros datasets o generadores no esta garantizada.",
            "- Los scores no son probabilidades calibradas.",
            "- No hay medicion comparable extremo a extremo por cancion.",
            "- La agregacion de fragmentos para el MVP permanece abierta.",
            "- Los limites de duracion, tamano y timeout permanecen pendientes de pruebas de integracion.",
            "",
            "## 12. Conclusion comparativa",
            "",
            "La evidencia disponible permite seleccionar `MFCC + StandardScaler + SVM RBF` como modelo inicial para el MVP web. La seleccion se apoya en el resultado final de test y en la menor complejidad operacional. MERT no se considera un experimento fallido: fue viable en CPU, obtuvo mejores resultados en validacion, alcanzo resultados proximos al baseline en test y mejoro ligeramente el recall IA. Se conserva como parte de la comparacion experimental del TFM, pero no se selecciona para el despliegue inicial.",
            "",
        ]
    )
    return "\n".join(lines)


def render_decision(comparison: dict[str, Any]) -> str:
    test = comparison["test_metrics"]
    counts = comparison["error_counts"]
    op = comparison["operational_metrics"]
    return "\n".join(
        [
            "# Seleccion del modelo de despliegue para el MVP",
            "",
            "- Estado: aceptada para el MVP web inicial.",
            "- Issue relacionada: \"Comparar enfoques y seleccionar el modelo de despliegue\".",
            "- Fase cubierta: comparacion de resultados existentes y decision de integracion.",
            "",
            "## Contexto",
            "",
            "El TFM dispone de dos enfoques ya implementados y evaluados sobre el mismo manifiesto AIME: `MFCC + StandardScaler + SVM RBF` y `MERT congelado + StandardScaler + SVM lineal` sobre embeddings. La seleccion para el MVP debe basarse exclusivamente en la evidencia existente, sin reentrenar, recalcular embeddings, optimizar umbrales ni consultar nuevas metricas experimentales.",
            "",
            "La particion de test contiene 150 clips balanceados y se utiliza como evidencia principal de rendimiento final. La validacion se usa para explicar la seleccion de configuraciones, no para sustituir el resultado de test.",
            "",
            "## Alternativas consideradas",
            "",
            "1. `MFCC + StandardScaler + SVM RBF`, con `C=10` y `gamma=0.01`.",
            "2. `m-a-p/MERT-v1-95M` congelado + `StandardScaler + SVM lineal`, con `C=0.1`.",
            "",
            "## Evidencia predictiva",
            "",
            f"- MFCC + SVM test: balanced accuracy `{test['mfcc_svm']['balanced_accuracy']:.4f}`, precision IA `{test['mfcc_svm']['precision_ai']:.4f}`, recall IA `{test['mfcc_svm']['recall_ai']:.4f}`, F1 IA `{test['mfcc_svm']['f1_ai']:.4f}`, ROC-AUC `{test['mfcc_svm']['roc_auc']:.4f}`, matriz `{test['mfcc_svm']['confusion_matrix']}`.",
            f"- MERT + SVM test: balanced accuracy `{test['mert_svm']['balanced_accuracy']:.4f}`, precision IA `{test['mert_svm']['precision_ai']:.4f}`, recall IA `{test['mert_svm']['recall_ai']:.4f}`, F1 IA `{test['mert_svm']['f1_ai']:.4f}`, ROC-AUC `{test['mert_svm']['roc_auc']:.4f}`, matriz `{test['mert_svm']['confusion_matrix']}`.",
            f"- MFCC obtiene `{counts['mfcc_svm_test']['n_correct']}` aciertos, `{counts['mfcc_svm_test']['false_positives']}` falsos positivos y `{counts['mfcc_svm_test']['false_negatives']}` falsos negativos.",
            f"- MERT obtiene `{counts['mert_svm_test']['n_correct']}` aciertos, `{counts['mert_svm_test']['false_positives']}` falsos positivos y `{counts['mert_svm_test']['false_negatives']}` falsos negativos.",
            "- MERT fue mejor en validacion y obtiene un recall IA ligeramente superior en test, pero la ventaja de validacion no se mantiene en el resultado final.",
            "",
            "## Evidencia operacional",
            "",
            f"- El joblib de MFCC ocupa `{op['mfcc_svm']['pipeline_joblib_size_bytes']}` bytes y evita cargar un encoder profundo.",
            f"- La SVM de MERT ocupa `{op['mert_svm']['svm_joblib_size_bytes']}` bytes, pero ese valor corresponde solo al clasificador sobre embeddings y no al pipeline completo.",
            f"- El snapshot local aproximado del encoder MERT ocupa `{op['mert_svm']['encoder_snapshot_size_bytes']}` bytes.",
            f"- La inferencia registrada del encoder MERT es aproximadamente `{op['mert_svm']['encoder_inference_seconds_mean_per_10s_clip']:.4f} s/clip` de 10 segundos en CPU.",
            "- No existe todavia una medicion extremo a extremo comparable por cancion.",
            "- Los tiempos con streaming remoto no son comparables con inferencia local.",
            "",
            "## Decision",
            "",
            "Se selecciona `MFCC + StandardScaler + SVM RBF` como modelo inicial para el MVP web.",
            "",
            "## Justificacion",
            "",
            "MFCC + SVM se selecciona porque obtiene mejores resultados en la mayoria de las metricas finales de test, logra dos aciertos mas, produce menos falsos positivos, presenta un artefacto mucho mas pequeno, tiene menos dependencias, evita cargar un encoder profundo y reduce el riesgo de cold start, memoria y despliegue. Para la integracion inicial con FastAPI y Hugging Face Spaces, MERT no aporta una mejora predictiva global que compense su mayor coste operacional.",
            "",
            "Esta decision no afirma que MFCC generalice mejor. Los datos solo permiten afirmar que rindio ligeramente mejor dentro del test de AIME utilizado en el protocolo experimental.",
            "",
            "## Consecuencias",
            "",
            "- El backend reproducira exactamente el preprocesamiento MFCC usado en los experimentos.",
            "- El pipeline joblib se cargara una vez durante el arranque de FastAPI.",
            "- El flujo sincrono sera la opcion inicial por simplicidad, pendiente de confirmacion durante la integracion.",
            "- Redis no se incorpora en esta fase.",
            "- La salida no se presentara como probabilidad calibrada.",
            "- La interfaz debera mostrar una estimacion o puntuacion acompanada de advertencias.",
            "- El sistema no se presentara como detector forense.",
            "",
            "## Riesgos",
            "",
            "- Los resultados proceden de clips AIME de 10 segundos.",
            "- El comportamiento sobre canciones completas requiere una estrategia de agregacion aun no definida.",
            "- La generalizacion a otros datasets, generadores o condiciones de audio no esta demostrada.",
            "- El preprocesamiento del MVP debe reproducir fielmente el pipeline experimental para evitar desviaciones.",
            "",
            "## Limitaciones",
            "",
            "- No se ha realizado una prueba inferencial de significacion estadistica.",
            "- Los scores proceden de `decision_function` y no son probabilidades calibradas.",
            "- No hay benchmark extremo a extremo del MVP.",
            "- No se fijan todavia duracion maxima, tamano maximo, timeout, numero de fragmentos ni metodo definitivo de agregacion.",
            "",
            "## Cuestiones abiertas",
            "",
            "- Limites de tamano y duracion de subida.",
            "- Timeout aceptable para la experiencia web.",
            "- Numero de fragmentos por cancion.",
            "- Estrategia de agregacion por cancion.",
            "- Validacion operacional real en Hugging Face Spaces.",
            "",
            "## Condiciones para revisar la decision",
            "",
            "La decision debera revisarse si una evaluacion posterior sobre datos externos favorece claramente a MERT, si el MVP exige maximizar recall IA por encima del resto de metricas, si se obtiene una medicion extremo a extremo donde MERT sea operacionalmente viable sin degradar la experiencia web, o si el baseline MFCC muestra fallos sistematicos durante la integracion.",
            "",
            "## Estado de MERT",
            "",
            "MERT no se considera un experimento fallido. Fue viable en CPU, obtuvo mejores resultados en validacion, alcanzo resultados proximos al baseline en test y mejoro ligeramente el recall IA. Se conserva como parte de la comparacion experimental del TFM y como referencia para trabajos posteriores.",
            "",
        ]
    )


def write_outputs(comparison: dict[str, Any]) -> None:
    OUTPUT_JSON_PATH.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_SUMMARY_PATH.write_text(render_summary(comparison), encoding="utf-8")
    OUTPUT_DECISION_PATH.write_text(render_decision(comparison), encoding="utf-8")


def main() -> None:
    comparison = build_comparison()
    write_outputs(comparison)
    print(f"Wrote {rel(OUTPUT_JSON_PATH)}")
    print(f"Wrote {rel(OUTPUT_SUMMARY_PATH)}")
    print(f"Wrote {rel(OUTPUT_DECISION_PATH)}")


if __name__ == "__main__":
    main()
