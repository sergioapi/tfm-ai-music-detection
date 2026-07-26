from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_mert_svm_test import (  # noqa: E402
    MertSvmTestEvaluationError,
    ai_generator_breakdown,
    build_test_prediction_frame,
    evaluate_test,
    final_evaluation_config,
    load_and_validate_selected_model,
    validate_prediction_confusion_consistency,
)
from scripts.train_mert_svm_classifier import (  # noqa: E402
    classification_metrics,
    embedding_columns,
    positive_decision_scores,
    split_train_val_test,
)


def tiny_eval_config(tmp_path: Path, *, expected_dim: int = 4) -> dict[str, object]:
    return {
        "experiment": {"name": "test", "phase": "test_evaluation", "seed": 42},
        "input": {
            "embeddings_path": str(tmp_path / "embeddings.parquet"),
            "expected_rows": 180,
            "expected_embedding_dim": expected_dim,
            "expected_embedding_dtype": "float32",
            "embedding_prefix": "mert_",
            "metadata_columns": ["id", "description", "model", "label", "split"],
            "label_column": "label",
            "split_column": "split",
            "id_column": "id",
            "positive_label": 1,
            "splits": {"train": "train", "validation": "val", "test": "test"},
            "expected_split_distribution": {"train": 20, "val": 10, "test": 150},
            "expected_label_distribution": {0: 90, 1: 90},
        },
        "classifier": {
            "pipeline": ["StandardScaler", "SVC"],
            "svc_probability": False,
            "candidates": [{"name": "svm_linear", "kernel": "linear", "C": [0.1]}],
        },
        "selection_rule": [{"maximize": "balanced_accuracy"}],
        "outputs": {
            "model_path": str(tmp_path / "selection_model.joblib"),
            "results_path": str(tmp_path / "selection_results.json"),
            "validation_predictions_path": str(tmp_path / "validation_predictions.csv"),
            "validation_confusion_matrix_path": str(tmp_path / "validation_confusion.png"),
            "report_path": str(tmp_path / "selection_summary.md"),
        },
        "final_evaluation": {
            "protocol": {
                "fit_split": "train",
                "selection_split": "val",
                "retrain_with_train_val": False,
                "evaluation_split": "test",
                "positive_label": 1,
                "decision_threshold": 0.0,
                "evaluate_once": True,
            },
            "selected_candidate": {
                "name": "svm_linear",
                "kernel": "linear",
                "C": 0.1,
                "gamma": None,
                "probability": False,
            },
            "expected_validation_metrics": {},
            "outputs": {
                "metrics_path": str(tmp_path / "test_metrics.json"),
                "predictions_path": str(tmp_path / "test_predictions.csv"),
                "confusion_matrix_path": str(tmp_path / "test_confusion.png"),
                "report_path": str(tmp_path / "test_summary.md"),
            },
        },
    }


def synthetic_embeddings(expected_dim: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_labels = {
        "train": [0] * 10 + [1] * 10,
        "val": [0] * 5 + [1] * 5,
        "test": [0] * 75 + [1] * 75,
    }
    index = 0
    for split, labels in split_labels.items():
        for split_index, label in enumerate(labels):
            base = -1.0 if label == 0 else 1.0
            split_offset = {"train": 0.0, "val": 0.2, "test": -0.15}[split]
            values = np.full(expected_dim, base + split_offset, dtype=np.float32)
            values += np.linspace(0.0, 0.05, expected_dim, dtype=np.float32)
            model = "MTG-Jamendo"
            if label == 1:
                model = ["GeneratorA", "GeneratorB", "GeneratorC"][split_index % 3]
            row = {
                "id": f"{index:05d}",
                "description": f"description-{split}-{split_index}",
                "model": model,
                "label": label,
                "split": split,
            }
            row.update({column: value for column, value in zip(embedding_columns(expected_dim), values)})
            rows.append(row)
            index += 1
    return pd.DataFrame(rows)


def write_embeddings(path: Path, frame: pd.DataFrame) -> None:
    frame = frame.copy()
    columns = [column for column in frame.columns if column.startswith("mert_")]
    frame[columns] = frame[columns].astype(np.float32)
    frame.to_parquet(path, index=False)


def write_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def prepare_selection_artifacts(tmp_path: Path) -> tuple[Path, dict[str, object], pd.DataFrame, dict[str, object]]:
    config = tiny_eval_config(tmp_path)
    frame = synthetic_embeddings()
    write_embeddings(Path(config["input"]["embeddings_path"]), frame)
    splits = split_train_val_test(frame, config)
    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear", C=0.1, random_state=42))])
    columns = embedding_columns(4)
    pipeline.fit(splits["train"][columns].to_numpy(dtype=np.float32), splits["train"]["label"].to_numpy(dtype=int))
    val_X = splits["val"][columns].to_numpy(dtype=np.float32)
    val_predictions = pipeline.predict(val_X).astype(int)
    val_scores = positive_decision_scores(pipeline, val_X)
    val_metrics = classification_metrics(splits["val"]["label"].to_numpy(dtype=int), val_predictions, val_scores)
    config["final_evaluation"]["expected_validation_metrics"] = val_metrics
    artifact = {
        "pipeline": pipeline,
        "feature_columns": columns,
        "metadata_columns": ["id", "description", "model", "label", "split"],
        "selected_candidate": {"name": "svm_linear", "kernel": "linear", "C": 0.1, "gamma": None},
        "positive_label": 1,
        "seed": 42,
        "fit_split": "train",
        "selection_split": "val",
        "test_locked": True,
    }
    joblib.dump(artifact, config["outputs"]["model_path"])
    selection_results = {
        "selected_candidate": {
            "candidate": artifact["selected_candidate"],
            "fit_split": "train",
            "selection_split": "val",
            "probability": False,
        },
        "validation_metrics": val_metrics,
    }
    Path(config["outputs"]["results_path"]).write_text(json.dumps(selection_results), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    write_config(config_path, config)
    return config_path, config, frame, artifact


def test_load_and_validate_selected_model_accepts_closed_artifact(tmp_path: Path) -> None:
    config_path, config, _, _ = prepare_selection_artifacts(tmp_path)
    selection_results = json.loads(Path(config["outputs"]["results_path"]).read_text(encoding="utf-8"))

    _, validation, load_seconds = load_and_validate_selected_model(
        Path(config["outputs"]["model_path"]),
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        selection_results,
    )

    assert validation["fit_split"] == "train"
    assert validation["selection_split"] == "val"
    assert validation["test_locked"] is True
    assert validation["svm"]["kernel"] == "linear"
    assert validation["svm"]["C"] == 0.1
    assert validation["svm"]["probability"] is False
    assert load_seconds >= 0.0


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("retrain_with_train_val", True),
        ("evaluation_split", "val"),
        ("decision_threshold", 0.5),
        ("evaluate_once", False),
    ],
)
def test_final_evaluation_protocol_rejects_drift(tmp_path: Path, field: str, bad_value: object) -> None:
    config = tiny_eval_config(tmp_path)
    config["final_evaluation"]["protocol"][field] = bad_value

    with pytest.raises(MertSvmTestEvaluationError, match=field):
        final_evaluation_config(config)


def test_rejects_model_not_fitted_only_with_train(tmp_path: Path) -> None:
    config_path, config, _, artifact = prepare_selection_artifacts(tmp_path)
    artifact["fit_split"] = "train_val"
    joblib.dump(artifact, config["outputs"]["model_path"])
    selection_results = json.loads(Path(config["outputs"]["results_path"]).read_text(encoding="utf-8"))

    with pytest.raises(MertSvmTestEvaluationError, match="train"):
        load_and_validate_selected_model(
            Path(config["outputs"]["model_path"]),
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            selection_results,
        )


def test_rejects_kernel_or_c_mismatch(tmp_path: Path) -> None:
    config_path, config, _, artifact = prepare_selection_artifacts(tmp_path)
    artifact["pipeline"].named_steps["svm"].C = 1.0
    joblib.dump(artifact, config["outputs"]["model_path"])
    selection_results = json.loads(Path(config["outputs"]["results_path"]).read_text(encoding="utf-8"))

    with pytest.raises(MertSvmTestEvaluationError, match="C="):
        load_and_validate_selected_model(
            Path(config["outputs"]["model_path"]),
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            selection_results,
        )


def test_rejects_probability_true(tmp_path: Path) -> None:
    config_path, config, _, artifact = prepare_selection_artifacts(tmp_path)
    artifact["pipeline"].named_steps["svm"].probability = True
    joblib.dump(artifact, config["outputs"]["model_path"])
    selection_results = json.loads(Path(config["outputs"]["results_path"]).read_text(encoding="utf-8"))

    with pytest.raises(MertSvmTestEvaluationError, match="probability"):
        load_and_validate_selected_model(
            Path(config["outputs"]["model_path"]),
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            selection_results,
        )


def test_evaluate_test_uses_only_test_rows_and_does_not_call_fit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path, config, _, _ = prepare_selection_artifacts(tmp_path)

    def forbidden_fit(self, X, y=None, **kwargs):  # noqa: ANN001
        raise AssertionError("fit must not be called during test evaluation")

    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    result = evaluate_test(config_path)

    predictions = pd.read_csv(config["final_evaluation"]["outputs"]["predictions_path"])
    assert result["test"]["n_examples"] == 150
    assert len(predictions) == 150
    assert set(predictions["split"]) == {"test"}
    assert result["protocol"]["retrain_with_train_val"] is False


def test_metrics_and_roc_auc_use_decision_function(tmp_path: Path) -> None:
    config_path, _, _, _ = prepare_selection_artifacts(tmp_path)
    result = evaluate_test(config_path)

    assert result["test"]["metrics"]["balanced_accuracy"] == 1.0
    assert result["test"]["metrics"]["roc_auc"] == 1.0
    assert result["test"]["metrics"]["confusion_matrix"] == [[75, 0], [0, 75]]


def test_prediction_frame_contains_exactly_test_predictions() -> None:
    frame = synthetic_embeddings()
    test_frame = frame[frame["split"].eq("test")].copy()
    predictions = test_frame["label"].to_numpy(dtype=int)
    scores = np.where(predictions == 1, 1.0, -1.0)

    output = build_test_prediction_frame(test_frame, predictions, scores)

    assert len(output) == 150
    assert set(output["split"]) == {"test"}
    assert not output["id"].astype(str).duplicated().any()
    assert "probability" not in output.columns


def test_prediction_confusion_consistency() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    metrics = classification_metrics(y_true, y_pred, np.array([-1.0, 0.2, -0.1, 0.9]))

    counts = validate_prediction_confusion_consistency(y_true, y_pred, metrics)

    assert counts == {"n_correct": 2, "n_errors": 2, "false_positives": 1, "false_negatives": 1}


def test_ai_generator_breakdown_counts_only_ai_rows() -> None:
    frame = synthetic_embeddings()
    test_frame = frame[frame["split"].eq("test")].copy()
    predictions = test_frame["label"].to_numpy(dtype=int)
    scores = np.where(predictions == 1, 1.0, -1.0)

    breakdown = ai_generator_breakdown(test_frame, predictions, scores)

    assert sum(item["n_examples"] for item in breakdown) == 75
    assert {item["model"] for item in breakdown} == {"GeneratorA", "GeneratorB", "GeneratorC"}
    assert all(item["recall_ai"] == 1.0 for item in breakdown)


def test_json_serialization_and_reproducible_synthetic_evaluation(tmp_path: Path) -> None:
    config_path, config, _, _ = prepare_selection_artifacts(tmp_path)

    first = evaluate_test(config_path)
    Path(config["final_evaluation"]["outputs"]["metrics_path"]).unlink()
    Path(config["final_evaluation"]["outputs"]["predictions_path"]).unlink()
    Path(config["final_evaluation"]["outputs"]["confusion_matrix_path"]).unlink()
    Path(config["final_evaluation"]["outputs"]["report_path"]).unlink()
    second = evaluate_test(config_path)

    saved = json.loads(Path(config["final_evaluation"]["outputs"]["metrics_path"]).read_text(encoding="utf-8"))
    assert saved["test"]["metrics"] == first["test"]["metrics"]
    assert first["test"]["metrics"] == second["test"]["metrics"]


def test_refuses_to_overwrite_existing_test_artifacts(tmp_path: Path) -> None:
    config_path, _, _, _ = prepare_selection_artifacts(tmp_path)
    evaluate_test(config_path)

    with pytest.raises(MertSvmTestEvaluationError, match="overwrite"):
        evaluate_test(config_path)
