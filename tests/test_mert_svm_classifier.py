from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_mert_svm_classifier import (  # noqa: E402
    MertSvmClassifierError,
    candidate_grid,
    classification_metrics,
    embedding_columns,
    evaluate_candidate,
    load_config,
    positive_decision_scores,
    prediction_frame,
    run_selection,
    select_best_candidate,
    split_train_val_test,
    validate_input_artifact,
)


def tiny_config(tmp_path: Path, *, expected_dim: int = 4) -> dict[str, object]:
    return {
        "experiment": {"name": "test", "phase": "selection", "seed": 42},
        "input": {
            "embeddings_path": str(tmp_path / "embeddings.parquet"),
            "expected_rows": 12,
            "expected_embedding_dim": expected_dim,
            "expected_embedding_dtype": "float32",
            "embedding_prefix": "mert_",
            "metadata_columns": ["id", "description", "model", "label", "split"],
            "label_column": "label",
            "split_column": "split",
            "id_column": "id",
            "positive_label": 1,
            "splits": {"train": "train", "validation": "val", "test": "test"},
            "expected_split_distribution": {"train": 6, "val": 4, "test": 2},
            "expected_label_distribution": {0: 6, 1: 6},
        },
        "classifier": {
            "pipeline": ["StandardScaler", "SVC"],
            "svc_probability": False,
            "candidates": [
                {"name": "svm_linear", "kernel": "linear", "C": [0.1, 1]},
                {"name": "svm_rbf", "kernel": "rbf", "C": [1], "gamma": ["scale", 0.01]},
            ],
        },
        "selection_rule": [
            {"maximize": "balanced_accuracy"},
            {"maximize": "roc_auc"},
            {"maximize": "f1_ai"},
            {"minimize": "prediction_latency_seconds_per_example"},
            {"prefer": "linear"},
        ],
        "outputs": {
            "model_path": str(tmp_path / "model.joblib"),
            "results_path": str(tmp_path / "results.json"),
            "validation_predictions_path": str(tmp_path / "val_predictions.csv"),
            "validation_confusion_matrix_path": str(tmp_path / "val_confusion.png"),
            "report_path": str(tmp_path / "summary.md"),
        },
    }


def write_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def synthetic_embeddings(expected_dim: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_labels = {
        "train": [0, 0, 0, 1, 1, 1],
        "val": [0, 0, 1, 1],
        "test": [0, 1],
    }
    index = 0
    for split, labels in split_labels.items():
        for label in labels:
            base = -1.0 if label == 0 else 1.0
            split_offset = {"train": 0.0, "val": 0.3, "test": -0.4}[split]
            values = np.full(expected_dim, base, dtype=np.float32)
            values += np.float32(split_offset)
            values += np.linspace(0.0, 0.1, expected_dim, dtype=np.float32)
            row = {
                "id": f"{index:05d}",
                "description": f"description-{split}-{index}",
                "model": "MTG-Jamendo" if label == 0 else "Generator",
                "label": label,
                "split": split,
            }
            row.update({column: value for column, value in zip(embedding_columns(expected_dim), values)})
            rows.append(row)
            index += 1
    return pd.DataFrame(rows)


def write_embeddings(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    embedding_cols = [column for column in frame.columns if column.startswith("mert_")]
    frame = frame.copy()
    frame[embedding_cols] = frame[embedding_cols].astype(np.float32)
    frame.to_parquet(path, index=False)


def test_load_config_reads_machine_readable_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    config = tiny_config(tmp_path)
    write_config(path, config)

    loaded = load_config(path)

    assert loaded["input"]["expected_embedding_dim"] == 4
    assert loaded["classifier"]["svc_probability"] is False
    assert loaded["config_path"] == str(path)


def test_load_config_rejects_probability_true(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    config = tiny_config(tmp_path)
    config["classifier"]["svc_probability"] = True
    write_config(path, config)

    with pytest.raises(MertSvmClassifierError, match="probability"):
        load_config(path)


def test_embedding_columns_identifies_exact_768_columns() -> None:
    columns = embedding_columns(768)

    assert len(columns) == 768
    assert columns[0] == "mert_000"
    assert columns[-1] == "mert_767"


def test_validate_input_artifact_checks_dtype_finitude_and_distributions(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    frame = synthetic_embeddings()
    write_embeddings(Path(config["input"]["embeddings_path"]), frame)

    loaded_frame, validation = validate_input_artifact(config)

    assert len(loaded_frame) == 12
    assert validation["embedding_shape"] == [12, 4]
    assert validation["parquet_embedding_dtypes"] == ["float"]
    assert validation["all_finite"] is True
    assert validation["split_distribution"] == {"test": 2, "train": 6, "val": 4}
    assert validation["label_distribution"] == {"0": 6, "1": 6}


def test_validate_input_artifact_rejects_wrong_dtype(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    frame = synthetic_embeddings()
    embedding_cols = embedding_columns(4)
    frame[embedding_cols] = frame[embedding_cols].astype(np.float64)
    frame.to_parquet(Path(config["input"]["embeddings_path"]), index=False)

    with pytest.raises(MertSvmClassifierError, match="float32"):
        validate_input_artifact(config)


def test_validate_input_artifact_rejects_non_finite_values(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    frame = synthetic_embeddings()
    frame.loc[0, "mert_000"] = np.inf
    write_embeddings(Path(config["input"]["embeddings_path"]), frame)

    with pytest.raises(MertSvmClassifierError, match="NaN or infinite"):
        validate_input_artifact(config)


def test_split_train_val_test_keeps_test_separate(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    splits = split_train_val_test(synthetic_embeddings(), config)

    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 4
    assert len(splits["test"]) == 2
    assert set(splits["val"]["split"]) == {"val"}


def test_candidate_grid_contains_only_linear_and_rbf(tmp_path: Path) -> None:
    grid = candidate_grid(tiny_config(tmp_path))

    assert len(grid) == 4
    assert {candidate["kernel"] for candidate in grid} == {"linear", "rbf"}
    assert all(candidate.get("probability") is None for candidate in grid)


def test_evaluate_candidate_fits_scaler_only_with_train(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    splits = split_train_val_test(synthetic_embeddings(), config)
    candidate = {"name": "svm_linear", "kernel": "linear", "C": 1.0, "gamma": None}

    result = evaluate_candidate(
        candidate,
        splits["train"],
        splits["val"],
        embedding_columns(4),
        "label",
        seed=42,
    )

    train_mean = splits["train"][embedding_columns(4)].to_numpy(dtype=np.float32).mean(axis=0)
    all_mean = synthetic_embeddings()[embedding_columns(4)].to_numpy(dtype=np.float32).mean(axis=0)
    scaler_mean = result["pipeline"].named_steps["scaler"].mean_
    assert np.allclose(scaler_mean, train_mean)
    assert not np.allclose(scaler_mean, all_mean)
    assert result["probability"] is False


def test_metrics_use_decision_function_for_roc_auc(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    splits = split_train_val_test(synthetic_embeddings(), config)
    candidate = {"name": "svm_linear", "kernel": "linear", "C": 1.0, "gamma": None}

    result = evaluate_candidate(candidate, splits["train"], splits["val"], embedding_columns(4), "label", seed=42)
    scores = positive_decision_scores(
        result["pipeline"],
        splits["val"][embedding_columns(4)].to_numpy(dtype=np.float32),
    )
    metrics = classification_metrics(splits["val"]["label"].to_numpy(dtype=int), result["validation_predictions"], scores)

    assert np.array_equal(scores, result["validation_decision_scores"])
    assert metrics["roc_auc"] == result["validation_metrics"]["roc_auc"]
    assert metrics["balanced_accuracy"] == result["validation_metrics"]["balanced_accuracy"]


def test_selection_rule_is_deterministic_and_prefers_linear_on_full_tie() -> None:
    base_metrics = {
        "balanced_accuracy": 0.8,
        "roc_auc": 0.9,
        "f1_ai": 0.75,
        "precision_ai": 0.7,
        "recall_ai": 0.8,
        "confusion_matrix": [[2, 1], [1, 2]],
    }
    rbf = {
        "candidate_index": 0,
        "candidate": {"name": "svm_rbf", "kernel": "rbf", "C": 1.0, "gamma": "scale"},
        "validation_metrics": base_metrics,
        "prediction_latency_seconds_per_example": 0.001,
    }
    linear = {
        "candidate_index": 1,
        "candidate": {"name": "svm_linear", "kernel": "linear", "C": 1.0, "gamma": None},
        "validation_metrics": base_metrics,
        "prediction_latency_seconds_per_example": 0.001,
    }

    assert select_best_candidate([rbf, linear]) is linear


def test_prediction_frame_contains_only_validation_rows(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    splits = split_train_val_test(synthetic_embeddings(), config)
    predictions = np.array([0, 0, 1, 1])
    scores = np.array([-1.0, -0.5, 0.5, 1.0])

    frame = prediction_frame(splits["val"], predictions, scores)

    assert len(frame) == 4
    assert set(frame["split"]) == {"val"}
    assert "is_correct" in frame.columns
    assert "probability" not in frame.columns


def test_run_selection_writes_artifacts_without_test_metrics(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    write_embeddings(Path(config["input"]["embeddings_path"]), synthetic_embeddings())
    write_config(config_path, config)

    result = run_selection(config_path)

    assert result["status"] == "satisfactory"
    assert result["data_usage"]["train_examples_used_for_fit"] == 6
    assert result["data_usage"]["validation_examples_used_for_selection"] == 4
    assert result["data_usage"]["test_examples_used_for_fit"] == 0
    assert result["test_lock"]["test_predictive_metrics_computed"] is False
    assert "test_metrics" not in result

    predictions = pd.read_csv(config["outputs"]["validation_predictions_path"])
    assert len(predictions) == 4
    assert set(predictions["split"]) == {"val"}

    loaded = joblib.load(config["outputs"]["model_path"])
    assert loaded["fit_split"] == "train"
    assert loaded["selection_split"] == "val"
    assert loaded["test_locked"] is True
    assert loaded["pipeline"].predict(synthetic_embeddings()[embedding_columns(4)].to_numpy(dtype=np.float32)).shape == (12,)


def test_run_selection_is_reproducible_on_synthetic_data(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    write_embeddings(Path(config["input"]["embeddings_path"]), synthetic_embeddings())
    write_config(config_path, config)

    first = run_selection(config_path)
    second = run_selection(config_path)

    assert first["selected_candidate"]["candidate"] == second["selected_candidate"]["candidate"]
    assert first["selected_candidate"]["validation_metrics"] == second["selected_candidate"]["validation_metrics"]
