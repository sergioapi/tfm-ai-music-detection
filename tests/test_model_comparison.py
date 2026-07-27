from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from scripts import build_model_comparison as comparison


ROOT = Path(__file__).resolve().parents[1]


def test_prediction_artifacts_share_test_ids_labels_and_have_no_duplicates() -> None:
    manifest = comparison.read_csv_rows(comparison.MANIFEST_PATH)
    mfcc_rows = comparison.read_csv_rows(comparison.MFCC_PREDICTIONS_PATH)
    mert_rows = comparison.read_csv_rows(comparison.MERT_TEST_PREDICTIONS_PATH)

    comparison.ensure_no_duplicate_ids(manifest, "manifest")
    comparison.ensure_no_duplicate_ids(mfcc_rows, "mfcc")
    comparison.ensure_no_duplicate_ids(mert_rows, "mert")

    mfcc_test = comparison.rows_by_split(mfcc_rows, "test")
    mert_test = comparison.rows_by_split(mert_rows, "test")

    assert len(mfcc_test) == 150
    assert len(mert_test) == 150
    assert {row["id"] for row in mfcc_test} == {row["id"] for row in mert_test}

    comparison.validate_manifest_alignment(manifest, mfcc_test, "mfcc")
    comparison.validate_manifest_alignment(manifest, mert_test, "mert")

    mfcc_by_id = {row["id"]: row for row in mfcc_test}
    for row in mert_test:
        assert row["label"] == mfcc_by_id[row["id"]]["label"]
        assert row["description"] == mfcc_by_id[row["id"]]["description"]


def test_reconstructed_matrices_counts_and_metrics_match_json() -> None:
    built = comparison.build_comparison()

    assert built["confusion_matrices"]["mfcc_svm_test"] == [[64, 11], [14, 61]]
    assert built["confusion_matrices"]["mert_svm_validation"] == [[62, 13], [7, 68]]
    assert built["confusion_matrices"]["mert_svm_test"] == [[61, 14], [13, 62]]

    assert built["error_counts"]["mfcc_svm_test"]["n_correct"] == 125
    assert built["error_counts"]["mfcc_svm_test"]["false_positives"] == 11
    assert built["error_counts"]["mfcc_svm_test"]["false_negatives"] == 14

    assert built["error_counts"]["mert_svm_test"]["n_correct"] == 123
    assert built["error_counts"]["mert_svm_test"]["false_positives"] == 14
    assert built["error_counts"]["mert_svm_test"]["false_negatives"] == 13


def test_mert_minus_mfcc_differences_are_correct() -> None:
    built = comparison.build_comparison()
    mfcc = built["test_metrics"]["mfcc_svm"]
    mert = built["test_metrics"]["mert_svm"]
    diffs = built["differences_mert_minus_mfcc"]["test"]

    for metric in comparison.METRIC_KEYS:
        assert diffs[metric] == mert[metric] - mfcc[metric]

    count_diffs = built["differences_mert_minus_mfcc"]["test_counts"]
    assert count_diffs == {
        "n_correct": -2,
        "false_positives": 3,
        "false_negatives": -1,
    }


def test_json_has_expected_structure_selection_and_reasons() -> None:
    built = comparison.build_comparison()

    required_keys = {
        "schema_version",
        "sources",
        "protocol",
        "final_configurations",
        "validation_metrics",
        "test_metrics",
        "confusion_matrices",
        "error_counts",
        "differences_mert_minus_mfcc",
        "operational_metrics",
        "comparability_warnings",
        "data_not_available",
        "selected_model",
        "selection_reasons",
    }
    assert required_keys.issubset(built)
    assert built["selected_model"]["id"] == "mfcc_svm"
    assert built["selected_model"]["name"] == "MFCC + StandardScaler + SVM RBF"
    assert len(built["selection_reasons"]) >= 8
    assert any("Fewer false positives" in reason for reason in built["selection_reasons"])


def test_operational_warnings_avoid_misrepresenting_mert_size_and_memory() -> None:
    built = comparison.build_comparison()
    mert_op = built["operational_metrics"]["mert_svm"]
    warnings = " ".join(built["comparability_warnings"]).lower()

    assert mert_op["svm_joblib_size_bytes"] == 1693580
    assert mert_op["svm_joblib_size_is_full_pipeline"] is False
    assert "not to the full deep pipeline" in warnings

    assert mert_op["rss_peak_extraction_bytes"] > 10_000_000_000
    assert "bulk embedding extraction" in mert_op["rss_peak_scope"]
    assert "not the memory required for one mvp request" in warnings


def test_scores_are_not_presented_as_probabilities() -> None:
    built = comparison.build_comparison()

    assert built["protocol"]["score_source"] == "decision_function"
    assert built["protocol"]["scores_are_calibrated_probabilities"] is False
    assert built["mvp_implications"]["output_calibrated_probability"] is False

    mfcc_rows = comparison.read_csv_rows(comparison.MFCC_PREDICTIONS_PATH)
    mert_rows = comparison.read_csv_rows(comparison.MERT_TEST_PREDICTIONS_PATH)
    assert all("probability" not in row for row in mfcc_rows)
    assert all("probability" not in row for row in mert_rows)


def test_generated_json_file_contains_builder_core_values() -> None:
    built = comparison.build_comparison()
    json_path = ROOT / "docs" / "model_comparison.json"

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == built["schema_version"]
    assert saved["selected_model"] == built["selected_model"]
    assert saved["test_metrics"] == built["test_metrics"]
    assert saved["confusion_matrices"] == built["confusion_matrices"]
    assert saved["error_counts"] == built["error_counts"]
    for split in ["validation", "test"]:
        for metric in comparison.METRIC_KEYS:
            assert math.isclose(
                saved["differences_mert_minus_mfcc"][split][metric],
                built["differences_mert_minus_mfcc"][split][metric],
                abs_tol=1e-12,
            )
    assert saved["differences_mert_minus_mfcc"]["test_counts"] == built[
        "differences_mert_minus_mfcc"
    ]["test_counts"]


def test_generation_is_deterministic() -> None:
    built = comparison.build_comparison()
    first = json.dumps(built, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    second_built = comparison.build_comparison()
    second = json.dumps(second_built, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    assert first == second
    assert comparison.render_summary(copy.deepcopy(built)) == comparison.render_summary(copy.deepcopy(second_built))
    assert comparison.render_decision(copy.deepcopy(built)) == comparison.render_decision(copy.deepcopy(second_built))
