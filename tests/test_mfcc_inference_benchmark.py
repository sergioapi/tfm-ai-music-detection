from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.benchmark_mfcc_inference import (
    determinism_check,
    summarize_values,
    to_jsonable,
)


def test_summarize_values_computes_core_statistics() -> None:
    summary = summarize_values([1.0, 2.0, 3.0, 4.0])

    assert summary["mean"] == pytest.approx(2.5)
    assert summary["median"] == pytest.approx(2.5)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(4.0)
    assert summary["std"] == pytest.approx(float(np.std([1.0, 2.0, 3.0, 4.0])))


def test_determinism_check_accepts_stable_repetitions() -> None:
    repetitions = [
        {"n_fragments": 2, "predicted_label": 1, "ai_score": 0.25},
        {"n_fragments": 2, "predicted_label": 1, "ai_score": 0.25 + 1e-10},
    ]

    result = determinism_check(repetitions)

    assert result["is_deterministic"] is True


def test_determinism_check_flags_score_changes() -> None:
    repetitions = [
        {"n_fragments": 2, "predicted_label": 1, "ai_score": 0.25},
        {"n_fragments": 2, "predicted_label": 1, "ai_score": 0.35},
    ]

    result = determinism_check(repetitions)

    assert result["is_deterministic"] is False
    assert result["same_ai_score_within_tolerance"] is False
    assert result["values"]["ai_score"] == [0.25, 0.35]


def test_determinism_check_flags_fragment_or_label_changes() -> None:
    repetitions = [
        {"n_fragments": 2, "predicted_label": 1, "ai_score": 0.25},
        {"n_fragments": 3, "predicted_label": 0, "ai_score": 0.25},
    ]

    result = determinism_check(repetitions)

    assert result["is_deterministic"] is False
    assert result["same_fragment_count"] is False
    assert result["same_predicted_label"] is False


def test_json_conversion_handles_numpy_and_paths() -> None:
    payload = {
        "integer": np.int64(3),
        "floating": np.float32(1.5),
        "boolean": np.bool_(True),
        "array": np.array([1, 2]),
        "path": Path("backend/tests"),
    }

    converted = to_jsonable(payload)

    assert converted == {
        "integer": 3,
        "floating": pytest.approx(1.5),
        "boolean": True,
        "array": [1, 2],
        "path": "backend\\tests" if "\\" in str(Path("backend/tests")) else "backend/tests",
    }
