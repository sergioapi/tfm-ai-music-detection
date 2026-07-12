from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_aime_splits import (  # noqa: E402
    AimeSplitError,
    DATASET_REVISION,
    FINAL_COLUMNS,
    HUMAN_MODEL,
    SELECTION_STRATEGY,
    create_manifest,
    write_manifest,
)

AI_MODELS = [f"Generator-{index:02d}" for index in range(1, 13)]


@pytest.fixture()
def synthetic_metadata() -> pd.DataFrame:
    rows = []
    for description_index in range(500):
        description = f"description-{description_index:03d}"
        rows.append(
            {
                "id": f"h-{description_index:03d}",
                "model": HUMAN_MODEL,
                "description": description,
            }
        )
        for model_index, model in enumerate(AI_MODELS, start=1):
            rows.append(
                {
                    "id": f"ai-{description_index:03d}-{model_index:02d}",
                    "model": model,
                    "description": description,
                }
            )
    return pd.DataFrame(rows)


def assert_valid_manifest(manifest: pd.DataFrame) -> None:
    assert list(manifest.columns) == FINAL_COLUMNS
    assert len(manifest) == 1000
    assert manifest["label"].value_counts().to_dict() == {0: 500, 1: 500}
    assert manifest.isna().sum().sum() == 0
    assert not manifest["id"].duplicated().any()
    assert manifest["selection_seed"].unique().tolist() == [42]
    assert manifest["selection_strategy"].unique().tolist() == [SELECTION_STRATEGY]
    assert manifest["dataset_revision"].unique().tolist() == [DATASET_REVISION]


def test_create_manifest_global_counts(synthetic_metadata: pd.DataFrame) -> None:
    manifest = create_manifest(synthetic_metadata)

    assert_valid_manifest(manifest)
    assert manifest["split"].value_counts().to_dict() == {
        "train": 700,
        "val": 150,
        "test": 150,
    }


def test_split_and_label_counts(synthetic_metadata: pd.DataFrame) -> None:
    manifest = create_manifest(synthetic_metadata)

    counts = manifest.groupby(["split", "label"], observed=False).size().to_dict()
    assert counts == {
        ("train", 0): 350,
        ("train", 1): 350,
        ("val", 0): 75,
        ("val", 1): 75,
        ("test", 0): 75,
        ("test", 1): 75,
    }


def test_description_groups_are_intact(synthetic_metadata: pd.DataFrame) -> None:
    manifest = create_manifest(synthetic_metadata)

    grouped = manifest.groupby("description", observed=False)
    assert grouped.size().eq(2).all()
    assert grouped["label"].apply(lambda values: sorted(values) == [0, 1]).all()
    assert grouped["split"].nunique().eq(1).all()
    assert grouped["selected_ai_model"].nunique().eq(1).all()


def test_generator_quotas(synthetic_metadata: pd.DataFrame) -> None:
    manifest = create_manifest(synthetic_metadata)
    ai_rows = manifest[manifest["label"].eq(1)]

    totals = ai_rows["selected_ai_model"].value_counts()
    assert totals.eq(42).sum() == 8
    assert totals.eq(41).sum() == 4

    split_model_counts = (
        ai_rows.groupby(["split", "selected_ai_model"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    assert split_model_counts.loc["train"].value_counts().to_dict() == {29: 10, 30: 2}
    assert split_model_counts.loc["val"].value_counts().to_dict() == {6: 9, 7: 3}
    assert split_model_counts.loc["test"].value_counts().to_dict() == {6: 9, 7: 3}
    assert split_model_counts.gt(0).all().all()


def test_reproducibility(synthetic_metadata: pd.DataFrame) -> None:
    first = create_manifest(synthetic_metadata)
    second = create_manifest(synthetic_metadata)

    pd.testing.assert_frame_equal(first, second)


def test_input_order_independence(synthetic_metadata: pd.DataFrame) -> None:
    original = create_manifest(synthetic_metadata)
    shuffled = synthetic_metadata.sample(frac=1, random_state=123).reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(original, create_manifest(shuffled))


def test_stable_serialization(tmp_path: Path, synthetic_metadata: pd.DataFrame) -> None:
    manifest = create_manifest(synthetic_metadata)
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    write_manifest(manifest, first_path)
    write_manifest(manifest, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()


def test_missing_required_column_raises_clear_error(
    synthetic_metadata: pd.DataFrame,
) -> None:
    metadata = synthetic_metadata.drop(columns=["description"])

    with pytest.raises(AimeSplitError, match="Missing required metadata columns"):
        create_manifest(metadata)


def test_invalid_precondition_raises_clear_error(
    synthetic_metadata: pd.DataFrame,
) -> None:
    metadata = synthetic_metadata[synthetic_metadata["id"] != "h-000"].copy()

    with pytest.raises(AimeSplitError, match="one MTG-Jamendo row per description"):
        create_manifest(metadata)
