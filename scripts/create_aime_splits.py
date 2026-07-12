from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import pandas as pd

HUMAN_MODEL = "MTG-Jamendo"
DEFAULT_INPUT = Path("data/interim/aime_metadata_minimal.parquet")
DEFAULT_OUTPUT = Path("data/aime_splits.csv")
DEFAULT_SEED = 42
DATASET_REVISION = "b84d4be5eda830b6eb714998569dba73530f2601"
SELECTION_STRATEGY = "balanced_generator_group_split_v1"
FINAL_COLUMNS = [
    "id",
    "model",
    "description",
    "label",
    "split",
    "selected_ai_model",
    "selection_seed",
    "selection_strategy",
    "dataset_revision",
]
SPLIT_ORDER = ["train", "val", "test"]


class AimeSplitError(ValueError):
    """Raised when AIME metadata or split invariants are invalid."""


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. Regenerate "
            "data/interim/aime_metadata_minimal.parquet with the audited "
            "metadata-only procedure. This script will not download or decode audio."
        )

    try:
        metadata = pd.read_parquet(path, columns=["id", "model", "description"])
    except ImportError as exc:
        raise RuntimeError(
            "Unable to read Parquet metadata because no usable pandas Parquet "
            "engine is installed. Install the repository Parquet engine and retry."
        ) from exc

    metadata = metadata.copy()
    metadata["id"] = metadata["id"].astype("string")
    metadata["model"] = metadata["model"].astype("string")
    metadata["description"] = metadata["description"].astype("string")
    return metadata


def validate_input_metadata(metadata: pd.DataFrame) -> list[str]:
    required_columns = {"id", "model", "description"}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise AimeSplitError(f"Missing required metadata columns: {sorted(missing)}")

    if metadata[list(required_columns)].isna().any().any():
        null_counts = metadata[list(required_columns)].isna().sum().to_dict()
        raise AimeSplitError(f"Null values found in required columns: {null_counts}")

    if metadata["id"].duplicated().any():
        duplicated = metadata.loc[metadata["id"].duplicated(), "id"].head(5).tolist()
        raise AimeSplitError(
            f"Duplicated ids found in metadata, examples: {duplicated}"
        )

    descriptions = sorted(metadata["description"].unique().tolist())
    if len(descriptions) != 500:
        raise AimeSplitError(f"Expected 500 descriptions, found {len(descriptions)}")

    models = sorted(metadata["model"].unique().tolist())
    if HUMAN_MODEL not in models:
        raise AimeSplitError(f"Human model {HUMAN_MODEL!r} not found in metadata")

    ai_models = [model for model in models if model != HUMAN_MODEL]
    if len(ai_models) != 12:
        raise AimeSplitError(f"Expected 12 AI generators, found {len(ai_models)}")

    human_counts = (
        metadata.loc[metadata["model"] == HUMAN_MODEL]
        .groupby("description", observed=False)
        .size()
    )
    invalid_human = human_counts[human_counts != 1]
    if len(human_counts) != 500 or not invalid_human.empty:
        raise AimeSplitError(
            "Expected exactly one MTG-Jamendo row per description; "
            f"invalid descriptions: {invalid_human.head(5).to_dict()}"
        )

    counts = (
        metadata.groupby(["description", "model"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=descriptions, columns=models, fill_value=0)
    )
    invalid_pairs = counts.ne(1)
    if invalid_pairs.any().any():
        invalid_description = invalid_pairs.any(axis=1).idxmax()
        invalid_models = counts.loc[invalid_description][
            counts.loc[invalid_description] != 1
        ].to_dict()
        raise AimeSplitError(
            "Each description must contain exactly one row for every model; "
            f"{invalid_description!r} has invalid counts {invalid_models}"
        )

    return ai_models


def derive_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_generator_quotas(
    ai_models: list[str], seed: int
) -> dict[str, dict[str, int]]:
    shuffled_models = sorted(ai_models)
    random.Random(derive_seed(seed, "models")).shuffle(shuffled_models)

    quota_patterns = (
        [{"train": 29, "val": 6, "test": 6}] * 4
        + [{"train": 30, "val": 6, "test": 6}] * 2
        + [{"train": 29, "val": 7, "test": 6}] * 3
        + [{"train": 29, "val": 6, "test": 7}] * 3
    )
    return {
        model: dict(pattern) for model, pattern in zip(shuffled_models, quota_patterns)
    }


def build_tickets(quotas: dict[str, dict[str, int]], seed: int) -> list[dict[str, str]]:
    tickets = []
    for model in sorted(quotas):
        for split in SPLIT_ORDER:
            tickets.extend(
                {"split": split, "selected_ai_model": model}
                for _ in range(quotas[model][split])
            )

    random.Random(derive_seed(seed, "tickets")).shuffle(tickets)
    return tickets


def create_manifest(metadata: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["id"] = metadata["id"].astype("string")
    ai_models = validate_input_metadata(metadata)
    quotas = build_generator_quotas(ai_models, seed)
    tickets = build_tickets(quotas, seed)

    descriptions = sorted(metadata["description"].unique().tolist())
    random.Random(derive_seed(seed, "descriptions")).shuffle(descriptions)
    if len(descriptions) != len(tickets):
        raise AimeSplitError(
            f"Cannot assign {len(tickets)} tickets to {len(descriptions)} descriptions"
        )

    sorted_metadata = metadata.sort_values(["description", "model", "id"]).reset_index(
        drop=True
    )
    rows = []
    for description, ticket in zip(descriptions, tickets):
        description_rows = sorted_metadata[
            sorted_metadata["description"].eq(description)
        ]
        selected_model = ticket["selected_ai_model"]
        for label, model in ((0, HUMAN_MODEL), (1, selected_model)):
            source_row = description_rows.loc[description_rows["model"].eq(model)]
            if len(source_row) != 1:
                raise AimeSplitError(
                    f"Expected one row for description={description!r}, model={model!r}; "
                    f"found {len(source_row)}"
                )
            record = source_row.iloc[0][["id", "model", "description"]].to_dict()
            record.update(
                {
                    "label": label,
                    "split": ticket["split"],
                    "selected_ai_model": selected_model,
                    "selection_seed": seed,
                    "selection_strategy": SELECTION_STRATEGY,
                    "dataset_revision": DATASET_REVISION,
                }
            )
            rows.append(record)

    manifest = pd.DataFrame(rows, columns=FINAL_COLUMNS)
    manifest["label"] = manifest["label"].astype(int)
    manifest["selection_seed"] = manifest["selection_seed"].astype(int)
    manifest["split"] = pd.Categorical(
        manifest["split"], categories=SPLIT_ORDER, ordered=True
    )
    manifest = manifest.sort_values(["split", "description", "label"]).reset_index(
        drop=True
    )
    manifest["split"] = manifest["split"].astype(str)
    validate_manifest(manifest, ai_models, seed)
    return manifest


def _assert_count(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise AimeSplitError(f"{name}: expected {expected}, found {actual}")


def validate_manifest(
    manifest: pd.DataFrame, ai_models: list[str], seed: int = DEFAULT_SEED
) -> None:
    if list(manifest.columns) != FINAL_COLUMNS:
        raise AimeSplitError(f"Unexpected manifest columns: {list(manifest.columns)}")

    _assert_count("manifest rows", len(manifest), 1000)
    _assert_count("label 0 rows", int(manifest["label"].eq(0).sum()), 500)
    _assert_count("label 1 rows", int(manifest["label"].eq(1).sum()), 500)
    _assert_count("unique descriptions", manifest["description"].nunique(), 500)

    if manifest.isna().any().any():
        raise AimeSplitError("Manifest contains null values")

    if manifest["id"].duplicated().any():
        duplicated = manifest.loc[manifest["id"].duplicated(), "id"].head(5).tolist()
        raise AimeSplitError(f"Manifest contains duplicated ids: {duplicated}")

    split_counts = manifest["split"].value_counts().to_dict()
    _assert_count("train rows", split_counts.get("train", 0), 700)
    _assert_count("val rows", split_counts.get("val", 0), 150)
    _assert_count("test rows", split_counts.get("test", 0), 150)

    split_label_counts = (
        manifest.groupby(["split", "label"], observed=False).size().to_dict()
    )
    for split, human_count, ai_count in (
        ("train", 350, 350),
        ("val", 75, 75),
        ("test", 75, 75),
    ):
        _assert_count(
            f"{split} human rows", split_label_counts.get((split, 0), 0), human_count
        )
        _assert_count(
            f"{split} AI rows", split_label_counts.get((split, 1), 0), ai_count
        )

    description_counts = manifest.groupby("description", observed=False).size()
    invalid_description_counts = description_counts[description_counts != 2]
    if not invalid_description_counts.empty:
        raise AimeSplitError(
            "Each description must appear exactly twice; invalid examples: "
            f"{invalid_description_counts.head(5).to_dict()}"
        )

    grouped = manifest.groupby("description", observed=False)
    for description, rows in grouped:
        labels = sorted(rows["label"].tolist())
        if labels != [0, 1]:
            raise AimeSplitError(
                f"Description {description!r} must contain one human and one AI row"
            )
        if rows["split"].nunique() != 1:
            raise AimeSplitError(f"Description {description!r} spans multiple splits")
        if rows["selected_ai_model"].nunique() != 1:
            raise AimeSplitError(
                f"Description {description!r} spans multiple selected AI models"
            )

    human_rows = manifest[manifest["label"].eq(0)]
    ai_rows = manifest[manifest["label"].eq(1)]
    if not human_rows["model"].eq(HUMAN_MODEL).all():
        raise AimeSplitError("All human rows must have model == MTG-Jamendo")
    if not ai_rows["model"].eq(ai_rows["selected_ai_model"]).all():
        raise AimeSplitError("All AI rows must have model == selected_ai_model")

    selected_ai_counts = ai_rows["selected_ai_model"].value_counts().sort_index()
    selected_models = sorted(selected_ai_counts.index.tolist())
    if selected_models != sorted(ai_models):
        raise AimeSplitError(
            f"Expected selected AI models {sorted(ai_models)}, found {selected_models}"
        )
    _assert_count("AI generators with 42 rows", int(selected_ai_counts.eq(42).sum()), 8)
    _assert_count("AI generators with 41 rows", int(selected_ai_counts.eq(41).sum()), 4)

    split_model_counts = (
        ai_rows.groupby(["split", "selected_ai_model"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=SPLIT_ORDER, columns=sorted(ai_models), fill_value=0)
    )
    expected_split_shapes = {
        "train": {30: 2, 29: 10},
        "val": {7: 3, 6: 9},
        "test": {7: 3, 6: 9},
    }
    for split, expected_counts in expected_split_shapes.items():
        counts = split_model_counts.loc[split].value_counts().to_dict()
        for per_model_count, expected_generators in expected_counts.items():
            _assert_count(
                f"{split} generators with {per_model_count} AI rows",
                int(counts.get(per_model_count, 0)),
                expected_generators,
            )
        if split_model_counts.loc[split].le(0).any():
            raise AimeSplitError(f"All AI generators must appear in {split}")

    _assert_count("selection_seed values", manifest["selection_seed"].nunique(), 1)
    _assert_count("selection_seed", int(manifest["selection_seed"].iloc[0]), seed)
    _assert_count(
        "selection_strategy values", manifest["selection_strategy"].nunique(), 1
    )
    _assert_count(
        "selection_strategy",
        manifest["selection_strategy"].iloc[0],
        SELECTION_STRATEGY,
    )
    _assert_count("dataset_revision values", manifest["dataset_revision"].nunique(), 1)
    _assert_count(
        "dataset_revision", manifest["dataset_revision"].iloc[0], DATASET_REVISION
    )


def write_manifest(manifest: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")


def print_summary(manifest: pd.DataFrame, output_path: Path) -> None:
    ai_distribution = (
        manifest.loc[manifest["label"].eq(1), "selected_ai_model"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    split_counts = {
        split: int(manifest["split"].eq(split).sum()) for split in SPLIT_ORDER
    }
    print(f"rows: {len(manifest)}")
    print(f"split_counts: {split_counts}")
    print(f"label_counts: {manifest['label'].value_counts().sort_index().to_dict()}")
    print(f"ai_generator_distribution: {ai_distribution}")
    print(f"output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the reproducible AIME experimental split manifest."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.input)
    manifest = create_manifest(metadata, seed=args.seed)
    write_manifest(manifest, args.output)
    print_summary(manifest, args.output)


if __name__ == "__main__":
    main()
