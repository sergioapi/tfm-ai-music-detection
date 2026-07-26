from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_mert_embeddings import (  # noqa: E402
    MertEmbeddingExtractionError,
    append_embedding_row,
    embedding_columns,
    load_existing_embedding_ids,
    open_csv_appender,
    output_columns,
    process_streaming_rows,
    validate_and_consolidate_embeddings,
)


def tiny_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "00001",
                "model": "MTG-Jamendo",
                "description": "ambient, piano",
                "label": 0,
                "split": "train",
                "selected_ai_model": "Generator",
                "selection_seed": 42,
                "selection_strategy": "test",
                "dataset_revision": "dataset-sha",
            },
            {
                "id": "00002",
                "model": "Generator",
                "description": "ambient, piano",
                "label": 1,
                "split": "train",
                "selected_ai_model": "Generator",
                "selection_seed": 42,
                "selection_strategy": "test",
                "dataset_revision": "dataset-sha",
            },
        ]
    )


def write_embedding_csv(
    path: Path,
    manifest: pd.DataFrame,
    *,
    expected_dim: int = 2,
    records: list[dict[str, object]] | None = None,
    embedding: np.ndarray | None = None,
) -> None:
    handle, writer = open_csv_appender(
        path,
        output_columns(list(manifest.columns), expected_dim),
        overwrite=False,
    )
    try:
        rows = records if records is not None else manifest.to_dict(orient="records")
        values = embedding if embedding is not None else np.array([1.0, 2.0], dtype=np.float32)
        for record in rows:
            append_embedding_row(
                writer,
                handle,
                record,
                values,
                expected_dim=expected_dim,
            )
    finally:
        handle.close()


def test_embedding_and_output_columns_are_stable() -> None:
    manifest_columns = ["id", "model", "split"]

    assert embedding_columns(4) == ["mert_000", "mert_001", "mert_002", "mert_003"]
    assert output_columns(manifest_columns, 2) == ["id", "model", "split", "mert_000", "mert_001"]


def test_append_embedding_row_and_load_existing_ids(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.csv"
    manifest = tiny_manifest()
    columns = output_columns(list(manifest.columns), 3)
    handle, writer = open_csv_appender(path, columns, overwrite=False)
    try:
        append_embedding_row(
            writer,
            handle,
            manifest.iloc[0].to_dict(),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            expected_dim=3,
        )
    finally:
        handle.close()

    assert load_existing_embedding_ids(path, expected_dim=3) == {"00001"}


def test_append_embedding_row_rejects_non_finite(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.csv"
    manifest = tiny_manifest()
    handle, writer = open_csv_appender(
        path,
        output_columns(list(manifest.columns), 2),
        overwrite=False,
    )
    try:
        with pytest.raises(MertEmbeddingExtractionError, match="NaN or infinite"):
            append_embedding_row(
                writer,
                handle,
                manifest.iloc[0].to_dict(),
                np.array([1.0, np.nan], dtype=np.float32),
                expected_dim=2,
            )
    finally:
        handle.close()


def test_process_streaming_rows_single_pass_stops_when_pending_found() -> None:
    stream = ({"id": f"{index:05d}", "audio": object()} for index in range(10))
    processed: list[str] = []

    stats = process_streaming_rows(
        stream,
        {"00002", "00005"},
        lambda row: processed.append(row["id"]),
        progress=False,
    )

    assert processed == ["00002", "00005"]
    assert stats == {"rows_scanned": 6, "matched_rows": 2, "missing_ids": []}


def test_process_streaming_rows_reports_missing_ids() -> None:
    stream = ({"id": f"{index:05d}", "audio": object()} for index in range(3))

    stats = process_streaming_rows(
        stream,
        {"00002", "00005"},
        lambda row: None,
        progress=False,
    )

    assert stats["rows_scanned"] == 3
    assert stats["matched_rows"] == 1
    assert stats["missing_ids"] == ["00005"]


def test_validate_and_consolidate_embeddings_writes_parquet(tmp_path: Path) -> None:
    manifest = tiny_manifest()
    csv_path = tmp_path / "embeddings.csv"
    parquet_path = tmp_path / "embeddings.parquet"
    write_embedding_csv(csv_path, manifest)

    result = validate_and_consolidate_embeddings(
        csv_path,
        parquet_path,
        manifest,
        expected_dim=2,
    )

    assert result["n_rows"] == 2
    assert result["embedding_shape"] == [2, 2]
    assert result["embedding_dtype"] == "float32"
    assert result["ids_match_manifest"] is True
    assert result["metadata_match_manifest"] is True
    assert result["split_distribution"] == {"train": 2}
    assert result["label_distribution"] == {"0": 1, "1": 1}
    assert result["all_finite"] is True
    assert parquet_path.exists()

    schema = pq.read_schema(parquet_path)
    assert {str(schema.field(column).type) for column in embedding_columns(2)} == {"float"}


def test_validate_and_consolidate_embeddings_rejects_missing_ids(tmp_path: Path) -> None:
    manifest = tiny_manifest()
    csv_path = tmp_path / "embeddings.csv"
    parquet_path = tmp_path / "embeddings.parquet"
    write_embedding_csv(csv_path, manifest, records=[manifest.iloc[0].to_dict()])

    with pytest.raises(MertEmbeddingExtractionError, match="missing ids"):
        validate_and_consolidate_embeddings(csv_path, parquet_path, manifest, expected_dim=2)


def test_validate_and_consolidate_embeddings_rejects_additional_ids(tmp_path: Path) -> None:
    manifest = tiny_manifest()
    csv_path = tmp_path / "embeddings.csv"
    parquet_path = tmp_path / "embeddings.parquet"
    extra_record = manifest.iloc[0].to_dict()
    extra_record["id"] = "99999"
    write_embedding_csv(
        csv_path,
        manifest,
        records=[*manifest.to_dict(orient="records"), extra_record],
    )

    with pytest.raises(MertEmbeddingExtractionError, match="additional ids"):
        validate_and_consolidate_embeddings(csv_path, parquet_path, manifest, expected_dim=2)


def test_validate_and_consolidate_embeddings_rejects_metadata_mismatch(tmp_path: Path) -> None:
    manifest = tiny_manifest()
    csv_path = tmp_path / "embeddings.csv"
    parquet_path = tmp_path / "embeddings.parquet"
    records = manifest.to_dict(orient="records")
    records[1] = {**records[1], "model": "DifferentGenerator"}
    write_embedding_csv(csv_path, manifest, records=records)

    with pytest.raises(MertEmbeddingExtractionError, match="Metadata mismatch"):
        validate_and_consolidate_embeddings(csv_path, parquet_path, manifest, expected_dim=2)


def test_validate_and_consolidate_embeddings_rejects_non_finite_values(tmp_path: Path) -> None:
    manifest = tiny_manifest()
    csv_path = tmp_path / "embeddings.csv"
    parquet_path = tmp_path / "embeddings.parquet"
    write_embedding_csv(csv_path, manifest)
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(text.replace("2.0", "inf", 1), encoding="utf-8")

    with pytest.raises(MertEmbeddingExtractionError, match="NaN or infinite"):
        validate_and_consolidate_embeddings(csv_path, parquet_path, manifest, expected_dim=2)
