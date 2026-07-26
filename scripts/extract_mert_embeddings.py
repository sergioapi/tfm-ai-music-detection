from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

try:
    from scripts.mfcc_svm_baseline import (  # type: ignore
        PreprocessConfig,
        _decode_remote_audio_payload,
        _load_hf_audio_stream,
        find_local_audio,
        load_manifest,
        preprocess_audio_file,
        validate_description_splits,
    )
    from scripts.smoke_test_mert import (  # type: ignore
        MertSmokeTestError,
        SmokeTestConfig,
        _json_default,
        _log,
        current_rss_bytes,
        cuda_info,
        estimate_snapshot_size_bytes,
        extract_clip_embedding,
        load_config,
        load_processor_and_model,
        resolve_device,
        review_remote_code,
        version_info,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from scripts/.
    from mfcc_svm_baseline import (  # type: ignore
        PreprocessConfig,
        _decode_remote_audio_payload,
        _load_hf_audio_stream,
        find_local_audio,
        load_manifest,
        preprocess_audio_file,
        validate_description_splits,
    )
    from smoke_test_mert import (  # type: ignore
        MertSmokeTestError,
        SmokeTestConfig,
        _json_default,
        _log,
        current_rss_bytes,
        cuda_info,
        estimate_snapshot_size_bytes,
        extract_clip_embedding,
        load_config,
        load_processor_and_model,
        resolve_device,
        review_remote_code,
        version_info,
    )


DEFAULT_CONFIG = Path("configs/mert_frozen_embeddings.yaml")
DEFAULT_OUTPUT_CSV = Path("data/processed/aime_mert_embeddings.csv")
DEFAULT_OUTPUT_PARQUET = Path("data/processed/aime_mert_embeddings.parquet")
DEFAULT_FAILURES = Path("data/processed/aime_mert_embedding_failures.csv")
DEFAULT_SUMMARY = Path("data/processed/aime_mert_embedding_extraction_summary.json")


class MertEmbeddingExtractionError(RuntimeError):
    """Raised when full MERT embedding extraction cannot be completed."""


def load_extraction_paths(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise MertEmbeddingExtractionError("PyYAML is required to read the config") from exc

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    extraction = (raw or {}).get("embedding_extraction", {})
    return {
        "output_csv_path": Path(extraction.get("output_csv_path", DEFAULT_OUTPUT_CSV)),
        "output_parquet_path": Path(extraction.get("output_parquet_path", DEFAULT_OUTPUT_PARQUET)),
        "failures_path": Path(extraction.get("failures_path", DEFAULT_FAILURES)),
        "summary_path": Path(extraction.get("summary_path", DEFAULT_SUMMARY)),
        "progress_every_rows": int(extraction.get("progress_every_rows", 250)),
        "expected_rows": int(extraction["expected_rows"]) if "expected_rows" in extraction else None,
        "expected_embedding_dim": (
            int(extraction["expected_embedding_dim"]) if "expected_embedding_dim" in extraction else None
        ),
        "expected_output_dtype": str(extraction.get("expected_output_dtype", "float32")),
    }


def embedding_columns(expected_dim: int) -> list[str]:
    return [f"mert_{index:03d}" for index in range(expected_dim)]


def output_columns(manifest_columns: list[str], expected_dim: int) -> list[str]:
    return manifest_columns + embedding_columns(expected_dim)


def load_existing_embedding_ids(path: Path, expected_dim: int) -> set[str]:
    if not path.exists():
        return set()
    header = pd.read_csv(path, nrows=0)
    required = {"id", *embedding_columns(expected_dim)}
    missing = required.difference(header.columns)
    if missing:
        raise MertEmbeddingExtractionError(
            f"Existing embedding file is missing columns: {sorted(missing)[:10]}"
        )
    ids = pd.read_csv(path, usecols=["id"], dtype={"id": "string"})["id"].astype(str)
    duplicates = ids[ids.duplicated()].tolist()
    if duplicates:
        raise MertEmbeddingExtractionError(
            f"Existing embedding file contains duplicate ids: {sorted(set(duplicates))[:10]}"
        )
    return set(ids.tolist())


def open_csv_appender(path: Path, columns: list[str], *, overwrite: bool) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    write_header = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    if write_header:
        writer.writeheader()
        handle.flush()
    return handle, writer


def append_embedding_row(
    writer: csv.DictWriter,
    handle: Any,
    metadata: dict[str, Any],
    embedding: np.ndarray,
    expected_dim: int,
) -> None:
    if embedding.shape != (expected_dim,):
        raise MertEmbeddingExtractionError(
            f"Expected embedding shape ({expected_dim},), found {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise MertEmbeddingExtractionError("Embedding contains NaN or infinite values")
    row = dict(metadata)
    row.update({column: float(value) for column, value in zip(embedding_columns(expected_dim), embedding)})
    writer.writerow(row)
    handle.flush()


def append_failure(path: Path, metadata: dict[str, Any], phase: str, exc: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "description", "model", "label", "split", "phase", "reason"]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "id": metadata.get("id", ""),
                "description": metadata.get("description", ""),
                "model": metadata.get("model", ""),
                "label": metadata.get("label", ""),
                "split": metadata.get("split", ""),
                "phase": phase,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )


def metadata_by_id(manifest: pd.DataFrame) -> dict[str, dict[str, Any]]:
    ids = manifest["id"].astype(str)
    duplicates = ids[ids.duplicated()].tolist()
    if duplicates:
        raise MertEmbeddingExtractionError(
            f"Manifest contains duplicate ids: {sorted(set(duplicates))[:10]}"
        )
    return {
        str(record["id"]): {column: record[column] for column in manifest.columns}
        for _, record in manifest.iterrows()
    }


def process_streaming_rows(
    stream: Any,
    pending_ids: set[str],
    process_row: Callable[[dict[str, Any]], None],
    *,
    progress_every_rows: int = 250,
    progress: bool = True,
) -> dict[str, Any]:
    pending = set(pending_ids)
    rows_scanned = 0
    matched_rows = 0
    for row in stream:
        rows_scanned += 1
        audio_id = str(row["id"])
        if audio_id in pending:
            process_row(row)
            pending.remove(audio_id)
            matched_rows += 1
            _log(
                f"Processed id={audio_id}; processed={matched_rows} pending={len(pending)} rows_scanned={rows_scanned}",
                enabled=progress,
            )
            if not pending:
                break
        elif progress_every_rows > 0 and rows_scanned % progress_every_rows == 0:
            _log(
                f"Scanned {rows_scanned} AIME rows; processed={matched_rows} pending={len(pending)}",
                enabled=progress,
            )
    return {
        "rows_scanned": rows_scanned,
        "matched_rows": matched_rows,
        "missing_ids": sorted(pending),
    }


def preprocess_remote_audio(row: dict[str, Any], config: SmokeTestConfig) -> np.ndarray:
    preprocess_config = PreprocessConfig(
        target_sample_rate=config.sample_rate_hz,
        duration_seconds=config.total_duration_seconds,
    )
    return _decode_remote_audio_payload(row["audio"], preprocess_config)


def preprocess_local_audio(record: dict[str, Any], config: SmokeTestConfig, audio_dir: Path) -> np.ndarray:
    preprocess_config = PreprocessConfig(
        target_sample_rate=config.sample_rate_hz,
        duration_seconds=config.total_duration_seconds,
    )
    return preprocess_audio_file(
        find_local_audio(audio_dir, str(record["id"])),
        config=preprocess_config,
    )


def validate_and_consolidate_embeddings(
    csv_path: Path,
    parquet_path: Path,
    manifest: pd.DataFrame,
    expected_dim: int,
    expected_output_dtype: str = "float32",
) -> dict[str, Any]:
    if expected_output_dtype != "float32":
        raise MertEmbeddingExtractionError(
            f"Only float32 embedding output is supported, found {expected_output_dtype!r}"
        )
    if not csv_path.exists():
        raise MertEmbeddingExtractionError(f"Embedding CSV was not created: {csv_path}")
    frame = pd.read_csv(csv_path, dtype={"id": "string"})
    columns = embedding_columns(expected_dim)
    expected_columns = output_columns(list(manifest.columns), expected_dim)
    missing_columns = set(expected_columns).difference(frame.columns)
    if missing_columns:
        raise MertEmbeddingExtractionError(
            f"Embedding file is missing columns: {sorted(missing_columns)[:10]}"
        )
    actual_embedding_columns = [column for column in frame.columns if column.startswith("mert_")]
    if actual_embedding_columns != columns:
        raise MertEmbeddingExtractionError(
            f"Expected exactly {expected_dim} embedding columns named mert_000..mert_{expected_dim - 1:03d}"
        )

    frame_ids = frame["id"].astype(str)
    duplicate_ids = sorted(set(frame_ids[frame_ids.duplicated()].tolist()))
    if duplicate_ids:
        raise MertEmbeddingExtractionError(f"Embedding file contains duplicate ids: {duplicate_ids[:10]}")

    manifest_ids = manifest["id"].astype(str)
    missing_ids = sorted(set(manifest_ids).difference(frame_ids))
    additional_ids = sorted(set(frame_ids).difference(manifest_ids))
    if missing_ids or additional_ids:
        raise MertEmbeddingExtractionError(
            f"Embedding ids do not match manifest; missing ids: {missing_ids[:10]}, "
            f"additional ids: {additional_ids[:10]}"
        )
    if len(frame) != len(manifest):
        raise MertEmbeddingExtractionError(
            f"Expected {len(manifest)} embeddings, found {len(frame)}"
        )

    non_numeric_columns = [column for column in columns if not is_numeric_dtype(frame[column])]
    if non_numeric_columns:
        raise MertEmbeddingExtractionError(
            f"Embedding columns must be numeric: {non_numeric_columns[:10]}"
        )

    embeddings = frame[columns].to_numpy(dtype=np.float32)
    if embeddings.shape != (len(manifest), expected_dim):
        raise MertEmbeddingExtractionError(
            f"Expected embedding matrix {(len(manifest), expected_dim)}, found {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise MertEmbeddingExtractionError("Embedding matrix contains NaN or infinite values")

    frame[columns] = frame[columns].astype(np.float32)
    manifest_by_id = manifest.assign(id=manifest["id"].astype(str)).set_index("id")
    frame_by_id = frame.assign(id=frame["id"].astype(str)).set_index("id")
    metadata_columns = [column for column in manifest.columns if column != "id" and column in frame.columns]
    for column in metadata_columns:
        expected_values = manifest_by_id.loc[sorted(manifest_by_id.index), column].astype("string").fillna("<NA>")
        actual_values = frame_by_id.loc[sorted(manifest_by_id.index), column].astype("string").fillna("<NA>")
        if not actual_values.equals(expected_values):
            mismatch_id = next(
                str(audio_id)
                for audio_id, actual, expected in zip(
                    sorted(manifest_by_id.index),
                    actual_values.tolist(),
                    expected_values.tolist(),
                )
                if actual != expected
            )
            raise MertEmbeddingExtractionError(
                f"Metadata mismatch for column {column!r} at id {mismatch_id!r}"
            )

    split_distribution = frame["split"].astype(str).value_counts().sort_index().to_dict()
    expected_split_distribution = manifest["split"].astype(str).value_counts().sort_index().to_dict()
    label_distribution = frame["label"].astype(str).value_counts().sort_index().to_dict()
    expected_label_distribution = manifest["label"].astype(str).value_counts().sort_index().to_dict()
    if split_distribution != expected_split_distribution:
        raise MertEmbeddingExtractionError(
            f"Split distribution mismatch: {split_distribution} != {expected_split_distribution}"
        )
    if label_distribution != expected_label_distribution:
        raise MertEmbeddingExtractionError(
            f"Label distribution mismatch: {label_distribution} != {expected_label_distribution}"
        )

    frame = frame.sort_values(["split", "description", "label", "id"]).reset_index(drop=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)

    import pyarrow.parquet as pq

    parquet_schema = pq.read_schema(parquet_path)
    parquet_embedding_dtypes = sorted({str(parquet_schema.field(column).type) for column in columns})
    if parquet_embedding_dtypes != ["float"]:
        raise MertEmbeddingExtractionError(
            f"Expected Parquet embedding dtype float32, found {parquet_embedding_dtypes}"
        )
    return {
        "n_rows": int(len(frame)),
        "embedding_shape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
        "embedding_dtype": "float32",
        "parquet_embedding_dtypes": parquet_embedding_dtypes,
        "all_finite": True,
        "ids_match_manifest": True,
        "metadata_match_manifest": True,
        "split_distribution": split_distribution,
        "label_distribution": label_distribution,
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path),
    }


def extract_embeddings(
    config: SmokeTestConfig,
    *,
    output_csv_path: Path,
    output_parquet_path: Path,
    failures_path: Path,
    summary_path: Path,
    device_name: str,
    audio_dir: Path | None,
    overwrite: bool,
    limit: int | None,
    progress_every_rows: int,
    expected_rows: int | None,
    expected_output_dtype: str,
    progress: bool = True,
) -> dict[str, Any]:
    import torch

    total_start = time.perf_counter()
    rss_before = current_rss_bytes()
    manifest = load_manifest(config.manifest_path)
    validate_description_splits(manifest)
    if limit is not None:
        if limit <= 0:
            raise MertEmbeddingExtractionError("--limit must be positive")
        manifest = manifest.head(limit).copy()
        _log(f"Debug limit active: extracting first {len(manifest)} manifest rows.", enabled=progress)
    elif expected_rows is not None and len(manifest) != expected_rows:
        raise MertEmbeddingExtractionError(
            f"Expected {expected_rows} manifest rows from config, found {len(manifest)}"
        )

    dataset_revision = str(manifest["dataset_revision"].dropna().iloc[0])
    metadata = metadata_by_id(manifest)
    expected_ids = set(metadata)

    if overwrite:
        for path in [output_csv_path, output_parquet_path, failures_path, summary_path]:
            if path.exists():
                path.unlink()
    elif failures_path.exists():
        failures_path.unlink()

    existing_ids = load_existing_embedding_ids(output_csv_path, config.expected_embedding_dim)
    unknown_existing = sorted(existing_ids.difference(expected_ids))
    if unknown_existing:
        raise MertEmbeddingExtractionError(
            f"Existing embedding file contains ids outside current manifest: {unknown_existing[:10]}"
        )
    pending_ids = expected_ids.difference(existing_ids)
    _log(
        f"Manifest rows={len(manifest)} existing={len(existing_ids)} pending={len(pending_ids)}",
        enabled=progress,
    )

    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _log(f"Resolved device: {device}", enabled=progress)

    code_review = review_remote_code(config)
    rss_before_load = current_rss_bytes()
    processor, model, load_timings = load_processor_and_model(config, device)
    rss_after_load = current_rss_bytes()
    model_frozen = all(not parameter.requires_grad for parameter in model.parameters())
    model_eval = not bool(model.training)
    if not model_frozen or not model_eval:
        raise MertEmbeddingExtractionError("Model must be frozen and in eval mode")

    columns = output_columns(list(manifest.columns), config.expected_embedding_dim)
    processed_this_run = 0
    failed_this_run = 0
    preprocessing_seconds: list[float] = []
    inference_window_seconds: list[float] = []
    inference_clip_seconds: list[float] = []
    peak_candidates = [rss_before, rss_before_load, rss_after_load]
    peak_rss = max(value for value in peak_candidates if value is not None) if any(peak_candidates) else None

    handle, writer = open_csv_appender(output_csv_path, columns, overwrite=False)
    try:
        def process_signal(audio_id: str, signal: np.ndarray) -> None:
            nonlocal processed_this_run, failed_this_run, peak_rss
            record = metadata[audio_id]
            try:
                start = time.perf_counter()
                embedding, window_times = extract_clip_embedding(
                    signal,
                    processor=processor,
                    model=model,
                    device=device,
                    config=config,
                )
                inference_elapsed = time.perf_counter() - start
                append_embedding_row(
                    writer,
                    handle,
                    record,
                    embedding,
                    config.expected_embedding_dim,
                )
                processed_this_run += 1
                inference_window_seconds.extend(window_times)
                inference_clip_seconds.append(float(sum(window_times)))
                # Keep wall-clock extraction overhead visible separately from model latency.
                if not window_times:
                    inference_clip_seconds.append(inference_elapsed)
            except Exception as exc:  # noqa: BLE001 - every failed id must be recorded.
                failed_this_run += 1
                append_failure(failures_path, record, "embedding_extraction", exc)
            rss_now = current_rss_bytes()
            if rss_now is not None:
                peak_rss = rss_now if peak_rss is None else max(peak_rss, rss_now)

        if pending_ids:
            if audio_dir is None:
                stream = _load_hf_audio_stream(config.dataset_name, dataset_revision)

                def process_remote_row(row: dict[str, Any]) -> None:
                    nonlocal failed_this_run
                    audio_id = str(row["id"])
                    record = metadata[audio_id]
                    try:
                        start = time.perf_counter()
                        signal = preprocess_remote_audio(row, config)
                        preprocessing_seconds.append(time.perf_counter() - start)
                        process_signal(audio_id, signal)
                    except Exception as exc:  # noqa: BLE001
                        failed_this_run += 1
                        append_failure(failures_path, record, "preprocessing", exc)
                    finally:
                        row.clear()

                stream_start = time.perf_counter()
                stream_stats = process_streaming_rows(
                    stream,
                    pending_ids,
                    process_remote_row,
                    progress_every_rows=progress_every_rows,
                    progress=progress,
                )
                streaming_seconds = time.perf_counter() - stream_start
            else:
                stream_stats = {
                    "rows_scanned": None,
                    "matched_rows": 0,
                    "missing_ids": [],
                }
                streaming_seconds = None
                for index, record in enumerate(manifest.to_dict(orient="records"), start=1):
                    audio_id = str(record["id"])
                    if audio_id not in pending_ids:
                        continue
                    _log(
                        f"Processing local id={audio_id}; {index}/{len(manifest)}",
                        enabled=progress,
                    )
                    try:
                        start = time.perf_counter()
                        signal = preprocess_local_audio(record, config, audio_dir)
                        preprocessing_seconds.append(time.perf_counter() - start)
                        process_signal(audio_id, signal)
                        stream_stats["matched_rows"] += 1
                    except Exception as exc:  # noqa: BLE001
                        failed_this_run += 1
                        append_failure(failures_path, record, "local_audio", exc)
        else:
            stream_stats = {"rows_scanned": 0, "matched_rows": 0, "missing_ids": []}
            streaming_seconds = 0.0
    finally:
        handle.close()

    if stream_stats["missing_ids"]:
        for missing_id in stream_stats["missing_ids"]:
            append_failure(
                failures_path,
                metadata[missing_id],
                "dataset_streaming",
                MertEmbeddingExtractionError(f"Audio id {missing_id!r} was not found"),
            )

    validation = None
    status = "failed"
    try:
        validation = validate_and_consolidate_embeddings(
            output_csv_path,
            output_parquet_path,
            manifest,
            config.expected_embedding_dim,
            expected_output_dtype,
        )
        status = "satisfactory" if failed_this_run == 0 and not stream_stats["missing_ids"] else "failed"
    except Exception as exc:  # noqa: BLE001
        append_failure(
            failures_path,
            {"id": "", "description": "", "model": "", "label": "", "split": ""},
            "final_validation",
            exc,
        )
        status = "failed"

    vram = cuda_info(device)
    result = {
        "status": status,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "identifier": config.model_id,
            "revision": config.revision,
            "trust_remote_code": config.trust_remote_code,
            "expected_embedding_dim": config.expected_embedding_dim,
        },
        "dataset": {
            "name": config.dataset_name,
            "manifest_path": str(config.manifest_path),
            "dataset_revision_from_manifest": dataset_revision,
            "n_manifest_rows": int(len(manifest)),
        },
        "device": {
            "requested": device_name,
            "resolved": str(device),
            "type": device.type,
            "batch_size": config.batch_size,
            "mixed_precision": False,
            "quantization": False,
        },
        "model_state": {
            "eval_mode": model_eval,
            "all_parameters_frozen": model_frozen,
        },
        "audio": {
            "sample_rate_hz": config.sample_rate_hz,
            "total_duration_seconds": config.total_duration_seconds,
            "window_duration_seconds": config.window_duration_seconds,
            "num_windows": config.num_windows,
            "total_samples": config.total_samples,
            "window_samples": config.window_samples,
        },
        "counts": {
            "existing_before_run": int(len(existing_ids)),
            "pending_before_run": int(len(pending_ids)),
            "processed_this_run": int(processed_this_run),
            "failed_this_run": int(failed_this_run),
            "missing_after_stream": int(len(stream_stats["missing_ids"])),
        },
        "streaming": {
            **stream_stats,
            "seconds": streaming_seconds,
            "mode": "remote_streaming_once" if audio_dir is None else "local_audio_dir",
        },
        "validation": validation,
        "timings": {
            **load_timings,
            "preprocessing_seconds_total": float(sum(preprocessing_seconds)),
            "preprocessing_seconds_mean": _mean(preprocessing_seconds),
            "inference_window_seconds_mean": _mean(inference_window_seconds),
            "inference_clip_seconds_mean": _mean(inference_clip_seconds),
            "total_seconds": time.perf_counter() - total_start,
        },
        "memory": {
            "rss_before_bytes": rss_before,
            "rss_before_load_bytes": rss_before_load,
            "rss_after_load_bytes": rss_after_load,
            "rss_peak_approx_bytes": peak_rss,
            "rss_note": "RSS aproximado medido con psutil cuando esta disponible.",
            "vram": vram,
            "local_snapshot_size_bytes": estimate_snapshot_size_bytes(code_review),
        },
        "versions": version_info(),
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "paths": {
            "output_csv": str(output_csv_path),
            "output_parquet": str(output_parquet_path),
            "failures": str(failures_path),
            "summary": str(summary_path),
        },
        "remote_code_review": code_review,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return result


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frozen MERT embeddings for AIME manifest rows.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-parquet", type=Path, default=None)
    parser.add_argument("--failures", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        paths = load_extraction_paths(args.config)
        expected_embedding_dim = paths["expected_embedding_dim"]
        if expected_embedding_dim is not None and expected_embedding_dim != config.expected_embedding_dim:
            raise MertEmbeddingExtractionError(
                "embedding_extraction.expected_embedding_dim must match model.expected_embedding_dim"
            )
        result = extract_embeddings(
            config,
            output_csv_path=args.output_csv or paths["output_csv_path"],
            output_parquet_path=args.output_parquet or paths["output_parquet_path"],
            failures_path=args.failures or paths["failures_path"],
            summary_path=args.summary or paths["summary_path"],
            device_name=args.device,
            audio_dir=args.audio_dir,
            overwrite=args.overwrite,
            limit=args.limit,
            progress_every_rows=paths["progress_every_rows"],
            expected_rows=paths["expected_rows"],
            expected_output_dtype=paths["expected_output_dtype"],
            progress=not args.quiet,
        )
        print(json.dumps({key: result[key] for key in ["status", "counts", "validation", "timings"]}, indent=2, default=_json_default))
        return 0 if result["status"] == "satisfactory" else 1
    except Exception as exc:  # noqa: BLE001 - CLI should fail loudly for automation.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
