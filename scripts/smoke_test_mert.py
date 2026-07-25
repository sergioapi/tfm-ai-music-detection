from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.mfcc_svm_baseline import (  # type: ignore
        DATASET_NAME,
        PreprocessConfig,
        _decode_remote_audio_payload,
        _load_hf_audio_stream,
        find_local_audio,
        load_manifest,
        preprocess_audio_file,
        validate_description_splits,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from scripts/.
    from mfcc_svm_baseline import (  # type: ignore
        DATASET_NAME,
        PreprocessConfig,
        _decode_remote_audio_payload,
        _load_hf_audio_stream,
        find_local_audio,
        load_manifest,
        preprocess_audio_file,
        validate_description_splits,
    )


DEFAULT_CONFIG = Path("configs/mert_frozen_embeddings.yaml")
DEFAULT_OUTPUT = Path("data/processed/mert_smoke_test_result.json")
DEFAULT_SUMMARY = Path("docs/mert_smoke_test_summary.md")
REMOTE_CODE_FILES = (
    "configuration_MERT.py",
    "modeling_MERT.py",
    "config.json",
    "preprocessor_config.json",
)
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class MertSmokeTestError(RuntimeError):
    """Raised when the MERT smoke test cannot continue safely."""


@dataclass(frozen=True)
class SmokeTestConfig:
    model_id: str
    revision: str
    trust_remote_code: bool
    manifest_path: Path
    dataset_name: str
    sample_rate_hz: int
    total_duration_seconds: float
    window_duration_seconds: float
    num_windows: int
    expected_embedding_dim: int
    seed: int
    sample_pairs: int
    allowed_split: str
    preferred_ai_generators: tuple[str, ...]
    batch_size: int
    determinism_tolerance_abs: float
    output_path: Path
    summary_path: Path

    @property
    def total_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.total_duration_seconds))

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.window_duration_seconds))


def load_config(path: Path = DEFAULT_CONFIG) -> SmokeTestConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise MertSmokeTestError("PyYAML is required to read the smoke test config") from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise MertSmokeTestError(f"Invalid YAML config: {path}")

    model = raw["model"]
    dataset = raw["dataset"]
    audio = raw["audio_input"]
    experiment = raw["experiment"]
    smoke = raw.get("smoke_test", {})

    revision = str(model["immutable_revision"])
    validate_revision_sha(revision)
    trust_remote_code = bool(model.get("trust_remote_code", False))
    if not trust_remote_code:
        raise MertSmokeTestError("MERT must be loaded with trust_remote_code=True")

    return SmokeTestConfig(
        model_id=str(model["identifier"]),
        revision=revision,
        trust_remote_code=trust_remote_code,
        manifest_path=Path(dataset["manifest_path"]),
        dataset_name=str(dataset.get("name", DATASET_NAME)),
        sample_rate_hz=int(audio["sample_rate_hz"]),
        total_duration_seconds=float(audio["total_duration_seconds"]),
        window_duration_seconds=float(audio["window_duration_seconds"]),
        num_windows=int(audio["num_windows"]),
        expected_embedding_dim=int(model["expected_embedding_dim"]),
        seed=int(experiment.get("seed", 42)),
        sample_pairs=int(smoke.get("sample_pairs", 6)),
        allowed_split=str(smoke.get("allowed_split", "train")),
        preferred_ai_generators=tuple(smoke.get("preferred_ai_generators", ["Udio", "Riffusion"])),
        batch_size=int(smoke.get("batch_size", 1)),
        determinism_tolerance_abs=float(smoke.get("determinism_tolerance_abs", 1e-5)),
        output_path=Path(smoke.get("output_path", DEFAULT_OUTPUT)),
        summary_path=Path(smoke.get("summary_path", DEFAULT_SUMMARY)),
    )


def validate_revision_sha(revision: str) -> None:
    if revision == "main" or not REVISION_RE.match(revision):
        raise MertSmokeTestError(
            "Model revision must be an immutable 40-character hexadecimal SHA"
        )


def select_smoke_sample(
    manifest: pd.DataFrame,
    *,
    n_pairs: int,
    seed: int,
    allowed_split: str = "train",
    preferred_ai_generators: tuple[str, ...] = ("Udio", "Riffusion"),
) -> pd.DataFrame:
    if n_pairs <= 0:
        raise MertSmokeTestError("n_pairs must be positive")

    train = manifest[manifest["split"].eq(allowed_split)].copy()
    if train.empty:
        raise MertSmokeTestError(f"No rows found for split {allowed_split!r}")

    ai_rows = train[train["label"].eq(1)].copy()
    human_rows = train[train["label"].eq(0)].copy()
    human_by_description = {str(row["description"]): row for _, row in human_rows.iterrows()}
    ai_rows = ai_rows[ai_rows["description"].astype(str).isin(human_by_description)]
    if ai_rows.empty:
        raise MertSmokeTestError("No train AI rows have a paired human description")

    available_models = sorted(str(value) for value in ai_rows["model"].unique())
    ordered_models: list[str] = []
    for model in preferred_ai_generators:
        if model in available_models and model not in ordered_models:
            ordered_models.append(model)
    for model in available_models:
        if model not in ordered_models:
            ordered_models.append(model)
    ordered_models = ordered_models[:n_pairs]
    if len(ordered_models) < n_pairs:
        raise MertSmokeTestError(
            f"Requested {n_pairs} AI generators, only {len(ordered_models)} available"
        )

    rng = random.Random(seed)
    selected_rows: list[dict[str, Any]] = []
    for model in ordered_models:
        candidates = ai_rows[ai_rows["model"].eq(model)].sort_values(["description", "id"])
        ai_record = candidates.iloc[rng.randrange(len(candidates))].to_dict()
        human_record = dict(human_by_description[str(ai_record["description"])])
        selected_rows.append(human_record)
        selected_rows.append(ai_record)

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    validate_paired_sample(selected, allowed_split=allowed_split)
    return selected


def validate_paired_sample(sample: pd.DataFrame, *, allowed_split: str = "train") -> None:
    if sample.empty:
        raise MertSmokeTestError("Smoke test sample is empty")
    if not sample["split"].eq(allowed_split).all():
        raise MertSmokeTestError(f"Smoke test sample must contain only {allowed_split} rows")
    grouped = sample.groupby("description", observed=False)
    if not grouped.size().eq(2).all():
        raise MertSmokeTestError("Each selected description must have exactly two rows")
    labels_ok = grouped["label"].apply(lambda values: sorted(values.astype(int).tolist()) == [0, 1])
    if not labels_ok.all():
        raise MertSmokeTestError("Each selected description must pair one human and one AI row")


def sample_records(sample: pd.DataFrame) -> list[dict[str, Any]]:
    columns = ["id", "description", "label", "model", "split"]
    return [
        {column: _json_value(row[column]) for column in columns}
        for _, row in sample[columns].iterrows()
    ]


def split_clip_into_windows(signal: np.ndarray, config: SmokeTestConfig) -> list[np.ndarray]:
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim != 1:
        raise MertSmokeTestError(f"Expected mono signal, found shape {array.shape}")
    if array.shape[0] != config.total_samples:
        raise MertSmokeTestError(
            f"Expected {config.total_samples} samples, found {array.shape[0]}"
        )
    expected_total = config.window_samples * config.num_windows
    if expected_total != config.total_samples:
        raise MertSmokeTestError(
            f"Window configuration yields {expected_total} samples, expected {config.total_samples}"
        )
    windows = np.split(array, config.num_windows)
    for window in windows:
        if window.shape != (config.window_samples,):
            raise MertSmokeTestError(
                f"Expected window shape ({config.window_samples},), found {window.shape}"
            )
    return [window.astype(np.float32, copy=False) for window in windows]


def pool_last_hidden_state(hidden_state: Any, expected_dim: int) -> np.ndarray:
    shape = tuple(int(value) for value in hidden_state.shape)
    if len(shape) != 3 or shape[0] != 1 or shape[-1] != expected_dim:
        raise MertSmokeTestError(
            f"Expected last_hidden_state shape (1, time, {expected_dim}), found {shape}"
        )
    if hasattr(hidden_state, "detach"):
        pooled = hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()
    else:
        pooled = np.asarray(hidden_state).mean(axis=1).squeeze(0)
    pooled = np.asarray(pooled, dtype=np.float32)
    if pooled.shape != (expected_dim,):
        raise MertSmokeTestError(f"Expected pooled shape ({expected_dim},), found {pooled.shape}")
    ensure_finite(pooled, "window embedding")
    return pooled


def aggregate_window_embeddings(embeddings: list[np.ndarray], expected_dim: int) -> np.ndarray:
    if not embeddings:
        raise MertSmokeTestError("No window embeddings were produced")
    stacked = np.stack([np.asarray(item, dtype=np.float32) for item in embeddings], axis=0)
    if stacked.shape != (len(embeddings), expected_dim):
        raise MertSmokeTestError(
            f"Expected stacked embeddings (*, {expected_dim}), found {stacked.shape}"
        )
    ensure_finite(stacked, "window embeddings")
    clip_embedding = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
    if clip_embedding.shape != (expected_dim,):
        raise MertSmokeTestError(
            f"Expected clip embedding shape ({expected_dim},), found {clip_embedding.shape}"
        )
    ensure_finite(clip_embedding, "clip embedding")
    return clip_embedding


def ensure_finite(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise MertSmokeTestError(f"{name} contains NaN or infinite values")


def preprocess_record_audio(
    record: pd.Series,
    config: SmokeTestConfig,
    *,
    audio_dir: Path | None,
    stream_by_id: dict[str, Any] | None,
) -> np.ndarray:
    preprocess_config = PreprocessConfig(
        target_sample_rate=config.sample_rate_hz,
        duration_seconds=config.total_duration_seconds,
    )
    audio_id = str(record["id"])
    if audio_dir is not None:
        return preprocess_audio_file(find_local_audio(audio_dir, audio_id), config=preprocess_config)
    if stream_by_id is None or audio_id not in stream_by_id:
        raise MertSmokeTestError(f"Audio id {audio_id!r} was not found in AIME stream")
    return _decode_remote_audio_payload(stream_by_id[audio_id]["audio"], preprocess_config)


def load_remote_rows_for_sample(
    sample: pd.DataFrame,
    *,
    dataset_name: str,
    dataset_revision: str,
    progress: bool = True,
) -> dict[str, Any]:
    pending = {str(value) for value in sample["id"].tolist()}
    rows: dict[str, Any] = {}
    rows_scanned = 0
    stream = _load_hf_audio_stream(dataset_name, dataset_revision)
    _log(
        f"Streaming AIME until {len(pending)} selected audio rows are found...",
        enabled=progress,
    )
    for row in stream:
        rows_scanned += 1
        audio_id = str(row["id"])
        if audio_id in pending:
            rows[audio_id] = row
            pending.remove(audio_id)
            _log(
                f"Found audio id {audio_id}; pending={len(pending)} rows_scanned={rows_scanned}",
                enabled=progress,
            )
            if not pending:
                break
        elif rows_scanned % 500 == 0:
            _log(
                f"Scanned {rows_scanned} AIME rows; pending={len(pending)}",
                enabled=progress,
            )
    if pending:
        raise MertSmokeTestError(f"Could not find audio ids in AIME stream: {sorted(pending)}")
    _log(f"Finished AIME streaming after {rows_scanned} rows.", enabled=progress)
    return rows


def review_remote_code(config: SmokeTestConfig) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise MertSmokeTestError("huggingface_hub is required for remote code review") from exc

    files = []
    for filename in REMOTE_CODE_FILES:
        start = time.perf_counter()
        path = Path(
            hf_hub_download(
                repo_id=config.model_id,
                filename=filename,
                revision=config.revision,
            )
        )
        text = path.read_text(encoding="utf-8")
        item: dict[str, Any] = {
            "filename": filename,
            "local_path": str(path),
            "sha256": sha256_file(path),
            "download_or_cache_seconds": time.perf_counter() - start,
        }
        if filename.endswith(".py"):
            item.update(inspect_python_source(text))
        elif filename == "config.json":
            parsed = json.loads(text)
            item.update(
                {
                    "auto_map": parsed.get("auto_map", {}),
                    "feature_extractor_cqt": parsed.get("feature_extractor_cqt"),
                    "hidden_size": parsed.get("hidden_size"),
                    "sample_rate": parsed.get("sample_rate"),
                }
            )
        elif filename == "preprocessor_config.json":
            parsed = json.loads(text)
            item.update(
                {
                    "feature_extractor_type": parsed.get("feature_extractor_type"),
                    "sampling_rate": parsed.get("sampling_rate"),
                    "do_normalize": parsed.get("do_normalize"),
                }
            )
        files.append(item)
    return {
        "files": files,
        "findings": summarize_remote_code_review(files),
        "conclusion": (
            "Se inspeccionaron los archivos fijados antes de cargar con trust_remote_code=True. "
            "Esta revision no demuestra seguridad absoluta; solo registra imports, auto_map y "
            "patrones de riesgo observados."
        ),
    }


def inspect_python_source(text: str) -> dict[str, Any]:
    imports = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
    risk_patterns = [
        "subprocess",
        "os.system",
        "Popen",
        "requests",
        "urllib",
        "socket",
        "exec(",
        "eval(",
    ]
    observed = [pattern for pattern in risk_patterns if pattern in text]
    io_patterns = [pattern for pattern in ["open(", "Path(", "load_state_dict"] if pattern in text]
    optional_dependencies = ["nnAudio"] if "nnAudio" in text else []
    return {
        "imports": imports,
        "risk_patterns_observed": observed,
        "io_patterns_observed": io_patterns,
        "optional_dependencies": optional_dependencies,
    }


def summarize_remote_code_review(files: list[dict[str, Any]]) -> dict[str, Any]:
    imports = sorted({item for file_info in files for item in file_info.get("imports", [])})
    risk_patterns = sorted(
        {item for file_info in files for item in file_info.get("risk_patterns_observed", [])}
    )
    optional_dependencies = sorted(
        {item for file_info in files for item in file_info.get("optional_dependencies", [])}
    )
    auto_map = {}
    feature_extractor_cqt = None
    for file_info in files:
        if file_info["filename"] == "config.json":
            auto_map = file_info.get("auto_map", {})
            feature_extractor_cqt = file_info.get("feature_extractor_cqt")
    return {
        "imports": imports,
        "auto_map": auto_map,
        "risk_patterns_observed": risk_patterns,
        "optional_dependencies": optional_dependencies,
        "feature_extractor_cqt": feature_extractor_cqt,
        "nnAudio_required_by_config": bool(feature_extractor_cqt),
        "unexpected_system_or_network_patterns_observed": bool(risk_patterns),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise MertSmokeTestError("CUDA was requested but is not available")
    return torch.device(requested)


def load_processor_and_model(config: SmokeTestConfig, device: Any) -> tuple[Any, Any, dict[str, float]]:
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise MertSmokeTestError("transformers is required to load MERT") from exc

    timings: dict[str, float] = {}
    start = time.perf_counter()
    processor = AutoFeatureExtractor.from_pretrained(
        config.model_id,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    timings["processor_load_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    model = AutoModel.from_pretrained(
        config.model_id,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    timings["model_load_seconds"] = time.perf_counter() - start
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model, timings


def extract_window_embedding(
    window: np.ndarray,
    *,
    processor: Any,
    model: Any,
    device: Any,
    config: SmokeTestConfig,
) -> tuple[np.ndarray, float]:
    import torch

    if window.shape != (config.window_samples,):
        raise MertSmokeTestError(
            f"Expected window shape ({config.window_samples},), found {window.shape}"
        )
    inputs = processor(window, sampling_rate=config.sample_rate_hz, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return pool_last_hidden_state(outputs.last_hidden_state, config.expected_embedding_dim), elapsed


def extract_clip_embedding(
    signal: np.ndarray,
    *,
    processor: Any,
    model: Any,
    device: Any,
    config: SmokeTestConfig,
) -> tuple[np.ndarray, list[float]]:
    window_embeddings = []
    window_seconds = []
    for window in split_clip_into_windows(signal, config):
        embedding, elapsed = extract_window_embedding(
            window,
            processor=processor,
            model=model,
            device=device,
            config=config,
        )
        window_embeddings.append(embedding)
        window_seconds.append(elapsed)
    return aggregate_window_embeddings(window_embeddings, config.expected_embedding_dim), window_seconds


def max_abs_difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def current_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def version_info() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for module_name in ["torch", "transformers", "huggingface_hub", "yaml", "psutil"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except ImportError:
            versions[module_name] = None
    return versions


def cuda_info(device: Any) -> dict[str, Any]:
    import torch

    if device.type != "cuda":
        return {
            "available": torch.cuda.is_available(),
            "device": None,
            "name": None,
            "total_memory_bytes": None,
            "max_memory_allocated_bytes": None,
            "max_memory_reserved_bytes": None,
        }
    props = torch.cuda.get_device_properties(device)
    return {
        "available": True,
        "device": str(device),
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def estimate_snapshot_size_bytes(code_review: dict[str, Any]) -> int | None:
    paths = [Path(item["local_path"]) for item in code_review.get("files", []) if item.get("local_path")]
    snapshot_dirs = {path.parent for path in paths if path.exists()}
    if not snapshot_dirs:
        return None
    total = 0
    for snapshot_dir in snapshot_dirs:
        for child in snapshot_dir.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    return int(total)


def run_smoke_test(
    config: SmokeTestConfig,
    *,
    device_name: str,
    audio_dir: Path | None,
    max_pairs: int | None,
    progress: bool = True,
) -> dict[str, Any]:
    import torch

    total_start = time.perf_counter()
    _log("Loading manifest and selecting deterministic train sample...", enabled=progress)
    rss_before = current_rss_bytes()
    manifest = load_manifest(config.manifest_path)
    validate_description_splits(manifest)
    dataset_revision = str(manifest["dataset_revision"].dropna().iloc[0])
    sample = select_smoke_sample(
        manifest,
        n_pairs=max_pairs or config.sample_pairs,
        seed=config.seed,
        allowed_split=config.allowed_split,
        preferred_ai_generators=config.preferred_ai_generators,
    )
    _log(
        f"Selected {len(sample)} train rows ({len(sample) // 2} pairs).",
        enabled=progress,
    )

    _log("Reviewing fixed remote code/config files from Hugging Face...", enabled=progress)
    code_review = review_remote_code(config)
    device = resolve_device(device_name)
    _log(f"Resolved device: {device}", enabled=progress)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rss_before_load = current_rss_bytes()
    _log("Loading feature extractor and MERT model; this may download model files if not cached...", enabled=progress)
    processor, model, load_timings = load_processor_and_model(config, device)
    _log(
        "Loaded processor/model "
        f"(processor={load_timings['processor_load_seconds']:.2f}s, "
        f"model={load_timings['model_load_seconds']:.2f}s).",
        enabled=progress,
    )
    rss_after_load = current_rss_bytes()
    model_frozen = all(not parameter.requires_grad for parameter in model.parameters())
    model_eval = not bool(model.training)
    if not model_frozen or not model_eval:
        raise MertSmokeTestError("Model must be frozen and in eval mode")

    stream_rows = None
    download_seconds = None
    if audio_dir is None:
        start = time.perf_counter()
        stream_rows = load_remote_rows_for_sample(
            sample,
            dataset_name=config.dataset_name,
            dataset_revision=dataset_revision,
            progress=progress,
        )
        download_seconds = time.perf_counter() - start
    else:
        _log(f"Using local audio directory: {audio_dir}", enabled=progress)

    records = []
    errors = []
    preprocessing_seconds = []
    inference_clip_seconds = []
    inference_window_seconds = []
    peak_candidates = [rss_before, rss_before_load, rss_after_load]
    peak_rss = max(value for value in peak_candidates if value is not None) if any(peak_candidates) else None

    warmup_record = sample.iloc[0]
    _log(f"Running warm-up on id={warmup_record['id']}...", enabled=progress)
    warmup_signal = preprocess_record_audio(warmup_record, config, audio_dir=audio_dir, stream_by_id=stream_rows)
    warmup_embedding, warmup_windows = extract_clip_embedding(
        warmup_signal,
        processor=processor,
        model=model,
        device=device,
        config=config,
    )
    warmup = {
        "id": str(warmup_record["id"]),
        "embedding_shape": list(warmup_embedding.shape),
        "window_inference_seconds": warmup_windows,
        "clip_inference_seconds": float(sum(warmup_windows)),
    }

    _log("Running determinism check on the warm-up sample...", enabled=progress)
    determinism_start = time.perf_counter()
    det_first, _ = extract_clip_embedding(warmup_signal, processor=processor, model=model, device=device, config=config)
    det_second, _ = extract_clip_embedding(warmup_signal, processor=processor, model=model, device=device, config=config)
    determinism_max_diff = max_abs_difference(det_first, det_second)
    determinism = {
        "sample_id": str(warmup_record["id"]),
        "max_abs_difference": determinism_max_diff,
        "tolerance_abs": config.determinism_tolerance_abs,
        "passed": bool(determinism_max_diff <= config.determinism_tolerance_abs),
        "seconds": time.perf_counter() - determinism_start,
    }

    _log("Processing selected smoke-test records...", enabled=progress)
    for index, record in sample.iterrows():
        item = {key: _json_value(record[key]) for key in ["id", "description", "label", "model", "split"]}
        _log(
            f"[{index + 1}/{len(sample)}] id={record['id']} label={record['label']} model={record['model']}",
            enabled=progress,
        )
        try:
            start = time.perf_counter()
            signal = preprocess_record_audio(record, config, audio_dir=audio_dir, stream_by_id=stream_rows)
            preprocessing_elapsed = time.perf_counter() - start
            preprocessing_seconds.append(preprocessing_elapsed)
            windows = split_clip_into_windows(signal, config)
            embedding, window_elapsed = extract_clip_embedding(
                signal,
                processor=processor,
                model=model,
                device=device,
                config=config,
            )
            inference_window_seconds.extend(window_elapsed)
            inference_clip_seconds.append(sum(window_elapsed))
            item.update(
                {
                    "clip_samples": int(signal.shape[0]),
                    "window_shapes": [list(window.shape) for window in windows],
                    "embedding_shape": list(embedding.shape),
                    "finite": bool(np.isfinite(embedding).all()),
                    "mean": float(np.mean(embedding)),
                    "std": float(np.std(embedding)),
                    "sha256": hashlib.sha256(embedding.tobytes()).hexdigest(),
                    "preprocessing_seconds": preprocessing_elapsed,
                    "window_inference_seconds": window_elapsed,
                    "clip_inference_seconds": float(sum(window_elapsed)),
                }
            )
        except Exception as exc:  # noqa: BLE001 - every sample error is part of the report.
            error = {"id": str(record["id"]), "type": type(exc).__name__, "message": str(exc)}
            item["error"] = error
            errors.append(error)
        records.append(item)
        rss_now = current_rss_bytes()
        if rss_now is not None:
            peak_rss = rss_now if peak_rss is None else max(peak_rss, rss_now)

    vram = cuda_info(device)
    processed = len(records) - len(errors)
    all_shapes_ok = all(item.get("embedding_shape") == [config.expected_embedding_dim] for item in records if "error" not in item)
    all_finite = all(item.get("finite") for item in records if "error" not in item) and processed > 0
    verdict = "satisfactory" if not errors and all_shapes_ok and all_finite and determinism["passed"] else "failed"

    _log(f"Smoke test finished with verdict={verdict}.", enabled=progress)
    return {
        "model": {
            "identifier": config.model_id,
            "revision": config.revision,
            "trust_remote_code": config.trust_remote_code,
            "expected_embedding_dim": config.expected_embedding_dim,
        },
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "versions": version_info(),
        "system": {"platform": platform.platform(), "processor": platform.processor()},
        "device": {
            "requested": device_name,
            "resolved": str(device),
            "type": device.type,
            "batch_size": config.batch_size,
            "mixed_precision": False,
            "quantization": False,
        },
        "dataset": {
            "name": config.dataset_name,
            "manifest_path": str(config.manifest_path),
            "dataset_revision_from_manifest": dataset_revision,
            "allowed_split": config.allowed_split,
        },
        "remote_code_review": code_review,
        "sample_selection": {
            "seed": config.seed,
            "requested_pairs": max_pairs or config.sample_pairs,
            "n_rows": int(len(sample)),
            "records": sample_records(sample),
        },
        "model_state": {"eval_mode": model_eval, "all_parameters_frozen": model_frozen},
        "audio": {
            "sample_rate_hz": config.sample_rate_hz,
            "total_duration_seconds": config.total_duration_seconds,
            "window_duration_seconds": config.window_duration_seconds,
            "num_windows": config.num_windows,
            "total_samples": config.total_samples,
            "window_samples": config.window_samples,
        },
        "warmup": warmup,
        "determinism": determinism,
        "records": records,
        "embedding_checks": {
            "processed": processed,
            "failed": len(errors),
            "expected_shape": [config.expected_embedding_dim],
            "all_shapes_ok": bool(all_shapes_ok),
            "all_finite": bool(all_finite),
        },
        "timings": {
            "download_sample_audio_seconds": download_seconds,
            **load_timings,
            "preprocessing_seconds_total": float(sum(preprocessing_seconds)),
            "preprocessing_seconds_mean": _mean(preprocessing_seconds),
            "inference_window_seconds_mean": _mean(inference_window_seconds),
            "inference_clip_seconds_mean": _mean(inference_clip_seconds),
            "total_smoke_test_seconds": time.perf_counter() - total_start,
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
        "errors": errors,
        "verdict": verdict,
    }


def write_json_result(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def write_markdown_summary(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Smoke test tecnico de MERT",
        "",
        f"- Estado: `{result.get('verdict', 'unknown')}`.",
        f"- Fecha UTC: `{result.get('date_utc', 'no disponible')}`.",
        f"- Modelo: `{result.get('model', {}).get('identifier')}`.",
        f"- Revision: `{result.get('model', {}).get('revision')}`.",
        f"- Dispositivo: `{result.get('device', {}).get('resolved')}`.",
        f"- Muestras procesadas: `{result.get('embedding_checks', {}).get('processed')}`.",
        f"- Fallos: `{result.get('embedding_checks', {}).get('failed')}`.",
        "",
        "## Resultado",
        "",
        f"- Formas correctas: `{result.get('embedding_checks', {}).get('all_shapes_ok')}`.",
        f"- Embeddings finitos: `{result.get('embedding_checks', {}).get('all_finite')}`.",
        f"- Determinismo: `{result.get('determinism', {}).get('passed')}`.",
        f"- Diferencia maxima absoluta: `{result.get('determinism', {}).get('max_abs_difference')}`.",
        "",
        "## Tiempos",
        "",
        f"- Carga del procesador: `{result.get('timings', {}).get('processor_load_seconds')}` s.",
        f"- Carga del modelo: `{result.get('timings', {}).get('model_load_seconds')}` s.",
        f"- Inferencia media por ventana: `{result.get('timings', {}).get('inference_window_seconds_mean')}` s.",
        f"- Inferencia media por clip: `{result.get('timings', {}).get('inference_clip_seconds_mean')}` s.",
        f"- Total: `{result.get('timings', {}).get('total_smoke_test_seconds')}` s.",
        "",
        "## Memoria",
        "",
        f"- RSS antes de cargar: `{result.get('memory', {}).get('rss_before_load_bytes')}` bytes.",
        f"- RSS despues de cargar: `{result.get('memory', {}).get('rss_after_load_bytes')}` bytes.",
        f"- Pico RSS aproximado: `{result.get('memory', {}).get('rss_peak_approx_bytes')}` bytes.",
        f"- VRAM maxima asignada: `{result.get('memory', {}).get('vram', {}).get('max_memory_allocated_bytes')}` bytes.",
        f"- VRAM maxima reservada: `{result.get('memory', {}).get('vram', {}).get('max_memory_reserved_bytes')}` bytes.",
        "",
        "## Observaciones",
        "",
        "Este smoke test comprueba viabilidad tecnica, no rendimiento predictivo. No calcula balanced accuracy, F1, ROC-AUC ni matriz de confusion.",
        "",
    ]
    errors = result.get("errors") or []
    if errors:
        lines.extend(["## Errores", ""])
        for error in errors:
            lines.append(f"- `{error.get('id')}`: {error.get('type')}: {error.get('message')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pending_summary(path: Path, reason: str) -> None:
    lines = [
        "# Smoke test tecnico de MERT",
        "",
        "- Estado: implementacion preparada; ejecucion pendiente.",
        "- Modelo previsto: `m-a-p/MERT-v1-95M`.",
        "- Revision prevista: `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.",
        "- Split permitido: `train`.",
        "",
        "## Bloqueo",
        "",
        reason,
        "",
        "## Comandos para ejecutar",
        "",
        "```powershell",
        "python scripts/smoke_test_mert.py --device cpu --max-pairs 1",
        "python scripts/smoke_test_mert.py --device cpu",
        "python scripts/smoke_test_mert.py --device auto",
        "python scripts/smoke_test_mert.py --device cuda",
        "python scripts/smoke_test_mert.py --device cpu --audio-dir data/audio/aime_raw",
        "```",
        "",
        "El resultado estructurado se escribira por defecto en `data/processed/mert_smoke_test_result.json`, una ruta ignorada por Git.",
        "",
        "La fase no debe marcarse como satisfactoria hasta cargar el modelo real y procesar al menos una muestra real de AIME en CPU.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _log(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(f"[mert-smoke] {message}", flush=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a controlled MERT smoke test.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--write-pending-summary", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        output_path = args.output or config.output_path
        summary_path = args.summary or config.summary_path
        if args.write_pending_summary:
            write_pending_summary(
                summary_path,
                "Ejecucion no realizada en este comando; se ha generado solo el resumen pendiente.",
            )
            return 0
        result = run_smoke_test(
            config,
            device_name=args.device,
            audio_dir=args.audio_dir,
            max_pairs=args.max_pairs,
            progress=not args.quiet,
        )
        write_json_result(result, output_path)
        write_markdown_summary(result, summary_path)
        return 0 if result["verdict"] == "satisfactory" else 1
    except Exception as exc:  # noqa: BLE001 - CLI must return non-zero on critical failures.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
