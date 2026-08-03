from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import statistics
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import psutil
import sklearn
import soundfile as sf

from app.inference.service import AudioInferenceService


DEFAULT_AUDIO_PATHS = (
    Path("data/audio/aime_acceptance_raw/audioldm-2-large_01631.wav"),
    Path("data/audio/aime_acceptance_raw/riffusion_02631.wav"),
    Path("data/audio/aime_acceptance_raw/udio_04501.wav"),
    Path("data/audio/aime_acceptance_raw/suno-v3_05001.wav"),
    Path("data/audio/aime_acceptance_raw/mtg-jamendo_06001.wav"),
)
DEFAULT_OUTPUT_JSON = Path("docs/benchmarks/mfcc_svm_inference_local.json")
DEFAULT_OUTPUT_MARKDOWN = Path("docs/benchmarks/mfcc_svm_inference_local.md")
SCORE_TOLERANCE = 1.0e-9


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    numbers = [float(value) for value in values]
    return {
        "mean": float(statistics.fmean(numbers)),
        "median": float(statistics.median(numbers)),
        "min": float(min(numbers)),
        "max": float(max(numbers)),
        "std": float(statistics.pstdev(numbers)),
    }


def determinism_check(
    repetitions: list[dict[str, Any]],
    score_tolerance: float = SCORE_TOLERANCE,
) -> dict[str, Any]:
    if not repetitions:
        return {
            "is_deterministic": False,
            "reason": "No repetitions were recorded",
            "values": {},
        }

    first = repetitions[0]
    fragment_counts = [item["n_fragments"] for item in repetitions]
    labels = [item["predicted_label"] for item in repetitions]
    scores = [float(item["ai_score"]) for item in repetitions]
    same_fragments = all(value == first["n_fragments"] for value in fragment_counts)
    same_labels = all(value == first["predicted_label"] for value in labels)
    same_scores = all(abs(value - scores[0]) <= score_tolerance for value in scores)
    return {
        "is_deterministic": bool(same_fragments and same_labels and same_scores),
        "same_fragment_count": bool(same_fragments),
        "same_predicted_label": bool(same_labels),
        "same_ai_score_within_tolerance": bool(same_scores),
        "score_tolerance": score_tolerance,
        "values": {
            "n_fragments": fragment_counts,
            "predicted_label": labels,
            "ai_score": scores,
        },
    }


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class RssSampler:
    def __init__(self, process: psutil.Process, interval_seconds: float = 0.005) -> None:
        self.process = process
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_rss_bytes = int(process.memory_info().rss)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        self.peak_rss_bytes = max(self.peak_rss_bytes, int(self.process.memory_info().rss))
        return self.peak_rss_bytes

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_bytes = max(self.peak_rss_bytes, int(self.process.memory_info().rss))
            self._stop.wait(self.interval_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MFCC + SVM audio inference.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--audio", type=Path, action="append", default=None)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    return parser.parse_args(argv)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be greater than zero")

    repo_root = Path.cwd().resolve()
    model_path = resolve_model_path(args.model_path)
    audio_paths = tuple(path.expanduser().resolve() for path in (args.audio or DEFAULT_AUDIO_PATHS))
    validate_input_files((model_path, *audio_paths))

    process = psutil.Process()
    rss_initial = int(process.memory_info().rss)
    environment = collect_environment()
    artifact = collect_artifact_info(model_path, repo_root)

    load_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        service = AudioInferenceService(model_path=model_path)
    service_load_seconds = time.perf_counter() - load_start
    rss_after_load = int(process.memory_info().rss)
    load_warnings = warning_records(caught)
    fragment_duration_seconds = service.config.fragment_duration_seconds
    audio_metadata = [
        collect_audio_info(path, fragment_duration_seconds, repo_root)
        for path in audio_paths
    ]

    first_inference = measure_prediction(
        service=service,
        audio_path=audio_paths[0],
        audio_info=audio_metadata[0],
        process=process,
    )

    audio_results = []
    for path, info in zip(audio_paths, audio_metadata):
        repetitions = [
            measure_prediction(service, path, info, process)
            for _ in range(args.repetitions)
        ]
        audio_results.append(
            {
                "audio": info,
                "repetitions": repetitions,
                "summary": summarize_repetitions(repetitions),
                "determinism": determinism_check(repetitions),
            }
        )

    return {
        "environment": environment,
        "artifact": artifact,
        "service_load": {
            "load_seconds": float(service_load_seconds),
            "rss_before_bytes": rss_initial,
            "rss_after_bytes": rss_after_load,
            "rss_increase_bytes": int(rss_after_load - rss_initial),
            "warnings": load_warnings,
        },
        "first_inference": first_inference,
        "audios": audio_results,
        "benchmark": {
            "repetitions": int(args.repetitions),
            "first_inference_excluded_from_hot_statistics": True,
            "single_service_instance": True,
            "score_tolerance": SCORE_TOLERANCE,
        },
    }


def resolve_model_path(model_path: Path | None) -> Path:
    candidate = model_path
    if candidate is None:
        env_path = os.environ.get("MODEL_PATH")
        if not env_path:
            raise ValueError("Model path was not provided and MODEL_PATH is not set")
        candidate = Path(env_path)
    return candidate.expanduser().resolve()


def validate_input_files(paths: tuple[Path, ...]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist: {missing}")
    not_files = [str(path) for path in paths if not path.is_file()]
    if not_files:
        raise FileNotFoundError(f"Input paths are not files: {not_files}")


def collect_environment() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "librosa": librosa.__version__,
        "soundfile": sf.__version__,
        "libsndfile": getattr(sf, "__libsndfile_version__", None),
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "psutil": psutil.__version__,
        "cpu": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_bytes": int(memory.total),
    }


def collect_artifact_info(model_path: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "path": relative_to_root(model_path, repo_root),
        "size_bytes": int(model_path.stat().st_size),
        "sha256": sha256_file(model_path),
    }


def collect_audio_info(
    path: Path,
    fragment_duration_seconds: float,
    repo_root: Path,
) -> dict[str, Any]:
    with sf.SoundFile(path) as handle:
        duration = float(handle.frames / handle.samplerate)
        return {
            "path": relative_to_root(path, repo_root),
            "name": path.name,
            "extension": path.suffix,
            "size_bytes": int(path.stat().st_size),
            "duration_seconds": duration,
            "sample_rate": int(handle.samplerate),
            "channels": int(handle.channels),
            "expected_fragments": int(math.ceil(duration / fragment_duration_seconds)),
        }


def measure_prediction(
    service: AudioInferenceService,
    audio_path: Path,
    audio_info: dict[str, Any],
    process: psutil.Process,
) -> dict[str, Any]:
    rss_before = int(process.memory_info().rss)
    sampler = RssSampler(process)
    wall_start = time.perf_counter()
    sampler.start()
    try:
        prediction = service.predict_file(audio_path)
    finally:
        peak_rss = sampler.stop()
    wall_clock_seconds = time.perf_counter() - wall_start
    rss_after = int(process.memory_info().rss)
    total_seconds = float(prediction.timings.total_seconds)
    duration_seconds = float(audio_info["duration_seconds"])
    real_time_factor = total_seconds / duration_seconds
    audio_seconds_per_processing_second = (
        duration_seconds / total_seconds if total_seconds > 0.0 else float("inf")
    )
    return {
        "ai_score": float(prediction.ai_score),
        "predicted_label": int(prediction.predicted_label),
        "predicted_class": prediction.predicted_class,
        "n_fragments": int(prediction.n_fragments),
        "wall_clock_seconds": float(wall_clock_seconds),
        "decode_seconds": float(prediction.timings.decode_seconds),
        "segmentation_seconds": float(prediction.timings.segmentation_seconds),
        "preprocessing_seconds": float(prediction.timings.preprocessing_seconds),
        "mfcc_seconds": float(prediction.timings.mfcc_seconds),
        "prediction_seconds": float(prediction.timings.prediction_seconds),
        "aggregation_seconds": float(prediction.timings.aggregation_seconds),
        "total_seconds": total_seconds,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_peak_bytes": int(peak_rss),
        "rss_peak_increase_bytes": int(max(0, peak_rss - rss_before)),
        "real_time_factor": float(real_time_factor),
        "audio_seconds_per_processing_second": float(audio_seconds_per_processing_second),
    }


def summarize_repetitions(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "wall_clock_seconds",
        "total_seconds",
        "decode_seconds",
        "preprocessing_seconds",
        "mfcc_seconds",
        "prediction_seconds",
        "real_time_factor",
        "rss_peak_bytes",
    )
    return {
        field: summarize_values([float(item[field]) for item in repetitions])
        for field in fields
    }


def warning_records(caught: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in caught
    ]


def render_markdown(result: dict[str, Any]) -> str:
    first = result["first_inference"]
    lines = [
        "# MFCC + SVM inference benchmark",
        "",
        "## Entorno",
        "",
        f"- Fecha UTC: `{result['environment']['timestamp_utc']}`.",
        f"- Sistema: `{result['environment']['platform']}`.",
        f"- Python: `{result['environment']['python']}`.",
        f"- NumPy: `{result['environment']['numpy']}`.",
        f"- librosa: `{result['environment']['librosa']}`.",
        f"- soundfile/libsndfile: `{result['environment']['soundfile']}` / `{result['environment']['libsndfile']}`.",
        f"- scikit-learn: `{result['environment']['scikit_learn']}`.",
        f"- joblib: `{result['environment']['joblib']}`.",
        f"- psutil: `{result['environment']['psutil']}`.",
        f"- RAM total: `{format_bytes(result['environment']['ram_total_bytes'])}`.",
        "",
        "## Artefacto",
        "",
        f"- Ruta: `{result['artifact']['path']}`.",
        f"- Tamaño: `{format_bytes(result['artifact']['size_bytes'])}`.",
        f"- SHA-256: `{result['artifact']['sha256']}`.",
        "",
        "## Reproducción",
        "",
        "```powershell",
        '$env:PYTHONPATH="backend"',
        '$env:MODEL_PATH=(Resolve-Path "data/models/mfcc_svm_baseline.joblib").Path',
        r".\.venv\Scripts\python.exe scripts/benchmark_mfcc_inference.py",
        "```",
        "",
        "## Tiempo y memoria de carga",
        "",
        f"- Carga del servicio: `{result['service_load']['load_seconds']:.4f} s`.",
        f"- RSS inicial: `{format_bytes(result['service_load']['rss_before_bytes'])}`.",
        f"- RSS después de cargar: `{format_bytes(result['service_load']['rss_after_bytes'])}`.",
        f"- Incremento RSS: `{format_bytes(result['service_load']['rss_increase_bytes'])}`.",
        "",
        "## Advertencias de carga",
        "",
    ]
    load_warnings = result["service_load"]["warnings"]
    if load_warnings:
        for warning in load_warnings:
            lines.append(f"- `{warning['category']}`: {warning['message']}")
    else:
        lines.append("- No se registraron warnings de carga.")

    lines.extend(
        [
            "",
            "## Primera inferencia",
            "",
            f"- Archivo: `{result['audios'][0]['audio']['name']}`.",
            f"- Tiempo total: `{first['total_seconds']:.4f} s`.",
            f"- Wall clock: `{first['wall_clock_seconds']:.4f} s`.",
            f"- Fragmentos: `{first['n_fragments']}`.",
            f"- RSS pico: `{format_bytes(first['rss_peak_bytes'])}`.",
            "",
            "## Resultados en caliente por audio",
            "",
            "| archivo | duración | tamaño | canales | frecuencia | fragmentos | mediana total | máximo total | RTF medio | pico RSS máximo |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in result["audios"]:
        audio = item["audio"]
        summary = item["summary"]
        lines.append(
            "| "
            f"`{audio['name']}` | "
            f"{audio['duration_seconds']:.3f} s | "
            f"{format_bytes(audio['size_bytes'])} | "
            f"{audio['channels']} | "
            f"{audio['sample_rate']} | "
            f"{audio['expected_fragments']} | "
            f"{summary['total_seconds']['median']:.4f} s | "
            f"{summary['total_seconds']['max']:.4f} s | "
            f"{summary['real_time_factor']['mean']:.5f} | "
            f"{format_bytes(summary['rss_peak_bytes']['max'])} |"
        )

    lines.extend(
        [
            "",
            "## Tiempo por etapa",
            "",
            "| archivo | decode mediana | preprocesado mediana | MFCC mediana | predicción mediana |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in result["audios"]:
        audio = item["audio"]
        summary = item["summary"]
        lines.append(
            "| "
            f"`{audio['name']}` | "
            f"{summary['decode_seconds']['median']:.4f} s | "
            f"{summary['preprocessing_seconds']['median']:.4f} s | "
            f"{summary['mfcc_seconds']['median']:.4f} s | "
            f"{summary['prediction_seconds']['median']:.4f} s |"
        )

    lines.extend(
        [
            "",
            "## Memoria",
            "",
            "| archivo | pico RSS mediano | pico RSS máximo | incremento pico máximo |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for item in result["audios"]:
        audio = item["audio"]
        summary = item["summary"]
        peak_increase = max(rep["rss_peak_increase_bytes"] for rep in item["repetitions"])
        lines.append(
            "| "
            f"`{audio['name']}` | "
            f"{format_bytes(summary['rss_peak_bytes']['median'])} | "
            f"{format_bytes(summary['rss_peak_bytes']['max'])} | "
            f"{format_bytes(peak_increase)} |"
        )

    lines.extend(
        [
            "",
            "## Determinismo",
            "",
            "| archivo | determinista |",
            "| --- | ---: |",
        ]
    )
    for item in result["audios"]:
        lines.append(
            f"| `{item['audio']['name']}` | `{item['determinism']['is_deterministic']}` |"
        )

    lines.extend(
        [
            "",
            "## Observaciones y limitaciones",
            "",
            "- La primera inferencia se mide aparte y no se incluye en las estadísticas en caliente.",
            "- Los scores proceden de `decision_function`; no son probabilidades calibradas.",
            "- Este benchmark no evalúa precisión predictiva ni etiquetas reales a nivel de canción.",
            "- El benchmark debe repetirse en Docker o Hugging Face Spaces con versiones fijadas.",
            "- Los límites de tamaño de subida quedan pendientes de pruebas de formatos comprimidos y de la API.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_bytes(value: float | int) -> str:
    units = ("B", "KB", "MB", "GB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} GB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_outputs(result: dict[str, Any], output_json: Path, output_markdown: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(to_jsonable(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_markdown.write_text(render_markdown(result), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_benchmark(args)
    write_outputs(result, args.output_json, args.output_markdown)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
