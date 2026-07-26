from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_test_mert import (  # noqa: E402
    MertSmokeTestError,
    aggregate_window_embeddings,
    ensure_finite,
    load_config,
    pool_last_hidden_state,
    select_smoke_sample,
    split_clip_into_windows,
    validate_revision_sha,
)


REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"


@pytest.fixture()
def smoke_manifest() -> pd.DataFrame:
    rows = []
    ai_models = [
        "Udio",
        "Riffusion",
        "AudioLDM 2 Large",
        "MusicGen Small",
        "Suno v3",
        "Stable Audio v2",
        "Mustango",
    ]
    for index, model in enumerate(ai_models):
        description = f"description-{index:02d}"
        rows.append(
            {
                "id": f"h-{index:02d}",
                "model": "MTG-Jamendo",
                "description": description,
                "label": 0,
                "split": "train",
            }
        )
        rows.append(
            {
                "id": f"ai-{index:02d}",
                "model": model,
                "description": description,
                "label": 1,
                "split": "train",
            }
        )
    rows.append(
        {
            "id": "val-h",
            "model": "MTG-Jamendo",
            "description": "validation-description",
            "label": 0,
            "split": "val",
        }
    )
    rows.append(
        {
            "id": "test-ai",
            "model": "Udio",
            "description": "test-description",
            "label": 1,
            "split": "test",
        }
    )
    return pd.DataFrame(rows)


def test_load_config_reads_smoke_test_settings() -> None:
    config = load_config(Path("configs/mert_frozen_embeddings.yaml"))

    assert config.model_id == "m-a-p/MERT-v1-95M"
    assert config.revision == REVISION
    assert config.sample_pairs == 6
    assert config.allowed_split == "train"
    assert config.batch_size == 1
    assert config.window_samples == 120000
    assert config.total_samples == 240000


def test_select_smoke_sample_is_deterministic_train_only(
    smoke_manifest: pd.DataFrame,
) -> None:
    first = select_smoke_sample(
        smoke_manifest,
        n_pairs=6,
        seed=42,
        allowed_split="train",
        preferred_ai_generators=("Udio", "Riffusion"),
    )
    second = select_smoke_sample(
        smoke_manifest,
        n_pairs=6,
        seed=42,
        allowed_split="train",
        preferred_ai_generators=("Udio", "Riffusion"),
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["split"].unique().tolist() == ["train"]
    assert len(first) == 12


def test_select_smoke_sample_pairs_by_description(
    smoke_manifest: pd.DataFrame,
) -> None:
    sample = select_smoke_sample(
        smoke_manifest,
        n_pairs=3,
        seed=42,
        allowed_split="train",
        preferred_ai_generators=("Udio", "Riffusion"),
    )

    grouped = sample.groupby("description", observed=False)
    assert grouped.size().eq(2).all()
    assert grouped["label"].apply(lambda values: sorted(values.tolist()) == [0, 1]).all()
    ai_models = sample[sample["label"].eq(1)]["model"].tolist()
    assert ai_models[:2] == ["Udio", "Riffusion"]


def test_split_clip_into_two_exact_windows() -> None:
    config = load_config(Path("configs/mert_frozen_embeddings.yaml"))
    signal = np.zeros(config.total_samples, dtype=np.float32)

    windows = split_clip_into_windows(signal, config)

    assert len(windows) == 2
    assert [window.shape for window in windows] == [(120000,), (120000,)]


def test_split_clip_rejects_wrong_length() -> None:
    config = load_config(Path("configs/mert_frozen_embeddings.yaml"))
    signal = np.zeros(config.total_samples - 1, dtype=np.float32)

    with pytest.raises(MertSmokeTestError, match="Expected 240000 samples"):
        split_clip_into_windows(signal, config)


def test_pooling_and_aggregation_produce_768_embedding() -> None:
    first_hidden = np.ones((1, 10, 768), dtype=np.float32)
    second_hidden = np.full((1, 10, 768), 3.0, dtype=np.float32)

    first = pool_last_hidden_state(first_hidden, expected_dim=768)
    second = pool_last_hidden_state(second_hidden, expected_dim=768)
    clip = aggregate_window_embeddings([first, second], expected_dim=768)

    assert first.shape == (768,)
    assert second.shape == (768,)
    assert clip.shape == (768,)
    assert np.allclose(clip, 2.0)


def test_finite_validation_rejects_nan_and_infinite() -> None:
    with pytest.raises(MertSmokeTestError, match="contains NaN or infinite"):
        ensure_finite(np.array([0.0, np.nan], dtype=np.float32), "embedding")

    with pytest.raises(MertSmokeTestError, match="contains NaN or infinite"):
        ensure_finite(np.array([0.0, np.inf], dtype=np.float32), "embedding")


def test_validate_revision_sha_rejects_main_and_invalid_values() -> None:
    validate_revision_sha(REVISION)

    with pytest.raises(MertSmokeTestError, match="immutable"):
        validate_revision_sha("main")

    with pytest.raises(MertSmokeTestError, match="immutable"):
        validate_revision_sha("12af15f")
