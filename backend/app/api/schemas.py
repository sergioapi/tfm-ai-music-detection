from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_ready: bool


class ModelInfoResponse(BaseModel):
    model_id: str
    sha256: str
    classes: tuple[int, ...]
    positive_label: int
    score_type: Literal["decision_function"]
    score_is_calibrated_probability: bool
    decision_threshold: float
    target_sample_rate: int
    fragment_duration_seconds: float
    n_mfcc: int
    n_features: int
    aggregation_strategy: str
    usage_warning: str
