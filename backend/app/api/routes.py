from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from app.api.schemas import HealthResponse, ModelInfoResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    model_ready = bool(getattr(request.app.state, "model_ready", False))
    if model_ready:
        return HealthResponse(status="ok", model_ready=True)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="degraded", model_ready=False)


@router.get("/api/v1/model", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    service = getattr(request.app.state, "inference_service", None)
    model_ready = bool(getattr(request.app.state, "model_ready", False))
    if not model_ready or service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available",
        )

    metadata = service.model.metadata
    return ModelInfoResponse(
        model_id=metadata.model_id,
        sha256=metadata.sha256,
        classes=metadata.classes,
        positive_label=metadata.positive_label,
        score_type="decision_function",
        score_is_calibrated_probability=metadata.score_is_calibrated_probability,
        decision_threshold=metadata.decision_threshold,
        target_sample_rate=metadata.target_sample_rate,
        fragment_duration_seconds=metadata.fragment_duration_seconds,
        n_mfcc=metadata.n_mfcc,
        n_features=metadata.n_features,
        aggregation_strategy=metadata.aggregation_strategy,
        usage_warning=service.config.usage_warning,
    )
