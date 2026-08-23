from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import ApiSettings
from app.inference.errors import ModelArtifactError
from app.inference.interfaces import InferenceService
from app.inference.memory import MemoryProfiler
from app.inference.service import AudioInferenceService


logger = logging.getLogger(__name__)
ServiceFactory = Callable[[], InferenceService]


def _default_service_factory(memory_profiling_enabled: bool) -> AudioInferenceService:
    return AudioInferenceService(
        memory_profiler=MemoryProfiler(enabled=memory_profiling_enabled),
    )


def create_app(
    service_factory: ServiceFactory | None = None,
    settings: ApiSettings | None = None,
) -> FastAPI:
    api_settings = settings or ApiSettings.from_env()
    factory = service_factory or (
        lambda: _default_service_factory(api_settings.memory_profiling_enabled)
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.inference_service = None
        application.state.model_ready = False
        try:
            service = factory()
        except ModelArtifactError as exc:
            logger.error(
                "Inference service could not be loaded; API starts degraded: %s",
                exc,
            )
        else:
            application.state.inference_service = service
            application.state.model_ready = True

        yield

    application = FastAPI(
        title="AI Music Detection API",
        version="0.1.0",
        lifespan=lifespan,
    )
    if api_settings.cors_allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(api_settings.cors_allowed_origins),
            allow_credentials=False,
            allow_methods=["POST"],
            allow_headers=[],
        )
    application.state.settings = api_settings
    application.include_router(router)
    return application


app = create_app()
