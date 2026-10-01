"""
Production FastAPI application assembly.

This module is the application entry point for the completed production API.
It creates the FastAPI application, registers the centralized API error
handlers, and mounts every finalized production router.

Application assembly belongs here. Workflow orchestration, Run Management,
filesystem operations, and Parts A-E implementation remain below the API
boundary.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    eda_router,
    feature_engineering_router,
    health_router,
    inference_router,
    pipeline_router,
    preprocessing_router,
    register_error_handlers,
    runs_router,
    training_router,
)
from app.core import settings
from app.services.background_runner import get_background_runner


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


register_error_handlers(
    app
)


# Application-level liveness.
app.include_router(
    health_router
)


# Read-only run / attempt inspection.
app.include_router(
    runs_router
)


# Standalone production training stages.
app.include_router(
    eda_router
)
app.include_router(
    preprocessing_router
)
app.include_router(
    feature_engineering_router
)
app.include_router(
    training_router
)


# Complete production training pipeline.
app.include_router(
    pipeline_router
)


# Production Part E inference.
app.include_router(
    inference_router
)


@app.on_event("shutdown")
def shutdown_background_runner() -> None:
    """Wait for accepted in-process work before application shutdown."""
    get_background_runner().shutdown()

__all__ = [
    "app",
]
