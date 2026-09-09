"""
Public import surface for the production API package.

This module exposes the production routers and the central FastAPI exception
handler registration function. It deliberately does not create the FastAPI
application or register routers by side effect; application assembly belongs
to ``app.main``.
"""

from __future__ import annotations

from .eda import router as eda_router
from .error_handlers import register_error_handlers
from .feature_engineering import router as feature_engineering_router
from .health import router as health_router
from .inference import router as inference_router
from .pipeline import router as pipeline_router
from .preprocessing import router as preprocessing_router
from .runs import router as runs_router
from .training import router as training_router


__all__ = [
    "register_exception_handlers",
    "health_router",
    "runs_router",
    "eda_router",
    "preprocessing_router",
    "feature_engineering_router",
    "training_router",
    "pipeline_router",
    "inference_router",
]
