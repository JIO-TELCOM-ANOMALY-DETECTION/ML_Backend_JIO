"""
Create and configure the temporary FastAPI application used to test
Part A EDA and Part B preprocessing.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.test_pipeline_routes import (
    router as test_pipeline_router,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)


app = FastAPI(
    title="Telecom Anomaly Backend",
    description=(
        "Temporary FastAPI server for testing Part A EDA "
        "and Part B preprocessing."
    ),
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Return the basic application health status.

    Returns:
        dict[str, str]:
            Confirmation that the FastAPI application is running.
    """

    return {
        "status": "healthy",
    }


app.include_router(
    test_pipeline_router,
    prefix="/api/v1",
    tags=["Pipeline Test"],
)