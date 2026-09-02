"""
Create and configure the temporary FastAPI application used to test
Parts A through E.

Intended project location::

    test_main_part_E.py

The application exposes two independent endpoints:

- ``POST /api/v1/test-pipeline/run``
    A -> B -> C -> D training plus saved Part-E state bootstrap.

- ``POST /api/v1/test-inference/run``
    Pure Part-E inference against a previously completed saved run.

The temporary inference endpoint uses an explicitly supplied run directory in
place of the future database ``run_id -> run_directory`` lookup.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.test_pipeline_routes_part_E import (
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
        "Temporary FastAPI server for testing Part A EDA, Part B "
        "preprocessing, Part C feature engineering, Part D model training "
        "with fixed or Optuna tuning, and Part E saved-model inference."
    ),
    version="0.2.0",
)


@app.get(
    "/health"
)
def health_check() -> dict[
    str,
    str,
]:
    """Return the basic application health status."""

    return {
        "status": (
            "healthy"
        ),
    }


app.include_router(
    test_pipeline_router,
    prefix="/api/v1",
    tags=[
        "Pipeline Test"
    ],
)
