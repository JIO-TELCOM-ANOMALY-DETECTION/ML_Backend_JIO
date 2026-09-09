"""
Shared FastAPI dependencies for the production API layer.

This module is intentionally infrastructure-only. It does not execute
workflows, access request bodies, translate exceptions into HTTP responses, or
contain any Part A-E business logic.

It provides:

- one lazily-created process-wide RunManager dependency;
- one lightweight process-level API logger for requests that do not yet have a
  run_id;
- helpers for obtaining run-scoped component loggers once a real run_id exists;
- FastAPI Annotated aliases used by route modules.

The shared RunManager is safe for the current V1 architecture because run and
registry mutation is protected by the Run Management locking layer. Keeping one
manager instance also avoids reconstructing the local registry facade for every
HTTP request.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from logging import Logger
from typing import Annotated

from fastapi import Depends

from app.core import (
    get_run_logger,
    settings,
)
from app.run_management import RunManager


API_LOGGER_NAME = "telecom_anomaly_backend.api"


@lru_cache(maxsize=1)
def get_run_manager() -> RunManager:
    """
    Return the shared production RunManager.

    Construction is lazy rather than module-global so importing the API package
    does not immediately initialize the configured run-registry backend.
    """
    return RunManager()


@lru_cache(maxsize=1)
def get_api_logger() -> Logger:
    """
    Return the process-level API logger.

    This logger is intentionally not a fake run-scoped logger. New Part A and
    master-pipeline requests may not have a run_id yet, so inventing one here
    would write logs under a misleading run identity.

    Handler ownership is left to application/server logging configuration.
    Once a real run_id exists, route code may use get_run_component_logger()
    for the existing run-aware logging infrastructure.
    """
    logger = logging.getLogger(
        API_LOGGER_NAME
    )
    logger.setLevel(
        settings.log_level
    )

    return logger


def get_run_component_logger(
    run_id: str,
    component_name: str = "pipeline",
    *,
    include_console: bool = True,
) -> Logger:
    """
    Return a run-scoped logger for one configured backend component.

    This is a normal helper rather than a FastAPI dependency because the run_id
    usually comes from the already-validated request body or route path.
    """
    return get_run_logger(
        run_id,
        component_name,
        include_console=include_console,
    )


def get_workflow_logger(
    run_id: str | None,
    *,
    component_name: str = "pipeline",
    include_console: bool = True,
) -> Logger:
    """
    Resolve the most accurate logger available for one workflow call.

    Existing run:
        use the normal run-scoped/component-scoped logger.

    New run where run_id is not known yet:
        use the process-level API logger rather than inventing a synthetic
        run identifier.
    """
    if run_id is None:
        return get_api_logger()

    return get_run_component_logger(
        run_id,
        component_name,
        include_console=include_console,
    )


RunManagerDependency = Annotated[
    RunManager,
    Depends(
        get_run_manager
    ),
]

ApiLoggerDependency = Annotated[
    Logger,
    Depends(
        get_api_logger
    ),
]


__all__ = [
    "API_LOGGER_NAME",
    "RunManagerDependency",
    "ApiLoggerDependency",
    "get_run_manager",
    "get_api_logger",
    "get_run_component_logger",
    "get_workflow_logger",
]