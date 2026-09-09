"""
Production FastAPI route for standalone Part B preprocessing execution.

This module is intentionally a thin HTTP boundary.

Responsibilities
----------------
- receive and validate ``PreprocessingStageRequest`` through Pydantic/FastAPI;
- establish whether the caller-supplied run and attempt genuinely exist so
  public missing resources can use the frozen API_* 404 contract;
- obtain the existing run-scoped workflow logger;
- delegate the complete Part B lifecycle to ``run_part_b_workflow``.

This module deliberately does NOT:
- read or validate an external CSV;
- inspect Part A result files directly;
- check Part A completion itself;
- decide Part B retry legality;
- create/complete/fail stage executions;
- manipulate Manifest V2;
- execute preprocessing ML logic directly;
- reshape workflow success/failure responses.

Run Management and the Part B workflow remain authoritative for prerequisite
resolution, completed-stage immutability, active-execution conflicts, artifact
resolution, stage persistence, and preprocessing orchestration.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core import (
    AttemptNotFoundError,
    RunNotFoundError,
)
from app.schemas.workflows import (
    PreprocessingStageRequest,
    StandaloneStageResponse,
)
from app.workflows import (
    run_part_b_workflow,
)

from .dependencies import (
    RunManagerDependency,
    get_workflow_logger,
)


router = APIRouter(
    prefix="/api/v1/preprocessing",
    tags=["preprocessing"],
)


def _validate_public_target(
    *,
    request: PreprocessingStageRequest,
    manager: Any,
) -> None:
    """
    Establish genuine caller-owned run/attempt absence before Part B starts.

    The route translates only public lookup absence:

    - unknown caller-supplied run
        -> API_RUN_NOT_FOUND;
    - unknown caller-supplied attempt inside a known run
        -> API_ATTEMPT_NOT_FOUND.

    Once these checks pass, later Run Management/workflow failures propagate
    unchanged. A registered run whose manifest/directory/artifacts are broken
    is an internal backend problem, not a caller-owned 404.
    """
    if not manager.registry.exists(
        request.run_id
    ):
        raise RunNotFoundError(
            "The caller-requested run is not registered.",
            error_code="API_RUN_NOT_FOUND",
            details={
                "run_id": request.run_id,
            },
        )

    try:
        manager.resolve_attempt(
            request.run_id,
            request.attempt_id,
        )

    except AttemptNotFoundError as error:
        raise AttemptNotFoundError(
            "The caller-requested training attempt is not registered.",
            error_code="API_ATTEMPT_NOT_FOUND",
            details={
                "run_id": request.run_id,
                "attempt_id": (
                    request.attempt_id
                ),
            },
        ) from error


@router.post(
    "/run",
    response_model=StandaloneStageResponse,
    summary="Run standalone Part B preprocessing",
)
def run_preprocessing_stage(
    request: PreprocessingStageRequest,
    manager: RunManagerDependency,
) -> StandaloneStageResponse:
    """
    Execute one standalone Part B preprocessing workflow.

    The request always targets an existing run + attempt.

    The API validates only caller-owned resource identity. It deliberately does
    not inspect Part A completion or Part B lifecycle state. The Part B workflow
    calls Run Management's ``prepare_stage_execution()``, which is authoritative
    for:

    - successful Part A prerequisite resolution;
    - Part B retry legality;
    - completed-stage immutability;
    - already-running execution conflicts.

    Once a Part B execution starts, the workflow owns preprocessing execution,
    B->C handoff persistence, stage success/failure persistence, and the
    returned ``StandaloneStageResponse``.
    """
    _validate_public_target(
        request=request,
        manager=manager,
    )

    logger = get_workflow_logger(
        request.run_id,
        component_name="pipeline",
    )

    return run_part_b_workflow(
        request=request,
        manager=manager,
        logger=logger,
    )


__all__ = [
    "router",
]
