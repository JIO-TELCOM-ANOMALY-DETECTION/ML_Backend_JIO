"""
Production FastAPI route for the master training pipeline.

The master endpoint always runs one fresh training attempt through:

    Part A
      ↓
    Part B
      ↓
    Part C
      ↓
    Part D
      ↓
    inference_bootstrap

The HTTP route remains intentionally thin.

Responsibilities
----------------
- receive and validate ``TrainingPipelineRequest`` through Pydantic/FastAPI;
- establish genuine absence for an optional caller-supplied existing ``run_id``;
- validate the caller-owned external CSV before any new run/attempt is created;
- obtain the most accurate workflow logger available;
- delegate the complete A -> B -> C -> D -> inference_bootstrap lifecycle to
  ``run_training_pipeline``.

This module deliberately does NOT:
- accept or resolve an attempt_id;
- create attempts directly;
- call Part A/B/C/D/bootstrap workflows individually;
- construct Part C model-profile defaults;
- inspect stage state or retry legality;
- manipulate Manifest V2;
- resolve stage artifacts;
- activate inference directly;
- execute Part E inference;
- reshape the master workflow response.

Every master request creates a fresh attempt. Failed attempts are continued
through the appropriate standalone stage endpoint rather than by resubmitting
the master pipeline request.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core import (
    InputValidationError,
    RunNotFoundError,
    TelecomBackendError,
)
from app.schemas.workflows import (
    MasterPipelineRequestAcknowledgementReturnResponse,
    TrainingPipelineRequest,
    TrainingPipelineResponse,
)
from app.services.file_service import (
    read_csv_file,
)
from app.workflows import (
    run_training_pipeline,
)

from app.services.background_runner import get_background_runner

from .dependencies import (
    RunManagerDependency,
    get_workflow_logger,
)


router = APIRouter(
    prefix="/api/v1/pipeline",
    tags=["pipeline"],
)


# ---------------------------------------------------------------------------
# Caller-owned input-file boundary
# ---------------------------------------------------------------------------


_INPUT_FILE_ERROR_MAP: dict[
    str,
    tuple[
        str,
        tuple[str, ...],
    ],
] = {
    "FILE_NOT_FOUND": (
        "API_INPUT_FILE_NOT_FOUND",
        (),
    ),
    "PATH_IS_NOT_FILE": (
        "API_INPUT_PATH_NOT_FILE",
        (),
    ),
    "INVALID_FILE_EXTENSION": (
        "API_INPUT_FILE_EXTENSION_INVALID",
        (
            "received_extension",
            "allowed_extensions",
        ),
    ),
    "EMPTY_FILE": (
        "API_INPUT_FILE_EMPTY",
        (),
    ),
    "FILE_TOO_LARGE": (
        "API_INPUT_FILE_TOO_LARGE",
        (
            "file_size_bytes",
            "maximum_size_bytes",
            "maximum_size_gb",
        ),
    ),
    "CSV_PARSE_FAILED": (
        "API_INPUT_CSV_PARSE_FAILED",
        (),
    ),
    "CSV_ENCODING_ERROR": (
        "API_INPUT_CSV_ENCODING_INVALID",
        (),
    ),
    "EMPTY_CSV": (
        "API_INPUT_CSV_EMPTY",
        (),
    ),
}


def _normalized_error_code(
    error: TelecomBackendError,
) -> str:
    """Return one normalized backend application error code."""
    if not isinstance(
        error.error_code,
        str,
    ):
        return ""

    return error.error_code.strip().upper()


def _translate_external_input_error(
    error: TelecomBackendError,
) -> None:
    """
    Translate only known validation failures for the caller-owned master CSV.

    This helper is used only during preflight of
    ``request.dataset.input_file``. File/artifact failures raised later after
    Run Management has copied and persisted backend-owned state therefore stay
    internal and are not incorrectly exposed as API_INPUT_* 422 responses.
    """
    mapping = _INPUT_FILE_ERROR_MAP.get(
        _normalized_error_code(
            error
        )
    )

    if mapping is None:
        raise error

    (
        public_code,
        allowed_detail_keys,
    ) = mapping

    source_details = (
        error.details
        if isinstance(
            error.details,
            dict,
        )
        else {}
    )

    public_details = {
        key: source_details[key]
        for key in allowed_detail_keys
        if key in source_details
    }

    raise InputValidationError(
        "The caller-supplied master-pipeline input file failed API boundary "
        "validation.",
        error_code=public_code,
        details=public_details,
    ) from error


def _validate_external_dataset_file(
    request: TrainingPipelineRequest,
) -> None:
    """
    Fully validate the external CSV before the master workflow creates state.

    Every ``TrainingPipelineRequest`` creates a fresh attempt. Therefore
    ``request.dataset.input_file`` is always a caller-owned external boundary,
    even when ``run_id`` points to an existing logical run.

    ``read_csv_file`` reuses the shared Services contract and validates:
    - file existence/type;
    - .csv extension;
    - non-empty file and configured size limit;
    - decoding;
    - Pandas parsing;
    - readable columns / EMPTY_CSV.

    A successfully parsed DataFrame containing zero rows is also rejected.

    The DataFrame is intentionally discarded. The workflow later creates the
    immutable attempt-owned copy and Part A reads from that persisted copy.
    The duplicate CSV read is accepted in V1 to ensure caller-owned 422 failures
    happen before a run/attempt is created.
    """
    try:
        dataframe = read_csv_file(
            request.dataset.input_file
        )

    except TelecomBackendError as error:
        _translate_external_input_error(
            error
        )

    if dataframe.empty:
        raise InputValidationError(
            "The caller-supplied master-pipeline CSV contains no usable rows.",
            error_code="API_INPUT_CSV_EMPTY",
            details={},
        )

    del dataframe


# ---------------------------------------------------------------------------
# Caller-owned run boundary
# ---------------------------------------------------------------------------


def _validate_public_run(
    *,
    request: TrainingPipelineRequest,
    manager: Any,
) -> None:
    """
    Prove genuine caller-owned absence for an optional existing run.

    The master request intentionally contains no attempt_id, so there is no
    public attempt lookup here. If ``run_id`` is supplied and registered, all
    later technical resolution/lifecycle failures propagate unchanged.
    """
    if request.run_id is None:
        return

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


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=TrainingPipelineResponse,
    summary="Run the full A-to-D training pipeline and inference bootstrap",
)
def run_pipeline(
    request: TrainingPipelineRequest,
    manager: RunManagerDependency,
) -> TrainingPipelineResponse:
    """
    Run one fresh master training attempt.

    Request semantics are frozen by ``TrainingPipelineRequest``:

    - no ``run_id``:
        create a new logical run + fresh attempt;
    - ``run_id`` supplied:
        create a fresh attempt below that existing logical run.

    The request intentionally has no ``attempt_id``. Resubmitting the master
    endpoint never resumes a failed attempt; recovery uses the corresponding
    standalone stage endpoint.

    The API validates only:
    - optional caller-owned run existence; and
    - the caller-owned external CSV.

    ``run_training_pipeline`` remains authoritative for:
    - Part A run/attempt creation;
    - A -> B -> C -> D -> inference_bootstrap sequencing;
    - Part C model-matched default-profile construction when its explicit
      configuration is omitted;
    - stage failure propagation;
    - final inference-readiness reporting.
    """
    _validate_public_run(
        request=request,
        manager=manager,
    )

    _validate_external_dataset_file(
        request
    )

    logger = get_workflow_logger(
        request.run_id,
        component_name="pipeline",
    )

    return run_training_pipeline(
        request=request,
        manager=manager,
        logger=logger,
    )


def _run_accepted_master_pipeline(
    *,
    request: TrainingPipelineRequest,
    manager: Any,
) -> None:
    """Enter the existing master pipeline exactly once in the worker."""
    logger = get_workflow_logger(request.run_id, component_name="pipeline")
    run_training_pipeline(request=request, manager=manager, logger=logger)


@router.post(
    "/master-pipeline-request-acknowledgement-return",
    response_model=MasterPipelineRequestAcknowledgementReturnResponse,
    summary="Accept the full master pipeline for background execution",
)
def acknowledge_master_pipeline_request(
    request: TrainingPipelineRequest,
    manager: RunManagerDependency,
) -> MasterPipelineRequestAcknowledgementReturnResponse:
    """Create only a run, submit the unchanged master request, and return."""
    if request.run_id is not None:
        raise InputValidationError(
            "This endpoint always creates a new run and does not accept run_id.",
            error_code="API_RUN_ID_NOT_ALLOWED",
            details={},
        )

    _validate_external_dataset_file(request)
    run_context = manager.create_run(run_metadata=request.run_metadata)
    request_with_run_id = request.model_copy(
        deep=True,
        update={"run_id": run_context.run_id},
    )

    try:
        get_background_runner().submit(
            _run_accepted_master_pipeline,
            request=request_with_run_id,
            manager=manager,
        )
    except Exception as error:
        logger = get_workflow_logger(run_context.run_id, component_name="pipeline")
        logger.exception(
            "Background master-pipeline submission failed for run %s.",
            run_context.run_id,
        )
        raise RuntimeError(
            "The run was created, but background execution could not be accepted."
        ) from error

    return MasterPipelineRequestAcknowledgementReturnResponse(
        run_id=run_context.run_id,
        attempt_id=None,
        status="created",
        message="Master pipeline request accepted for background execution.",
    )

__all__ = [
    "router",
]
