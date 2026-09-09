"""
Production FastAPI route for standalone Part A EDA execution.

This module is intentionally thin. It owns only HTTP-boundary concerns that
cannot safely be inferred later:

- validate the caller-owned external CSV file before workflow execution;
- establish whether caller-supplied run/attempt identifiers genuinely exist so
  missing public resources can use the frozen API_* 404 contract;
- select the appropriate shared/run-scoped workflow logger;
- delegate the complete Part A lifecycle to ``run_part_a_workflow``.

The route does NOT implement EDA logic, create or finalize stage executions,
write Manifest V2 directly, scan run folders, or reshape workflow results.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core import (
    AttemptNotFoundError,
    InputValidationError,
    RunNotFoundError,
    TelecomBackendError,
)
from app.schemas.workflows import (
    EDAStageRequest,
    StandaloneStageResponse,
)
from app.services.file_service import (
    read_csv_file,
)
from app.workflows import (
    run_part_a_workflow,
)

from .dependencies import (
    RunManagerDependency,
    get_workflow_logger,
)


router = APIRouter(
    prefix="/api/v1/eda",
    tags=["eda"],
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
    Translate only known validation failures for the caller-owned Part A file.

    This helper is deliberately called only around the API preflight validation
    of ``request.dataset.input_file``. Generic file-service errors raised later
    while the workflow operates on backend-owned copies/artifacts are therefore
    never incorrectly translated into public 422 responses.
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
        "The caller-supplied Part A input file failed API boundary validation.",
        error_code=public_code,
        details=public_details,
    ) from error


def _validate_external_dataset_file(
    request: EDAStageRequest,
) -> None:
    """
    Fully validate the caller-owned CSV only when Part A will create a new
    attempt.

    New-attempt requests must establish all caller-owned file failures before
    Run Management creates a run/attempt. ``read_csv_file`` reuses the shared
    Services contract and checks:

    - file existence/type;
    - .csv extension;
    - non-empty file and size limit;
    - configured/default decoding;
    - Pandas CSV parsing;
    - no-readable-columns / EMPTY_CSV;
    - successfully parsed CSVs containing zero usable rows.

    This intentionally parses the CSV once at the API boundary and Part A reads
    it again from the immutable attempt copy. The duplicate read is accepted in
    V1 because it preserves the frozen API_INPUT_* 422 contract without
    creating an attempt that the eventual 422 response cannot identify.

    Existing-attempt retries skip this entirely. Their authoritative input is
    ``attempt_context.input_file``; the original external file may legitimately
    have been moved or deleted after attempt creation.
    """
    if request.attempt_id is not None:
        return

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
            "The caller-supplied Part A CSV contains no usable rows.",
            error_code="API_INPUT_CSV_EMPTY",
            details={},
        )

    # The DataFrame is intentionally not passed into the workflow. Part A owns
    # the immutable attempt copy and remains unchanged.
    del dataframe


# ---------------------------------------------------------------------------
# Caller-owned run / attempt boundary
# ---------------------------------------------------------------------------


def _validate_public_target(
    *,
    request: EDAStageRequest,
    manager: Any,
) -> None:
    """
    Establish genuine public run/attempt absence before workflow execution.

    Once these boundary checks pass, later Run Management failures are allowed
    to propagate unchanged. This prevents an internal manifest/registry failure
    during Part A finalization from being incorrectly reclassified as a public
    404.
    """
    if request.run_id is None:
        return

    # Targeted registry membership is used only to establish caller-owned
    # resource absence. Detailed technical resolution remains RunManager's job.
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

    if request.attempt_id is None:
        return

    # resolve_attempt() is read-only. Catch only the exact absence condition
    # while it is unquestionably the caller-requested attempt lookup.
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


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=StandaloneStageResponse,
    summary="Run standalone Part A EDA",
)
def run_eda_stage(
    request: EDAStageRequest,
    manager: RunManagerDependency,
) -> StandaloneStageResponse:
    """
    Run one standalone Part A EDA workflow.

    Request selection semantics come directly from ``EDAStageRequest``:

    - no ``run_id``:
        create a new run and new attempt;
    - ``run_id`` only:
        create a fresh attempt under that existing run;
    - ``run_id`` + ``attempt_id``:
        target that existing attempt, normally to retry failed Part A.

    Run Management remains authoritative for whether a retry is legal.
    """
    _validate_public_target(
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

    # The workflow owns all Part A lifecycle behavior:
    # run/attempt creation or resolution, immutable input copy use, execution
    # creation, EDA runner invocation, A->B handoff persistence, success/failure
    # finalization, and the standard StandaloneStageResponse.
    return run_part_a_workflow(
        request=request,
        manager=manager,
        logger=logger,
    )


__all__ = [
    "router",
]