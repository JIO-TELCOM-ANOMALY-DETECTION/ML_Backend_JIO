"""
Production FastAPI route for Part E inference.

The inference endpoint consumes one already inference-ready logical run and a
new caller-owned raw CSV. It deliberately does not accept an attempt_id.
Run Management resolves the authoritative active inference-ready training
attempt internally through the Part E workflow.

Responsibilities
----------------
- receive and validate ``InferenceRequest`` through Pydantic/FastAPI;
- establish genuine absence for the caller-supplied ``run_id``;
- validate the caller-owned inference CSV before Run Management creates an
  inference request snapshot;
- obtain the run-scoped pipeline logger;
- delegate the complete Part E lifecycle to ``run_part_e_workflow``.

This module deliberately does NOT:
- accept or resolve an attempt_id;
- inspect inference readiness or Manifest V2 directly;
- resolve saved inference-state paths;
- read ``inference_config.json`` directly;
- interpret late data;
- perform A/B/C reconstruction;
- load models or execute predictions;
- generate invocation IDs;
- mutate history/cache state;
- reshape workflow responses.

All saved-state resolution, inference invocation lifecycle, Part E execution,
history/cache updates, and sequential-inference behavior remain below the API
boundary.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core import (
    InputValidationError,
    RunNotFoundError,
    TelecomBackendError,
)
from app.schemas.inference.request import (
    InferenceRequest,
)
from app.schemas.workflows import (
    InferenceWorkflowResponse,
)
from app.services.file_service import (
    read_csv_file,
)
from app.workflows import (
    run_part_e_workflow,
)

from .dependencies import (
    RunManagerDependency,
    get_workflow_logger,
)


router = APIRouter(
    prefix="/api/v1/inference",
    tags=["inference"],
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
    Translate only known validation failures for the caller-owned inference CSV.

    This helper is used only during HTTP preflight of
    ``request.input_file``. Once ``run_part_e_workflow`` starts, Run Management
    creates and owns the inference request snapshot and copied input. File,
    model, history, cache, manifest, or saved-state failures after that boundary
    remain internal unless explicitly mapped by the central API error policy.
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
        "The caller-supplied inference input file failed API boundary "
        "validation.",
        error_code=public_code,
        details=public_details,
    ) from error


def _validate_external_input_file(
    request: InferenceRequest,
) -> None:
    """
    Fully validate the caller-owned inference CSV before Part E creates state.

    ``read_csv_file`` reuses the shared Services contract and validates:
    - file existence/type;
    - .csv extension;
    - non-empty file and configured size limit;
    - decoding;
    - Pandas parsing;
    - readable columns / EMPTY_CSV.

    A successfully parsed DataFrame containing zero rows is also rejected.

    The DataFrame is deliberately discarded. ``run_part_e_workflow`` passes the
    original external path to Run Management, which creates the authoritative
    request-owned copy before the pure Part E runner executes.
    """
    try:
        dataframe = read_csv_file(
            request.input_file
        )

    except TelecomBackendError as error:
        _translate_external_input_error(
            error
        )

    if dataframe.empty:
        raise InputValidationError(
            "The caller-supplied inference CSV contains no usable rows.",
            error_code="API_INPUT_CSV_EMPTY",
            details={},
        )

    del dataframe


# ---------------------------------------------------------------------------
# Caller-owned run boundary
# ---------------------------------------------------------------------------


def _validate_public_run(
    *,
    request: InferenceRequest,
    manager: Any,
) -> None:
    """
    Establish genuine caller-owned run absence before workflow execution.

    ``InferenceRequest`` intentionally contains no attempt_id. The API therefore
    proves only that the requested logical run is registered.

    If the run exists, all later inference-readiness, active-attempt,
    persisted-state, manifest, model, history, cache, and artifact-resolution
    failures remain authoritative Run Management / workflow failures and
    propagate unchanged to the central error handlers.
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


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=InferenceWorkflowResponse,
    summary="Run Part E inference using the active inference-ready attempt",
)
def run_inference(
    request: InferenceRequest,
    manager: RunManagerDependency,
) -> InferenceWorkflowResponse:
    """
    Execute one production Part E inference request.

    The caller supplies only:

    - ``run_id``;
    - ``input_file``;
    - ``inference_days``.

    The caller cannot select a training attempt. ``run_part_e_workflow`` and
    Run Management resolve the authoritative active inference-ready attempt and
    own the complete request/invocation lifecycle.

    The API validates only:
    - caller-owned run existence; and
    - the caller-owned external inference CSV.

    The Part E workflow remains authoritative for:
    - active inference-ready attempt resolution;
    - request snapshot/input copying;
    - inference request locking;
    - saved-contract/state resolution;
    - normal vs late-data reconstruction behavior;
    - pure Part E execution;
    - Part E-owned invocation_id validation/registration;
    - history/cache commit;
    - safe structured workflow response generation.
    """
    _validate_public_run(
        request=request,
        manager=manager,
    )

    _validate_external_input_file(
        request
    )

    logger = get_workflow_logger(
        request.run_id,
        component_name="pipeline",
    )

    return run_part_e_workflow(
        request=request,
        manager=manager,
        logger=logger,
    )


__all__ = [
    "router",
]
