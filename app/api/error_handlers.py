"""
Central FastAPI exception-to-HTTP translation.

This module is the single public HTTP error boundary for the backend.

Final fail-closed policy
------------------------
Every TelecomBackendError is HTTP 500 by default.

Only explicitly approved application error codes may become:

    404 -> API-boundary public resource genuinely does not exist
    409 -> known valid request conflicts with healthy lifecycle state
    422 -> API-boundary or confirmed caller-fixable input/data condition

Low-level exception classes and generic service error codes do NOT determine
HTTP status because the same code/class can be raised for both caller-owned
input and backend-owned state.

Public information policy
-------------------------
Approved 4xx:
    - fixed safe public message
    - original application error_code
    - only explicitly approved detail fields

500:
    - original error_type
    - original application error_code
    - generic public message
    - details = {}
    - full diagnostics only in server logs

Known 500 failures are classified into a server-side DiagnosticFamily to narrow
investigation without exposing internal state to the caller.

This module does NOT execute workflows, validate/copy API files, translate
RunManager exceptions into API_* codes, persist failures, mutate manifests, or
perform recovery. Those responsibilities remain in routes, Run Management, and
workflows.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core import (
    AgentExecutionError,
    ArtifactResolutionError,
    ConfigurationError,
    EmailDeliveryError,
    FileProcessingError,
    InputValidationError,
    ModelExecutionError,
    ModelNotFoundError,
    PipelineExecutionError,
    PreprocessingError,
    ReportingError,
    ResultCombinationError,
    RunDirectoryError,
    RunLockError,
    RunManagementError,
    RunManifestError,
    RunRegistryError,
    TelecomBackendError,
)
from app.schemas.workflows import (
    WorkflowErrorDetail,
    WorkflowErrorResponse,
)

from .dependencies import get_api_logger


# ---------------------------------------------------------------------------
# Explicit public 4xx policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PublicErrorPolicy:
    """One deliberately approved public backend-error mapping."""

    status_code: int
    message: str
    detail_keys: tuple[str, ...] = ()


# This table is the ONLY path by which TelecomBackendError becomes 4xx.
#
# API_* codes are intentionally created at the HTTP/request boundary, where
# ownership is unambiguous. Generic service/RunManager codes remain internal
# and therefore fail closed to 500 unless they are explicitly proven to be a
# safe lifecycle or caller-data condition.
_PUBLIC_ERROR_POLICIES: dict[str, _PublicErrorPolicy] = {
    # ------------------------------------------------------------------
    # 404 — API-boundary public logical resource genuinely does not exist.
    # ------------------------------------------------------------------
    "API_RUN_NOT_FOUND": _PublicErrorPolicy(
        status_code=404,
        message="The requested run was not found.",
        detail_keys=("run_id",),
    ),
    "API_ATTEMPT_NOT_FOUND": _PublicErrorPolicy(
        status_code=404,
        message="The requested training attempt was not found.",
        detail_keys=("run_id", "attempt_id"),
    ),

    # ------------------------------------------------------------------
    # 409 — request is valid; healthy current lifecycle state blocks it.
    # ------------------------------------------------------------------
    "COMPLETED_STAGE_IS_IMMUTABLE": _PublicErrorPolicy(
        status_code=409,
        message=(
            "The completed stage cannot be changed inside the current attempt."
        ),
        detail_keys=(
            "attempt_id",
            "stage_name",
            "successful_execution_id",
            "required_action",
        ),
    ),
    "STAGE_EXECUTION_ALREADY_RUNNING": _PublicErrorPolicy(
        status_code=409,
        message="The requested stage already has an active execution.",
        detail_keys=(
            "attempt_id",
            "stage_name",
            "execution_id",
        ),
    ),
    "STAGE_PREREQUISITE_NOT_COMPLETED": _PublicErrorPolicy(
        status_code=409,
        message=(
            "The requested stage cannot start until its prerequisite has completed."
        ),
        detail_keys=(
            "attempt_id",
            "stage_name",
            "required_stage",
        ),
    ),
    "RUN_NOT_INFERENCE_READY": _PublicErrorPolicy(
        status_code=409,
        message=(
            "The run does not currently have an inference-ready training attempt."
        ),
        detail_keys=("run_id",),
    ),
    "TRAINING_ATTEMPT_ALREADY_INFERENCE_READY": _PublicErrorPolicy(
        status_code=409,
        message="The training attempt is already inference-ready.",
        detail_keys=(
            "run_id",
            "attempt_id",
            "part_d_execution_id",
            "bootstrap_execution_id",
        ),
    ),

    # ------------------------------------------------------------------
    # 422 — API-boundary file validation.
    #
    # Routes should translate genuinely caller-owned file failures into these
    # API_* codes before they reach this handler. Generic file_service codes
    # such as EMPTY_FILE / FILE_NOT_FOUND / FILE_TOO_LARGE remain 500.
    # ------------------------------------------------------------------
    "API_INPUT_FILE_NOT_FOUND": _PublicErrorPolicy(
        status_code=422,
        message="The supplied input file was not found.",
    ),
    "API_INPUT_PATH_NOT_FILE": _PublicErrorPolicy(
        status_code=422,
        message="The supplied input path is not a regular file.",
    ),
    "API_INPUT_FILE_EXTENSION_INVALID": _PublicErrorPolicy(
        status_code=422,
        message="The supplied input file type is not supported.",
        detail_keys=(
            "received_extension",
            "allowed_extensions",
        ),
    ),
    "API_INPUT_FILE_EMPTY": _PublicErrorPolicy(
        status_code=422,
        message="The supplied input file is empty.",
    ),
    "API_INPUT_FILE_TOO_LARGE": _PublicErrorPolicy(
        status_code=422,
        message="The supplied input file exceeds the permitted size.",
        detail_keys=(
            "file_size_bytes",
            "maximum_size_bytes",
            "maximum_size_gb",
        ),
    ),
    "API_INPUT_FILENAME_INVALID": _PublicErrorPolicy(
        status_code=422,
        message="The supplied input file name is invalid.",
    ),
    "API_INPUT_CSV_PARSE_FAILED": _PublicErrorPolicy(
        status_code=422,
        message="The supplied CSV file could not be parsed.",
    ),
    "API_INPUT_CSV_ENCODING_INVALID": _PublicErrorPolicy(
        status_code=422,
        message="The supplied CSV file encoding is not supported.",
    ),
    "API_INPUT_CSV_EMPTY": _PublicErrorPolicy(
        status_code=422,
        message="The supplied CSV contains no usable data.",
    ),

    # ------------------------------------------------------------------
    # 422 — confirmed Part E caller-data conditions.
    # ------------------------------------------------------------------
    "NO_FORWARD_INFERENCE_DATA": _PublicErrorPolicy(
        status_code=422,
        message=(
            "The supplied data does not contain new forward timestamps beyond "
            "the finalized inference history."
        ),
    ),
    "NO_FORWARD_DATA_IN_REQUESTED_INFERENCE_WINDOW": _PublicErrorPolicy(
        status_code=422,
        message=(
            "The supplied data contains no forward rows inside the requested "
            "inference window."
        ),
        detail_keys=(
            "new_period_start",
            "new_period_end_exclusive",
            "inference_days",
        ),
    ),
    "INFERENCE_HISTORY_MAX_GAP_DAYS_EXCEEDED": _PublicErrorPolicy(
        status_code=422,
        message=(
            "The supplied inference data exceeds the permitted history-gap duration."
        ),
        detail_keys=(
            "series",
            "gap_points",
            "gap_days",
            "max_gap_days",
        ),
    ),
    "INFERENCE_HISTORY_MAX_GAP_RATIO_EXCEEDED": _PublicErrorPolicy(
        status_code=422,
        message=(
            "The supplied inference data exceeds the permitted history-gap ratio."
        ),
        detail_keys=(
            "series",
            "gap_points",
            "gap_ratio",
            "max_gap_ratio",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Server-side diagnostic families
# ---------------------------------------------------------------------------


class DiagnosticFamily(str, Enum):
    """Internal investigation family for backend failures."""

    RUN_STATE = "RUN_STATE"
    REGISTRY = "REGISTRY"
    MANIFEST = "MANIFEST"
    ARTIFACT = "ARTIFACT"
    LOCK_INTERNAL = "LOCK_INTERNAL"
    CONFIGURATION = "CONFIGURATION"
    FILE_IO = "FILE_IO"
    VALIDATION = "VALIDATION"
    MODEL = "MODEL"
    PIPELINE = "PIPELINE"
    INFERENCE_CONTRACT = "INFERENCE_CONTRACT"
    HISTORY_STATE = "HISTORY_STATE"
    REPORTING = "REPORTING"
    AGENT = "AGENT"
    EMAIL = "EMAIL"
    UNKNOWN = "UNKNOWN"


_FILE_SERVICE_VALIDATION_CODES = {
    "EMPTY_FILENAME",
    "INVALID_FILENAME",
    "FILE_NOT_FOUND",
    "PATH_IS_NOT_FILE",
    "INVALID_FILE_EXTENSION",
    "INVALID_FILE_SIZE_LIMIT",
    "EMPTY_FILE",
    "FILE_TOO_LARGE",
    "INVALID_CHECKSUM_CHUNK_SIZE",
    "INVALID_CSV_OUTPUT_EXTENSION",
    "EMPTY_CSV",
    "INVALID_JSON_EXTENSION",
    "INVALID_JSON_OUTPUT_EXTENSION",
}

_HISTORY_CODE_PREFIXES = (
    "HISTORY_",
    "INFERENCE_HISTORY_",
    "INFERENCE_PROCESSED_HISTORY_",
    "INFERENCE_PROCESSED_CACHE_",
    "EMPTY_INFERENCE_HISTORY_",
    "INVALID_INFERENCE_HISTORY_",
    "RECONSTRUCTION_",
    "LATE_DATA_",
    "UNKNOWN_HISTORICAL_",
)

_INFERENCE_CONTRACT_CODE_PREFIXES = (
    "PART_E_",
    "INFERENCE_MODEL_",
    "INFERENCE_PART_C_",
    "XGBOOST_INFERENCE_",
    "ISOLATION_FOREST_INFERENCE_",
    "INVALID_XGBOOST_INFERENCE_",
    "INVALID_ISOLATION_FOREST_INFERENCE_",
)

_RUN_STATE_CODES = {
    "MISSING_STAGE_RECORD",
    "INCONSISTENT_SUCCESSFUL_EXECUTION",
    "INCONSISTENT_STAGE_LINEAGE",
    "INCONSISTENT_TRAINING_LIFECYCLE",
    "REGISTERED_RUN_DIRECTORY_NOT_FOUND",
    "EXECUTION_IS_NOT_ACTIVE",
    "INFERENCE_SESSION_NOT_STARTED",
    "INFERENCE_SESSION_ALREADY_FINALIZED",
    "INFERENCE_SESSION_NOT_FINALIZED",
    "RUN_NOT_FOUND",
    "ATTEMPT_NOT_FOUND",
    "ATTEMPT_NOT_INFERENCE_READY",
}


def _normalized_error_code(
    error: TelecomBackendError,
) -> str:
    """Return a normalized application error code for internal classification."""
    if not isinstance(error.error_code, str):
        return ""

    return error.error_code.strip().upper()


def diagnostic_family(
    error: TelecomBackendError,
) -> DiagnosticFamily:
    """
    Classify one backend failure for server-side investigation.

    The family never changes the HTTP status. It only narrows where operators
    should investigate a known or unclassified 500.
    """
    code = _normalized_error_code(error)

    if isinstance(error, RunRegistryError):
        return DiagnosticFamily.REGISTRY

    if isinstance(error, RunManifestError):
        return DiagnosticFamily.MANIFEST

    if isinstance(error, ArtifactResolutionError):
        return DiagnosticFamily.ARTIFACT

    if isinstance(error, RunLockError):
        return DiagnosticFamily.LOCK_INTERNAL

    if isinstance(error, ConfigurationError):
        return DiagnosticFamily.CONFIGURATION

    if isinstance(error, (FileProcessingError, RunDirectoryError)):
        return DiagnosticFamily.FILE_IO

    if code in _FILE_SERVICE_VALIDATION_CODES:
        return DiagnosticFamily.FILE_IO

    if isinstance(error, ReportingError):
        return DiagnosticFamily.REPORTING

    if isinstance(error, AgentExecutionError):
        return DiagnosticFamily.AGENT

    if isinstance(error, EmailDeliveryError):
        return DiagnosticFamily.EMAIL

    if code in _RUN_STATE_CODES or code.startswith("INCONSISTENT_"):
        return DiagnosticFamily.RUN_STATE

    if code.startswith(_HISTORY_CODE_PREFIXES) or code == (
        "DISABLED_INFERENCE_USABILITY_ENFORCEMENT"
    ):
        return DiagnosticFamily.HISTORY_STATE

    if code.startswith(_INFERENCE_CONTRACT_CODE_PREFIXES):
        return DiagnosticFamily.INFERENCE_CONTRACT

    if isinstance(error, (ModelExecutionError, ModelNotFoundError)):
        return DiagnosticFamily.MODEL

    if isinstance(
        error,
        (
            PipelineExecutionError,
            PreprocessingError,
            ResultCombinationError,
        ),
    ):
        return DiagnosticFamily.PIPELINE

    if isinstance(error, RunManagementError):
        return DiagnosticFamily.RUN_STATE

    if isinstance(error, InputValidationError):
        return DiagnosticFamily.VALIDATION

    return DiagnosticFamily.UNKNOWN


# ---------------------------------------------------------------------------
# Public HTTP-status resolution
# ---------------------------------------------------------------------------


def _public_policy(
    error: TelecomBackendError,
) -> _PublicErrorPolicy | None:
    """Return the explicit public policy for one application error code."""
    code = _normalized_error_code(error)

    if not code:
        return None

    return _PUBLIC_ERROR_POLICIES.get(code)


def backend_http_status(
    error: TelecomBackendError,
) -> int:
    """
    Return the HTTP status for one backend exception.

    Fail-closed rule:
        explicitly allowlisted code -> approved 404 / 409 / 422
        anything else              -> 500
    """
    policy = _public_policy(error)

    if policy is not None:
        return policy.status_code

    return 500


# ---------------------------------------------------------------------------
# Safe public-detail handling and policy self-validation
# ---------------------------------------------------------------------------


_SENSITIVE_DETAIL_SUFFIXES = (
    "_path",
    "_paths",
    "_file",
    "_files",
    "_directory",
    "_directories",
    "_root",
)

_SENSITIVE_DETAIL_KEYS = {
    "path",
    "paths",
    "file",
    "files",
    "directory",
    "directories",
    "root",
    "run_directory",
    "attempt_directory",
    "execution_directory",
    "state_directory",
    "staging_directory",
    "output_directory",
    "input_file",
    "output_file",
    "manifest_file",
    "request_file",
    "result_file",
    "error_file",
    "model_file",
    "model_files",
}

_FORBIDDEN_PUBLIC_DETAIL_KEYS = {
    "original_error",
    "traceback",
    "stack_trace",
    "exception",
    "raw_error",
}


def _is_sensitive_detail_key(
    key: object,
) -> bool:
    """Return whether one structured-detail key represents a server path."""
    cleaned = str(key).strip().lower()

    if cleaned in _SENSITIVE_DETAIL_KEYS:
        return True

    return any(
        cleaned.endswith(suffix)
        for suffix in _SENSITIVE_DETAIL_SUFFIXES
    )


def _validate_public_error_policies() -> None:
    """
    Validate the static public 4xx policy table.

    Any unsafe policy is a programming/configuration defect and should fail
    immediately during import/tests rather than silently exposing information.
    """
    allowed_statuses = {
        404,
        409,
        422,
    }

    for code, policy in _PUBLIC_ERROR_POLICIES.items():
        if not code or code != code.strip() or code != code.upper():
            raise RuntimeError(
                f"Invalid public error code declaration: {code!r}."
            )

        if policy.status_code not in allowed_statuses:
            raise RuntimeError(
                f"Public error code {code!r} uses unsupported status "
                f"{policy.status_code!r}."
            )

        if not isinstance(policy.message, str) or not policy.message.strip():
            raise RuntimeError(
                f"Public error code {code!r} has an empty public message."
            )

        if len(policy.detail_keys) != len(set(policy.detail_keys)):
            raise RuntimeError(
                f"Public error code {code!r} contains duplicate detail keys."
            )

        for detail_key in policy.detail_keys:
            cleaned = str(detail_key).strip().lower()

            if not cleaned:
                raise RuntimeError(
                    f"Public error code {code!r} contains an empty detail key."
                )

            if (
                cleaned in _FORBIDDEN_PUBLIC_DETAIL_KEYS
                or _is_sensitive_detail_key(cleaned)
            ):
                raise RuntimeError(
                    f"Public error code {code!r} exposes forbidden detail key "
                    f"{detail_key!r}."
                )


_validate_public_error_policies()


def _public_json_value(
    value: Any,
) -> Any:
    """
    Convert one approved 4xx detail value into conservative JSON-safe data.

    Unknown opaque objects expose only their type name, never repr().
    """
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return "[redacted]"

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Enum):
        return _public_json_value(value.value)

    if isinstance(value, BaseModel):
        return _public_json_value(
            value.model_dump(mode="python")
        )

    if is_dataclass(value) and not isinstance(value, type):
        return _public_json_value(asdict(value))

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}

        for key, child in value.items():
            public_key = str(key)

            if _is_sensitive_detail_key(public_key):
                result[public_key] = "[redacted]"
            else:
                result[public_key] = _public_json_value(child)

        return result

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _public_json_value(child)
            for child in value
        ]

    if isinstance(value, (set, frozenset)):
        return [
            _public_json_value(child)
            for child in value
        ]

    return {
        "type": type(value).__name__,
    }


def _public_details(
    *,
    policy: _PublicErrorPolicy,
    details: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return only explicitly approved detail fields for one public 4xx."""
    if not details or not policy.detail_keys:
        return {}

    selected = {
        key: details[key]
        for key in policy.detail_keys
        if key in details
    }

    converted = _public_json_value(selected)

    return (
        converted
        if isinstance(converted, dict)
        else {}
    )


# ---------------------------------------------------------------------------
# Common response renderer
# ---------------------------------------------------------------------------


def _error_response(
    *,
    status_code: int,
    detail: WorkflowErrorDetail,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Render one WorkflowErrorResponse."""
    body = WorkflowErrorResponse(
        error=detail,
    )

    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=(
            dict(headers)
            if headers
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Project-specific backend exceptions
# ---------------------------------------------------------------------------


async def telecom_backend_exception_handler(
    request: Request,
    error: TelecomBackendError,
) -> JSONResponse:
    """
    Handle all known project-specific backend exceptions.

    Explicitly allowlisted application codes receive their approved 4xx policy.
    Every other backend error receives a safe 500.
    """
    policy = _public_policy(error)
    logger = get_api_logger()

    if policy is not None:
        logger.warning(
            (
                "Public backend error during %s %s: "
                "status=%s code=%s type=%s"
            ),
            request.method,
            request.url.path,
            policy.status_code,
            error.error_code,
            type(error).__name__,
        )

        return _error_response(
            status_code=policy.status_code,
            detail=WorkflowErrorDetail(
                error_type=type(error).__name__,
                message=policy.message,
                error_code=_normalized_error_code(error),
                details=_public_details(
                    policy=policy,
                    details=error.details,
                ),
            ),
        )

    family = diagnostic_family(error)

    logger.error(
        (
            "Backend failure during %s %s: "
            "family=%s code=%s type=%s error=%s"
        ),
        request.method,
        request.url.path,
        family.value,
        error.error_code,
        type(error).__name__,
        error,
        exc_info=(
            error.__traceback__
            is not None
        ),
    )

    return _error_response(
        status_code=500,
        detail=WorkflowErrorDetail(
            error_type=type(error).__name__,
            message=(
                "An internal backend error occurred while processing the request."
            ),
            error_code=error.error_code,
            details={},
        ),
    )


# ---------------------------------------------------------------------------
# FastAPI/Pydantic request validation
# ---------------------------------------------------------------------------


def _validation_error_details(
    error: RequestValidationError,
) -> dict[str, Any]:
    """
    Return a compact safe view of FastAPI/Pydantic validation failures.

    Raw ``input`` and ``ctx`` values are intentionally excluded.
    """
    failures = []

    for item in error.errors():
        failures.append(
            {
                "type": str(
                    item.get(
                        "type",
                        "validation_error",
                    )
                ),
                "loc": [
                    str(part)
                    for part in item.get(
                        "loc",
                        (),
                    )
                ],
                "msg": str(
                    item.get(
                        "msg",
                        "Request validation failed.",
                    )
                ),
            }
        )

    return {
        "validation_errors": failures,
    }


async def request_validation_exception_handler(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    """Normalize FastAPI request validation into the public error schema."""
    logger = get_api_logger()

    logger.warning(
        "Request validation failed during %s %s.",
        request.method,
        request.url.path,
    )

    return _error_response(
        status_code=422,
        detail=WorkflowErrorDetail(
            error_type="RequestValidationError",
            message="The request did not satisfy the API schema.",
            error_code="REQUEST_VALIDATION_FAILED",
            details=_validation_error_details(error),
        ),
    )


# ---------------------------------------------------------------------------
# Framework HTTP exceptions
# ---------------------------------------------------------------------------


_ALLOWED_HTTP_EXCEPTION_HEADERS = {
    "allow",
    "retry-after",
    "www-authenticate",
}


def _public_http_exception_headers(
    headers: Mapping[str, str] | None,
) -> dict[str, str] | None:
    """
    Return only explicitly approved framework HTTP response headers.

    Arbitrary headers are discarded because route/framework exceptions may
    otherwise expose internal implementation details.
    """
    if not headers:
        return None

    public_headers = {
        key: value
        for key, value in headers.items()
        if str(key).strip().lower() in _ALLOWED_HTTP_EXCEPTION_HEADERS
    }

    return public_headers or None


def _http_exception_message(
    status_code: int,
) -> str:
    """
    Return a fixed framework message derived only from HTTP status.

    Arbitrary HTTPException.detail is intentionally never reflected.
    """
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return (
            "An internal server error occurred."
            if status_code >= 500
            else "HTTP request failed."
        )


async def http_exception_handler(
    request: Request,
    error: StarletteHTTPException,
) -> JSONResponse:
    """Normalize framework/route HTTP failures into the public error schema."""
    logger = get_api_logger()

    if error.status_code >= 500:
        logger.error(
            "Framework HTTP failure during %s %s: status=%s",
            request.method,
            request.url.path,
            error.status_code,
        )
        public_message = "An internal server error occurred."
    else:
        logger.warning(
            "HTTP request rejected during %s %s: status=%s",
            request.method,
            request.url.path,
            error.status_code,
        )
        public_message = _http_exception_message(
            error.status_code
        )

    return _error_response(
        status_code=error.status_code,
        detail=WorkflowErrorDetail(
            error_type="HTTPException",
            message=public_message,
            error_code=f"HTTP_{error.status_code}",
            details={},
        ),
        headers=_public_http_exception_headers(
            error.headers
        ),
    )


# ---------------------------------------------------------------------------
# Unknown Python exceptions
# ---------------------------------------------------------------------------


async def unhandled_exception_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions without exposing raw exception content."""
    logger = get_api_logger()

    logger.exception(
        "Unhandled exception during %s %s.",
        request.method,
        request.url.path,
        exc_info=error,
    )

    return _error_response(
        status_code=500,
        detail=WorkflowErrorDetail(
            error_type="InternalServerError",
            message="An unexpected internal server error occurred.",
            error_code="INTERNAL_SERVER_ERROR",
            details={},
        ),
    )


# ---------------------------------------------------------------------------
# Application registration
# ---------------------------------------------------------------------------


def register_error_handlers(
    app: FastAPI,
) -> None:
    """Register the complete API exception policy on one FastAPI application."""
    app.add_exception_handler(
        TelecomBackendError,
        telecom_backend_exception_handler,
    )
    app.add_exception_handler(
        RequestValidationError,
        request_validation_exception_handler,
    )
    app.add_exception_handler(
        StarletteHTTPException,
        http_exception_handler,
    )
    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )


__all__ = [
    "DiagnosticFamily",
    "backend_http_status",
    "diagnostic_family",
    "telecom_backend_exception_handler",
    "request_validation_exception_handler",
    "http_exception_handler",
    "unhandled_exception_handler",
    "register_error_handlers",
]
