"""
Production Part E workflow.

Part E is intentionally a thin workflow wrapper around the already-tested pure
inference runner.

Responsibilities:

- ask Run Management to resolve the active inference-ready training attempt;
- hold the run lock for the complete Part E operation through
  ``inference_session()``;
- replace the external caller input path with Run Management's request-owned
  copied input;
- call the existing ``run_inference_pipeline()`` unchanged;
- validate the returned invocation identity and published output location;
- register/finalize the exact Part E-owned invocation_id through Run Management;
- return a public response without exposing physical server paths.

This workflow does not reimplement history preparation, A/B/C reconstruction,
model loading, prediction, anomaly decisions, history mutation, output staging,
or result publication.
"""

from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any

from app.core.exceptions import RunManagementError
from app.ml.inference.runner import (
    InferenceRunnerResult,
    run_inference_pipeline,
)
from app.run_management import RunManager
from app.schemas.inference.request import InferenceRequest
from app.schemas.workflows import (
    InferenceWorkflowResponse,
    WorkflowErrorDetail,
)
from app.services.run_manifest_service import (
    convert_exception_to_record,
)

from .common import to_json_safe


def _error_detail(
    error: Exception,
) -> WorkflowErrorDetail:
    """Convert one backend exception into the public structured error shape."""
    return WorkflowErrorDetail.model_validate(
        convert_exception_to_record(
            error
        )
    )


def _runtime_request(
    request: InferenceRequest,
    *,
    copied_input_file: str | Path,
) -> InferenceRequest:
    """
    Rebuild the runtime Part E request using the run-owned request input copy.

    ``run_id`` and ``inference_days`` remain unchanged. The external caller path
    is no longer authoritative after Run Management has created the inference
    request snapshot.
    """
    payload = request.model_dump(
        mode="python"
    )
    payload["input_file"] = Path(
        copied_input_file
    ).expanduser().resolve()

    return InferenceRequest.model_validate(
        payload
    )


def _normalized_invocation_id(
    value: object,
) -> str:
    """
    Require the real Part E runner to return one usable invocation identifier.

    Run Management's internal request_id is deliberately not substituted here;
    invocation_id remains owned by Part E.
    """
    if not isinstance(
        value,
        str,
    ):
        raise RunManagementError(
            "Part E returned an invalid invocation_id type.",
            error_code="PART_E_INVOCATION_ID_INVALID",
            details={
                "received_type": (
                    type(
                        value
                    ).__name__
                ),
            },
        )

    invocation_id = value.strip()

    if not invocation_id:
        raise RunManagementError(
            "Part E returned a blank invocation_id.",
            error_code="PART_E_INVOCATION_ID_BLANK",
        )

    return invocation_id


def _result_summary(
    result: InferenceRunnerResult,
) -> dict[str, Any]:
    """
    Build safe headline information for the public workflow response.

    Physical output, manifest, history, cache, and model paths remain internal.
    """
    return {
        "model_type": (
            result.model_type
        ),
        "inference_start": (
            result
            .inference_start
            .isoformat()
        ),
        "inference_end_exclusive": (
            result
            .inference_end_exclusive
            .isoformat()
        ),
        "total_saved_series": int(
            result.total_saved_series
        ),
        "completed_series": int(
            result.completed_series
        ),
        "skipped_series_count": int(
            result.skipped_series_count
        ),
        "unmodelled_part_c_series": int(
            result.unmodelled_part_c_series
        ),
        "output_count": len(
            result.outputs
        ),
        # InferenceRunnerResult is returned only after commit_history_pipeline()
        # has succeeded and the final result directory has been published.
        "history_committed": True,
    }


def _validate_runner_result(
    *,
    request: InferenceRequest,
    result: InferenceRunnerResult,
    invocation_id: str,
    inference_state_directory: str | Path,
) -> None:
    """
    Verify the identity/location returned by the existing Part E runner.

    Run Management independently revalidates output containment during
    ``session.complete()``. These checks catch workflow/runner contract drift
    before invocation registration.
    """
    if result.run_id != request.run_id:
        raise RunManagementError(
            "Part E returned a run_id different from the requested run.",
            error_code="PART_E_RUN_ID_MISMATCH",
            details={
                "requested_run_id": (
                    request.run_id
                ),
                "returned_run_id": (
                    result.run_id
                ),
                "invocation_id": (
                    invocation_id
                ),
            },
        )

    state_root = Path(
        inference_state_directory
    ).expanduser().resolve()

    if not state_root.is_dir():
        raise RunManagementError(
            "The active inference-state directory does not exist.",
            error_code="PART_E_STATE_DIRECTORY_MISSING",
            details={
                "state_directory": str(
                    state_root
                ),
                "invocation_id": (
                    invocation_id
                ),
            },
        )

    output_directory = Path(
        result.output_directory
    ).expanduser().resolve()

    try:
        output_directory.relative_to(
            state_root
        )
    except ValueError as error:
        raise RunManagementError(
            "Part E returned an output directory outside the active "
            "inference-state root.",
            error_code="PART_E_OUTPUT_OUTSIDE_STATE",
            details={
                "invocation_id": (
                    invocation_id
                ),
                "output_directory": str(
                    output_directory
                ),
                "state_directory": str(
                    state_root
                ),
            },
        ) from error

    if not output_directory.is_dir():
        raise RunManagementError(
            "Part E returned an output directory that does not exist.",
            error_code="PART_E_OUTPUT_DIRECTORY_MISSING",
            details={
                "invocation_id": (
                    invocation_id
                ),
                "output_directory": str(
                    output_directory
                ),
            },
        )

    manifest_file = Path(
        result.manifest_file
    ).expanduser().resolve()

    if not manifest_file.is_file():
        raise RunManagementError(
            "Part E returned a manifest file that does not exist.",
            error_code="PART_E_MANIFEST_MISSING",
            details={
                "invocation_id": (
                    invocation_id
                ),
                "manifest_file": str(
                    manifest_file
                ),
            },
        )

    if manifest_file.parent != output_directory:
        raise RunManagementError(
            "Part E manifest_file is not inside its published invocation "
            "output directory.",
            error_code="PART_E_MANIFEST_DIRECTORY_MISMATCH",
            details={
                "invocation_id": (
                    invocation_id
                ),
                "manifest_file": str(
                    manifest_file
                ),
                "output_directory": str(
                    output_directory
                ),
            },
        )


def _failed_response(
    *,
    request: InferenceRequest,
    training_attempt_id: str,
    invocation_id: str,
    error: Exception,
) -> InferenceWorkflowResponse:
    """
    Build a failed Part E response only when a valid Part E invocation_id exists.

    The current public InferenceWorkflowResponse schema requires invocation_id.
    Failures that happen before Part E produces a valid invocation identifier
    therefore propagate to the API error boundary instead of creating an
    invalid/misleading response.
    """
    return InferenceWorkflowResponse(
        status="failed",
        run_id=request.run_id,
        training_attempt_id=(
            training_attempt_id
        ),
        invocation_id=(
            invocation_id
        ),
        message="Part E inference failed.",
        summary={},
        error=_error_detail(
            error
        ),
    )


def run_part_e_workflow(
    *,
    request: InferenceRequest,
    manager: RunManager,
    logger: Logger,
) -> InferenceWorkflowResponse:
    """
    Execute one production Part E invocation against the active saved attempt.

    Failure boundary:

    - before Run Management resolves an active training attempt:
        propagate to API error handling;

    - after the session starts but before Part E returns a valid invocation_id:
        Run Management persists the request-level failure, then the original
        error propagates because InferenceWorkflowResponse cannot truthfully be
        formed without invocation_id;

    - after a valid invocation_id exists:
        Run Management persists the appropriate request/invocation failure and
        this workflow may return a normal failed InferenceWorkflowResponse.
    """
    training_attempt_id: str | None = None
    returned_invocation_id: str | None = None

    try:
        with manager.inference_session(
            request.run_id,
            request_payload=request,
            input_file=request.input_file,
            metadata={
                "workflow": "part_e",
            },
        ) as session:
            context = session.context

            if context is None:
                raise RunManagementError(
                    "Run Management did not provide an inference request "
                    "context.",
                    error_code="PART_E_CONTEXT_MISSING",
                )

            training_attempt_id = (
                context.training_attempt_id
            )

            runtime_request = _runtime_request(
                request,
                copied_input_file=(
                    context.input_file
                ),
            )

            result = run_inference_pipeline(
                request=runtime_request,
                run_directory=(
                    context
                    .inference_state_directory
                ),
                logger_override=logger,
            )

            # Establish a schema-valid invocation identity immediately after the
            # real runner returns. Any later contract/registration failure can
            # then be represented by InferenceWorkflowResponse.
            returned_invocation_id = (
                _normalized_invocation_id(
                    result.invocation_id
                )
            )

            _validate_runner_result(
                request=runtime_request,
                result=result,
                invocation_id=(
                    returned_invocation_id
                ),
                inference_state_directory=(
                    context
                    .inference_state_directory
                ),
            )

            summary = _result_summary(
                result
            )

            session.complete(
                invocation_id=(
                    returned_invocation_id
                ),
                result_payload=to_json_safe(
                    result
                ),
                output_directory=(
                    result.output_directory
                ),
                metrics=summary,
            )

            return InferenceWorkflowResponse(
                status="completed",
                run_id=request.run_id,
                training_attempt_id=(
                    training_attempt_id
                ),
                invocation_id=(
                    returned_invocation_id
                ),
                message=(
                    "Part E inference completed successfully."
                ),
                summary=summary,
            )

    except Exception as error:
        # No active saved attempt/request context was resolved.
        if training_attempt_id is None:
            raise

        # Run Management's context manager has already persisted the
        # request-level failure. The current public response schema requires a
        # real Part E invocation_id, so do not mask the original failure with a
        # Pydantic response-validation error or invent a synthetic identifier.
        if returned_invocation_id is None:
            raise

        return _failed_response(
            request=request,
            training_attempt_id=(
                training_attempt_id
            ),
            invocation_id=(
                returned_invocation_id
            ),
            error=error,
        )


__all__ = [
    "run_part_e_workflow",
]
