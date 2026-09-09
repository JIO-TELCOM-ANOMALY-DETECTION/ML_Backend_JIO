"""
Identifier generation for attempts, stage executions, and inference requests.

The logical run ID remains owned by run_directory_service.py. Part E's final
invocation_id remains owned by the existing Part E runner.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.core.exceptions import (
    RunManagementError,
)


# Part E builds: <safe_run_id>_<22-char UTC stamp>_<8-char UUID>.
# Two underscores make the suffix 32 characters. Manifest V2 record IDs
# are capped at 128 characters, leaving 96 characters for run_id.
PART_E_INVOCATION_SUFFIX_LENGTH = 32
MAX_PART_E_COMPATIBLE_RUN_ID_LENGTH = 128 - PART_E_INVOCATION_SUFFIX_LENGTH


def _next_sequential_id(
    *,
    prefix: str,
    existing_ids: Iterable[str],
    width: int,
) -> str:
    """
    Return the next monotonic identifier for one scoped collection.

    Unknown/nonmatching identifiers are ignored so imported historical records
    do not prevent normal numbering.
    """
    if width <= 0:
        raise RunManagementError(
            "Identifier width must be positive.",
            error_code="INVALID_IDENTIFIER_WIDTH",
            details={
                "prefix": prefix,
                "width": width,
            },
        )

    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+)$"
    )

    highest = 0

    for value in existing_ids:
        match = pattern.fullmatch(
            str(value)
        )

        if match is None:
            continue

        highest = max(
            highest,
            int(
                match.group(1)
            ),
        )

    next_value = highest + 1

    return (
        f"{prefix}_"
        f"{next_value:0{width}d}"
    )



def validate_part_e_compatible_run_id(
    run_id: str,
) -> str:
    """
    Ensure a run ID cannot make Part E's internally generated invocation ID
    exceed Manifest V2's 128-character record-ID limit.
    """
    cleaned = run_id.strip()

    if (
        len(cleaned)
        > MAX_PART_E_COMPATIBLE_RUN_ID_LENGTH
    ):
        raise RunManagementError(
            "The run ID is too long for the existing Part E invocation-ID format.",
            error_code="RUN_ID_TOO_LONG_FOR_PART_E",
            details={
                "run_id_length": len(
                    cleaned
                ),
                "maximum_run_id_length": (
                    MAX_PART_E_COMPATIBLE_RUN_ID_LENGTH
                ),
                "reason": (
                    "Part E appends a UTC timestamp and UUID suffix "
                    "to run_id when generating invocation_id."
                ),
            },
        )

    return cleaned


def generate_attempt_id(
    existing_ids: Iterable[str],
) -> str:
    """Generate attempt_0001, attempt_0002, ... within one run."""
    return _next_sequential_id(
        prefix="attempt",
        existing_ids=existing_ids,
        width=4,
    )


def generate_execution_id(
    existing_ids: Iterable[str],
) -> str:
    """Generate execution_0001, execution_0002, ... within one stage."""
    return _next_sequential_id(
        prefix="execution",
        existing_ids=existing_ids,
        width=4,
    )


def generate_inference_request_id(
    existing_ids: Iterable[str],
) -> str:
    """
    Generate request_000001, request_000002, ... for manager-side Part E
    request snapshots.

    This is not the Part E invocation_id.
    """
    return _next_sequential_id(
        prefix="request",
        existing_ids=existing_ids,
        width=6,
    )


__all__ = [
    "PART_E_INVOCATION_SUFFIX_LENGTH",
    "MAX_PART_E_COMPATIBLE_RUN_ID_LENGTH",
    "validate_part_e_compatible_run_id",
    "generate_attempt_id",
    "generate_execution_id",
    "generate_inference_request_id",
]
