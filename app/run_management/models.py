"""
Internal data contracts for the run-management layer.

These are runtime dataclasses, not public API schemas. They carry resolved
filesystem locations and registry state between Run Management and workflows
without exposing physical paths to API callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RunRegistryRecord:
    """Search/index record for one logical backend run."""

    run_id: str
    run_directory: Path

    status: str
    run_metadata: Mapping[str, Any]

    latest_attempt_id: str | None
    active_inference_attempt_id: str | None
    inference_ready: bool

    created_at_utc: str
    updated_at_utc: str

    manifest_version: str
    workflow_schema_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the registry record into JSON-compatible data."""
        return {
            "run_id": self.run_id,
            "run_directory": str(
                self.run_directory
            ),
            "status": self.status,
            "run_metadata": dict(
                self.run_metadata
            ),
            "latest_attempt_id": (
                self.latest_attempt_id
            ),
            "active_inference_attempt_id": (
                self.active_inference_attempt_id
            ),
            "inference_ready": (
                self.inference_ready
            ),
            "created_at_utc": (
                self.created_at_utc
            ),
            "updated_at_utc": (
                self.updated_at_utc
            ),
            "manifest_version": (
                self.manifest_version
            ),
            "workflow_schema_version": (
                self.workflow_schema_version
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "RunRegistryRecord":
        """Build a registry record from persisted JSON data."""
        return cls(
            run_id=str(
                data["run_id"]
            ),
            run_directory=Path(
                data["run_directory"]
            ).expanduser().resolve(),
            status=str(
                data["status"]
            ),
            run_metadata=dict(
                data.get(
                    "run_metadata",
                    {},
                )
            ),
            latest_attempt_id=(
                str(
                    data[
                        "latest_attempt_id"
                    ]
                )
                if data.get(
                    "latest_attempt_id"
                ) is not None
                else None
            ),
            active_inference_attempt_id=(
                str(
                    data[
                        "active_inference_attempt_id"
                    ]
                )
                if data.get(
                    "active_inference_attempt_id"
                ) is not None
                else None
            ),
            inference_ready=bool(
                data.get(
                    "inference_ready",
                    False,
                )
            ),
            created_at_utc=str(
                data["created_at_utc"]
            ),
            updated_at_utc=str(
                data["updated_at_utc"]
            ),
            manifest_version=str(
                data["manifest_version"]
            ),
            workflow_schema_version=str(
                data[
                    "workflow_schema_version"
                ]
            ),
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
    ) -> "RunRegistryRecord":
        """Build the lightweight registry entry from Manifest V2."""
        return cls(
            run_id=str(
                manifest["run_id"]
            ),
            run_directory=Path(
                manifest["run_directory"]
            ).expanduser().resolve(),
            status=str(
                manifest["status"]
            ),
            run_metadata=dict(
                manifest.get(
                    "run_metadata",
                    {},
                )
            ),
            latest_attempt_id=(
                str(
                    manifest[
                        "latest_attempt_id"
                    ]
                )
                if manifest.get(
                    "latest_attempt_id"
                ) is not None
                else None
            ),
            active_inference_attempt_id=(
                str(
                    manifest[
                        "active_inference_attempt_id"
                    ]
                )
                if manifest.get(
                    "active_inference_attempt_id"
                ) is not None
                else None
            ),
            inference_ready=bool(
                manifest.get(
                    "inference_ready",
                    False,
                )
            ),
            created_at_utc=str(
                manifest["created_at_utc"]
            ),
            updated_at_utc=str(
                manifest["updated_at_utc"]
            ),
            manifest_version=str(
                manifest["manifest_version"]
            ),
            workflow_schema_version=str(
                manifest[
                    "workflow_schema_version"
                ]
            ),
        )


@dataclass(frozen=True, slots=True)
class RunContext:
    """Resolved filesystem/registry context for one logical run."""

    run_id: str
    run_directory: Path
    manifest_path: Path
    registry_record: RunRegistryRecord


@dataclass(frozen=True, slots=True)
class AttemptContext:
    """Resolved context for one immutable training-attempt snapshot."""

    run_id: str
    attempt_id: str

    run_directory: Path
    attempt_directory: Path
    input_directory: Path
    stages_directory: Path

    request_file: Path
    input_file: Path


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """One safely resolved artifact produced by a successful execution."""

    path: Path
    artifact_type: str

    attempt_id: str
    stage_name: str
    execution_id: str

    output_type: str | None = None


@dataclass(frozen=True, slots=True)
class OutputRegistration:
    """Output file that a workflow asks Run Management to register."""

    file_path: Path
    output_type: str
    include_checksum: bool = False


@dataclass(frozen=True, slots=True)
class StageExecutionContext:
    """Resolved execution locations supplied to one training-stage workflow."""

    run_id: str
    attempt_id: str
    stage_name: str
    execution_id: str

    run_directory: Path
    attempt_directory: Path
    stage_directory: Path
    execution_directory: Path

    request_file: Path
    output_directory: Path
    result_file: Path
    error_file: Path

    prerequisite_results: tuple[
        ArtifactReference,
        ...,
    ] = ()

    # Only populated for inference_bootstrap. During execution this is an
    # execution-specific staging root. Run Management publishes it to the
    # stable run/inference/<attempt_id>/ root only after successful bootstrap.
    inference_state_directory: Path | None = None


@dataclass(frozen=True, slots=True)
class InferenceRequestContext:
    """
    Run-managed request snapshot for one Part E call.

    request_id is internal to Run Management. It is intentionally distinct from
    Part E's invocation_id, which the existing Part E runner generates itself.
    """

    run_id: str
    training_attempt_id: str
    request_id: str

    run_directory: Path
    inference_state_directory: Path

    request_directory: Path
    request_file: Path
    input_file: Path
    result_file: Path
    error_file: Path

    prepared_at_utc: str


__all__ = [
    "RunRegistryRecord",
    "RunContext",
    "AttemptContext",
    "ArtifactReference",
    "OutputRegistration",
    "StageExecutionContext",
    "InferenceRequestContext",
]
