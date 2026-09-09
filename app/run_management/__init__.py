"""
Production Run Management layer.

This package owns run lookup, attempts, immutable stage lineage, execution
folders, artifact resolution, registry persistence, and run-level locking.

It deliberately does not execute Parts A-E.
"""

from .artifact_resolver import (
    get_inference_state_directory,
    resolve_active_inference_state,
    resolve_attempt_input,
    resolve_prerequisite_results,
    resolve_stage_outputs,
    resolve_stage_result,
    resolve_successful_execution_record,
)
from .identifiers import (
    MAX_PART_E_COMPATIBLE_RUN_ID_LENGTH,
    PART_E_INVOCATION_SUFFIX_LENGTH,
    generate_attempt_id,
    generate_execution_id,
    generate_inference_request_id,
    validate_part_e_compatible_run_id,
)
from .lineage import (
    PREREQUISITE_STAGE,
    STAGE_ORDER,
    derive_attempt_status,
    ensure_inference_activation_allowed,
    ensure_stage_can_start,
    get_active_inference_attempt_id,
    get_attempt_record,
    get_next_stage,
    get_stage_record,
    get_successful_execution_id,
    is_stage_completed,
)
from .locking import (
    FileLock,
    RunLock,
    get_registry_lock_file,
)
from .manager import (
    RunManager,
)
from .models import (
    ArtifactReference,
    AttemptContext,
    InferenceRequestContext,
    OutputRegistration,
    RunContext,
    RunRegistryRecord,
    StageExecutionContext,
)
from .registry import (
    REGISTRY_VERSION,
    LocalJsonRunRegistry,
    RunRegistry,
)


__all__ = [
    # Runtime models
    "RunRegistryRecord",
    "RunContext",
    "AttemptContext",
    "ArtifactReference",
    "OutputRegistration",
    "StageExecutionContext",
    "InferenceRequestContext",

    # Identifier generation
    "PART_E_INVOCATION_SUFFIX_LENGTH",
    "MAX_PART_E_COMPATIBLE_RUN_ID_LENGTH",
    "validate_part_e_compatible_run_id",
    "generate_attempt_id",
    "generate_execution_id",
    "generate_inference_request_id",

    # Lineage
    "STAGE_ORDER",
    "PREREQUISITE_STAGE",
    "get_attempt_record",
    "get_stage_record",
    "get_successful_execution_id",
    "is_stage_completed",
    "ensure_stage_can_start",
    "ensure_inference_activation_allowed",
    "get_active_inference_attempt_id",
    "derive_attempt_status",
    "get_next_stage",

    # Locking
    "FileLock",
    "RunLock",
    "get_registry_lock_file",

    # Artifact resolution
    "resolve_attempt_input",
    "resolve_successful_execution_record",
    "resolve_stage_result",
    "resolve_stage_outputs",
    "resolve_prerequisite_results",
    "get_inference_state_directory",
    "resolve_active_inference_state",

    # Registry
    "RunRegistry",
    "REGISTRY_VERSION",
    "LocalJsonRunRegistry",

    # Facade
    "RunManager",
]
