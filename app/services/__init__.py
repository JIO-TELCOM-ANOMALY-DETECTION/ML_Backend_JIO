"""
Shared services for the telecom anomaly detection backend.

This package exposes the run-directory, file-handling, and run-manifest
utilities used by workflows, API routes, run-management components,
preprocessing modules, model runners, and reporting components.
"""

# ---------------------------------------------------------------------------
# RUN DIRECTORY SERVICES
# ---------------------------------------------------------------------------
from app.services.run_directory_service import (
    RUN_SUBDIRECTORIES,
    create_run_directories,
    generate_run_id,
    get_run_directory,
    get_run_log_directory,
    get_run_paths,
    get_run_subdirectory,
    list_run_ids,
    run_directory_exists,
    validate_run_directory,
    validate_run_id,
)

# ---------------------------------------------------------------------------
# FILE SERVICES
# ---------------------------------------------------------------------------
from app.services.file_service import (
    calculate_file_sha256,
    copy_input_file,
    ensure_directory,
    get_file_metadata,
    read_csv_file,
    read_json_file,
    sanitise_filename,
    validate_file_exists,
    validate_file_extension,
    validate_file_size,
    validate_input_file,
    write_csv_file,
    write_json_file,
)

# ---------------------------------------------------------------------------
# RUN MANIFEST SERVICES
# ---------------------------------------------------------------------------
from app.services.run_manifest_service import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    RECORD_ID_PATTERN,
    TRAINING_STAGE_NAMES,
    VALID_ATTEMPT_STATUSES,
    VALID_EXECUTION_STATUSES,
    VALID_INFERENCE_INVOCATION_STATUSES,
    VALID_RUN_STATUSES,
    VALID_STAGE_STATUSES,
    WORKFLOW_SCHEMA_VERSION,
    append_manifest_event,
    cancel_stage_execution,
    clear_active_inference_attempt,
    complete_inference_invocation,
    complete_stage_execution,
    convert_exception_to_record,
    create_attempt_record,
    create_run_manifest,
    fail_inference_invocation,
    fail_stage_execution,
    get_current_utc_timestamp,
    get_manifest_path,
    load_run_manifest,
    record_output_file,
    save_run_manifest,
    set_active_inference_attempt,
    set_attempt_status,
    set_run_status,
    start_inference_invocation,
    start_stage_execution,
    update_attempt_summary,
    update_run_metadata,
    update_run_summary,
    validate_attempt_status,
    validate_execution_status,
    validate_inference_invocation_status,
    validate_manifest_structure,
    validate_record_id,
    validate_run_status,
    validate_stage_name,
    validate_stage_status,
)


__all__ = [
    # Run-directory constants and functions
    "RUN_SUBDIRECTORIES",
    "generate_run_id",
    "validate_run_id",
    "get_run_directory",
    "get_run_log_directory",
    "get_run_paths",
    "get_run_subdirectory",
    "create_run_directories",
    "validate_run_directory",
    "run_directory_exists",
    "list_run_ids",

    # File-service functions
    "ensure_directory",
    "sanitise_filename",
    "validate_file_exists",
    "validate_file_extension",
    "validate_file_size",
    "validate_input_file",
    "calculate_file_sha256",
    "get_file_metadata",
    "copy_input_file",
    "read_csv_file",
    "write_csv_file",
    "read_json_file",
    "write_json_file",

    # Manifest constants
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "WORKFLOW_SCHEMA_VERSION",
    "TRAINING_STAGE_NAMES",
    "VALID_RUN_STATUSES",
    "VALID_ATTEMPT_STATUSES",
    "VALID_STAGE_STATUSES",
    "VALID_EXECUTION_STATUSES",
    "VALID_INFERENCE_INVOCATION_STATUSES",
    "RECORD_ID_PATTERN",

    # Manifest validation / general helpers
    "get_current_utc_timestamp",
    "get_manifest_path",
    "validate_record_id",
    "validate_stage_name",
    "validate_run_status",
    "validate_attempt_status",
    "validate_stage_status",
    "validate_execution_status",
    "validate_inference_invocation_status",
    "validate_manifest_structure",
    "convert_exception_to_record",
    "append_manifest_event",

    # Run-level manifest operations
    "create_run_manifest",
    "load_run_manifest",
    "save_run_manifest",
    "set_run_status",
    "update_run_metadata",
    "update_run_summary",

    # Attempt operations
    "create_attempt_record",
    "set_attempt_status",
    "update_attempt_summary",

    # Stage-execution operations
    "start_stage_execution",
    "complete_stage_execution",
    "fail_stage_execution",
    "cancel_stage_execution",
    "record_output_file",

    # Inference-ready pointer
    "set_active_inference_attempt",
    "clear_active_inference_attempt",

    # Inference-invocation operations
    "start_inference_invocation",
    "complete_inference_invocation",
    "fail_inference_invocation",
]
