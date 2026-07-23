"""
Shared services for the telecom anomaly detection backend.

This package exposes the run-directory, file-handling, and run-manifest
utilities used by workflows, API routes, preprocessing modules, model
runners, and reporting components.
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
    VALID_RUN_STATUSES,
    VALID_STAGE_STATUSES,
    append_manifest_event,
    complete_pipeline_stage,
    convert_exception_to_record,
    create_run_manifest,
    fail_pipeline_stage,
    get_current_utc_timestamp,
    get_manifest_path,
    load_run_manifest,
    record_output_file,
    save_run_manifest,
    set_run_status,
    skip_pipeline_stage,
    start_pipeline_stage,
    update_run_summary,
    validate_manifest_structure,
    validate_run_status,
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

    # Manifest constants and functions
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "VALID_RUN_STATUSES",
    "VALID_STAGE_STATUSES",
    "get_current_utc_timestamp",
    "get_manifest_path",
    "validate_run_status",
    "validate_stage_status",
    "validate_manifest_structure",
    "append_manifest_event",
    "create_run_manifest",
    "load_run_manifest",
    "save_run_manifest",
    "set_run_status",
    "start_pipeline_stage",
    "complete_pipeline_stage",
    "skip_pipeline_stage",
    "convert_exception_to_record",
    "fail_pipeline_stage",
    "record_output_file",
    "update_run_summary",
]