"""
Shared core utilities for the telecom anomaly detection backend.

This package exposes the global configuration object, custom exception
classes, and run-specific logging utilities used throughout the backend.
"""

from app.core.config import Settings, settings
from app.core.exceptions import (
    AgentExecutionError,
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
    RunManifestError,
    TelecomBackendError,
)
from app.core.logging_config import (
    close_run_logging,
    get_run_logger,
    setup_run_logging,
)


__all__ = [
    "Settings",
    "settings",
    "TelecomBackendError",
    "ConfigurationError",
    "InputValidationError",
    "FileProcessingError",
    "RunDirectoryError",
    "RunManifestError",
    "PipelineExecutionError",
    "PreprocessingError",
    "ModelExecutionError",
    "ModelNotFoundError",
    "ResultCombinationError",
    "ReportingError",
    "AgentExecutionError",
    "EmailDeliveryError",
    "get_run_logger",
    "setup_run_logging",
    "close_run_logging",
]