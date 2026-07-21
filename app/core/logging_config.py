"""
Run-specific logging configuration for the telecom anomaly backend.

This module creates separate log files for every pipeline run and backend
component. Each run receives its own directory under:

    logs/<run_id>/

The pipeline orchestrator, preprocessing stages, machine-learning models,
reporting modules, agents, and API routes can each write to an independent
log file.
"""

import logging
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import ConfigurationError, RunDirectoryError


LOGGER_NAMESPACE = "telecom_anomaly_backend"


def validate_run_id(run_id: str) -> str:
    """
    Validate a pipeline run identifier before using it in a directory path.

    This validation prevents empty identifiers and values containing path
    separators from creating unintended nested directories.

    Args:
        run_id: Unique identifier assigned to a pipeline run.

    Returns:
        The validated run identifier.

    Raises:
        RunDirectoryError: If the run identifier is empty or unsafe.
    """
    cleaned_run_id = run_id.strip()

    if not cleaned_run_id:
        raise RunDirectoryError(
            "The run ID cannot be empty.",
            error_code="EMPTY_RUN_ID",
        )

    forbidden_values = {
        ".",
        "..",
    }

    if cleaned_run_id in forbidden_values:
        raise RunDirectoryError(
            "The run ID cannot be '.' or '..'.",
            error_code="INVALID_RUN_ID",
            details={
                "run_id": cleaned_run_id,
            },
        )

    if "/" in cleaned_run_id or "\\" in cleaned_run_id:
        raise RunDirectoryError(
            "The run ID cannot contain path separators.",
            error_code="INVALID_RUN_ID",
            details={
                "run_id": cleaned_run_id,
            },
        )

    return cleaned_run_id


def validate_component_name(component_name: str) -> str:
    """
    Validate that a backend component has a configured log filename.

    Args:
        component_name: Name of the backend component requesting a logger.

    Returns:
        The normalised component name.

    Raises:
        ConfigurationError: If the component is not configured.
    """
    normalised_name = component_name.strip().lower()

    if normalised_name not in settings.log_file_names:
        raise ConfigurationError(
            f"Unknown logging component '{component_name}'.",
            error_code="UNKNOWN_LOG_COMPONENT",
            details={
                "received_component": component_name,
                "available_components": sorted(
                    settings.log_file_names
                ),
            },
        )

    return normalised_name


def create_log_formatter() -> logging.Formatter:
    """
    Create the common formatter used by all backend log handlers.

    Returns:
        A logging formatter using the format and date format from settings.
    """
    return logging.Formatter(
        fmt=settings.log_format,
        datefmt=settings.log_date_format,
    )


def create_file_handler(
    log_file_path: Path,
) -> logging.FileHandler:
    """
    Create a file handler for one component-specific log file.

    Args:
        log_file_path: Destination path of the log file.

    Returns:
        A configured file handler.

    Raises:
        RunDirectoryError: If the log file cannot be created.
    """
    try:
        log_file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = logging.FileHandler(
            filename=log_file_path,
            mode="a",
            encoding="utf-8",
        )

    except OSError as error:
        raise RunDirectoryError(
            "Unable to create the component log file.",
            error_code="LOG_FILE_CREATION_FAILED",
            details={
                "log_file_path": str(log_file_path),
                "original_error": str(error),
            },
        ) from error

    file_handler.setLevel(settings.log_level)
    file_handler.setFormatter(create_log_formatter())

    return file_handler


def create_console_handler() -> logging.StreamHandler:
    """
    Create a console handler for displaying logs in the terminal.

    Returns:
        A configured console logging handler.
    """
    console_handler = logging.StreamHandler()

    console_handler.setLevel(settings.log_level)
    console_handler.setFormatter(create_log_formatter())

    return console_handler


def build_logger_name(
    run_id: str,
    component_name: str,
) -> str:
    """
    Build a unique logger name for one run and one backend component.

    Args:
        run_id: Unique identifier assigned to the pipeline run.
        component_name: Backend component using the logger.

    Returns:
        Logger name containing the project namespace, run ID, and component.
    """
    return (
        f"{LOGGER_NAMESPACE}."
        f"{run_id}."
        f"{component_name}"
    )


def get_run_logger(
    run_id: str,
    component_name: str,
    *,
    include_console: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a run-specific and component-specific logger.

    The logger writes to:

        logs/<run_id>/<component_log_file>

    A previously configured logger is returned without adding duplicate
    handlers.

    Args:
        run_id: Unique identifier assigned to the pipeline run.
        component_name: Backend component requesting the logger.
        include_console: Whether logs should also appear in the terminal.

    Returns:
        Configured Python logger for the requested run and component.
    """
    validated_run_id = validate_run_id(run_id)

    validated_component = validate_component_name(
        component_name
    )

    logger_name = build_logger_name(
        run_id=validated_run_id,
        component_name=validated_component,
    )

    logger = logging.getLogger(logger_name)

    logger.setLevel(settings.log_level)

    # Prevent messages from being passed to the root logger and printed twice.
    logger.propagate = False

    # If handlers already exist, this logger has already been configured.
    if logger.handlers:
        return logger

    log_file_path = settings.get_component_log_path(
        run_id=validated_run_id,
        component_name=validated_component,
    )

    file_handler = create_file_handler(
        log_file_path=log_file_path,
    )

    logger.addHandler(file_handler)

    if include_console:
        console_handler = create_console_handler()
        logger.addHandler(console_handler)

    return logger


def setup_run_logging(
    run_id: str,
    *,
    include_console: bool = True,
) -> dict[str, logging.Logger]:
    """
    Configure all component loggers for one pipeline run.

    This function creates the run-specific log directory and prepares one
    logger for every component listed in the global configuration.

    Args:
        run_id: Unique identifier assigned to the pipeline run.
        include_console: Whether logs should also appear in the terminal.

    Returns:
        Dictionary mapping component names to configured logger objects.

    Raises:
        RunDirectoryError: If the run log directory cannot be created.
    """
    validated_run_id = validate_run_id(run_id)

    run_log_directory = settings.get_run_log_directory(
        validated_run_id
    )

    try:
        run_log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as error:
        raise RunDirectoryError(
            "Unable to create the run logging directory.",
            error_code="RUN_LOG_DIRECTORY_CREATION_FAILED",
            details={
                "run_id": validated_run_id,
                "directory": str(run_log_directory),
                "original_error": str(error),
            },
        ) from error

    loggers: dict[str, logging.Logger] = {}

    for component_name in settings.log_file_names:
        loggers[component_name] = get_run_logger(
            run_id=validated_run_id,
            component_name=component_name,
            include_console=include_console,
        )

    return loggers


def close_run_logging(run_id: str) -> None:
    """
    Close and remove every logging handler associated with one run.

    This should be called after a pipeline run finishes when handlers need to
    release their file resources.

    Args:
        run_id: Unique identifier assigned to the pipeline run.
    """
    validated_run_id = validate_run_id(run_id)

    logger_prefix = (
        f"{LOGGER_NAMESPACE}."
        f"{validated_run_id}."
    )

    logger_names = list(
        logging.Logger.manager.loggerDict.keys()
    )

    for logger_name in logger_names:
        if not logger_name.startswith(logger_prefix):
            continue

        logger = logging.getLogger(logger_name)

        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)