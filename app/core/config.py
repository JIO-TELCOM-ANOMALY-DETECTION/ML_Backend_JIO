"""
Central configuration for the telecom anomaly detection backend.

This module stores project-wide settings such as API metadata, directory
locations, input-file restrictions, and logging defaults. Settings can be
overridden through environment variables or the project's .env file.
"""

from pathlib import Path
from typing import Final

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# DEFAULT PROJECT LOCATION
# ---------------------------------------------------------------------------
# config.py is expected at:
# telecom_anomaly_backend/app/core/config.py
#
# parents[0] -> core
# parents[1] -> app
# parents[2] -> telecom_anomaly_backend
DEFAULT_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Store and validate shared backend configuration.

    This class contains settings used across the API, preprocessing pipeline,
    machine-learning modules, reporting components, agents, and services.

    Values can be overridden using environment variables or the root .env file.
    For example:

        PROJECT_ROOT=C:/path/to/telecom_anomaly_backend
        LOG_LEVEL=DEBUG
        MAX_INPUT_FILE_SIZE_GB=5
    """

    # -----------------------------------------------------------------------
    # PYDANTIC SETTINGS CONFIGURATION
    # -----------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=DEFAULT_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # APPLICATION METADATA
    # -----------------------------------------------------------------------
    app_name: str = "Telecom Anomaly Detection Backend"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"

    environment: str = "local"
    debug: bool = False

    # -----------------------------------------------------------------------
    # PROJECT PATH
    # -----------------------------------------------------------------------
    project_root: Path = Field(
        default=DEFAULT_PROJECT_ROOT,
        description="Root directory of the backend project.",
    )

    # -----------------------------------------------------------------------
    # INPUT FILE SETTINGS
    # -----------------------------------------------------------------------
    max_input_file_size_gb: float = Field(
        default=5.0,
        gt=0,
        description="Maximum permitted input-file size in gigabytes.",
    )

    allowed_input_extensions: tuple[str, ...] = (
        ".csv",
    )

    default_file_encoding: str = "utf-8"

    # -----------------------------------------------------------------------
    # LOGGING SETTINGS
    # -----------------------------------------------------------------------
    log_level: str = "INFO"

    # all the 5 lines representing different parts of the log will be joined 
    # and saved as a singular line in the final log
    log_format: str = (
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(filename)s:%(lineno)d | "
        "%(message)s"
    )

    log_date_format: str = "%Y-%m-%d %H:%M:%S"

    log_file_names: dict[str, str] = Field(
        default_factory=lambda: {            # default_factory used instead of default for future change and flexibitlity
            "pipeline": "pipeline.log",
            "preprocessing": "preprocessing.log",
            "xgboost": "xgboost.log",
            "isolation_forest": "isolation_forest.log",
            "reporting": "reporting.log",
            "agents": "agents.log",
            "api": "api.log",
        },
        description="Log filename assigned to each backend component.",
    )

    # -----------------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------------
    @field_validator("project_root", mode="before") # it's a decorator and the function tells what variable to validate and when, changes the variabel value if anything in env before pydantic performs other things like conversion of str to path on it.
    @classmethod                                    # this decorator makes this function a method of the class
    def normalise_project_root(cls, value: str | Path) -> Path: # a class method receives the class itself as it's first parameter; it is working with cls(class) instead of self(an already created object) because a object might not be initated yet.
        """
        Validate and normalise the configured project root path.

        Args:
            value: Project-root path supplied through code or environment.

        Returns:
            An expanded and resolved Path object.
        """
        return Path(value).expanduser().resolve()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """
        Validate the configured Python logging level.

        Args:
            value: Logging level supplied through code or environment.

        Returns:
            The validated logging level in uppercase.

        Raises:
            ValueError: If the supplied logging level is unsupported.
        """
        supported_levels = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        normalised_value = value.upper().strip()

        if normalised_value not in supported_levels:
            raise ValueError(
                "LOG_LEVEL must be one of: "
                f"{', '.join(sorted(supported_levels))}."
            )

        return normalised_value

    # -----------------------------------------------------------------------
    # DERIVED DIRECTORY PATHS
    # -----------------------------------------------------------------------
    @property # allows use to call it as app_directory instead of app_directory(), bascically it let's us call it like an attribute, this is not a class method this is a object method which is for every object
    def app_directory(self) -> Path:
        """
        Return the main application package directory.

        Returns:
            Path to the project's app directory.
        """
        return self.project_root / "app"

    @property
    def runs_directory(self) -> Path:
        """
        Return the directory containing pipeline-run outputs.

        Returns:
            Path to the project's runs directory.
        """
        return self.project_root / "runs"

    @property
    def logs_directory(self) -> Path:
        """
        Return the parent directory containing all run logs.

        Returns:
            Path to the project's logs directory.
        """
        return self.project_root / "logs"

    @property
    def model_artifacts_directory(self) -> Path:
        """
        Return the parent directory containing trained model artifacts.

        Returns:
            Path to the project's model_artifacts directory.
        """
        return self.project_root / "model_artifacts"

    @property
    def xgboost_artifacts_directory(self) -> Path:
        """
        Return the directory containing XGBoost model artifacts.

        Returns:
            Path to the XGBoost artifact directory.
        """
        return self.model_artifacts_directory / "xgboost"

    @property
    def isolation_forest_artifacts_directory(self) -> Path:
        """
        Return the directory containing Isolation Forest model artifacts.

        Returns:
            Path to the Isolation Forest artifact directory.
        """
        return self.model_artifacts_directory / "isolation_forest"

    @property
    def model_metadata_directory(self) -> Path:
        """
        Return the directory containing model metadata files.

        Returns:
            Path to the model metadata directory.
        """
        return self.model_artifacts_directory / "model_metadata"

    @property
    def max_input_file_size_bytes(self) -> int:
        """
        Convert the configured maximum file size from gigabytes to bytes.

        Returns:
            Maximum accepted input-file size in bytes.
        """
        return int(self.max_input_file_size_gb * 1024**3)

    # -----------------------------------------------------------------------
    # RUN-SPECIFIC PATH HELPERS
    # -----------------------------------------------------------------------
    def get_run_directory(self, run_id: str) -> Path:
        """
        Build the output directory path for a particular pipeline run.

        Args:
            run_id: Unique identifier assigned to the pipeline run.

        Returns:
            Path in the form runs/<run_id>.
        """
        return self.runs_directory / run_id

    def get_run_log_directory(self, run_id: str) -> Path:
        """
        Build the logging directory path for a particular pipeline run.

        Args:
            run_id: Unique identifier assigned to the pipeline run.

        Returns:
            Path in the form logs/<run_id>.
        """
        return self.logs_directory / run_id

    def get_component_log_path(
        self,
        run_id: str,
        component_name: str,
    ) -> Path:
        """
        Build the log-file path for a component within a pipeline run.

        Args:
            run_id: Unique identifier assigned to the pipeline run.
            component_name: Component whose log file is required.

        Returns:
            Path to the component-specific log file.

        Raises:
            ValueError: If the component does not have a configured log file.
        """
        if component_name not in self.log_file_names:
            available_components = ", ".join(
                sorted(self.log_file_names)
            )

            raise ValueError(
                f"Unknown logging component '{component_name}'. "
                f"Available components: {available_components}."
            )

        return (
            self.get_run_log_directory(run_id)
            / self.log_file_names[component_name]
        )

    # -----------------------------------------------------------------------
    # BASE DIRECTORY INITIALISATION
    # -----------------------------------------------------------------------
    def create_base_directories(self) -> None:
        """
        Create the backend's shared storage directories when missing.

        This creates only the common parent directories. Individual run
        directories will later be created by run_directory_service.py.
        """
        directories = (
            self.runs_directory,
            self.logs_directory,
            self.xgboost_artifacts_directory,
            self.isolation_forest_artifacts_directory,
            self.model_metadata_directory,
        )

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


# Shared configuration object imported throughout the backend.
settings = Settings()