"""
Custom exception classes for the telecom anomaly detection backend.

These exceptions allow different parts of the backend to report failures
using meaningful error types instead of relying only on generic Python
exceptions such as ValueError, RuntimeError, or FileNotFoundError.

The API layer can later catch these exceptions and convert them into
appropriate HTTP responses.
"""


class TelecomBackendError(Exception):
    """
    Base exception for all custom backend errors.

    Every custom exception in this project inherits from this class. This
    allows the backend to catch either one specific error type or all known
    backend errors together.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        details: dict | None = None,
    ) -> None:
        """
        Initialise a backend exception with optional structured information.

        Args:
            message: Human-readable explanation of the error.
            error_code: Optional machine-readable identifier for the error.
            details: Optional dictionary containing additional error context.
        """
        super().__init__(message)

        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        """
        Return the human-readable representation of the exception.

        Returns:
            The exception message, including the error code when available.
        """
        if self.error_code:
            return f"[{self.error_code}] {self.message}"

        return self.message

    def to_dict(self) -> dict:
        """
        Convert the exception into a structured dictionary.

        This structure can later be used by FastAPI exception handlers,
        workflow manifests, run-management records, and logging functions.

        Returns:
            Dictionary containing the error type, message, code, and details.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ConfigurationError(TelecomBackendError):
    """
    Raised when backend configuration is missing, invalid, or inconsistent.

    Examples:
        - An invalid project-root path
        - An unsupported logging level
        - Missing required environment settings
    """


class InputValidationError(TelecomBackendError):
    """
    Raised when an input file or pipeline request fails validation.

    Examples:
        - Unsupported file extension
        - File larger than the permitted size
        - Missing required dataset columns
        - Invalid request parameters
    """


class FileProcessingError(TelecomBackendError):
    """
    Raised when a file cannot be read, written, copied, or transformed.

    Examples:
        - CSV reading failure
        - Permission error
        - Output file creation failure
        - Unsupported file encoding
    """


class RunDirectoryError(TelecomBackendError):
    """
    Raised when a pipeline run directory cannot be created or accessed.

    Examples:
        - Duplicate run directory
        - Missing run folder
        - Permission failure while creating run folders
    """


class RunManifestError(TelecomBackendError):
    """
    Raised when a pipeline run manifest cannot be created or updated.

    Examples:
        - Invalid manifest structure
        - Manifest file write failure
        - Missing run metadata
    """


# ---------------------------------------------------------------------------
# RUN-MANAGEMENT / WORKFLOW-LIFECYCLE ERRORS
# ---------------------------------------------------------------------------


class RunManagementError(TelecomBackendError):
    """
    Base exception for failures in the run-management layer.

    This groups errors related to run identity, attempts, execution lineage,
    registry access, artifact resolution, and run-level coordination.
    """


class RunNotFoundError(RunManagementError):
    """
    Raised when a requested run identifier cannot be resolved.

    Examples:
        - Unknown run_id
        - Registry entry points to a run that no longer exists
    """


class AttemptNotFoundError(RunManagementError):
    """
    Raised when a requested attempt cannot be found inside a known run.

    Examples:
        - Unknown attempt_id
        - Attempt metadata missing from the run manifest
    """


class StagePrerequisiteError(RunManagementError):
    """
    Raised when a workflow stage is requested before its prerequisite exists.

    Examples:
        - Part B requested before Part A completed
        - Part C requested before Part B completed
        - Part D requested before Part C completed
        - Inference requested before an inference-ready training attempt exists
    """


class StageConflictError(RunManagementError):
    """
    Raised when a request would violate the immutable attempt lineage.

    Examples:
        - Trying to change Part B after Part B already completed
        - Trying to replace a completed upstream stage inside the same attempt
        - Trying to run a stage while a conflicting execution is active
    """


class ArtifactResolutionError(RunManagementError):
    """
    Raised when the authoritative artifact required by a workflow cannot be
    resolved safely.

    Examples:
        - Completed stage has no registered result snapshot
        - Active execution points to a missing output
        - Required upstream artifact path is invalid or unavailable
    """


class RunRegistryError(RunManagementError):
    """
    Raised when the run registry cannot be read or updated.

    Examples:
        - Invalid local registry JSON
        - Registry persistence failure
        - Inconsistent run-to-directory mapping
    """


class RunLockError(RunManagementError):
    """
    Raised when a run-level lock cannot be acquired or released safely.

    Examples:
        - Another mutable operation already owns the run
        - Lock acquisition exceeds the configured timeout
        - Lock metadata is inconsistent
    """


class PipelineExecutionError(TelecomBackendError):
    """
    Raised when the overall pipeline execution fails.

    This exception should represent a workflow-level execution failure,
    especially when an orchestrator cannot complete the requested sequence.
    Run-management conflicts and prerequisite failures use the more specific
    run-management exceptions above.
    """


class PreprocessingError(TelecomBackendError):
    """
    Raised when a preprocessing stage fails.

    Examples:
        - Timeline generation failure
        - Outlier-detection failure
        - Missing-value filling failure
        - Feature-generation failure
    """


class ModelExecutionError(TelecomBackendError):
    """
    Raised when a machine-learning model cannot complete execution.

    Examples:
        - Model training failure
        - Prediction failure
        - Invalid model input features
        - Model artifact loading failure
    """


class ModelNotFoundError(TelecomBackendError):
    """
    Raised when a requested model or model artifact cannot be found.

    Examples:
        - Missing XGBoost model file
        - Missing Isolation Forest model file
        - Unknown model name in the model registry
    """


class ResultCombinationError(TelecomBackendError):
    """
    Raised when outputs from multiple models cannot be combined.

    Examples:
        - Missing model-result columns
        - Incompatible row identifiers
        - Conflicting result schemas
    """


class ReportingError(TelecomBackendError):
    """
    Raised when metrics, plots, or reports cannot be generated.

    Examples:
        - Plot creation failure
        - Missing summary data
        - Report-file write failure
    """


class AgentExecutionError(TelecomBackendError):
    """
    Raised when an internal summary or email agent fails.

    Examples:
        - Summary-generation failure
        - Invalid agent input
        - Email-content generation failure
    """


class EmailDeliveryError(TelecomBackendError):
    """
    Raised when an email cannot be prepared or delivered.

    This exception will become useful after an external email service is
    connected to the backend.
    """
