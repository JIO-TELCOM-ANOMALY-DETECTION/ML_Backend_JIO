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
        pipeline manifests, and logging functions.

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


class PipelineExecutionError(TelecomBackendError):
    """
    Raised when the overall pipeline execution fails.

    This exception should represent a workflow-level failure, especially when
    the pipeline orchestrator cannot complete the requested sequence.
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