"""
Public schema interface for production workflow and run-management contracts.

Existing A-E configuration schemas remain authoritative for ML behavior. This
package adds only the orchestration-facing contracts required by the final API,
Run Management layer, and workflow layer.
"""

from .common import (
    AttemptId,
    AttemptStatus,
    ExecutionId,
    ExecutionStatus,
    InferenceBootstrapConfig,
    InferenceInvocationStatus,
    InvocationId,
    RunId,
    RunMetadata,
    RunStatus,
    SafeRecordId,
    StageStatus,
    TrainingConfig,
    TrainingModelType,
    TrainingStageName,
    WorkflowErrorDetail,
    WorkflowErrorRecord,
    WorkflowSchemaModel,
    WorkflowStageName,
)
from .pipeline import (
    TrainingPipelineRequest,
)
from .responses import (
    InferenceWorkflowResponse,
    MasterPipelineRequestAcknowledgementReturnResponse,
    StageExecutionSummary,
    StandaloneStageResponse,
    TrainingPipelineResponse,
    WorkflowErrorResponse,
)
from .run import (
    AttemptDetailResponse,
    AttemptView,
    ExecutionView,
    InferenceInvocationView,
    RunSummaryResponse,
    StageView,
)
from .stage_requests import (
    EDAStageRequest,
    FeatureEngineeringStageRequest,
    PreprocessingStageRequest,
    TrainingStageRequest,
)


__all__ = [
    # Shared identifiers / types
    "SafeRecordId",
    "RunId",
    "AttemptId",
    "ExecutionId",
    "InvocationId",
    "TrainingStageName",
    "WorkflowStageName",
    "TrainingModelType",
    "RunStatus",
    "AttemptStatus",
    "StageStatus",
    "ExecutionStatus",
    "InferenceInvocationStatus",
    "TrainingConfig",

    # Shared models
    "WorkflowSchemaModel",
    "RunMetadata",
    "InferenceBootstrapConfig",
    "WorkflowErrorDetail",
    "WorkflowErrorRecord",

    # Standalone stage requests
    "EDAStageRequest",
    "PreprocessingStageRequest",
    "FeatureEngineeringStageRequest",
    "TrainingStageRequest",

    # Master training request
    "TrainingPipelineRequest",

    # Run inspection responses
    "ExecutionView",
    "InferenceInvocationView",
    "StageView",
    "AttemptView",
    "RunSummaryResponse",
    "AttemptDetailResponse",

    # Workflow responses
    "StageExecutionSummary",
    "StandaloneStageResponse",
    "TrainingPipelineResponse",
    "MasterPipelineRequestAcknowledgementReturnResponse",
    "InferenceWorkflowResponse",
    "WorkflowErrorResponse",
]