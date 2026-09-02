"""
Provide the temporary FastAPI route used to test Part A EDA,
Part B preprocessing, Part C feature engineering, and Part D training
through one JSON request.

This module handles only HTTP input and output. All pipeline execution is
delegated to the workflow layer.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.ml.training.loader import SplitRange
from app.schemas.test_pipeline_till_part_d import (
    PipelineRunRequest,
    TrainingConfig,
)
from app.workflows.test_pipeline_workflow_till_part_d import (
    TestPipelineWorkflowResult,
    run_test_pipeline_workflow,
)


router = APIRouter()

logger = logging.getLogger(
    "telecom_anomaly.test_pipeline"
)


def optional_path_to_string(
    value: Any,
) -> str | None:
    """
    Convert an optional path-like value into a JSON-compatible string.

    Args:
        value:
            Path-like value or None.

    Returns:
        str | None:
            String path when present, otherwise None.
    """

    if value is None:
        return None

    return str(value)


def split_range_to_response(
    split_range: SplitRange | None,
) -> dict[str, Any] | None:
    """Convert one optional chronological Part D range to JSON-safe data."""

    if split_range is None:
        return None

    return {
        "start_position": split_range.start_position,
        "end_position": split_range.end_position,
        "start_timestamp": split_range.start_timestamp.isoformat(),
        "end_timestamp": split_range.end_timestamp.isoformat(),
        "row_count": split_range.row_count,
    }


def build_training_response(
    training: Any,
    *,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    """
    Convert one Part D runner result into compact JSON-compatible data.

    The complete chronological split plan and all detailed model summaries are
    already persisted by Part D. The HTTP response therefore returns headline
    validation/split statistics plus exact artifact paths rather than copying
    every per-series training metric into the API payload.
    """

    validation = training.validation_result
    split_plan = training.split_plan
    output_paths = training.output_paths

    tuning_method = (
        training_config.hyperparameter_tuning_method
        if training_config.hyperparameter_tuning_enabled
        else None
    )

    series_split_summaries = []

    for series_plan in split_plan.series:
        series_split_summaries.append(
            {
                "identity": {
                    "circle": series_plan.identity.circle,
                    "cluster": series_plan.identity.cluster,
                    "site_name": series_plan.identity.site_name,
                },
                "total_points": series_plan.total_points,
                "history": split_range_to_response(
                    series_plan.history
                ),
                "train": split_range_to_response(
                    series_plan.train
                ),
                "validation": split_range_to_response(
                    series_plan.validation
                ),
                "calibration": split_range_to_response(
                    series_plan.calibration
                ),
                "test": split_range_to_response(
                    series_plan.test
                ),
            }
        )

    return {
        "status": training.status,
        "model_type": training.model_type,
        "input_file": str(training.input_file),
        "output_directory": str(training.output_directory),
        "minimum_history_points": (
            training.minimum_history_points
        ),
        "hyperparameter_tuning": {
            "enabled": (
                training_config.hyperparameter_tuning_enabled
            ),
            "method": tuning_method,
        },
        "series_statistics": {
            "total_series": training.total_series,
            "valid_series": training.valid_series,
            "trained_series": training.trained_series,
            "skipped_invalid_series": (
                training.skipped_invalid_series
            ),
        },
        "skipped_series": [
            {
                "identity": {
                    "circle": item.identity.circle,
                    "cluster": item.identity.cluster,
                    "site_name": item.identity.site_name,
                },
                "usability": item.usability,
                "invalid_reason": item.invalid_reason,
            }
            for item in training.skipped_series
        ],
        "validation": {
            "total_rows": validation.total_rows,
            "all_part_c_feature_columns": list(
                validation.all_part_c_feature_columns
            ),
            "model_feature_columns": list(
                validation.model_feature_columns
            ),
            "excluded_part_c_feature_columns": list(
                validation.excluded_part_c_feature_columns
            ),
            "target_column": validation.target_column,
            "site_identity_columns": list(
                validation.site_identity_columns
            ),
            "timestamp_column": validation.timestamp_column,
            "status_column": validation.status_column,
            "usability_column": validation.usability_column,
            "invalid_reason_column": (
                validation.invalid_reason_column
            ),
            "trusted_statuses": sorted(
                validation.trusted_statuses
            ),
            "require_complete_feature_rows": (
                validation.require_complete_feature_rows
            ),
            "require_non_null_target": (
                validation.require_non_null_target
            ),
            "minimum_train_points": (
                validation.minimum_train_points
            ),
            "minimum_validation_points": (
                validation.minimum_validation_points
            ),
            "minimum_calibration_points": (
                validation.minimum_calibration_points
            ),
            "minimum_test_points": (
                validation.minimum_test_points
            ),
        },
        "split_plan": {
            "hyperparameter_tuning_enabled": (
                split_plan.hyperparameter_tuning_enabled
            ),
            "minimum_history_points": (
                split_plan.minimum_history_points
            ),
            "post_history_percentages": [
                {
                    "stage": stage_name,
                    "percentage": percentage,
                }
                for stage_name, percentage
                in split_plan.post_history_percentages
            ],
            "series_count": len(split_plan.series),
            "series": series_split_summaries,
        },
        "output_files": {
            "training_directory": str(
                output_paths.training_directory
            ),
            "model_directory": str(
                output_paths.model_directory
            ),
            "model_files": [
                str(path)
                for path in output_paths.model_files
            ],
            "split_plan": str(
                output_paths.split_plan_file
            ),
            "training_config": str(
                output_paths.training_config_file
            ),
            "feature_contract": str(
                output_paths.feature_contract_file
            ),
            "final_hyperparameters": str(
                output_paths.final_hyperparameters_file
            ),
            "training_summary": str(
                output_paths.training_summary_file
            ),
            "decision_settings": str(
                output_paths.decision_settings_file
            ),
            "usable_row_counts": str(
                output_paths.usable_row_counts_file
            ),
            "metadata": str(
                output_paths.metadata_file
            ),
        },
    }


def build_pipeline_response(
    result: TestPipelineWorkflowResult,
) -> dict[str, Any]:
    """
    Convert the temporary workflow result into JSON-compatible data.

    Existing Part B and Part C response fields are retained. Part D output is
    added under the ``training`` key for each KPI.

    Args:
        result:
            Complete EDA, preprocessing, feature-engineering, and training
            workflow result.

    Returns:
        dict[str, Any]:
            Execution status, output paths, feature columns, model artifacts,
            split information, and processing statistics.
    """

    kpi_responses: list[dict[str, Any]] = []

    for kpi_workflow_result in result.kpi_results:
        preprocessing = (
            kpi_workflow_result.preprocessing_result
        )
        feature_engineering = (
            kpi_workflow_result.feature_engineering_result
        )
        training = (
            kpi_workflow_result.training_result
        )

        kpi_responses.append(
            {
                "kpi_name": (
                    kpi_workflow_result.kpi_name
                ),
                "source_file": str(
                    preprocessing.source_file
                ),
                "output_directory": str(
                    preprocessing.output_directory
                ),
                "final_dataset_file": str(
                    preprocessing.final_dataset_file
                ),
                "summary_files": {
                    "outlier": str(
                        preprocessing.outlier_summary_file
                    ),
                    "original_gaps": str(
                        preprocessing.gap_summary_file
                    ),
                    "phase_1": str(
                        preprocessing.phase_1_summary_file
                    ),
                    "phase_2": str(
                        preprocessing.phase_2_summary_file
                    ),
                    "phase_3": str(
                        preprocessing.phase_3_summary_file
                    ),
                    "phase_4": optional_path_to_string(
                        preprocessing.phase_4_summary_file
                    ),
                },
                "intermediate_files": {
                    "timeline_corrected": optional_path_to_string(
                        preprocessing.timeline_checkpoint_file
                    ),
                    "outlier_labelled": optional_path_to_string(
                        preprocessing.outlier_checkpoint_file
                    ),
                    "gap_classified": optional_path_to_string(
                        preprocessing.gap_checkpoint_file
                    ),
                    "phase_1_filled": optional_path_to_string(
                        preprocessing.phase_1_checkpoint_file
                    ),
                    "phase_2_filled": optional_path_to_string(
                        preprocessing.phase_2_checkpoint_file
                    ),
                    "phase_3_filled": optional_path_to_string(
                        preprocessing.phase_3_checkpoint_file
                    ),
                },
                "statistics": {
                    "total_input_rows": (
                        preprocessing.total_input_rows
                    ),
                    "final_rows": (
                        preprocessing.final_rows
                    ),
                    "duplicate_rows_removed": (
                        preprocessing
                        .duplicate_rows_removed
                    ),
                    "inserted_timestamp_rows": (
                        preprocessing
                        .inserted_timestamp_rows
                    ),
                    "invalid_rows_converted": (
                        preprocessing
                        .invalid_rows_converted
                    ),
                    "valid_series": (
                        preprocessing.valid_series
                    ),
                    "invalid_series": (
                        preprocessing.invalid_series
                    ),
                    "outliers_flagged": (
                        preprocessing.outliers_flagged
                    ),
                    "original_gaps": (
                        preprocessing.original_gaps
                    ),
                    "original_missing_points": (
                        preprocessing
                        .original_missing_points
                    ),
                    "phase_1_filled_points": (
                        preprocessing
                        .phase_1_filled_points
                    ),
                    "phase_2_filled_points": (
                        preprocessing
                        .phase_2_filled_points
                    ),
                    "phase_3_filled_points": (
                        preprocessing
                        .phase_3_filled_points
                    ),
                    "phase_4_safe_filled_points": (
                        preprocessing
                        .phase_4_safe_filled_points
                    ),
                    "phase_4_leakage_filled_points": (
                        preprocessing
                        .phase_4_leakage_filled_points
                    ),
                    "unresolved_points": (
                        preprocessing.unresolved_points
                    ),
                },
                "method_counts": {
                    "phase_2": (
                        preprocessing.phase_2_method_counts
                    ),
                    "phase_3": (
                        preprocessing.phase_3_method_counts
                    ),
                    "phase_4_pass_1": (
                        preprocessing.phase_4_pass_1_counts
                    ),
                    "phase_4_pass_2": (
                        preprocessing.phase_4_pass_2_counts
                    ),
                },
                "feature_engineering": {
                    "source_file": str(
                        feature_engineering.source_file
                    ),
                    "output_directory": str(
                        feature_engineering.output_directory
                    ),
                    "final_dataset_file": str(
                        feature_engineering.final_dataset_file
                    ),
                    "summary_directory": str(
                        feature_engineering.summary_directory
                    ),
                    "site_summary_file": str(
                        feature_engineering.site_summary_file
                    ),
                    "intermediate_directory": optional_path_to_string(
                        feature_engineering.intermediate_directory
                    ),
                    "corrected_history_audit_file": (
                        optional_path_to_string(
                            feature_engineering
                            .corrected_history_audit_file
                        )
                    ),
                    "feature_columns": list(
                        feature_engineering.feature_columns
                    ),
                    "minimum_history_points": (
                        feature_engineering
                        .resolved_config
                        .minimum_history_points
                    ),
                    "statistics": {
                        "total_rows": (
                            feature_engineering.total_rows
                        ),
                        "total_sites": (
                            feature_engineering.total_sites
                        ),
                        "processed_sites": (
                            feature_engineering.processed_sites
                        ),
                        "skipped_invalid_sites": (
                            feature_engineering
                            .skipped_invalid_sites
                        ),
                        "insufficient_history_sites": (
                            feature_engineering
                            .insufficient_history_sites
                        ),
                        "populated_feature_cells": (
                            feature_engineering
                            .populated_feature_cells
                        ),
                        "missing_feature_cells": (
                            feature_engineering
                            .missing_feature_cells
                        ),
                    },
                },
                "training": build_training_response(
                    training,
                    training_config=result.training_config,
                ),
            }
        )

    return {
        "status": "completed",
        "output_directory": str(
            result.output_directory
        ),
        "kpi_count": len(
            result.kpi_results
        ),
        "eda_outputs": {
            "output_directory": str(
                result.eda_result.output_directory
            ),
            "prepared_datasets_directory": str(
                result.eda_result
                .prepared_datasets_directory
            ),
            "summaries_directory": str(
                result.eda_result.summaries_directory
            ),
            "source_file": str(
                result.eda_result.preparation.source_file
            ),
            "total_input_rows": (
                result.eda_result
                .preparation
                .total_input_rows
            ),
            "prepared_kpi_count": len(
                result.eda_result
                .preparation
                .prepared_datasets
            ),
            "was_split": (
                result.eda_result
                .preparation
                .was_split
            ),
        },
        "kpi_results": kpi_responses,
    }


@router.post(
    "/test-pipeline/run",
    status_code=status.HTTP_200_OK,
)
def run_test_pipeline(
    request: PipelineRunRequest,
) -> dict[str, Any]:
    """
    Receive one JSON request and execute Part A, Part B, Part C, and Part D.

    Args:
        request:
            Validated dataset path, preprocessing settings, feature-engineering
            settings, training settings, and output root.

    Returns:
        dict[str, Any]:
            Success response containing EDA, preprocessing,
            feature-engineering, and training outputs and statistics.

    Raises:
        HTTPException:
            If the workflow fails during any pipeline stage.
    """

    try:
        result = run_test_pipeline_workflow(
            request=request,
            logger=logger,
        )

        return build_pipeline_response(
            result
        )

    except Exception as error:
        logger.exception(
            "The temporary pipeline test failed."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
            },
        ) from error


__all__ = [
    "router",
]
