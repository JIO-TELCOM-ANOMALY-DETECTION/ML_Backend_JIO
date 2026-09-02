"""
Provide the temporary FastAPI route used to test Part A EDA,
Part B preprocessing, and Part C feature engineering through one JSON request.

This module handles only HTTP input and output. All pipeline execution
is delegated to the workflow layer.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.schemas.test_pipeline_till_part_c import PipelineRunRequest
from app.workflows.test_pipeline_workflow_till_part_c import (
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


def build_pipeline_response(
    result: TestPipelineWorkflowResult,
) -> dict[str, Any]:
    """
    Convert the temporary workflow result into JSON-compatible data.

    Existing Part B response fields are retained unchanged. Part C outputs are
    added under the ``feature_engineering`` key for each KPI.

    Args:
        result:
            Complete EDA, preprocessing, and feature-engineering workflow
            result.

    Returns:
        dict[str, Any]:
            Execution status, output paths, feature columns, and processing
            statistics.
    """

    kpi_responses: list[dict[str, Any]] = []

    for kpi_workflow_result in result.kpi_results:
        preprocessing = (
            kpi_workflow_result.preprocessing_result
        )
        feature_engineering = (
            kpi_workflow_result.feature_engineering_result
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
    Receive one JSON request and execute Part A, Part B, and Part C.

    Args:
        request:
            Validated dataset path, preprocessing settings, feature-engineering
            settings, and output root.

    Returns:
        dict[str, Any]:
            Success response containing EDA, preprocessing, and
            feature-engineering outputs and statistics.

    Raises:
        HTTPException:
            If the workflow fails during EDA, preprocessing, or feature
            engineering.
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