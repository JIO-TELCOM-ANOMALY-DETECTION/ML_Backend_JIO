"""
Temporary FastAPI routes for testing Parts A through E.

Intended project location::

    app/api/test_pipeline_routes_part_E.py

Endpoints
---------
POST /test-pipeline/run
    Run A -> B -> C -> D and leave a complete filesystem-backed saved run
    ready for later Part-E inference.

POST /test-inference/run
    Run pure Part-E inference using the production inference runner. The
    request explicitly supplies ``run_directory`` as the temporary replacement
    for the future database run lookup.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.ml.inference import InferenceRunnerResult
from app.ml.training.loader import SplitRange
from app.schemas.test_pipeline_part_E import (
    PipelineRunRequest,
    TestInferenceRunRequest,
    TrainingConfig,
)
from app.workflows.test_pipeline_workflow_part_E import (
    TestPipelineWorkflowResult,
    run_test_inference_workflow,
    run_test_pipeline_workflow,
)


router = APIRouter()

logger = logging.getLogger(
    "telecom_anomaly.test_pipeline_part_E"
)


def optional_path_to_string(
    value: Any,
) -> str | None:
    if value is None:
        return None

    return str(
        value
    )


def split_range_to_response(
    split_range: SplitRange | None,
) -> dict[str, Any] | None:
    if split_range is None:
        return None

    return {
        "start_position": (
            split_range
            .start_position
        ),
        "end_position": (
            split_range
            .end_position
        ),
        "start_timestamp": (
            split_range
            .start_timestamp
            .isoformat()
        ),
        "end_timestamp": (
            split_range
            .end_timestamp
            .isoformat()
        ),
        "row_count": (
            split_range
            .row_count
        ),
    }


def build_training_response(
    training: Any,
    *,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    """Convert one Part-D result into the existing compact HTTP contract."""

    validation = (
        training
        .validation_result
    )

    split_plan = (
        training
        .split_plan
    )

    output_paths = (
        training
        .output_paths
    )

    tuning_method = (
        training_config
        .hyperparameter_tuning_method
        if (
            training_config
            .hyperparameter_tuning_enabled
        )
        else None
    )

    series_split_summaries = []

    for series_plan in split_plan.series:
        series_split_summaries.append(
            {
                "identity": {
                    "circle": (
                        series_plan
                        .identity
                        .circle
                    ),
                    "cluster": (
                        series_plan
                        .identity
                        .cluster
                    ),
                    "site_name": (
                        series_plan
                        .identity
                        .site_name
                    ),
                },
                "total_points": (
                    series_plan
                    .total_points
                ),
                "history": (
                    split_range_to_response(
                        series_plan
                        .history
                    )
                ),
                "train": (
                    split_range_to_response(
                        series_plan
                        .train
                    )
                ),
                "validation": (
                    split_range_to_response(
                        series_plan
                        .validation
                    )
                ),
                "calibration": (
                    split_range_to_response(
                        series_plan
                        .calibration
                    )
                ),
                "test": (
                    split_range_to_response(
                        series_plan
                        .test
                    )
                ),
            }
        )

    return {
        "status": (
            training.status
        ),
        "model_type": (
            training.model_type
        ),
        "input_file": str(
            training.input_file
        ),
        "output_directory": str(
            training.output_directory
        ),
        "minimum_history_points": (
            training
            .minimum_history_points
        ),
        "hyperparameter_tuning": {
            "enabled": (
                training_config
                .hyperparameter_tuning_enabled
            ),
            "method": (
                tuning_method
            ),
        },
        "series_statistics": {
            "total_series": (
                training
                .total_series
            ),
            "valid_series": (
                training
                .valid_series
            ),
            "trained_series": (
                training
                .trained_series
            ),
            "skipped_invalid_series": (
                training
                .skipped_invalid_series
            ),
        },
        "skipped_series": [
            {
                "identity": {
                    "circle": (
                        item
                        .identity
                        .circle
                    ),
                    "cluster": (
                        item
                        .identity
                        .cluster
                    ),
                    "site_name": (
                        item
                        .identity
                        .site_name
                    ),
                },
                "usability": (
                    item.usability
                ),
                "invalid_reason": (
                    item
                    .invalid_reason
                ),
            }
            for item
            in training.skipped_series
        ],
        "validation": {
            "total_rows": (
                validation
                .total_rows
            ),
            "all_part_c_feature_columns": list(
                validation
                .all_part_c_feature_columns
            ),
            "model_feature_columns": list(
                validation
                .model_feature_columns
            ),
            "excluded_part_c_feature_columns": list(
                validation
                .excluded_part_c_feature_columns
            ),
            "target_column": (
                validation
                .target_column
            ),
            "site_identity_columns": list(
                validation
                .site_identity_columns
            ),
            "timestamp_column": (
                validation
                .timestamp_column
            ),
            "status_column": (
                validation
                .status_column
            ),
            "usability_column": (
                validation
                .usability_column
            ),
            "invalid_reason_column": (
                validation
                .invalid_reason_column
            ),
            "trusted_statuses": sorted(
                validation
                .trusted_statuses
            ),
            "require_complete_feature_rows": (
                validation
                .require_complete_feature_rows
            ),
            "require_non_null_target": (
                validation
                .require_non_null_target
            ),
            "minimum_train_points": (
                validation
                .minimum_train_points
            ),
            "minimum_validation_points": (
                validation
                .minimum_validation_points
            ),
            "minimum_calibration_points": (
                validation
                .minimum_calibration_points
            ),
            "minimum_test_points": (
                validation
                .minimum_test_points
            ),
        },
        "split_plan": {
            "hyperparameter_tuning_enabled": (
                split_plan
                .hyperparameter_tuning_enabled
            ),
            "minimum_history_points": (
                split_plan
                .minimum_history_points
            ),
            "post_history_percentages": [
                {
                    "stage": (
                        stage_name
                    ),
                    "percentage": (
                        percentage
                    ),
                }
                for stage_name, percentage
                in split_plan.post_history_percentages
            ],
            "series_count": len(
                split_plan.series
            ),
            "series": (
                series_split_summaries
            ),
        },
        "output_files": {
            "training_directory": str(
                output_paths
                .training_directory
            ),
            "model_directory": str(
                output_paths
                .model_directory
            ),
            "model_files": [
                str(
                    path
                )
                for path
                in output_paths.model_files
            ],
            "split_plan": str(
                output_paths
                .split_plan_file
            ),
            "training_config": str(
                output_paths
                .training_config_file
            ),
            "feature_contract": str(
                output_paths
                .feature_contract_file
            ),
            "final_hyperparameters": str(
                output_paths
                .final_hyperparameters_file
            ),
            "training_summary": str(
                output_paths
                .training_summary_file
            ),
            "decision_settings": str(
                output_paths
                .decision_settings_file
            ),
            "usable_row_counts": str(
                output_paths
                .usable_row_counts_file
            ),
            "metadata": str(
                output_paths
                .metadata_file
            ),
        },
    }


def build_pipeline_response(
    result: TestPipelineWorkflowResult,
) -> dict[str, Any]:
    """
    Convert the A/B/C/D result plus saved Part-E bootstrap state to JSON.

    Existing Part-D response fields are preserved. The new ``saved_run`` block
    gives the later inference-suite scripts everything needed to locate and
    call the trained run without a database.
    """

    kpi_responses: list[
        dict[str, Any]
    ] = []

    for kpi_workflow_result in result.kpi_results:
        preprocessing = (
            kpi_workflow_result
            .preprocessing_result
        )

        feature_engineering = (
            kpi_workflow_result
            .feature_engineering_result
        )

        training = (
            kpi_workflow_result
            .training_result
        )

        kpi_responses.append(
            {
                "kpi_name": (
                    kpi_workflow_result
                    .kpi_name
                ),
                "source_file": str(
                    preprocessing
                    .source_file
                ),
                "output_directory": str(
                    preprocessing
                    .output_directory
                ),
                "final_dataset_file": str(
                    preprocessing
                    .final_dataset_file
                ),
                "summary_files": {
                    "outlier": str(
                        preprocessing
                        .outlier_summary_file
                    ),
                    "original_gaps": str(
                        preprocessing
                        .gap_summary_file
                    ),
                    "phase_1": str(
                        preprocessing
                        .phase_1_summary_file
                    ),
                    "phase_2": str(
                        preprocessing
                        .phase_2_summary_file
                    ),
                    "phase_3": str(
                        preprocessing
                        .phase_3_summary_file
                    ),
                    "phase_4": (
                        optional_path_to_string(
                            preprocessing
                            .phase_4_summary_file
                        )
                    ),
                },
                "intermediate_files": {
                    "timeline_corrected": (
                        optional_path_to_string(
                            preprocessing
                            .timeline_checkpoint_file
                        )
                    ),
                    "outlier_labelled": (
                        optional_path_to_string(
                            preprocessing
                            .outlier_checkpoint_file
                        )
                    ),
                    "gap_classified": (
                        optional_path_to_string(
                            preprocessing
                            .gap_checkpoint_file
                        )
                    ),
                    "phase_1_filled": (
                        optional_path_to_string(
                            preprocessing
                            .phase_1_checkpoint_file
                        )
                    ),
                    "phase_2_filled": (
                        optional_path_to_string(
                            preprocessing
                            .phase_2_checkpoint_file
                        )
                    ),
                    "phase_3_filled": (
                        optional_path_to_string(
                            preprocessing
                            .phase_3_checkpoint_file
                        )
                    ),
                },
                "statistics": {
                    "total_input_rows": (
                        preprocessing
                        .total_input_rows
                    ),
                    "final_rows": (
                        preprocessing
                        .final_rows
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
                        preprocessing
                        .valid_series
                    ),
                    "invalid_series": (
                        preprocessing
                        .invalid_series
                    ),
                    "outliers_flagged": (
                        preprocessing
                        .outliers_flagged
                    ),
                    "original_gaps": (
                        preprocessing
                        .original_gaps
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
                        preprocessing
                        .unresolved_points
                    ),
                },
                "method_counts": {
                    "phase_2": (
                        preprocessing
                        .phase_2_method_counts
                    ),
                    "phase_3": (
                        preprocessing
                        .phase_3_method_counts
                    ),
                    "phase_4_pass_1": (
                        preprocessing
                        .phase_4_pass_1_counts
                    ),
                    "phase_4_pass_2": (
                        preprocessing
                        .phase_4_pass_2_counts
                    ),
                },
                "feature_engineering": {
                    "source_file": str(
                        feature_engineering
                        .source_file
                    ),
                    "output_directory": str(
                        feature_engineering
                        .output_directory
                    ),
                    "final_dataset_file": str(
                        feature_engineering
                        .final_dataset_file
                    ),
                    "summary_directory": str(
                        feature_engineering
                        .summary_directory
                    ),
                    "site_summary_file": str(
                        feature_engineering
                        .site_summary_file
                    ),
                    "intermediate_directory": (
                        optional_path_to_string(
                            feature_engineering
                            .intermediate_directory
                        )
                    ),
                    "corrected_history_audit_file": (
                        optional_path_to_string(
                            feature_engineering
                            .corrected_history_audit_file
                        )
                    ),
                    "feature_columns": list(
                        feature_engineering
                        .feature_columns
                    ),
                    "minimum_history_points": (
                        feature_engineering
                        .resolved_config
                        .minimum_history_points
                    ),
                    "statistics": {
                        "total_rows": (
                            feature_engineering
                            .total_rows
                        ),
                        "total_sites": (
                            feature_engineering
                            .total_sites
                        ),
                        "processed_sites": (
                            feature_engineering
                            .processed_sites
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
                "training": (
                    build_training_response(
                        training,
                        training_config=(
                            result
                            .training_config
                        ),
                    )
                ),
            }
        )

    bootstrap = (
        result
        .inference_bootstrap
    )

    history = (
        bootstrap
        .history_contract
    )

    return {
        "status": (
            "completed"
        ),
        "output_directory": str(
            result.output_directory
        ),
        "kpi_count": len(
            result.kpi_results
        ),
        "eda_outputs": {
            "output_directory": str(
                result
                .eda_result
                .output_directory
            ),
            "prepared_datasets_directory": str(
                result
                .eda_result
                .prepared_datasets_directory
            ),
            "summaries_directory": str(
                result
                .eda_result
                .summaries_directory
            ),
            "source_file": str(
                result
                .eda_result
                .preparation
                .source_file
            ),
            "total_input_rows": (
                result
                .eda_result
                .preparation
                .total_input_rows
            ),
            "prepared_kpi_count": len(
                result
                .eda_result
                .preparation
                .prepared_datasets
            ),
            "was_split": (
                result
                .eda_result
                .preparation
                .was_split
            ),
        },
        "saved_run": {
            "run_id": (
                bootstrap.run_id
            ),
            "run_directory": str(
                bootstrap
                .run_directory
            ),
            "inference_config": str(
                bootstrap
                .config_file
            ),
            "inference_state_directory": str(
                bootstrap
                .state_directory
            ),
            "raw_history_file": str(
                bootstrap
                .raw_history_file
            ),
            "raw_cache_file": str(
                bootstrap
                .raw_cache_file
            ),
            "processed_history_by_kpi": {
                key: str(
                    value
                )
                for key, value
                in bootstrap
                .processed_history_by_kpi
                .items()
            },
            "processed_cache_by_kpi": {
                key: str(
                    value
                )
                for key, value
                in bootstrap
                .processed_cache_by_kpi
                .items()
            },
            "inference_results_directory": str(
                bootstrap
                .inference_results_directory
            ),
            "history_contract": {
                "part_b_required_history_points": (
                    history
                    .part_b_required_history_points
                ),
                "part_b_required_history_days": (
                    history
                    .part_b_required_history_days
                ),
                "part_c_required_history_points": (
                    history
                    .part_c_required_history_points
                ),
                "part_c_maximum_lookback_points": (
                    history
                    .part_c_maximum_lookback_points
                ),
                "final_required_history_points": (
                    history
                    .final_required_history_points
                ),
                "final_required_history_days": (
                    history
                    .final_required_history_days
                ),
                "cache_days": (
                    history
                    .cache_days
                ),
                "max_gap_days": (
                    history
                    .max_gap_days
                ),
                "max_gap_ratio": (
                    history
                    .max_gap_ratio
                ),
                "late_data_reconstruction_enabled": (
                    history
                    .late_data_reconstruction_enabled
                ),
            },
        },
        "kpi_results": (
            kpi_responses
        ),
    }


def build_inference_response(
    result: InferenceRunnerResult,
) -> dict[str, Any]:
    """Convert one completed Part-E result into JSON-compatible data."""

    history = (
        result
        .history_commit
    )

    return {
        "status": (
            "completed"
        ),
        "run_id": (
            result.run_id
        ),
        "model_type": (
            result.model_type
        ),
        "invocation_id": (
            result.invocation_id
        ),
        "inference_window": {
            "start": (
                result
                .inference_start
                .isoformat()
            ),
            "end_exclusive": (
                result
                .inference_end_exclusive
                .isoformat()
            ),
        },
        "output_directory": str(
            result
            .output_directory
        ),
        "manifest_file": str(
            result
            .manifest_file
        ),
        "series_statistics": {
            "total_saved_series": (
                result
                .total_saved_series
            ),
            "completed_series": (
                result
                .completed_series
            ),
            "skipped_series": (
                result
                .skipped_series_count
            ),
            "unmodelled_part_c_series": (
                result
                .unmodelled_part_c_series
            ),
        },
        "outputs": [
            {
                "model_type": (
                    item.model_type
                ),
                "kpi_name": (
                    item.kpi_name
                ),
                "series_key": (
                    item.series_key
                ),
                "identity": {
                    "circle": (
                        item.identity[
                            0
                        ]
                    ),
                    "cluster": (
                        item.identity[
                            1
                        ]
                    ),
                    "site_name": (
                        item.identity[
                            2
                        ]
                    ),
                },
                "result_file": str(
                    item.result_file
                ),
                "total_window_rows": (
                    item
                    .total_window_rows
                ),
                "scorable_rows": (
                    item
                    .scorable_rows
                ),
                "unscorable_rows": (
                    item
                    .unscorable_rows
                ),
                "anomaly_count": (
                    item
                    .anomaly_count
                ),
                "anomaly_rate": (
                    item
                    .anomaly_rate
                ),
                "model_anomaly_count": (
                    item
                    .model_anomaly_count
                ),
                "lower_anomaly_count": (
                    item
                    .lower_anomaly_count
                ),
                "upper_anomaly_count": (
                    item
                    .upper_anomaly_count
                ),
            }
            for item
            in result.outputs
        ],
        "skipped_series": [
            {
                "kpi_name": (
                    item.kpi_name
                ),
                "series_key": (
                    item.series_key
                ),
                "identity": {
                    "circle": (
                        item.identity[
                            0
                        ]
                    ),
                    "cluster": (
                        item.identity[
                            1
                        ]
                    ),
                    "site_name": (
                        item.identity[
                            2
                        ]
                    ),
                },
                "reason": (
                    item.reason
                ),
            }
            for item
            in result.skipped_series
        ],
        "history_commit": {
            "raw_history_file": str(
                history
                .raw_history_file
            ),
            "raw_cache_file": str(
                history
                .raw_cache_file
            ),
            "raw_history_rows": (
                history
                .raw_history_rows
            ),
            "raw_cache_rows": (
                history
                .raw_cache_rows
            ),
            "finalized_end": (
                history
                .finalized_end
                .isoformat()
            ),
            "cache_cutoff": (
                history
                .cache_cutoff
                .isoformat()
            ),
            "kpis": [
                {
                    "kpi_name": (
                        item
                        .kpi_name
                    ),
                    "mode": (
                        item.mode
                    ),
                    "processed_history_file": str(
                        item
                        .processed_history_file
                    ),
                    "processed_cache_file": str(
                        item
                        .processed_cache_file
                    ),
                    "processed_history_rows": (
                        item
                        .processed_history_rows
                    ),
                    "processed_cache_rows": (
                        item
                        .processed_cache_rows
                    ),
                    "update_start": (
                        item
                        .update_start
                        .isoformat()
                    ),
                    "update_end": (
                        item
                        .update_end
                        .isoformat()
                    ),
                }
                for item
                in history.kpis
            ],
        },
    }


@router.post(
    "/test-pipeline/run",
    status_code=(
        status.HTTP_200_OK
    ),
)
def run_test_pipeline(
    request: PipelineRunRequest,
) -> dict[str, Any]:
    """Run Part A through Part D and prepare the saved Part-E state."""

    try:
        result = (
            run_test_pipeline_workflow(
                request=request,
                logger=logger,
            )
        )

        return build_pipeline_response(
            result
        )

    except Exception as error:
        logger.exception(
            "The temporary Part A-to-E preparation pipeline failed."
        )

        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "status": (
                    "failed"
                ),
                "stage": (
                    "training_pipeline_or_inference_bootstrap"
                ),
                "error_type": (
                    type(
                        error
                    ).__name__
                ),
                "message": str(
                    error
                ),
            },
        ) from error


@router.post(
    "/test-inference/run",
    status_code=(
        status.HTTP_200_OK
    ),
)
def run_test_inference(
    request: TestInferenceRunRequest,
) -> dict[str, Any]:
    """Run Part-E pure inference against one previously completed saved run."""

    try:
        result = (
            run_test_inference_workflow(
                request=request,
                logger=logger,
            )
        )

        return build_inference_response(
            result
        )

    except Exception as error:
        logger.exception(
            "The temporary Part-E inference test failed."
        )

        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "status": (
                    "failed"
                ),
                "stage": (
                    "inference"
                ),
                "error_type": (
                    type(
                        error
                    ).__name__
                ),
                "message": str(
                    error
                ),
            },
        ) from error


__all__ = [
    "router",
]
