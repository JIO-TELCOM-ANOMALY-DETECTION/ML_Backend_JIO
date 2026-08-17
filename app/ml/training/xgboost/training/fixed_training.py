"""
Part D XGBoost direct/fixed-candidate training.

Intended project location::

    app/ml/training/xgboost/fixed_training.py

This module owns the XGBoost fitting stage for ONE Composite_ID/site series.

Training modes
--------------
Hyperparameter tuning enabled::

    train
      -> fit every complete candidate
      -> evaluate each candidate on validation
      -> select by configured primary metric, then secondary metric
      -> obtain best tree count from early stopping
      -> refit final model on trusted train + validation rows

Hyperparameter tuning disabled::

    train
      -> resolve one complete hyperparameter configuration
      -> fit one final model directly

The calibration and test ranges are deliberately untouched here.

Responsibilities
----------------
- load only the selected series' train/validation ranges from the Part C CSV;
- create X using the exact validated feature order;
- create y from the validated XGBoost target column;
- apply trusted-status, complete-feature, and non-null-target rules;
- enforce minimum usable train/validation row counts;
- perform optional candidate comparison;
- fit/refit the final XGBoost model; and
- return the model plus training summaries to runner.py.

This module does NOT:
- create chronological split boundaries;
- use calibration rows;
- use test rows;
- calculate residual anomaly cutoffs;
- make final anomaly decisions;
- save model files or JSON artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.ml.training.loader import (
    SeriesIdentity,
    SeriesSplitPlan,
    TrainingSplitPlan,
)
from app.ml.training.validation import TrainingValidationResult
from app.schemas.training.xgboost.defaults import (
    DEFAULT_XGBOOST_BASE_PARAMS,
)
from app.schemas.training.xgboost.xgboost import (
    XGBoostHyperparameterSet,
    XGBoostTrainingConfig,
)


TRAINING_READ_CHUNK_SIZE = 250_000


@dataclass(
    frozen=True,
    slots=True,
)
class XGBoostTrainingResult:
    """
    Result of training one XGBoost model for one site series.

    ``model`` is intentionally kept in memory and returned to runner.py.
    Saving is handled later by model_output.py.
    """

    identity: SeriesIdentity

    model: XGBRegressor

    selected_hyperparameters: dict[str, Any]
    final_model_parameters: dict[str, Any]

    selected_candidate_number: int | None
    best_tree_count: int

    tuning_performed: bool
    tuning_summary: tuple[dict[str, Any], ...]

    training_summary: dict[str, Any]
    usable_row_counts: dict[str, int]


def _validate_training_inputs(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: XGBoostTrainingConfig,
) -> None:
    """Ensure all supplied objects describe one XGBoost training request."""

    if validation_result.model_type != "xgboost":
        raise ValueError(
            "XGBoost training requires validation_result.model_type='xgboost'."
        )

    if split_plan.model_type != "xgboost":
        raise ValueError(
            "XGBoost training requires split_plan.model_type='xgboost'."
        )

    if training_config.model_type != "xgboost":
        raise ValueError(
            "XGBoost training requires XGBoostTrainingConfig."
        )

    if validation_result.input_file != split_plan.input_file:
        raise ValueError(
            "validation_result and split_plan reference different Part C "
            "datasets."
        )

    if (
        split_plan.hyperparameter_tuning_enabled
        != training_config.hyperparameter_tuning_enabled
    ):
        raise ValueError(
            "split_plan tuning mode does not match XGBoostTrainingConfig."
        )

    if validation_result.target_column is None:
        raise ValueError(
            "XGBoost training requires one validated target column."
        )

    if not validation_result.model_feature_columns:
        raise ValueError(
            "XGBoost training requires at least one validated model feature."
        )

    if training_config.hyperparameter_tuning_enabled:
        if series_plan.validation is None:
            raise ValueError(
                "XGBoost tuning requires a validation split range."
            )
    elif series_plan.validation is not None:
        raise ValueError(
            "XGBoost direct training must not receive a validation split."
        )


def _series_identity_mask(
    chunk: pd.DataFrame,
    *,
    identity_columns: tuple[str, str, str],
    identity: SeriesIdentity,
) -> pd.Series:
    """Return rows belonging to exactly one planned site identity."""

    expected_values = identity.as_tuple()

    mask = pd.Series(
        True,
        index=chunk.index,
        dtype=bool,
    )

    for column, expected in zip(
        identity_columns,
        expected_values,
        strict=True,
    ):
        mask &= (
            chunk[column]
            .astype("string")
            .eq(str(expected))
            .fillna(False)
        )

    return mask


def _load_training_window(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
) -> pd.DataFrame:
    """
    Load only raw train/validation rows for one series.

    The CSV is scanned in chunks. Calibration/test rows are not retained.
    """

    feature_columns = list(
        validation_result.model_feature_columns
    )

    target_column = validation_result.target_column

    if target_column is None:
        raise ValueError(
            "XGBoost target column is missing from validation result."
        )

    required_columns = list(
        dict.fromkeys(
            [
                *validation_result.site_identity_columns,
                validation_result.timestamp_column,
                validation_result.status_column,
                target_column,
                *feature_columns,
            ]
        )
    )

    start_timestamp = pd.Timestamp(
        series_plan.train.start_timestamp
    )

    end_timestamp = pd.Timestamp(
        (
            series_plan.validation.end_timestamp
            if series_plan.validation is not None
            else series_plan.train.end_timestamp
        )
    )

    selected_chunks: list[pd.DataFrame] = []

    reader = pd.read_csv(
        split_plan.input_file,
        usecols=required_columns,
        chunksize=TRAINING_READ_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        identity_mask = _series_identity_mask(
            chunk,
            identity_columns=validation_result.site_identity_columns,
            identity=series_plan.identity,
        )

        if not bool(identity_mask.any()):
            continue

        subset = chunk.loc[
            identity_mask
        ].copy()

        subset[
            validation_result.timestamp_column
        ] = pd.to_datetime(
            subset[
                validation_result.timestamp_column
            ],
            errors="raise",
        )

        time_mask = (
            subset[
                validation_result.timestamp_column
            ].ge(start_timestamp)
            & subset[
                validation_result.timestamp_column
            ].le(end_timestamp)
        )

        subset = subset.loc[
            time_mask
        ]

        if not subset.empty:
            selected_chunks.append(
                subset
            )

    if not selected_chunks:
        raise ValueError(
            "No XGBoost train/validation rows were found for series "
            f"{series_plan.identity.as_tuple()}."
        )

    rows = pd.concat(
        selected_chunks,
        ignore_index=True,
        sort=False,
    )

    rows = (
        rows
        .sort_values(
            validation_result.timestamp_column,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    duplicate_time = rows[
        validation_result.timestamp_column
    ].duplicated()

    if bool(
        duplicate_time.any()
    ):
        raise ValueError(
            "Duplicate timestamps were found while loading the XGBoost "
            f"training window for series {series_plan.identity.as_tuple()}."
        )

    return rows


def _select_raw_stage_rows(
    rows: pd.DataFrame,
    *,
    timestamp_column: str,
    start_timestamp: Any,
    end_timestamp: Any,
) -> pd.DataFrame:
    """Select one inclusive chronological stage from the loaded window."""

    start = pd.Timestamp(
        start_timestamp
    )

    end = pd.Timestamp(
        end_timestamp
    )

    mask = (
        rows[timestamp_column].ge(start)
        & rows[timestamp_column].le(end)
    )

    return (
        rows.loc[
            mask
        ]
        .sort_values(
            timestamp_column,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _validate_raw_stage_count(
    *,
    stage_name: str,
    rows: pd.DataFrame,
    expected_count: int,
    identity: SeriesIdentity,
) -> None:
    """
    Ensure the timestamp-boundary load exactly reproduces loader.py's range.

    This catches a changed source file or a boundary mismatch before fitting.
    """

    actual_count = len(
        rows
    )

    if actual_count != expected_count:
        raise ValueError(
            f"XGBoost raw {stage_name} row count does not match loader.py "
            f"for series {identity.as_tuple()}. "
            f"Expected {expected_count}, found {actual_count}."
        )


def _prepare_usable_rows(
    rows: pd.DataFrame,
    *,
    validation_result: TrainingValidationResult,
) -> pd.DataFrame:
    """
    Apply the fixed XGBoost row-eligibility rules.

    Trusted statuses apply to y because this is self-supervised regression.
    """

    feature_columns = list(
        validation_result.model_feature_columns
    )

    target_column = validation_result.target_column

    if target_column is None:
        raise ValueError(
            "XGBoost target column is missing."
        )

    status_column = validation_result.status_column

    prepared = rows.copy()

    for column in [
        *feature_columns,
        target_column,
        status_column,
    ]:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    mask = (
        prepared[
            status_column
        ]
        .isin(
            validation_result.trusted_statuses
        )
        .fillna(False)
    )

    if validation_result.require_complete_feature_rows:
        mask &= (
            prepared[
                feature_columns
            ]
            .notna()
            .all(axis=1)
        )

    if validation_result.require_non_null_target:
        mask &= prepared[
            target_column
        ].notna()

    usable = (
        prepared.loc[
            mask
        ]
        .sort_values(
            validation_result.timestamp_column,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return usable


def _require_minimum_usable_rows(
    *,
    rows: pd.DataFrame,
    stage_name: str,
    minimum_points: int,
    identity: SeriesIdentity,
) -> None:
    """Enforce minimum usable rows after all model-row filters."""

    if len(rows) < minimum_points:
        raise ValueError(
            f"Insufficient usable XGBoost {stage_name} rows for series "
            f"{identity.as_tuple()}: "
            f"{len(rows)} < minimum {minimum_points}."
        )


def _xy(
    rows: pd.DataFrame,
    *,
    validation_result: TrainingValidationResult,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build X and y using the exact validated feature order."""

    target_column = validation_result.target_column

    if target_column is None:
        raise ValueError(
            "XGBoost target column is missing."
        )

    x = rows.loc[
        :,
        list(
            validation_result.model_feature_columns
        ),
    ]

    y = rows.loc[
        :,
        target_column,
    ]

    return (
        x,
        y,
    )


def _regression_metrics(
    *,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate the regression diagnostics used for candidate comparison."""

    actual_values = np.asarray(
        actual,
        dtype=np.float64,
    )

    predicted_values = np.asarray(
        predicted,
        dtype=np.float64,
    )

    if actual_values.shape != predicted_values.shape:
        raise ValueError(
            "Actual and predicted arrays must have the same shape."
        )

    if actual_values.size == 0:
        raise ValueError(
            "Regression metrics require at least one row."
        )

    residual = (
        actual_values
        - predicted_values
    )

    residual_median = float(
        np.median(
            residual
        )
    )

    return {
        "mae": float(
            np.mean(
                np.abs(
                    residual
                )
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        residual
                    )
                )
            )
        ),
        "median_residual": (
            residual_median
        ),
        "mean_residual": float(
            np.mean(
                residual
            )
        ),
        "residual_mad": float(
            np.median(
                np.abs(
                    residual
                    - residual_median
                )
            )
        ),
    }


def _candidate_model(
    *,
    parameters: XGBoostHyperparameterSet,
    training_config: XGBoostTrainingConfig,
) -> XGBRegressor:
    """Create one temporary early-stopping candidate model."""

    return XGBRegressor(
        **dict(
            DEFAULT_XGBOOST_BASE_PARAMS
        ),
        **dict(
            parameters
        ),
        n_estimators=(
            training_config.tuning.max_estimators
        ),
        early_stopping_rounds=(
            training_config.tuning.early_stopping_rounds
        ),
    )


def _fit_and_select_candidate(
    *,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    validation_result: TrainingValidationResult,
    training_config: XGBoostTrainingConfig,
) -> tuple[
    XGBoostHyperparameterSet,
    int,
    int,
    dict[str, float],
    tuple[dict[str, Any], ...],
]:
    """Fit all candidates and select the best validation result."""

    x_train, y_train = _xy(
        train_rows,
        validation_result=validation_result,
    )

    x_validation, y_validation = _xy(
        validation_rows,
        validation_result=validation_result,
    )

    candidates = (
        training_config.resolve_hyperparameter_candidates()
    )

    candidate_results: list[
        dict[str, Any]
    ] = []

    passed_candidates: list[
        dict[str, Any]
    ] = []

    for candidate_number, parameters in enumerate(
        candidates,
        start=1,
    ):
        result: dict[str, Any] = {
            "candidate_number": candidate_number,
            "candidate_status": "failed",
            "hyperparameters": dict(
                parameters
            ),
            "best_iteration": None,
            "best_tree_count": None,
            "validation_metrics": None,
            "error_message": None,
        }

        try:
            model = _candidate_model(
                parameters=parameters,
                training_config=training_config,
            )

            model.fit(
                x_train,
                y_train,
                eval_set=[
                    (
                        x_validation,
                        y_validation,
                    )
                ],
                verbose=False,
            )

            prediction = model.predict(
                x_validation
            )

            metrics = _regression_metrics(
                actual=y_validation.to_numpy(),
                predicted=prediction,
            )

            best_iteration = getattr(
                model,
                "best_iteration",
                None,
            )

            if best_iteration is None:
                best_iteration = (
                    training_config.tuning.max_estimators
                    - 1
                )

            best_iteration = int(
                best_iteration
            )

            best_tree_count = (
                best_iteration
                + 1
            )

            result.update(
                {
                    "candidate_status": "passed",
                    "best_iteration": (
                        best_iteration
                    ),
                    "best_tree_count": (
                        best_tree_count
                    ),
                    "validation_metrics": (
                        metrics
                    ),
                    "error_message": None,
                }
            )

            passed_candidates.append(
                {
                    "candidate_number": (
                        candidate_number
                    ),
                    "hyperparameters": dict(
                        parameters
                    ),
                    "best_iteration": (
                        best_iteration
                    ),
                    "best_tree_count": (
                        best_tree_count
                    ),
                    "validation_metrics": (
                        metrics
                    ),
                }
            )

        except Exception as error:
            result[
                "error_message"
            ] = (
                f"{type(error).__name__}: {error}"
            )

        candidate_results.append(
            result
        )

    if not passed_candidates:
        raise RuntimeError(
            "Every XGBoost hyperparameter candidate failed."
        )

    primary_metric = (
        training_config
        .tuning
        .primary_selection_metric
    )

    secondary_metric = (
        training_config
        .tuning
        .secondary_selection_metric
    )

    best_candidate = min(
        passed_candidates,
        key=lambda item: (
            item[
                "validation_metrics"
            ][
                primary_metric
            ],
            item[
                "validation_metrics"
            ][
                secondary_metric
            ],
            item[
                "candidate_number"
            ],
        ),
    )

    return (
        dict(
            best_candidate[
                "hyperparameters"
            ]
        ),
        int(
            best_candidate[
                "candidate_number"
            ]
        ),
        int(
            best_candidate[
                "best_tree_count"
            ]
        ),
        dict(
            best_candidate[
                "validation_metrics"
            ]
        ),
        tuple(
            candidate_results
        ),
    )


def _fit_final_model(
    *,
    final_fit_rows: pd.DataFrame,
    validation_result: TrainingValidationResult,
    selected_hyperparameters: XGBoostHyperparameterSet,
    tree_count: int,
) -> XGBRegressor:
    """Fit the final XGBoost estimator without calibration/test access."""

    if tree_count <= 0:
        raise ValueError(
            "Final XGBoost tree_count must be positive."
        )

    x_final, y_final = _xy(
        final_fit_rows,
        validation_result=validation_result,
    )

    model = XGBRegressor(
        **dict(
            DEFAULT_XGBOOST_BASE_PARAMS
        ),
        **dict(
            selected_hyperparameters
        ),
        n_estimators=tree_count,
    )

    model.fit(
        x_final,
        y_final,
        verbose=False,
    )

    return model


def train_xgboost_series(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: XGBoostTrainingConfig,
) -> XGBoostTrainingResult:
    """
    Train the final XGBoost model for one planned site series.

    runner.py should call this once for each ``split_plan.series`` entry.
    """

    _validate_training_inputs(
        validation_result=validation_result,
        split_plan=split_plan,
        series_plan=series_plan,
        training_config=training_config,
    )

    loaded_rows = _load_training_window(
        validation_result=validation_result,
        split_plan=split_plan,
        series_plan=series_plan,
    )

    raw_train_rows = _select_raw_stage_rows(
        loaded_rows,
        timestamp_column=validation_result.timestamp_column,
        start_timestamp=series_plan.train.start_timestamp,
        end_timestamp=series_plan.train.end_timestamp,
    )

    _validate_raw_stage_count(
        stage_name="train",
        rows=raw_train_rows,
        expected_count=series_plan.train.row_count,
        identity=series_plan.identity,
    )

    train_rows = _prepare_usable_rows(
        raw_train_rows,
        validation_result=validation_result,
    )

    _require_minimum_usable_rows(
        rows=train_rows,
        stage_name="train",
        minimum_points=validation_result.minimum_train_points,
        identity=series_plan.identity,
    )

    tuning_performed = bool(
        training_config.hyperparameter_tuning_enabled
    )

    validation_rows = pd.DataFrame()
    validation_metrics: dict[str, float] | None = None
    tuning_summary: tuple[
        dict[str, Any],
        ...
    ] = tuple()
    selected_candidate_number: int | None = None

    if tuning_performed:
        if series_plan.validation is None:
            raise RuntimeError(
                "Internal XGBoost error: validation split disappeared."
            )

        raw_validation_rows = _select_raw_stage_rows(
            loaded_rows,
            timestamp_column=validation_result.timestamp_column,
            start_timestamp=(
                series_plan.validation.start_timestamp
            ),
            end_timestamp=(
                series_plan.validation.end_timestamp
            ),
        )

        _validate_raw_stage_count(
            stage_name="validation",
            rows=raw_validation_rows,
            expected_count=series_plan.validation.row_count,
            identity=series_plan.identity,
        )

        validation_rows = _prepare_usable_rows(
            raw_validation_rows,
            validation_result=validation_result,
        )

        _require_minimum_usable_rows(
            rows=validation_rows,
            stage_name="validation",
            minimum_points=(
                validation_result.minimum_validation_points
            ),
            identity=series_plan.identity,
        )

        (
            selected_hyperparameters,
            selected_candidate_number,
            best_tree_count,
            validation_metrics,
            tuning_summary,
        ) = _fit_and_select_candidate(
            train_rows=train_rows,
            validation_rows=validation_rows,
            validation_result=validation_result,
            training_config=training_config,
        )

        # After candidate selection, validation is no longer needed for model
        # selection. It may therefore join train for the final refit. The
        # dedicated calibration and untouched test ranges remain unseen.
        final_fit_rows = (
            pd.concat(
                [
                    train_rows,
                    validation_rows,
                ],
                ignore_index=True,
                sort=False,
            )
            .sort_values(
                validation_result.timestamp_column,
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    else:
        selected_hyperparameters = (
            training_config.resolve_hyperparameters()
        )

        # Without validation/early stopping there is no learned tree count.
        # The configured max_estimators therefore becomes the direct final
        # n_estimators value.
        best_tree_count = int(
            training_config.tuning.max_estimators
        )

        final_fit_rows = (
            train_rows.copy()
        )

    final_model = _fit_final_model(
        final_fit_rows=final_fit_rows,
        validation_result=validation_result,
        selected_hyperparameters=(
            selected_hyperparameters
        ),
        tree_count=best_tree_count,
    )

    x_final, y_final = _xy(
        final_fit_rows,
        validation_result=validation_result,
    )

    final_fit_prediction = (
        final_model.predict(
            x_final
        )
    )

    final_fit_metrics = _regression_metrics(
        actual=y_final.to_numpy(),
        predicted=final_fit_prediction,
    )

    final_model_parameters = {
        **dict(
            DEFAULT_XGBOOST_BASE_PARAMS
        ),
        **dict(
            selected_hyperparameters
        ),
        "n_estimators": (
            best_tree_count
        ),
    }

    usable_row_counts = {
        "train": len(
            train_rows
        ),
        "validation": len(
            validation_rows
        ),
        "final_fit": len(
            final_fit_rows
        ),
    }

    training_summary: dict[
        str,
        Any,
    ] = {
        "series_identity": {
            "circle": (
                series_plan.identity.circle
            ),
            "cluster": (
                series_plan.identity.cluster
            ),
            "site_name": (
                series_plan.identity.site_name
            ),
        },
        "tuning_performed": (
            tuning_performed
        ),
        "feature_columns": list(
            validation_result.model_feature_columns
        ),
        "feature_count": len(
            validation_result.model_feature_columns
        ),
        "target_column": (
            validation_result.target_column
        ),
        "raw_train_rows": len(
            raw_train_rows
        ),
        "usable_train_rows": len(
            train_rows
        ),
        "raw_validation_rows": (
            series_plan.validation.row_count
            if series_plan.validation is not None
            else 0
        ),
        "usable_validation_rows": len(
            validation_rows
        ),
        "final_fit_rows": len(
            final_fit_rows
        ),
        "selected_candidate_number": (
            selected_candidate_number
        ),
        "best_tree_count": (
            best_tree_count
        ),
        "selected_hyperparameters": dict(
            selected_hyperparameters
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "final_fit_metrics": (
            final_fit_metrics
        ),
    }

    return XGBoostTrainingResult(
        identity=series_plan.identity,
        model=final_model,
        selected_hyperparameters=dict(
            selected_hyperparameters
        ),
        final_model_parameters=(
            final_model_parameters
        ),
        selected_candidate_number=(
            selected_candidate_number
        ),
        best_tree_count=(
            best_tree_count
        ),
        tuning_performed=(
            tuning_performed
        ),
        tuning_summary=(
            tuning_summary
        ),
        training_summary=(
            training_summary
        ),
        usable_row_counts=(
            usable_row_counts
        ),
    )


__all__ = [
    "XGBoostTrainingResult",
    "train_xgboost_series",
]