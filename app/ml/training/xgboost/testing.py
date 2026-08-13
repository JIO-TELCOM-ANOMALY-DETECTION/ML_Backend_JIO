"""
Part D XGBoost final testing.

Intended project location::

    app/ml/training/xgboost/testing.py

This module evaluates ONE already-trained and calibrated XGBoost site-series
model on that series' untouched test split.

Residual definition
-------------------
    residual = actual_target - predicted_target

Decision behavior
-----------------
The thresholds are not recalculated here. They must come from
``residual_calibration.py``:

``lower_only``:
    anomaly when residual <= lower_threshold

``upper_only``:
    anomaly when residual >= upper_threshold

``both``:
    anomaly when residual <= lower_threshold
    OR residual >= upper_threshold

Important testing rule
----------------------
The training/calibration trusted-target-status filter is NOT applied to the
final test set. In particular, a status-2 outlier must remain available for
testing rather than being removed merely because it is not a trusted training
target.

The test stage only requires:
- complete model features when configured;
- a non-null target when configured; and
- the structural/status columns already validated upstream.

Status values are preserved for summary/analysis, but this module does not
hardcode a status-to-ground-truth anomaly classification contract.

This module does NOT:
- train/refit XGBoost;
- use train/validation/calibration rows;
- recalculate residual thresholds;
- save model/output files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.ml.training.loader import (
    SeriesIdentity,
    SeriesSplitPlan,
    TrainingSplitPlan,
)
from app.ml.training.validation import TrainingValidationResult
from app.schemas.training.xgboost.xgboost import (
    XGBoostTrainingConfig,
)


TEST_READ_CHUNK_SIZE = 250_000


@dataclass(
    frozen=True,
    slots=True,
)
class XGBoostTestingResult:
    """Final untouched-test output for one site-series XGBoost model."""

    identity: SeriesIdentity

    usable_row_count: int
    anomaly_count: int
    anomaly_rate: float

    lower_anomaly_count: int
    upper_anomaly_count: int

    test_summary: dict[str, Any]
    usable_row_counts: dict[str, int]


def _validate_testing_inputs(
    *,
    trained_model: XGBRegressor,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: XGBoostTrainingConfig,
    decision_settings: Mapping[str, Any],
) -> None:
    """Ensure all inputs describe one finalized XGBoost test request."""

    if validation_result.model_type != "xgboost":
        raise ValueError(
            "XGBoost testing requires validation_result.model_type='xgboost'."
        )

    if split_plan.model_type != "xgboost":
        raise ValueError(
            "XGBoost testing requires split_plan.model_type='xgboost'."
        )

    if training_config.model_type != "xgboost":
        raise ValueError(
            "XGBoost testing requires XGBoostTrainingConfig."
        )

    if validation_result.input_file != split_plan.input_file:
        raise ValueError(
            "validation_result and split_plan reference different Part C "
            "datasets."
        )

    if validation_result.target_column is None:
        raise ValueError(
            "XGBoost testing requires a validated target column."
        )

    if not validation_result.model_feature_columns:
        raise ValueError(
            "XGBoost testing requires validated model features."
        )

    if not callable(
        getattr(
            trained_model,
            "predict",
            None,
        )
    ):
        raise TypeError(
            "trained_model must provide a callable predict() method."
        )

    if decision_settings.get(
        "method"
    ) != "residual_quantile":
        raise ValueError(
            "XGBoost testing requires residual_quantile decision settings."
        )

    expected_direction = str(
        training_config.threshold_direction
    )

    supplied_direction = str(
        decision_settings.get(
            "threshold_direction"
        )
    )

    if supplied_direction != expected_direction:
        raise ValueError(
            "decision_settings threshold_direction does not match "
            "XGBoostTrainingConfig."
        )

    if decision_settings.get(
        "residual_definition"
    ) != "actual_minus_prediction":
        raise ValueError(
            "XGBoost testing requires residual_definition="
            "'actual_minus_prediction'."
        )

    if expected_direction in {
        "lower_only",
        "both",
    }:
        if decision_settings.get(
            "lower_threshold"
        ) is None:
            raise ValueError(
                f"{expected_direction} requires lower_threshold."
            )

    if expected_direction in {
        "upper_only",
        "both",
    }:
        if decision_settings.get(
            "upper_threshold"
        ) is None:
            raise ValueError(
                f"{expected_direction} requires upper_threshold."
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


def _load_test_rows(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
) -> pd.DataFrame:
    """Load only the untouched test range for one series."""

    target_column = validation_result.target_column

    if target_column is None:
        raise ValueError(
            "XGBoost target column is missing."
        )

    required_columns = list(
        dict.fromkeys(
            [
                *validation_result.site_identity_columns,
                validation_result.timestamp_column,
                validation_result.status_column,
                target_column,
                *validation_result.model_feature_columns,
            ]
        )
    )

    start_timestamp = pd.Timestamp(
        series_plan.test.start_timestamp
    )

    end_timestamp = pd.Timestamp(
        series_plan.test.end_timestamp
    )

    selected_chunks: list[pd.DataFrame] = []

    reader = pd.read_csv(
        split_plan.input_file,
        usecols=required_columns,
        chunksize=TEST_READ_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        identity_mask = _series_identity_mask(
            chunk,
            identity_columns=validation_result.site_identity_columns,
            identity=series_plan.identity,
        )

        if not bool(
            identity_mask.any()
        ):
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
            "No XGBoost test rows were found for series "
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

    if bool(
        rows[
            validation_result.timestamp_column
        ].duplicated().any()
    ):
        raise ValueError(
            "Duplicate timestamps were found while loading the XGBoost "
            f"test range for series {series_plan.identity.as_tuple()}."
        )

    if len(
        rows
    ) != series_plan.test.row_count:
        raise ValueError(
            "XGBoost raw test row count does not match loader.py "
            f"for series {series_plan.identity.as_tuple()}. "
            f"Expected {series_plan.test.row_count}, found {len(rows)}."
        )

    return rows


def _prepare_usable_test_rows(
    rows: pd.DataFrame,
    *,
    validation_result: TrainingValidationResult,
) -> pd.DataFrame:
    """
    Apply final test row eligibility.

    Unlike training/calibration, trusted_target_statuses are deliberately NOT
    used here. This keeps non-training statuses, including known outliers,
    available for final testing.
    """

    feature_columns = list(
        validation_result.model_feature_columns
    )

    target_column = validation_result.target_column

    if target_column is None:
        raise ValueError(
            "XGBoost target column is missing."
        )

    prepared = rows.copy()

    for column in [
        *feature_columns,
        target_column,
        validation_result.status_column,
    ]:
        prepared[
            column
        ] = pd.to_numeric(
            prepared[
                column
            ],
            errors="coerce",
        )

    mask = pd.Series(
        True,
        index=prepared.index,
        dtype=bool,
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


def _predict_residuals(
    *,
    trained_model: XGBRegressor,
    rows: pd.DataFrame,
    validation_result: TrainingValidationResult,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Generate actual, predicted, and actual-minus-prediction residuals."""

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

    actual = rows[
        target_column
    ].to_numpy(
        dtype=np.float64
    )

    predicted = np.asarray(
        trained_model.predict(
            x
        ),
        dtype=np.float64,
    )

    if predicted.shape != actual.shape:
        raise ValueError(
            "XGBoost test prediction shape does not match target shape."
        )

    if bool(
        np.isnan(
            predicted
        ).any()
    ) or bool(
        np.isinf(
            predicted
        ).any()
    ):
        raise ValueError(
            "XGBoost produced invalid test predictions."
        )

    residual = (
        actual
        - predicted
    )

    return (
        actual,
        predicted,
        residual,
    )


def _apply_decision_settings(
    *,
    residual: np.ndarray,
    decision_settings: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Apply already-calibrated thresholds.

    Returns:
        anomaly_mask,
        lower_anomaly_mask,
        upper_anomaly_mask
    """

    direction = str(
        decision_settings[
            "threshold_direction"
        ]
    )

    lower_mask = np.zeros(
        residual.shape,
        dtype=bool,
    )

    upper_mask = np.zeros(
        residual.shape,
        dtype=bool,
    )

    if direction in {
        "lower_only",
        "both",
    }:
        lower_threshold = float(
            decision_settings[
                "lower_threshold"
            ]
        )

        lower_mask = (
            residual
            <= lower_threshold
        )

    if direction in {
        "upper_only",
        "both",
    }:
        upper_threshold = float(
            decision_settings[
                "upper_threshold"
            ]
        )

        upper_mask = (
            residual
            >= upper_threshold
        )

    if direction == "lower_only":
        anomaly_mask = lower_mask

    elif direction == "upper_only":
        anomaly_mask = upper_mask

    elif direction == "both":
        anomaly_mask = (
            lower_mask
            | upper_mask
        )

    else:
        raise ValueError(
            "Unsupported XGBoost threshold_direction: "
            f"{direction}"
        )

    return (
        anomaly_mask,
        lower_mask,
        upper_mask,
    )


def _status_counts(
    rows: pd.DataFrame,
    *,
    status_column: str,
) -> dict[str, int]:
    """Return test status distribution without interpreting label semantics."""

    counts = (
        rows[
            status_column
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    return {
        str(
            key
        ): int(
            value
        )
        for key, value in counts.items()
    }


def test_xgboost_series(
    *,
    trained_model: XGBRegressor,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: XGBoostTrainingConfig,
    decision_settings: Mapping[str, Any],
) -> XGBoostTestingResult:
    """Run final untouched-test evaluation for one XGBoost site series."""

    _validate_testing_inputs(
        trained_model=trained_model,
        validation_result=validation_result,
        split_plan=split_plan,
        series_plan=series_plan,
        training_config=training_config,
        decision_settings=decision_settings,
    )

    raw_rows = _load_test_rows(
        validation_result=validation_result,
        split_plan=split_plan,
        series_plan=series_plan,
    )

    usable_rows = _prepare_usable_test_rows(
        raw_rows,
        validation_result=validation_result,
    )

    if len(
        usable_rows
    ) < validation_result.minimum_test_points:
        raise ValueError(
            "Insufficient usable XGBoost test rows for series "
            f"{series_plan.identity.as_tuple()}: "
            f"{len(usable_rows)} < minimum "
            f"{validation_result.minimum_test_points}."
        )

    (
        actual,
        predicted,
        residual,
    ) = _predict_residuals(
        trained_model=trained_model,
        rows=usable_rows,
        validation_result=validation_result,
    )

    (
        anomaly_mask,
        lower_anomaly_mask,
        upper_anomaly_mask,
    ) = _apply_decision_settings(
        residual=residual,
        decision_settings=decision_settings,
    )

    anomaly_count = int(
        np.sum(
            anomaly_mask
        )
    )

    lower_anomaly_count = int(
        np.sum(
            lower_anomaly_mask
        )
    )

    upper_anomaly_count = int(
        np.sum(
            upper_anomaly_mask
        )
    )

    residual_median = float(
        np.median(
            residual
        )
    )

    test_summary: dict[
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
        "raw_test_rows": len(
            raw_rows
        ),
        "usable_test_rows": len(
            usable_rows
        ),
        "test_start_timestamp": (
            series_plan.test.start_timestamp
        ),
        "test_end_timestamp": (
            series_plan.test.end_timestamp
        ),
        "feature_count": len(
            validation_result.model_feature_columns
        ),
        "target_column": (
            validation_result.target_column
        ),
        "residual_definition": (
            "actual_minus_prediction"
        ),
        "threshold_direction": (
            decision_settings[
                "threshold_direction"
            ]
        ),
        "lower_threshold": (
            decision_settings.get(
                "lower_threshold"
            )
        ),
        "upper_threshold": (
            decision_settings.get(
                "upper_threshold"
            )
        ),
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
        "residual_mean": float(
            np.mean(
                residual
            )
        ),
        "residual_median": (
            residual_median
        ),
        "residual_std": float(
            np.std(
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
        "residual_min": float(
            np.min(
                residual
            )
        ),
        "residual_max": float(
            np.max(
                residual
            )
        ),
        "anomaly_count": (
            anomaly_count
        ),
        "anomaly_rate": float(
            np.mean(
                anomaly_mask
            )
        ),
        "lower_anomaly_count": (
            lower_anomaly_count
        ),
        "upper_anomaly_count": (
            upper_anomaly_count
        ),
        "status_counts": _status_counts(
            usable_rows,
            status_column=validation_result.status_column,
        ),
    }

    usable_row_counts = {
        "test": len(
            usable_rows
        ),
    }

    return XGBoostTestingResult(
        identity=series_plan.identity,
        usable_row_count=len(
            usable_rows
        ),
        anomaly_count=(
            anomaly_count
        ),
        anomaly_rate=float(
            np.mean(
                anomaly_mask
            )
        ),
        lower_anomaly_count=(
            lower_anomaly_count
        ),
        upper_anomaly_count=(
            upper_anomaly_count
        ),
        test_summary=(
            test_summary
        ),
        usable_row_counts=(
            usable_row_counts
        ),
    )


__all__ = [
    "XGBoostTestingResult",
    "test_xgboost_series",
]