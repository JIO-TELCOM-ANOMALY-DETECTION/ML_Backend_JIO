"""
Part D Isolation Forest final testing.

Intended project location::

    app/ml/training/isolation_forest/testing.py

This module evaluates ONE already-trained Isolation Forest site-series model
on that series' untouched test split.

Isolation Forest first determines whether a row is a model anomaly:

    model.predict(X) == -1

The common threshold direction is then applied:

``lower_only``:
    keep model anomalies where residual < 0

``upper_only``:
    keep model anomalies where residual > 0

``both``:
    keep every model anomaly

For ``both``, the residual feature is not required solely for directional
filtering. If it exists, lower/upper model-anomaly subtype counts are also
reported.

The training trusted-status filter is NOT applied to final test rows. Known
or suspected outlier statuses must remain available for evaluation.

This module does NOT:
- train/refit Isolation Forest;
- use train/validation rows;
- alter contamination/model offset;
- save model/output files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.ml.training.loader import (
    SeriesIdentity,
    SeriesSplitPlan,
    TrainingSplitPlan,
)
from app.ml.training.validation import TrainingValidationResult
from app.schemas.training.isolation_forest.isolation_forest import (
    IsolationForestTrainingConfig,
)


TEST_READ_CHUNK_SIZE = 250_000


@dataclass(
    frozen=True,
    slots=True,
)
class IsolationForestTestingResult:
    """Final untouched-test output for one site-series Isolation Forest model."""

    identity: SeriesIdentity

    usable_row_count: int

    model_anomaly_count: int
    model_anomaly_rate: float

    anomaly_count: int
    anomaly_rate: float

    lower_anomaly_count: int
    upper_anomaly_count: int

    test_summary: dict[str, Any]
    usable_row_counts: dict[str, int]


def _validate_testing_inputs(
    *,
    trained_model: IsolationForest,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: IsolationForestTrainingConfig,
    decision_settings: Mapping[str, Any],
) -> None:
    """Ensure all supplied objects describe one final Isolation Forest test."""

    if validation_result.model_type != "isolation_forest":
        raise ValueError(
            "Isolation Forest testing requires "
            "validation_result.model_type='isolation_forest'."
        )

    if split_plan.model_type != "isolation_forest":
        raise ValueError(
            "Isolation Forest testing requires "
            "split_plan.model_type='isolation_forest'."
        )

    if training_config.model_type != "isolation_forest":
        raise ValueError(
            "Isolation Forest testing requires IsolationForestTrainingConfig."
        )

    if validation_result.input_file != split_plan.input_file:
        raise ValueError(
            "validation_result and split_plan reference different Part C "
            "datasets."
        )

    if validation_result.target_column is not None:
        raise ValueError(
            "Isolation Forest testing must not receive a y target."
        )

    if not validation_result.model_feature_columns:
        raise ValueError(
            "Isolation Forest testing requires validated model features."
        )

    if series_plan.calibration is not None:
        raise ValueError(
            "Isolation Forest testing must not receive a calibration split."
        )

    for method_name in (
        "predict",
        "decision_function",
        "score_samples",
    ):
        if not callable(
            getattr(
                trained_model,
                method_name,
                None,
            )
        ):
            raise TypeError(
                f"trained_model must provide callable {method_name}()."
            )

    if decision_settings.get(
        "method"
    ) != "isolation_forest":
        raise ValueError(
            "Isolation Forest testing requires isolation_forest "
            "decision settings."
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
            "IsolationForestTrainingConfig."
        )

    if expected_direction not in {
        "lower_only",
        "upper_only",
        "both",
    }:
        raise ValueError(
            "Unsupported Isolation Forest threshold_direction: "
            f"{expected_direction}"
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

    required_columns = list(
        dict.fromkeys(
            [
                *validation_result.site_identity_columns,
                validation_result.timestamp_column,
                validation_result.status_column,
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
            "No Isolation Forest test rows were found for series "
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
            "Duplicate timestamps were found while loading the Isolation "
            f"Forest test range for series {series_plan.identity.as_tuple()}."
        )

    if len(
        rows
    ) != series_plan.test.row_count:
        raise ValueError(
            "Isolation Forest raw test row count does not match loader.py "
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
    Keep final-test rows that satisfy feature completeness.

    No trusted-status filter is applied.
    """

    prepared = rows.copy()

    for column in [
        *validation_result.model_feature_columns,
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
                list(
                    validation_result.model_feature_columns
                )
            ]
            .notna()
            .all(axis=1)
        )

    return (
        prepared.loc[
            mask
        ]
        .sort_values(
            validation_result.timestamp_column,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _apply_direction(
    *,
    model_anomaly_mask: np.ndarray,
    rows: pd.DataFrame,
    validation_result: TrainingValidationResult,
    training_config: IsolationForestTrainingConfig,
    decision_settings: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
]:
    """
    Apply lower/upper/both filtering after Isolation Forest anomaly detection.

    Returns:
        final_anomaly_mask,
        lower_model_anomaly_mask,
        upper_model_anomaly_mask,
        direction_counts_available
    """

    direction = str(
        training_config.threshold_direction
    )

    direction_feature = decision_settings.get(
        "direction_feature"
    )

    if not isinstance(
        direction_feature,
        str,
    ) or not direction_feature:
        raise ValueError(
            "Isolation Forest decision settings contain no valid "
            "direction_feature."
        )

    direction_feature_available = (
        direction_feature
        in validation_result.model_feature_columns
    )

    lower_mask = np.zeros(
        model_anomaly_mask.shape,
        dtype=bool,
    )

    upper_mask = np.zeros(
        model_anomaly_mask.shape,
        dtype=bool,
    )

    if direction in {
        "lower_only",
        "upper_only",
    }:
        if not direction_feature_available:
            raise ValueError(
                f"Isolation Forest threshold_direction='{direction}' requires "
                f"feature '{direction_feature}'."
            )

        residual = rows[
            direction_feature
        ].to_numpy(
            dtype=np.float64
        )

        lower_mask = (
            model_anomaly_mask
            & (
                residual
                < 0.0
            )
        )

        upper_mask = (
            model_anomaly_mask
            & (
                residual
                > 0.0
            )
        )

        final_mask = (
            lower_mask
            if direction == "lower_only"
            else upper_mask
        )

        return (
            final_mask,
            lower_mask,
            upper_mask,
            True,
        )

    if direction == "both":
        if direction_feature_available:
            residual = rows[
                direction_feature
            ].to_numpy(
                dtype=np.float64
            )

            lower_mask = (
                model_anomaly_mask
                & (
                    residual
                    < 0.0
                )
            )

            upper_mask = (
                model_anomaly_mask
                & (
                    residual
                    > 0.0
                )
            )

        return (
            model_anomaly_mask.copy(),
            lower_mask,
            upper_mask,
            direction_feature_available,
        )

    raise ValueError(
        "Unsupported Isolation Forest threshold_direction: "
        f"{direction}"
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


def test_isolation_forest_series(
    *,
    trained_model: IsolationForest,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: IsolationForestTrainingConfig,
    decision_settings: Mapping[str, Any],
) -> IsolationForestTestingResult:
    """Run final untouched-test evaluation for one Isolation Forest series."""

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
            "Insufficient usable Isolation Forest test rows for series "
            f"{series_plan.identity.as_tuple()}: "
            f"{len(usable_rows)} < minimum "
            f"{validation_result.minimum_test_points}."
        )

    x_test = usable_rows.loc[
        :,
        list(
            validation_result.model_feature_columns
        ),
    ]

    predicted_labels = np.asarray(
        trained_model.predict(
            x_test
        ),
        dtype=np.int64,
    )

    decision_scores = np.asarray(
        trained_model.decision_function(
            x_test
        ),
        dtype=np.float64,
    )

    raw_scores = np.asarray(
        trained_model.score_samples(
            x_test
        ),
        dtype=np.float64,
    )

    if not (
        predicted_labels.shape
        == decision_scores.shape
        == raw_scores.shape
    ):
        raise ValueError(
            "Isolation Forest test prediction/score shapes differ."
        )

    if bool(
        np.isnan(
            decision_scores
        ).any()
    ) or bool(
        np.isinf(
            decision_scores
        ).any()
    ):
        raise ValueError(
            "Isolation Forest produced invalid test decision scores."
        )

    if bool(
        np.isnan(
            raw_scores
        ).any()
    ) or bool(
        np.isinf(
            raw_scores
        ).any()
    ):
        raise ValueError(
            "Isolation Forest produced invalid test score_samples values."
        )

    model_anomaly_mask = (
        predicted_labels
        == -1
    )

    (
        final_anomaly_mask,
        lower_anomaly_mask,
        upper_anomaly_mask,
        direction_counts_available,
    ) = _apply_direction(
        model_anomaly_mask=model_anomaly_mask,
        rows=usable_rows,
        validation_result=validation_result,
        training_config=training_config,
        decision_settings=decision_settings,
    )

    model_anomaly_count = int(
        np.sum(
            model_anomaly_mask
        )
    )

    anomaly_count = int(
        np.sum(
            final_anomaly_mask
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
        "target_column": None,
        "model_prediction_rule": (
            "predict == -1"
        ),
        "threshold_direction": str(
            training_config.threshold_direction
        ),
        "direction_feature": (
            decision_settings.get(
                "direction_feature"
            )
        ),
        "direction_counts_available": (
            direction_counts_available
        ),
        "model_offset": float(
            trained_model.offset_
        ),
        "contamination": (
            decision_settings.get(
                "contamination"
            )
        ),
        "model_anomaly_count": (
            model_anomaly_count
        ),
        "model_anomaly_rate": float(
            np.mean(
                model_anomaly_mask
            )
        ),
        "anomaly_count": (
            anomaly_count
        ),
        "anomaly_rate": float(
            np.mean(
                final_anomaly_mask
            )
        ),
        "lower_anomaly_count": (
            lower_anomaly_count
        ),
        "upper_anomaly_count": (
            upper_anomaly_count
        ),
        "decision_score_mean": float(
            np.mean(
                decision_scores
            )
        ),
        "decision_score_median": float(
            np.median(
                decision_scores
            )
        ),
        "decision_score_std": float(
            np.std(
                decision_scores
            )
        ),
        "decision_score_min": float(
            np.min(
                decision_scores
            )
        ),
        "decision_score_max": float(
            np.max(
                decision_scores
            )
        ),
        "score_samples_mean": float(
            np.mean(
                raw_scores
            )
        ),
        "score_samples_median": float(
            np.median(
                raw_scores
            )
        ),
        "score_samples_std": float(
            np.std(
                raw_scores
            )
        ),
        "score_samples_min": float(
            np.min(
                raw_scores
            )
        ),
        "score_samples_max": float(
            np.max(
                raw_scores
            )
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

    return IsolationForestTestingResult(
        identity=series_plan.identity,
        usable_row_count=len(
            usable_rows
        ),
        model_anomaly_count=(
            model_anomaly_count
        ),
        model_anomaly_rate=float(
            np.mean(
                model_anomaly_mask
            )
        ),
        anomaly_count=(
            anomaly_count
        ),
        anomaly_rate=float(
            np.mean(
                final_anomaly_mask
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
    "IsolationForestTestingResult",
    "test_isolation_forest_series",
]