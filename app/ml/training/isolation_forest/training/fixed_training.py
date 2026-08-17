"""
Part D Isolation Forest direct/fixed-candidate training.

Intended project location::

    app/ml/training/isolation_forest/training/fixed_training.py

This module owns the Isolation Forest fitting stage for ONE site/KPI series.

Training modes
--------------
Hyperparameter tuning enabled::

    trusted train rows
      -> fit every complete candidate
      -> evaluate each candidate on validation rows
      -> select the candidate with the largest configured score separation
      -> refit the final model on trusted train + trusted validation rows

Hyperparameter tuning disabled::

    trusted train rows
      -> resolve one complete hyperparameter configuration
      -> fit one final model directly

There is no separate calibration stage for Isolation Forest. The fitted
estimator's contamination/offset defines its anomaly boundary. The untouched
test split is not used here.

Tuning metric
-------------
The currently supported metric is ``score_separation``:

    mean(decision_function for model-labelled inliers)
    -
    mean(decision_function for model-labelled outliers)

Larger separation is preferred.

Training rows are restricted to the configured trusted statuses. Validation
rows used to compare candidates retain all complete-feature rows so anomalous
validation behaviour can remain visible. When the final model is refit,
validation rows are again restricted to trusted statuses.

This module does NOT:
- create chronological split boundaries;
- use test rows;
- apply final lower/upper directional anomaly filtering;
- save model/output files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.ml.training.loader import (
    SeriesIdentity,
    SeriesSplitPlan,
    TrainingSplitPlan,
)
from app.ml.training.validation import TrainingValidationResult
from app.schemas.training.isolation_forest.defaults import (
    DEFAULT_ISOLATION_FOREST_BASE_PARAMS,
)
from app.schemas.training.isolation_forest.isolation_forest import (
    IsolationForestHyperparameterSet,
    IsolationForestTrainingConfig,
)


TRAINING_READ_CHUNK_SIZE = 250_000


@dataclass(
    frozen=True,
    slots=True,
)
class IsolationForestTrainingResult:
    """Training output for one site-series Isolation Forest model."""

    identity: SeriesIdentity

    model: IsolationForest

    selected_hyperparameters: dict[str, Any]
    final_model_parameters: dict[str, Any]

    selected_candidate_number: int | None
    tuning_performed: bool
    tuning_summary: tuple[dict[str, Any], ...]

    decision_settings: dict[str, Any]
    training_summary: dict[str, Any]
    usable_row_counts: dict[str, int]


def _validate_training_inputs(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: IsolationForestTrainingConfig,
) -> None:
    """Ensure all supplied objects describe one Isolation Forest request."""

    if validation_result.model_type != "isolation_forest":
        raise ValueError(
            "Isolation Forest training requires "
            "validation_result.model_type='isolation_forest'."
        )

    if split_plan.model_type != "isolation_forest":
        raise ValueError(
            "Isolation Forest training requires "
            "split_plan.model_type='isolation_forest'."
        )

    if training_config.model_type != "isolation_forest":
        raise ValueError(
            "Isolation Forest training requires IsolationForestTrainingConfig."
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
            "split_plan tuning mode does not match "
            "IsolationForestTrainingConfig."
        )

    if validation_result.target_column is not None:
        raise ValueError(
            "Isolation Forest must not receive a y target column."
        )

    if not validation_result.model_feature_columns:
        raise ValueError(
            "Isolation Forest requires at least one validated model feature."
        )

    if training_config.hyperparameter_tuning_enabled:
        if series_plan.validation is None:
            raise ValueError(
                "Isolation Forest tuning requires a validation split."
            )
    elif series_plan.validation is not None:
        raise ValueError(
            "Isolation Forest direct training must not receive a "
            "validation split."
        )

    if series_plan.calibration is not None:
        raise ValueError(
            "Isolation Forest must not receive a calibration split."
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
    Load only train/validation rows for one series.

    Test rows are never retained by this stage.
    """

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
            "No Isolation Forest train/validation rows were found for series "
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
            f"Forest training window for series "
            f"{series_plan.identity.as_tuple()}."
        )

    return rows


def _select_raw_stage_rows(
    rows: pd.DataFrame,
    *,
    timestamp_column: str,
    start_timestamp: Any,
    end_timestamp: Any,
) -> pd.DataFrame:
    """Select one inclusive chronological stage."""

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
    """Ensure timestamp loading reproduces loader.py's planned range."""

    actual_count = len(
        rows
    )

    if actual_count != expected_count:
        raise ValueError(
            f"Isolation Forest raw {stage_name} row count does not match "
            f"loader.py for series {identity.as_tuple()}. "
            f"Expected {expected_count}, found {actual_count}."
        )


def _coerce_feature_and_status_columns(
    rows: pd.DataFrame,
    *,
    validation_result: TrainingValidationResult,
) -> pd.DataFrame:
    """Convert model features and status metadata to numeric values."""

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

    return prepared


def _complete_feature_mask(
    rows: pd.DataFrame,
    *,
    validation_result: TrainingValidationResult,
) -> pd.Series:
    """Build the configured complete-feature eligibility mask."""

    if not validation_result.require_complete_feature_rows:
        return pd.Series(
            True,
            index=rows.index,
            dtype=bool,
        )

    return (
        rows[
            list(
                validation_result.model_feature_columns
            )
        ]
        .notna()
        .all(axis=1)
    )


def _prepare_trusted_training_rows(
    rows: pd.DataFrame,
    *,
    validation_result: TrainingValidationResult,
) -> pd.DataFrame:
    """Keep complete-feature rows belonging to trusted training statuses."""

    prepared = _coerce_feature_and_status_columns(
        rows,
        validation_result=validation_result,
    )

    mask = _complete_feature_mask(
        prepared,
        validation_result=validation_result,
    )

    mask &= (
        prepared[
            validation_result.status_column
        ]
        .isin(
            validation_result.trusted_statuses
        )
        .fillna(False)
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


def _prepare_validation_evaluation_rows(
    rows: pd.DataFrame,
    *,
    validation_result: TrainingValidationResult,
) -> pd.DataFrame:
    """
    Keep complete-feature validation rows without trusted-status filtering.

    Candidate comparison is unsupervised and should not remove possible
    anomalous validation behaviour before scoring.
    """

    prepared = _coerce_feature_and_status_columns(
        rows,
        validation_result=validation_result,
    )

    mask = _complete_feature_mask(
        prepared,
        validation_result=validation_result,
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


def _require_minimum_usable_rows(
    *,
    rows: pd.DataFrame,
    stage_name: str,
    minimum_points: int,
    identity: SeriesIdentity,
) -> None:
    """Enforce minimum usable rows after stage-specific filtering."""

    if len(
        rows
    ) < minimum_points:
        raise ValueError(
            f"Insufficient usable Isolation Forest {stage_name} rows for "
            f"series {identity.as_tuple()}: "
            f"{len(rows)} < minimum {minimum_points}."
        )


def _x(
    rows: pd.DataFrame,
    *,
    validation_result: TrainingValidationResult,
) -> pd.DataFrame:
    """Build X using the exact validated Part C feature order."""

    return rows.loc[
        :,
        list(
            validation_result.model_feature_columns
        ),
    ]


def _new_model(
    parameters: IsolationForestHyperparameterSet,
) -> IsolationForest:
    """Create one Isolation Forest estimator from fixed + configurable params."""

    return IsolationForest(
        **dict(
            DEFAULT_ISOLATION_FOREST_BASE_PARAMS
        ),
        **dict(
            parameters
        ),
    )


def _score_separation(
    *,
    model: IsolationForest,
    x_validation: pd.DataFrame,
) -> tuple[
    float,
    dict[str, Any],
]:
    """
    Calculate the configured unsupervised candidate-comparison metric.

    Larger decision_function values are more normal. Separation is therefore
    defined as mean inlier score minus mean outlier score.
    """

    decision_scores = np.asarray(
        model.decision_function(
            x_validation
        ),
        dtype=np.float64,
    )

    predicted_labels = np.asarray(
        model.predict(
            x_validation
        ),
        dtype=np.int64,
    )

    if decision_scores.shape != predicted_labels.shape:
        raise ValueError(
            "Isolation Forest validation score/prediction shapes differ."
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
            "Isolation Forest produced invalid validation decision scores."
        )

    inlier_mask = (
        predicted_labels
        == 1
    )

    outlier_mask = (
        predicted_labels
        == -1
    )

    inlier_count = int(
        np.sum(
            inlier_mask
        )
    )

    outlier_count = int(
        np.sum(
            outlier_mask
        )
    )

    if inlier_count <= 0:
        raise ValueError(
            "Isolation Forest candidate labelled no validation inliers; "
            "score_separation is undefined."
        )

    if outlier_count <= 0:
        raise ValueError(
            "Isolation Forest candidate labelled no validation outliers; "
            "score_separation is undefined."
        )

    inlier_mean = float(
        np.mean(
            decision_scores[
                inlier_mask
            ]
        )
    )

    outlier_mean = float(
        np.mean(
            decision_scores[
                outlier_mask
            ]
        )
    )

    separation = (
        inlier_mean
        - outlier_mean
    )

    summary = {
        "score_separation": float(
            separation
        ),
        "validation_inlier_count": (
            inlier_count
        ),
        "validation_outlier_count": (
            outlier_count
        ),
        "validation_outlier_rate": float(
            outlier_count
            / len(
                predicted_labels
            )
        ),
        "validation_inlier_mean_decision_score": (
            inlier_mean
        ),
        "validation_outlier_mean_decision_score": (
            outlier_mean
        ),
        "validation_decision_score_mean": float(
            np.mean(
                decision_scores
            )
        ),
        "validation_decision_score_std": float(
            np.std(
                decision_scores
            )
        ),
        "validation_decision_score_min": float(
            np.min(
                decision_scores
            )
        ),
        "validation_decision_score_max": float(
            np.max(
                decision_scores
            )
        ),
    }

    return (
        float(
            separation
        ),
        summary,
    )


def _fit_and_select_candidate(
    *,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    validation_result: TrainingValidationResult,
    training_config: IsolationForestTrainingConfig,
) -> tuple[
    IsolationForestHyperparameterSet,
    int,
    dict[str, Any],
    tuple[dict[str, Any], ...],
]:
    """Fit every candidate and select the largest validation separation."""

    x_train = _x(
        train_rows,
        validation_result=validation_result,
    )

    x_validation = _x(
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

    selection_metric = (
        training_config
        .tuning
        .selection_metric
    )

    if selection_metric != "score_separation":
        raise ValueError(
            "Unsupported Isolation Forest tuning metric: "
            f"{selection_metric}"
        )

    for candidate_number, parameters in enumerate(
        candidates,
        start=1,
    ):
        result: dict[
            str,
            Any,
        ] = {
            "candidate_number": (
                candidate_number
            ),
            "candidate_status": (
                "failed"
            ),
            "hyperparameters": dict(
                parameters
            ),
            "validation_metrics": None,
            "model_offset": None,
            "error_message": None,
        }

        try:
            model = _new_model(
                parameters
            )

            model.fit(
                x_train
            )

            separation, metrics = _score_separation(
                model=model,
                x_validation=x_validation,
            )

            result.update(
                {
                    "candidate_status": (
                        "passed"
                    ),
                    "validation_metrics": (
                        metrics
                    ),
                    "model_offset": float(
                        model.offset_
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
                    "selection_value": (
                        separation
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
            "Every Isolation Forest hyperparameter candidate failed."
        )

    # Larger score separation is better. Candidate number provides a stable
    # deterministic tie-break in favour of the earlier supplied candidate.
    best_candidate = max(
        passed_candidates,
        key=lambda item: (
            item[
                "selection_value"
            ],
            -item[
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
    selected_hyperparameters: IsolationForestHyperparameterSet,
) -> IsolationForest:
    """Fit the final estimator without any access to the test split."""

    x_final = _x(
        final_fit_rows,
        validation_result=validation_result,
    )

    model = _new_model(
        selected_hyperparameters
    )

    model.fit(
        x_final
    )

    return model


def _direction_feature_name(
    training_config: IsolationForestTrainingConfig,
) -> str:
    """Resolve the configured Part C feature used for lower/upper direction."""

    rules = (
        training_config.get_validation_rules()
    )

    feature_name = (
        rules
        .direction_feature_rule
        .get(
            "feature_name"
        )
    )

    if not isinstance(
        feature_name,
        str,
    ) or not feature_name:
        raise ValueError(
            "Isolation Forest direction feature rule is invalid."
        )

    return feature_name


def train_isolation_forest_series(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: IsolationForestTrainingConfig,
) -> IsolationForestTrainingResult:
    """
    Train the final Isolation Forest model for one planned site series.

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

    train_rows = _prepare_trusted_training_rows(
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

    selected_candidate_number: int | None = None
    tuning_summary: tuple[
        dict[str, Any],
        ...
    ] = tuple()
    validation_metrics: dict[
        str,
        Any,
    ] | None = None

    validation_evaluation_rows = pd.DataFrame()
    trusted_validation_rows = pd.DataFrame()

    if tuning_performed:
        if series_plan.validation is None:
            raise RuntimeError(
                "Internal Isolation Forest error: validation split missing."
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

        validation_evaluation_rows = (
            _prepare_validation_evaluation_rows(
                raw_validation_rows,
                validation_result=validation_result,
            )
        )

        _require_minimum_usable_rows(
            rows=validation_evaluation_rows,
            stage_name="validation",
            minimum_points=(
                validation_result.minimum_validation_points
            ),
            identity=series_plan.identity,
        )

        (
            selected_hyperparameters,
            selected_candidate_number,
            validation_metrics,
            tuning_summary,
        ) = _fit_and_select_candidate(
            train_rows=train_rows,
            validation_rows=validation_evaluation_rows,
            validation_result=validation_result,
            training_config=training_config,
        )

        # The validation set may contain anomalous rows for candidate
        # comparison. Only trusted validation rows are allowed into final fit.
        trusted_validation_rows = (
            _prepare_trusted_training_rows(
                raw_validation_rows,
                validation_result=validation_result,
            )
        )

        final_fit_rows = (
            pd.concat(
                [
                    train_rows,
                    trusted_validation_rows,
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

        final_fit_rows = (
            train_rows.copy()
        )

    final_model = _fit_final_model(
        final_fit_rows=final_fit_rows,
        validation_result=validation_result,
        selected_hyperparameters=(
            selected_hyperparameters
        ),
    )

    x_final = _x(
        final_fit_rows,
        validation_result=validation_result,
    )

    final_predictions = np.asarray(
        final_model.predict(
            x_final
        ),
        dtype=np.int64,
    )

    final_decision_scores = np.asarray(
        final_model.decision_function(
            x_final
        ),
        dtype=np.float64,
    )

    final_model_outlier_count = int(
        np.sum(
            final_predictions
            == -1
        )
    )

    direction_feature = _direction_feature_name(
        training_config
    )

    final_model_parameters = {
        **dict(
            DEFAULT_ISOLATION_FOREST_BASE_PARAMS
        ),
        **dict(
            selected_hyperparameters
        ),
    }

    decision_settings: dict[
        str,
        Any,
    ] = {
        "method": (
            "isolation_forest"
        ),
        "model_prediction_rule": (
            "predict == -1"
        ),
        "decision_function_rule": (
            "decision_function < 0 indicates model anomaly"
        ),
        "model_offset": float(
            final_model.offset_
        ),
        "contamination": (
            selected_hyperparameters[
                "contamination"
            ]
        ),
        "threshold_direction": str(
            training_config.threshold_direction
        ),
        "direction_feature": (
            direction_feature
        ),
        "lower_direction_rule": (
            f"{direction_feature} < 0"
        ),
        "upper_direction_rule": (
            f"{direction_feature} > 0"
        ),
    }

    usable_row_counts = {
        "train": len(
            train_rows
        ),
        "validation_evaluation": len(
            validation_evaluation_rows
        ),
        "validation_trusted_refit": len(
            trusted_validation_rows
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
        "target_column": None,
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
        "usable_validation_evaluation_rows": len(
            validation_evaluation_rows
        ),
        "trusted_validation_refit_rows": len(
            trusted_validation_rows
        ),
        "final_fit_rows": len(
            final_fit_rows
        ),
        "selected_candidate_number": (
            selected_candidate_number
        ),
        "selected_hyperparameters": dict(
            selected_hyperparameters
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "final_model_offset": float(
            final_model.offset_
        ),
        "final_fit_model_outlier_count": (
            final_model_outlier_count
        ),
        "final_fit_model_outlier_rate": float(
            final_model_outlier_count
            / len(
                final_predictions
            )
        ),
        "final_fit_decision_score_mean": float(
            np.mean(
                final_decision_scores
            )
        ),
        "final_fit_decision_score_std": float(
            np.std(
                final_decision_scores
            )
        ),
        "final_fit_decision_score_min": float(
            np.min(
                final_decision_scores
            )
        ),
        "final_fit_decision_score_max": float(
            np.max(
                final_decision_scores
            )
        ),
    }

    return IsolationForestTrainingResult(
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
        tuning_performed=(
            tuning_performed
        ),
        tuning_summary=(
            tuning_summary
        ),
        decision_settings=(
            decision_settings
        ),
        training_summary=(
            training_summary
        ),
        usable_row_counts=(
            usable_row_counts
        ),
    )


__all__ = [
    "IsolationForestTrainingResult",
    "train_isolation_forest_series",
]