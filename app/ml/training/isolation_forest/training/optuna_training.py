"""
Part D Isolation Forest Optuna training.

Intended project location::

    app/ml/training/isolation_forest/training/optuna_training.py

This module owns the Optuna Isolation Forest fitting stage for ONE
site/KPI series.

Training mode
-------------
This module is used only when::

    hyperparameter_tuning_enabled = True
    hyperparameter_tuning_method = "optuna"

Flow::

    trusted train rows
      -> create an Optuna study
      -> generate hyperparameters from the resolved Optuna search space
      -> fit each trial on trusted train rows
      -> evaluate each trial on complete-feature validation rows
      -> maximize configured score separation
      -> select the best completed trial
      -> refit the final model on trusted train + trusted validation rows

There is no separate calibration stage for Isolation Forest. The fitted
estimator's contamination/offset defines its anomaly boundary. The untouched
test split is not used here.

Tuning metric
-------------
The currently supported metric is ``score_separation``::

    mean(decision_function for model-labelled inliers)
    -
    mean(decision_function for model-labelled outliers)

Larger separation is preferred.

This module reuses the same loading, row-filtering, scoring, final-fitting,
and directional-setting helpers as fixed training so both paths follow the
same model contract.

This module does NOT:
- create chronological split boundaries;
- use test rows;
- apply final lower/upper directional anomaly filtering;
- save model/output files.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from app.ml.training.loader import (
    SeriesSplitPlan,
    TrainingSplitPlan,
)
from app.ml.training.validation import TrainingValidationResult
from app.schemas.training.isolation_forest.defaults import (
    DEFAULT_ISOLATION_FOREST_BASE_PARAMS,
)
from app.schemas.training.isolation_forest.isolation_forest import (
    IsolationForestHyperparameterSet,
    IsolationForestOptunaSearchSpace,
    IsolationForestTrainingConfig,
)

from .fixed_training import (
    IsolationForestTrainingResult,
    _direction_feature_name,
    _fit_final_model,
    _load_training_window,
    _new_model,
    _prepare_trusted_training_rows,
    _prepare_validation_evaluation_rows,
    _require_minimum_usable_rows,
    _score_separation,
    _select_raw_stage_rows,
    _validate_raw_stage_count,
    _validate_training_inputs,
    _x,
)


def _validate_optuna_training_inputs(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: IsolationForestTrainingConfig,
) -> None:
    """Validate the shared Isolation Forest contract plus Optuna routing."""

    _validate_training_inputs(
        validation_result=validation_result,
        split_plan=split_plan,
        series_plan=series_plan,
        training_config=training_config,
    )

    if not training_config.hyperparameter_tuning_enabled:
        raise ValueError(
            "Isolation Forest Optuna training requires "
            "hyperparameter_tuning_enabled=True."
        )

    if training_config.hyperparameter_tuning_method != "optuna":
        raise ValueError(
            "Isolation Forest Optuna training requires "
            "hyperparameter_tuning_method='optuna'."
        )

    if series_plan.validation is None:
        raise ValueError(
            "Isolation Forest Optuna training requires a validation split."
        )


def _suggest_hyperparameters(
    *,
    trial: optuna.Trial,
    search_space: IsolationForestOptunaSearchSpace,
) -> IsolationForestHyperparameterSet:
    """Generate one complete Isolation Forest parameter set for one trial."""

    parameters: IsolationForestHyperparameterSet = {}

    for parameter_name, parameter_config in search_space.items():
        parameter_type = parameter_config.type

        if parameter_type == "int":
            if (
                parameter_config.low is None
                or parameter_config.high is None
            ):
                raise ValueError(
                    f"Optuna integer parameter '{parameter_name}' is "
                    "missing low/high boundaries."
                )

            parameters[
                parameter_name
            ] = trial.suggest_int(
                parameter_name,
                int(
                    parameter_config.low
                ),
                int(
                    parameter_config.high
                ),
                log=bool(
                    parameter_config.log
                ),
            )

            continue

        if parameter_type == "float":
            if (
                parameter_config.low is None
                or parameter_config.high is None
            ):
                raise ValueError(
                    f"Optuna float parameter '{parameter_name}' is "
                    "missing low/high boundaries."
                )

            parameters[
                parameter_name
            ] = trial.suggest_float(
                parameter_name,
                float(
                    parameter_config.low
                ),
                float(
                    parameter_config.high
                ),
                log=bool(
                    parameter_config.log
                ),
            )

            continue

        if parameter_type == "categorical":
            if parameter_config.choices is None:
                raise ValueError(
                    f"Optuna categorical parameter '{parameter_name}' "
                    "is missing choices."
                )

            parameters[
                parameter_name
            ] = trial.suggest_categorical(
                parameter_name,
                list(
                    parameter_config.choices
                ),
            )

            continue

        raise ValueError(
            "Unsupported Isolation Forest Optuna parameter type "
            f"'{parameter_type}' for '{parameter_name}'."
        )

    return parameters


def _fit_optuna_study(
    *,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    validation_result: TrainingValidationResult,
    training_config: IsolationForestTrainingConfig,
) -> tuple[
    IsolationForestHyperparameterSet,
    dict[str, Any],
    int,
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    """
    Run Optuna and return the selected trial information.

    Isolation Forest's configured ``score_separation`` metric is maximized.
    Trial number is used as a deterministic tie-break in favour of the
    earlier completed trial.
    """

    x_train = _x(
        train_rows,
        validation_result=validation_result,
    )

    x_validation = _x(
        validation_rows,
        validation_result=validation_result,
    )

    search_space = (
        training_config.resolve_optuna_search_space()
    )

    optuna_config = (
        training_config.resolve_optuna_config()
    )

    selection_metric = (
        training_config
        .tuning
        .selection_metric
    )

    if selection_metric != "score_separation":
        raise ValueError(
            "Unsupported Isolation Forest Optuna tuning metric: "
            f"{selection_metric}"
        )

    sampler = optuna.samplers.TPESampler(
        seed=optuna_config.sampler_seed,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        """Fit and evaluate one Optuna-generated Isolation Forest trial."""

        try:
            parameters = _suggest_hyperparameters(
                trial=trial,
                search_space=search_space,
            )

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

            trial.set_user_attr(
                "validation_metrics",
                dict(
                    metrics
                ),
            )

            trial.set_user_attr(
                "model_offset",
                float(
                    model.offset_
                ),
            )

            trial.set_user_attr(
                "error_message",
                None,
            )

            return float(
                separation
            )

        except Exception as error:
            trial.set_user_attr(
                "error_message",
                f"{type(error).__name__}: {error}",
            )
            raise

    # Sequential trials plus a fixed sampler seed keep equivalent searches
    # reproducible.
    study.optimize(
        objective,
        n_trials=optuna_config.n_trials,
        n_jobs=1,
        catch=(
            Exception,
        ),
        show_progress_bar=False,
    )

    completed_trials = [
        trial
        for trial in study.trials
        if (
            trial.state == TrialState.COMPLETE
            and trial.value is not None
            and trial.user_attrs.get(
                "validation_metrics"
            ) is not None
        )
    ]

    if not completed_trials:
        raise RuntimeError(
            "Every Isolation Forest Optuna trial failed."
        )

    selected_trial = max(
        completed_trials,
        key=lambda trial: (
            float(
                trial.value
            ),
            -int(
                trial.number
            ),
        ),
    )

    selected_hyperparameters: IsolationForestHyperparameterSet = dict(
        selected_trial.params
    )

    selected_validation_metrics = dict(
        selected_trial.user_attrs[
            "validation_metrics"
        ]
    )

    tuning_summary: list[
        dict[str, Any]
    ] = []

    for trial in study.trials:
        validation_metrics = trial.user_attrs.get(
            "validation_metrics"
        )

        tuning_summary.append(
            {
                "trial_number": int(
                    trial.number
                ),
                "trial_status": (
                    trial.state.name.lower()
                ),
                "hyperparameters": dict(
                    trial.params
                ),
                "objective_value": (
                    float(
                        trial.value
                    )
                    if trial.value is not None
                    else None
                ),
                "validation_metrics": (
                    dict(
                        validation_metrics
                    )
                    if validation_metrics is not None
                    else None
                ),
                "model_offset": (
                    float(
                        trial.user_attrs[
                            "model_offset"
                        ]
                    )
                    if trial.user_attrs.get(
                        "model_offset"
                    ) is not None
                    else None
                ),
                "error_message": (
                    trial.user_attrs.get(
                        "error_message"
                    )
                ),
            }
        )

    optuna_summary: dict[
        str,
        Any,
    ] = {
        "sampler": (
            type(
                study.sampler
            ).__name__
        ),
        "sampler_seed": (
            optuna_config.sampler_seed
        ),
        "requested_trials": int(
            optuna_config.n_trials
        ),
        "completed_trials": int(
            sum(
                trial.state == TrialState.COMPLETE
                for trial in study.trials
            )
        ),
        "failed_trials": int(
            sum(
                trial.state == TrialState.FAIL
                for trial in study.trials
            )
        ),
        "pruned_trials": int(
            sum(
                trial.state == TrialState.PRUNED
                for trial in study.trials
            )
        ),
        "selected_trial_number": int(
            selected_trial.number
        ),
        "objective_direction": "maximize",
        "selection_metric": (
            selection_metric
        ),
    }

    return (
        selected_hyperparameters,
        selected_validation_metrics,
        int(
            selected_trial.number
        ),
        tuple(
            tuning_summary
        ),
        optuna_summary,
    )


def train_isolation_forest_series(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: IsolationForestTrainingConfig,
) -> IsolationForestTrainingResult:
    """
    Train one Isolation Forest model using Optuna optimization.

    runner.py should call this only for tuning-enabled Isolation Forest
    requests whose hyperparameter_tuning_method is ``"optuna"``.
    """

    _validate_optuna_training_inputs(
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

    if series_plan.validation is None:
        raise RuntimeError(
            "Internal Isolation Forest Optuna error: "
            "validation split disappeared."
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
        validation_metrics,
        selected_trial_number,
        tuning_summary,
        optuna_summary,
    ) = _fit_optuna_study(
        train_rows=train_rows,
        validation_rows=validation_evaluation_rows,
        validation_result=validation_result,
        training_config=training_config,
    )

    # Candidate evaluation retains complete validation rows so anomalies remain
    # visible. Final refit, however, only receives trusted validation rows.
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
        "method": "isolation_forest",
        "model_prediction_rule": "predict == -1",
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
        "tuning_performed": True,
        "tuning_method": "optuna",
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
        "raw_validation_rows": len(
            raw_validation_rows
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
        # Preserve the existing result/output contract. Candidate numbering is
        # specific to fixed tuning.
        "selected_candidate_number": None,
        "selected_trial_number": (
            selected_trial_number
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
        "optuna": (
            optuna_summary
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
        selected_candidate_number=None,
        tuning_performed=True,
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
    "train_isolation_forest_series",
]