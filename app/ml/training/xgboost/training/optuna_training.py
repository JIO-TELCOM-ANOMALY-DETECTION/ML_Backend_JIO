"""
Part D XGBoost Optuna training.

Intended project location::

    app/ml/training/xgboost/optuna_training.py

This module owns the Optuna XGBoost fitting stage for ONE
Composite_ID/site series.

Training mode
-------------
This module is used only when::

    hyperparameter_tuning_enabled = True
    hyperparameter_tuning_method = "optuna"

Flow::

    train
      -> create an Optuna study
      -> generate hyperparameters from the resolved Optuna search space
      -> fit each trial with validation-based early stopping
      -> evaluate each completed trial on validation
      -> minimize the configured primary validation metric
      -> use the configured secondary metric as a deterministic tie-break
      -> obtain the selected trial's best tree count
      -> refit the final model on trusted train + validation rows

The calibration and test ranges are deliberately untouched here.

Responsibilities
----------------
- reuse the exact XGBoost train/validation loading and row-eligibility rules;
- resolve user-provided or default Optuna search ranges;
- run a reproducible Optuna TPE study;
- evaluate trials with the existing XGBoost validation metrics;
- select the best completed trial;
- refit the final XGBoost model on trusted train + validation rows; and
- return the same XGBoostTrainingResult contract used by fixed training.

This module does NOT:
- create chronological split boundaries;
- use calibration rows;
- use test rows;
- calculate residual anomaly cutoffs;
- make final anomaly decisions;
- save model files or JSON artifacts.
"""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd
from optuna.trial import TrialState

from app.ml.training.loader import (
    SeriesSplitPlan,
    TrainingSplitPlan,
)
from app.ml.training.validation import TrainingValidationResult
from app.schemas.training.xgboost.defaults import (
    DEFAULT_XGBOOST_BASE_PARAMS,
    XGBOOST_HYPERPARAMETER_RULES,
)
from app.schemas.training.xgboost.xgboost import (
    XGBoostHyperparameterSet,
    XGBoostOptunaSearchSpace,
    XGBoostTrainingConfig,
)

from .fixed_training import (
    XGBoostTrainingResult,
    _candidate_model,
    _fit_final_model,
    _load_training_window,
    _prepare_usable_rows,
    _regression_metrics,
    _require_minimum_usable_rows,
    _select_raw_stage_rows,
    _validate_raw_stage_count,
    _validate_training_inputs,
    _xy,
)


def _validate_optuna_training_inputs(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: XGBoostTrainingConfig,
) -> None:
    """Validate the shared XGBoost contract plus the Optuna routing mode."""

    _validate_training_inputs(
        validation_result=validation_result,
        split_plan=split_plan,
        series_plan=series_plan,
        training_config=training_config,
    )

    if not training_config.hyperparameter_tuning_enabled:
        raise ValueError(
            "XGBoost Optuna training requires "
            "hyperparameter_tuning_enabled=True."
        )

    if training_config.hyperparameter_tuning_method != "optuna":
        raise ValueError(
            "XGBoost Optuna training requires "
            "hyperparameter_tuning_method='optuna'."
        )

    if series_plan.validation is None:
        raise ValueError(
            "XGBoost Optuna training requires a validation split range."
        )


def _suggest_hyperparameters(
    *,
    trial: optuna.Trial,
    search_space: XGBoostOptunaSearchSpace,
) -> XGBoostHyperparameterSet:
    """
    Generate one complete XGBoost hyperparameter set from an Optuna trial.

    Parameter type is derived from XGBOOST_HYPERPARAMETER_RULES, matching the
    schema's validation contract.
    """

    parameters: XGBoostHyperparameterSet = {}

    for parameter_name, parameter_range in search_space.items():
        rule = XGBOOST_HYPERPARAMETER_RULES.get(
            parameter_name
        )

        if rule is None:
            raise ValueError(
                "Resolved Optuna search space contains unsupported XGBoost "
                f"hyperparameter '{parameter_name}'."
            )

        parameter_type = rule.get(
            "type"
        )

        if parameter_type == "int":
            parameters[
                parameter_name
            ] = trial.suggest_int(
                parameter_name,
                int(
                    parameter_range.low
                ),
                int(
                    parameter_range.high
                ),
                log=bool(
                    parameter_range.log
                ),
            )

            continue

        if parameter_type == "number":
            parameters[
                parameter_name
            ] = trial.suggest_float(
                parameter_name,
                float(
                    parameter_range.low
                ),
                float(
                    parameter_range.high
                ),
                log=bool(
                    parameter_range.log
                ),
            )

            continue

        raise ValueError(
            "XGBoost Optuna training supports only numeric configurable "
            f"hyperparameters. '{parameter_name}' uses rule type "
            f"'{parameter_type}'."
        )

    return parameters


def _fit_optuna_study(
    *,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    validation_result: TrainingValidationResult,
    training_config: XGBoostTrainingConfig,
) -> tuple[
    XGBoostHyperparameterSet,
    int,
    dict[str, float],
    int,
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    """
    Run the Optuna study and return the selected trial information.

    The study minimizes the configured primary validation metric.
    Completed trials are then compared using:
        primary metric -> secondary metric -> Optuna trial number

    This preserves the deterministic tie-break behaviour of fixed tuning.
    """

    x_train, y_train = _xy(
        train_rows,
        validation_result=validation_result,
    )

    x_validation, y_validation = _xy(
        validation_rows,
        validation_result=validation_result,
    )

    search_space = (
        training_config.resolve_optuna_search_space()
    )

    optuna_config = (
        training_config.resolve_optuna_config()
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

    sampler = optuna.samplers.TPESampler(
        seed=optuna_config.sampler_seed,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        """Fit and evaluate one Optuna-generated XGBoost configuration."""

        try:
            parameters = _suggest_hyperparameters(
                trial=trial,
                search_space=search_space,
            )

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

            # Trial user attributes are intentionally JSON-serializable so
            # they can later be copied into tuning_summary/model artifacts.
            trial.set_user_attr(
                "validation_metrics",
                {
                    key: float(
                        value
                    )
                    for key, value in metrics.items()
                },
            )

            trial.set_user_attr(
                "best_iteration",
                best_iteration,
            )

            trial.set_user_attr(
                "best_tree_count",
                best_tree_count,
            )

            trial.set_user_attr(
                "error_message",
                None,
            )

            return float(
                metrics[
                    primary_metric
                ]
            )

        except Exception as error:
            trial.set_user_attr(
                "error_message",
                f"{type(error).__name__}: {error}",
            )
            raise

    # n_jobs=1 is deliberate. Together with the fixed TPESampler seed, this
    # keeps equivalent searches reproducible.
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
            and trial.user_attrs.get(
                "best_tree_count"
            ) is not None
        )
    ]

    if not completed_trials:
        raise RuntimeError(
            "Every XGBoost Optuna trial failed."
        )

    selected_trial = min(
        completed_trials,
        key=lambda trial: (
            float(
                trial.user_attrs[
                    "validation_metrics"
                ][
                    primary_metric
                ]
            ),
            float(
                trial.user_attrs[
                    "validation_metrics"
                ][
                    secondary_metric
                ]
            ),
            int(
                trial.number
            ),
        ),
    )

    selected_hyperparameters: XGBoostHyperparameterSet = dict(
        selected_trial.params
    )

    selected_validation_metrics = {
        key: float(
            value
        )
        for (
            key,
            value,
        ) in selected_trial.user_attrs[
            "validation_metrics"
        ].items()
    }

    best_tree_count = int(
        selected_trial.user_attrs[
            "best_tree_count"
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
                "best_iteration": (
                    int(
                        trial.user_attrs[
                            "best_iteration"
                        ]
                    )
                    if trial.user_attrs.get(
                        "best_iteration"
                    ) is not None
                    else None
                ),
                "best_tree_count": (
                    int(
                        trial.user_attrs[
                            "best_tree_count"
                        ]
                    )
                    if trial.user_attrs.get(
                        "best_tree_count"
                    ) is not None
                    else None
                ),
                "validation_metrics": (
                    {
                        key: float(
                            value
                        )
                        for (
                            key,
                            value,
                        ) in validation_metrics.items()
                    }
                    if validation_metrics is not None
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
        "objective_direction": "minimize",
        "primary_selection_metric": (
            primary_metric
        ),
        "secondary_selection_metric": (
            secondary_metric
        ),
    }

    return (
        selected_hyperparameters,
        best_tree_count,
        selected_validation_metrics,
        int(
            selected_trial.number
        ),
        tuple(
            tuning_summary
        ),
        optuna_summary,
    )


def train_xgboost_series(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: XGBoostTrainingConfig,
) -> XGBoostTrainingResult:
    """
    Train one XGBoost model using Optuna hyperparameter optimization.

    runner.py should call this only for tuning-enabled XGBoost requests whose
    hyperparameter_tuning_method is ``"optuna"``.
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

    if series_plan.validation is None:
        raise RuntimeError(
            "Internal XGBoost Optuna error: validation split disappeared."
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
        best_tree_count,
        validation_metrics,
        selected_trial_number,
        tuning_summary,
        optuna_summary,
    ) = _fit_optuna_study(
        train_rows=train_rows,
        validation_rows=validation_rows,
        validation_result=validation_result,
        training_config=training_config,
    )

    # After optimization, validation may join training for the final refit.
    # Calibration and test remain untouched.
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

    final_fit_prediction = final_model.predict(
        x_final
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
        "tuning_performed": True,
        "tuning_method": "optuna",
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
        "raw_validation_rows": len(
            raw_validation_rows
        ),
        "usable_validation_rows": len(
            validation_rows
        ),
        "final_fit_rows": len(
            final_fit_rows
        ),
        # Keep the existing result contract intact. Candidate numbering is
        # specific to fixed tuning, so it remains None for Optuna.
        "selected_candidate_number": None,
        "selected_trial_number": (
            selected_trial_number
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
        "optuna": (
            optuna_summary
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
        # Preserve runner/model_output compatibility.
        selected_candidate_number=None,
        best_tree_count=(
            best_tree_count
        ),
        tuning_performed=True,
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
    "train_xgboost_series",
]