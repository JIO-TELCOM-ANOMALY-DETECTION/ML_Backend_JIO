"""
Pydantic schemas for Part D XGBoost training.

Intended project location::

    app/schemas/training/xgboost/xgboost.py

This module defines:

- the XGBoost hyperparameter input contract;
- the four supported hyperparameter modes;
- tuning-aware XGBoost chronological split percentages;
- hyperparameter-tuning controls;
- residual-calibration quantile controls;
- model-specific dataset-validation rules; and
- resolution of user-provided or default hyperparameters.

History is not percentage-configured here. The Part D loader must reserve the
minimum history required by the resolved Part C feature-engineering
configuration before applying the percentages in XGBoostSplitConfig to the
remaining rows.

Actual default values belong to::

    app/schemas/training/xgboost/defaults.py

This module does not load datasets, split dataframes, train XGBoost,
perform hyperparameter tuning, calculate residual thresholds, test models,
or save outputs. Those responsibilities belong to the Part D ML modules.
"""

from __future__ import annotations

from math import isclose
from typing import Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.training.common import CommonTrainingConfig

from .defaults import (
    DEFAULT_XGBOOST_CALIBRATION_PERCENTAGE,
    DEFAULT_XGBOOST_EARLY_STOPPING_ROUNDS,
    DEFAULT_XGBOOST_EXCLUDED_PART_C_FEATURE_TYPES,
    DEFAULT_XGBOOST_HYPERPARAMETER_CANDIDATES,
    DEFAULT_XGBOOST_HYPERPARAMETERS,
    DEFAULT_XGBOOST_MAX_ESTIMATORS,
    DEFAULT_XGBOOST_MINIMUM_CALIBRATION_POINTS,
    DEFAULT_XGBOOST_MINIMUM_TEST_POINTS,
    DEFAULT_XGBOOST_MINIMUM_TRAIN_POINTS,
    DEFAULT_XGBOOST_MINIMUM_VALIDATION_POINTS,
    DEFAULT_XGBOOST_PRIMARY_SELECTION_METRIC,
    DEFAULT_XGBOOST_RESIDUAL_QUANTILE,
    DEFAULT_XGBOOST_COLUMN_ROLE_RULES,
    DEFAULT_XGBOOST_REQUIRE_ALL_RESOLVED_FEATURES,
    DEFAULT_XGBOOST_REQUIRE_COMPLETE_FEATURE_ROWS,
    DEFAULT_XGBOOST_REQUIRE_NON_NULL_TARGET,
    DEFAULT_XGBOOST_USE_ONLY_RESOLVED_FEATURE_COLUMNS,
    DEFAULT_XGBOOST_SECONDARY_SELECTION_METRIC,
    DEFAULT_XGBOOST_TEST_PERCENTAGE,
    DEFAULT_XGBOOST_TRAIN_PERCENTAGE,
    DEFAULT_XGBOOST_TRUSTED_TARGET_STATUSES,
    DEFAULT_XGBOOST_VALIDATION_PERCENTAGE,
    XGBOOST_ALLOWED_SELECTION_METRICS,
    XGBOOST_HYPERPARAMETER_RULES,
)


XGBoostHyperparameterValue: TypeAlias = str | int | float | bool
XGBoostHyperparameterSet: TypeAlias = dict[
    str,
    XGBoostHyperparameterValue,
]


def _validate_parameter_value(
    *,
    parameter_name: str,
    value: XGBoostHyperparameterValue,
    field_name: str,
) -> None:
    """Validate one hyperparameter using data-only rules from defaults.py."""

    rule = XGBOOST_HYPERPARAMETER_RULES.get(
        parameter_name
    )

    if rule is None:
        raise ValueError(
            f"{field_name} contains unsupported XGBoost hyperparameter "
            f"'{parameter_name}'."
        )

    expected_type = rule["type"]

    if expected_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{field_name}.{parameter_name} must be an integer."
            )

    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{field_name}.{parameter_name} must be numeric."
            )

    elif expected_type == "str":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field_name}.{parameter_name} must be a non-empty string."
            )

    elif expected_type == "bool":
        if not isinstance(value, bool):
            raise ValueError(
                f"{field_name}.{parameter_name} must be a boolean."
            )

    else:
        raise ValueError(
            f"Unsupported validation rule type '{expected_type}' for "
            f"XGBoost hyperparameter '{parameter_name}'."
        )

    if expected_type in {"int", "number"}:
        numeric_value = float(value)

        minimum = rule.get("ge")
        strict_minimum = rule.get("gt")
        maximum = rule.get("le")
        strict_maximum = rule.get("lt")

        if minimum is not None and numeric_value < float(minimum):
            raise ValueError(
                f"{field_name}.{parameter_name} must be >= {minimum}."
            )

        if strict_minimum is not None and numeric_value <= float(
            strict_minimum
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} must be > "
                f"{strict_minimum}."
            )

        if maximum is not None and numeric_value > float(maximum):
            raise ValueError(
                f"{field_name}.{parameter_name} must be <= {maximum}."
            )

        if strict_maximum is not None and numeric_value >= float(
            strict_maximum
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} must be < "
                f"{strict_maximum}."
            )


def _validate_complete_hyperparameter_set(
    parameters: XGBoostHyperparameterSet,
    *,
    field_name: str,
) -> XGBoostHyperparameterSet:
    """
    Validate one complete user-configurable XGBoost parameter set.

    The keys of DEFAULT_XGBOOST_HYPERPARAMETERS define the complete contract.
    Adding or removing a configurable hyperparameter therefore requires
    changes only in defaults.py.
    """

    if not parameters:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    expected_parameters = set(
        DEFAULT_XGBOOST_HYPERPARAMETERS
    )

    supplied_parameters = set(
        parameters
    )

    missing_parameters = sorted(
        expected_parameters
        - supplied_parameters
    )

    unexpected_parameters = sorted(
        supplied_parameters
        - expected_parameters
    )

    if missing_parameters:
        raise ValueError(
            f"{field_name} must provide every configured XGBoost "
            f"hyperparameter. Missing: {missing_parameters}"
        )

    if unexpected_parameters:
        raise ValueError(
            f"{field_name} contains unsupported XGBoost "
            f"hyperparameters: {unexpected_parameters}"
        )

    for parameter_name, value in parameters.items():
        _validate_parameter_value(
            parameter_name=parameter_name,
            value=value,
            field_name=field_name,
        )

    return dict(parameters)


class ResolvedXGBoostSplitConfig(BaseModel):
    """Store the final post-history split percentages for one request."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    train_percentage: float = Field(
        gt=0.0,
        lt=100.0,
    )

    validation_percentage: float = Field(
        ge=0.0,
        lt=100.0,
    )

    calibration_percentage: float = Field(
        gt=0.0,
        lt=100.0,
    )

    test_percentage: float = Field(
        gt=0.0,
        lt=100.0,
    )

    uses_validation_split: bool


class XGBoostSplitConfig(BaseModel):
    """
    Define optional post-history split overrides.

    When hyperparameter tuning is enabled the resolved order is::

        train -> validation -> calibration -> test

    When hyperparameter tuning is disabled the resolved order is::

        train -> calibration -> test

    In direct-training mode, the validation share is absorbed into training.
    With the current defaults this converts 60/10/15/15 into 70/15/15.

    History is reserved separately from Part C's resolved
    ``minimum_history_points``.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    train_percentage: float | None = Field(
        default=None,
        gt=0.0,
        lt=100.0,
    )

    validation_percentage: float | None = Field(
        default=None,
        ge=0.0,
        lt=100.0,
    )

    calibration_percentage: float | None = Field(
        default=None,
        gt=0.0,
        lt=100.0,
    )

    test_percentage: float | None = Field(
        default=None,
        gt=0.0,
        lt=100.0,
    )

    def resolve_for_tuning(
        self,
        *,
        tuning_enabled: bool,
    ) -> ResolvedXGBoostSplitConfig:
        """Resolve split defaults and validate them against tuning mode."""

        if tuning_enabled:
            supplied = (
                self.train_percentage,
                self.validation_percentage,
                self.calibration_percentage,
                self.test_percentage,
            )

            supplied_count = sum(
                value is not None
                for value in supplied
            )

            if supplied_count not in {0, 4}:
                raise ValueError(
                    "When XGBoost tuning is enabled, either omit all split "
                    "percentages to use defaults or provide train, "
                    "validation, calibration, and test percentages together."
                )

            train_percentage = (
                self.train_percentage
                if self.train_percentage is not None
                else DEFAULT_XGBOOST_TRAIN_PERCENTAGE
            )

            validation_percentage = (
                self.validation_percentage
                if self.validation_percentage is not None
                else DEFAULT_XGBOOST_VALIDATION_PERCENTAGE
            )

            calibration_percentage = (
                self.calibration_percentage
                if self.calibration_percentage is not None
                else DEFAULT_XGBOOST_CALIBRATION_PERCENTAGE
            )

            test_percentage = (
                self.test_percentage
                if self.test_percentage is not None
                else DEFAULT_XGBOOST_TEST_PERCENTAGE
            )

            if validation_percentage <= 0.0:
                raise ValueError(
                    "XGBoost tuning requires a positive "
                    "validation_percentage."
                )

            total = (
                train_percentage
                + validation_percentage
                + calibration_percentage
                + test_percentage
            )

            if not isclose(
                total,
                100.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "XGBoost train, validation, calibration, and test "
                    "percentages must sum to exactly 100 when tuning is "
                    f"enabled. Received {total}."
                )

            return ResolvedXGBoostSplitConfig(
                train_percentage=train_percentage,
                validation_percentage=validation_percentage,
                calibration_percentage=calibration_percentage,
                test_percentage=test_percentage,
                uses_validation_split=True,
            )

        if (
            self.validation_percentage is not None
            and self.validation_percentage != 0.0
        ):
            raise ValueError(
                "validation_percentage is not used when XGBoost "
                "hyperparameter tuning is disabled. Its default share is "
                "absorbed into training."
            )

        train_supplied = self.train_percentage is not None
        calibration_supplied = self.calibration_percentage is not None
        test_supplied = self.test_percentage is not None

        supplied_count = sum(
            (
                train_supplied,
                calibration_supplied,
                test_supplied,
            )
        )

        if supplied_count not in {0, 3}:
            raise ValueError(
                "When XGBoost tuning is disabled, train_percentage, "
                "calibration_percentage, and test_percentage must either "
                "all be omitted or all be provided."
            )

        train_percentage = (
            self.train_percentage
            if self.train_percentage is not None
            else (
                DEFAULT_XGBOOST_TRAIN_PERCENTAGE
                + DEFAULT_XGBOOST_VALIDATION_PERCENTAGE
            )
        )

        calibration_percentage = (
            self.calibration_percentage
            if self.calibration_percentage is not None
            else DEFAULT_XGBOOST_CALIBRATION_PERCENTAGE
        )

        test_percentage = (
            self.test_percentage
            if self.test_percentage is not None
            else DEFAULT_XGBOOST_TEST_PERCENTAGE
        )

        total = (
            train_percentage
            + calibration_percentage
            + test_percentage
        )

        if not isclose(
            total,
            100.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "XGBoost train, calibration, and test percentages must sum "
                "to exactly 100 when tuning is disabled. "
                f"Received {total}."
            )

        return ResolvedXGBoostSplitConfig(
            train_percentage=train_percentage,
            validation_percentage=0.0,
            calibration_percentage=calibration_percentage,
            test_percentage=test_percentage,
            uses_validation_split=False,
        )


class XGBoostTuningConfig(BaseModel):
    """Define XGBoost candidate-selection and early-stopping controls."""

    model_config = ConfigDict(
        extra="forbid",
    )

    primary_selection_metric: str = Field(
        default=DEFAULT_XGBOOST_PRIMARY_SELECTION_METRIC,
        min_length=1,
        description=(
            "Primary validation metric used to select the best complete "
            "hyperparameter candidate."
        ),
    )

    secondary_selection_metric: str = Field(
        default=DEFAULT_XGBOOST_SECONDARY_SELECTION_METRIC,
        min_length=1,
        description=(
            "Tie-break validation metric used when primary candidate "
            "performance is equal."
        ),
    )

    max_estimators: int = Field(
        default=DEFAULT_XGBOOST_MAX_ESTIMATORS,
        gt=0,
        description=(
            "Maximum number of boosting trees available while evaluating "
            "a candidate with early stopping."
        ),
    )

    early_stopping_rounds: int = Field(
        default=DEFAULT_XGBOOST_EARLY_STOPPING_ROUNDS,
        gt=0,
        description=(
            "Validation rounds without improvement before candidate "
            "training stops."
        ),
    )

    @field_validator(
        "primary_selection_metric",
        "secondary_selection_metric",
    )
    @classmethod
    def validate_selection_metric(
        cls,
        value: str,
    ) -> str:
        """Validate candidate-selection metrics against defaults.py."""

        normalized = value.strip().lower()

        if normalized not in XGBOOST_ALLOWED_SELECTION_METRICS:
            raise ValueError(
                "Unsupported XGBoost hyperparameter selection metric. "
                f"Supported metrics: "
                f"{sorted(XGBOOST_ALLOWED_SELECTION_METRICS)}"
            )

        return normalized

    @model_validator(mode="after")
    def validate_distinct_selection_metrics(
        self,
    ) -> XGBoostTuningConfig:
        """Require primary and tie-break metrics to be different."""

        if (
            self.primary_selection_metric
            == self.secondary_selection_metric
        ):
            raise ValueError(
                "primary_selection_metric and "
                "secondary_selection_metric must be different."
            )

        return self


class XGBoostCalibrationConfig(BaseModel):
    """
    Configure XGBoost residual-quantile calibration.

    ``residual_quantile`` represents the upper residual-tail quantile.

    For example, with ``residual_quantile=0.99``:

    - upper_only uses the 0.99 residual quantile;
    - lower_only uses the complementary 0.01 residual quantile; and
    - both uses both 0.01 and 0.99 residual quantiles.

    The actual direction is controlled by the shared
    ``CommonTrainingConfig.threshold_direction`` setting. This schema only
    configures the quantile; residual calculation and threshold derivation
    belong to ``app/ml/training/xgboost/residual_calibration.py``.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    residual_quantile: float = Field(
        default=DEFAULT_XGBOOST_RESIDUAL_QUANTILE,
        gt=0.5,
        lt=1.0,
        description=(
            "Upper residual-tail quantile used for calibration. "
            "The lower-tail quantile is automatically 1 - residual_quantile."
        ),
    )



class XGBoostValidationRules(BaseModel):
    """
    Store fixed XGBoost dataset-validation and column-role rules.

    XGBoost is supervised/self-supervised regression in this pipeline:

    - X consists only of the resolved Part C feature columns;
    - y is the physical dataset value column resolved through
      DatasetColumnMapping.value_column; and
    - structural/status/usability columns are compulsory metadata but are
      never model features.

    These values are internal model rules rather than user-editable request
    fields. The shared Part D validation module retrieves them through
    ``get_validation_rules``.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    use_only_resolved_feature_columns: bool = Field(
        default=DEFAULT_XGBOOST_USE_ONLY_RESOLVED_FEATURE_COLUMNS,
        description=(
            "Build X exclusively from Part C's resolved feature_columns. "
            "Retained source/metadata columns are never inferred as model "
            "features."
        ),
    )

    require_all_resolved_features: bool = Field(
        default=DEFAULT_XGBOOST_REQUIRE_ALL_RESOLVED_FEATURES,
        description=(
            "Require every feature column declared by the resolved Part C "
            "configuration to exist in the engineered dataset."
        ),
    )

    excluded_part_c_feature_types: tuple[str, ...] = Field(
        default=DEFAULT_XGBOOST_EXCLUDED_PART_C_FEATURE_TYPES,
        description=(
            "Part C feature request families excluded from X because they "
            "directly contain the current KPI value used as y."
        ),
    )

    column_role_rules: dict[str, dict[str, object]] = Field(
        default_factory=lambda: {
            role: dict(rule)
            for role, rule in DEFAULT_XGBOOST_COLUMN_ROLE_RULES.items()
        },
        description=(
            "Logical rules defining X features, the required y target, and "
            "non-feature structural/status/usability columns."
        ),
    )

    trusted_target_statuses: frozenset[int] = Field(
        default=DEFAULT_XGBOOST_TRUSTED_TARGET_STATUSES,
    )

    require_complete_feature_rows: bool = Field(
        default=DEFAULT_XGBOOST_REQUIRE_COMPLETE_FEATURE_ROWS,
    )

    require_non_null_target: bool = Field(
        default=DEFAULT_XGBOOST_REQUIRE_NON_NULL_TARGET,
    )

    minimum_train_points: int = Field(
        default=DEFAULT_XGBOOST_MINIMUM_TRAIN_POINTS,
        gt=0,
    )

    minimum_validation_points: int = Field(
        default=DEFAULT_XGBOOST_MINIMUM_VALIDATION_POINTS,
        gt=0,
    )

    minimum_calibration_points: int = Field(
        default=DEFAULT_XGBOOST_MINIMUM_CALIBRATION_POINTS,
        gt=0,
    )

    minimum_test_points: int = Field(
        default=DEFAULT_XGBOOST_MINIMUM_TEST_POINTS,
        gt=0,
    )


class XGBoostTrainingConfig(CommonTrainingConfig):
    """
    Define the complete Part D XGBoost request.

    Four hyperparameter modes are supported:

    1. tuning enabled + candidate list supplied;
    2. tuning enabled + no candidate list -> internal default candidates;
    3. tuning disabled + single configuration supplied;
    4. tuning disabled + no configuration -> internal single default.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    model_type: Literal["xgboost"] = Field(
        default="xgboost",
    )

    split: XGBoostSplitConfig = Field(
        default_factory=XGBoostSplitConfig,
    )

    hyperparameters: XGBoostHyperparameterSet | None = Field(
        default=None,
        description=(
            "One complete configurable XGBoost hyperparameter set used when "
            "tuning is disabled. Omit to use the internal single default."
        ),
    )

    hyperparameter_candidates: (
        list[XGBoostHyperparameterSet]
        | None
    ) = Field(
        default=None,
        min_length=1,
        description=(
            "Complete configurable XGBoost hyperparameter candidates used "
            "when tuning is enabled. No maximum candidate count is imposed. "
            "Omit to use the internal default candidate list."
        ),
    )

    tuning: XGBoostTuningConfig = Field(
        default_factory=XGBoostTuningConfig,
    )

    calibration: XGBoostCalibrationConfig = Field(
        default_factory=XGBoostCalibrationConfig,
        description=(
            "Residual-quantile calibration settings used after final "
            "XGBoost training and before untouched test evaluation."
        ),
    )

    @field_validator(
        "hyperparameters",
    )
    @classmethod
    def validate_hyperparameters(
        cls,
        value: XGBoostHyperparameterSet | None,
    ) -> XGBoostHyperparameterSet | None:
        """Validate an explicitly supplied single configuration."""

        if value is None:
            return None

        return _validate_complete_hyperparameter_set(
            value,
            field_name="hyperparameters",
        )

    @field_validator(
        "hyperparameter_candidates",
    )
    @classmethod
    def validate_hyperparameter_candidates(
        cls,
        value: list[XGBoostHyperparameterSet] | None,
    ) -> list[XGBoostHyperparameterSet] | None:
        """Validate every explicitly supplied complete tuning candidate."""

        if value is None:
            return None

        return [
            _validate_complete_hyperparameter_set(
                candidate,
                field_name=(
                    f"hyperparameter_candidates[{index}]"
                ),
            )
            for index, candidate in enumerate(
                value
            )
        ]

    @model_validator(mode="after")
    def validate_hyperparameter_mode(
        self,
    ) -> XGBoostTrainingConfig:
        """Validate list-vs-single input against the tuning switch."""

        if self.hyperparameter_tuning_enabled:
            if self.hyperparameters is not None:
                raise ValueError(
                    "hyperparameters cannot be supplied when "
                    "hyperparameter_tuning_enabled=True. Supply "
                    "hyperparameter_candidates instead, or omit candidates "
                    "to use the internal default candidate list."
                )

            return self

        if self.hyperparameter_candidates is not None:
            raise ValueError(
                "hyperparameter_candidates cannot be supplied when "
                "hyperparameter_tuning_enabled=False. Supply "
                "hyperparameters instead, or omit them to use the internal "
                "single default configuration."
            )

        return self

    def resolve_hyperparameters(
        self,
    ) -> XGBoostHyperparameterSet:
        """Return the single configuration used when tuning is disabled."""

        if self.hyperparameter_tuning_enabled:
            raise ValueError(
                "resolve_hyperparameters cannot be used while "
                "hyperparameter tuning is enabled."
            )

        parameters = (
            self.hyperparameters
            if self.hyperparameters is not None
            else DEFAULT_XGBOOST_HYPERPARAMETERS
        )

        return _validate_complete_hyperparameter_set(
            dict(parameters),
            field_name="resolved hyperparameters",
        )

    def resolve_hyperparameter_candidates(
        self,
    ) -> tuple[XGBoostHyperparameterSet, ...]:
        """Return supplied candidates or the internal default candidate list."""

        if not self.hyperparameter_tuning_enabled:
            raise ValueError(
                "resolve_hyperparameter_candidates cannot be used while "
                "hyperparameter tuning is disabled."
            )

        candidates = (
            self.hyperparameter_candidates
            if self.hyperparameter_candidates is not None
            else DEFAULT_XGBOOST_HYPERPARAMETER_CANDIDATES
        )

        if not candidates:
            raise ValueError(
                "The resolved XGBoost hyperparameter candidate list "
                "cannot be empty."
            )

        return tuple(
            _validate_complete_hyperparameter_set(
                dict(candidate),
                field_name=(
                    f"resolved hyperparameter candidate[{index}]"
                ),
            )
            for index, candidate in enumerate(
                candidates
            )
        )

    def resolve_split(
        self,
    ) -> ResolvedXGBoostSplitConfig:
        """Return split percentages appropriate for the tuning mode."""

        return self.split.resolve_for_tuning(
            tuning_enabled=self.hyperparameter_tuning_enabled,
        )

    def get_validation_rules(
        self,
    ) -> XGBoostValidationRules:
        """Return fixed model-specific dataset-validation rules."""

        return XGBoostValidationRules()


__all__ = [
    "XGBoostHyperparameterSet",
    "XGBoostHyperparameterValue",
    "XGBoostCalibrationConfig",
    "XGBoostSplitConfig",
    "XGBoostTrainingConfig",
    "XGBoostTuningConfig",
    "XGBoostValidationRules",
    "ResolvedXGBoostSplitConfig",
]