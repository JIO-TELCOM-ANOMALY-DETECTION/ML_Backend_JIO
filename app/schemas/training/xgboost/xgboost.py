"""
Pydantic schemas for Part D XGBoost training.

Intended project location::

    app/schemas/training/xgboost/xgboost.py

This module defines:

- the XGBoost fixed hyperparameter input contract;
- the XGBoost Optuna search-space input contract;
- direct, fixed-tuning, and Optuna-tuning hyperparameter modes;
- tuning-aware XGBoost chronological split percentages;
- shared XGBoost tuning controls;
- Optuna-specific tuning controls;
- residual-calibration quantile controls;
- model-specific dataset-validation rules; and
- resolution of user-provided or default hyperparameter settings.

History is not percentage-configured here. The Part D loader must reserve the
minimum history required by the resolved Part C feature-engineering
configuration before applying the percentages in XGBoostSplitConfig to the
remaining rows.

Actual default values belong to::

    app/schemas/training/xgboost/defaults.py

This module does not load datasets, split dataframes, train XGBoost,
perform fixed hyperparameter tuning, perform Optuna optimization,
calculate residual thresholds, test models, or save outputs. Those
responsibilities belong to the Part D ML modules.
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
    DEFAULT_XGBOOST_COLUMN_ROLE_RULES,
    DEFAULT_XGBOOST_EARLY_STOPPING_ROUNDS,
    DEFAULT_XGBOOST_EXCLUDED_PART_C_FEATURE_TYPES,
    DEFAULT_XGBOOST_HYPERPARAMETER_CANDIDATES,
    DEFAULT_XGBOOST_HYPERPARAMETERS,
    DEFAULT_XGBOOST_MAX_ESTIMATORS,
    DEFAULT_XGBOOST_MINIMUM_CALIBRATION_POINTS,
    DEFAULT_XGBOOST_MINIMUM_TEST_POINTS,
    DEFAULT_XGBOOST_MINIMUM_TRAIN_POINTS,
    DEFAULT_XGBOOST_MINIMUM_VALIDATION_POINTS,
    DEFAULT_XGBOOST_OPTUNA_N_TRIALS,
    DEFAULT_XGBOOST_OPTUNA_SAMPLER_SEED,
    DEFAULT_XGBOOST_OPTUNA_SEARCH_SPACE,
    DEFAULT_XGBOOST_PRIMARY_SELECTION_METRIC,
    DEFAULT_XGBOOST_REQUIRE_ALL_RESOLVED_FEATURES,
    DEFAULT_XGBOOST_REQUIRE_COMPLETE_FEATURE_ROWS,
    DEFAULT_XGBOOST_REQUIRE_NON_NULL_TARGET,
    DEFAULT_XGBOOST_RESIDUAL_QUANTILE,
    DEFAULT_XGBOOST_SECONDARY_SELECTION_METRIC,
    DEFAULT_XGBOOST_TEST_PERCENTAGE,
    DEFAULT_XGBOOST_TRAIN_PERCENTAGE,
    DEFAULT_XGBOOST_TRUSTED_TARGET_STATUSES,
    DEFAULT_XGBOOST_USE_ONLY_RESOLVED_FEATURE_COLUMNS,
    DEFAULT_XGBOOST_VALIDATION_PERCENTAGE,
    XGBOOST_ALLOWED_SELECTION_METRICS,
    XGBOOST_HYPERPARAMETER_RULES,
)


# ---------------------------------------------------------------------------
# Fixed/direct hyperparameter contracts
# ---------------------------------------------------------------------------

XGBoostHyperparameterValue: TypeAlias = (
    str
    | int
    | float
    | bool
)

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
        if (
            isinstance(
                value,
                bool,
            )
            or not isinstance(
                value,
                int,
            )
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} must be an integer."
            )

    elif expected_type == "number":
        if (
            isinstance(
                value,
                bool,
            )
            or not isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} must be numeric."
            )

    elif expected_type == "str":
        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} must be a non-empty string."
            )

    elif expected_type == "bool":
        if not isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} must be a boolean."
            )

    else:
        raise ValueError(
            f"Unsupported validation rule type '{expected_type}' for "
            f"XGBoost hyperparameter '{parameter_name}'."
        )

    if expected_type in {
        "int",
        "number",
    }:
        numeric_value = float(
            value
        )

        minimum = rule.get(
            "ge"
        )

        strict_minimum = rule.get(
            "gt"
        )

        maximum = rule.get(
            "le"
        )

        strict_maximum = rule.get(
            "lt"
        )

        if (
            minimum is not None
            and numeric_value < float(
                minimum
            )
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} must be >= {minimum}."
            )

        if (
            strict_minimum is not None
            and numeric_value <= float(
                strict_minimum
            )
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} must be > "
                f"{strict_minimum}."
            )

        if (
            maximum is not None
            and numeric_value > float(
                maximum
            )
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} must be <= {maximum}."
            )

        if (
            strict_maximum is not None
            and numeric_value >= float(
                strict_maximum
            )
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

    return dict(
        parameters
    )


# ---------------------------------------------------------------------------
# Optuna hyperparameter search-space contracts
# ---------------------------------------------------------------------------


class XGBoostOptunaParameterRange(BaseModel):
    """
    Define the Optuna search range for one configurable XGBoost parameter.

    Parameter type is derived from XGBOOST_HYPERPARAMETER_RULES rather than
    supplied separately by the user.

    Integer XGBoost parameters are later sampled with suggest_int.
    Numeric XGBoost parameters are later sampled with suggest_float.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    low: int | float = Field(
        description=(
            "Inclusive lower boundary of the Optuna search range."
        ),
    )

    high: int | float = Field(
        description=(
            "Inclusive upper boundary of the Optuna search range."
        ),
    )

    log: bool = Field(
        default=False,
        description=(
            "When True, request logarithmic sampling across the range. "
            "The lower boundary must then be greater than zero."
        ),
    )


XGBoostOptunaSearchSpace: TypeAlias = dict[
    str,
    XGBoostOptunaParameterRange,
]


def _validate_optuna_parameter_range(
    *,
    parameter_name: str,
    parameter_range: XGBoostOptunaParameterRange,
    field_name: str,
) -> XGBoostOptunaParameterRange:
    """Validate one Optuna range against the XGBoost parameter contract."""

    rule = XGBOOST_HYPERPARAMETER_RULES.get(
        parameter_name
    )

    if rule is None:
        raise ValueError(
            f"{field_name} contains unsupported XGBoost hyperparameter "
            f"'{parameter_name}'."
        )

    expected_type = rule.get(
        "type"
    )

    if expected_type not in {
        "int",
        "number",
    }:
        raise ValueError(
            f"{field_name}.{parameter_name} cannot use a numeric Optuna "
            "range because its XGBoost validation type is "
            f"'{expected_type}'."
        )

    low = parameter_range.low
    high = parameter_range.high

    # Integer parameters require integer search boundaries.
    if expected_type == "int":
        if (
            isinstance(
                low,
                bool,
            )
            or not isinstance(
                low,
                int,
            )
        ):
            raise ValueError(
                f"{field_name}.{parameter_name}.low must be an integer."
            )

        if (
            isinstance(
                high,
                bool,
            )
            or not isinstance(
                high,
                int,
            )
        ):
            raise ValueError(
                f"{field_name}.{parameter_name}.high must be an integer."
            )

    # Reuse the existing XGBoost value rules for both range boundaries.
    _validate_parameter_value(
        parameter_name=parameter_name,
        value=low,
        field_name=(
            f"{field_name}.{parameter_name}.low"
        ),
    )

    _validate_parameter_value(
        parameter_name=parameter_name,
        value=high,
        field_name=(
            f"{field_name}.{parameter_name}.high"
        ),
    )

    if float(
        low
    ) >= float(
        high
    ):
        raise ValueError(
            f"{field_name}.{parameter_name} requires low < high."
        )

    if (
        parameter_range.log
        and float(
            low
        ) <= 0.0
    ):
        raise ValueError(
            f"{field_name}.{parameter_name} requires low > 0 when "
            "log=True."
        )

    return parameter_range


def _validate_complete_optuna_search_space(
    search_space: XGBoostOptunaSearchSpace,
    *,
    field_name: str,
) -> XGBoostOptunaSearchSpace:
    """
    Validate one complete Optuna XGBoost search space.

    User-supplied search spaces follow the same all-or-none contract as fixed
    hyperparameter candidate sets: when supplied, every configurable XGBoost
    hyperparameter must have a search range.

    When no search space is supplied, the internal default search space is
    resolved instead.
    """

    if not search_space:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    expected_parameters = set(
        DEFAULT_XGBOOST_HYPERPARAMETERS
    )

    supplied_parameters = set(
        search_space
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
            f"{field_name} must provide a range for every configurable "
            f"XGBoost hyperparameter. Missing: {missing_parameters}"
        )

    if unexpected_parameters:
        raise ValueError(
            f"{field_name} contains unsupported XGBoost "
            f"hyperparameters: {unexpected_parameters}"
        )

    validated: XGBoostOptunaSearchSpace = {}

    for parameter_name, parameter_range in search_space.items():
        validated[
            parameter_name
        ] = _validate_optuna_parameter_range(
            parameter_name=parameter_name,
            parameter_range=parameter_range,
            field_name=field_name,
        )

    return validated


def _parse_default_optuna_search_space() -> XGBoostOptunaSearchSpace:
    """Parse and validate the data-only default Optuna search space."""

    parsed: XGBoostOptunaSearchSpace = {}

    for (
        parameter_name,
        raw_range,
    ) in DEFAULT_XGBOOST_OPTUNA_SEARCH_SPACE.items():
        parsed[
            parameter_name
        ] = XGBoostOptunaParameterRange.model_validate(
            dict(
                raw_range
            )
        )

    return _validate_complete_optuna_search_space(
        parsed,
        field_name="default Optuna search space",
    )


class XGBoostOptunaConfig(BaseModel):
    """
    Define Optuna-specific optimization controls.

    Search ranges themselves are stored separately in optuna_search_space.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    n_trials: int = Field(
        default=DEFAULT_XGBOOST_OPTUNA_N_TRIALS,
        gt=0,
        description=(
            "Maximum number of Optuna trials executed for one XGBoost "
            "site-series tuning run."
        ),
    )

    sampler_seed: int | None = Field(
        default=DEFAULT_XGBOOST_OPTUNA_SAMPLER_SEED,
        description=(
            "Optional seed used by the Optuna sampler for reproducibility."
        ),
    )


# ---------------------------------------------------------------------------
# Chronological split configuration
# ---------------------------------------------------------------------------


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

    When hyperparameter tuning is enabled, regardless of whether fixed or
    Optuna tuning is selected, the resolved order is::

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

            if supplied_count not in {
                0,
                4,
            }:
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

        train_supplied = (
            self.train_percentage
            is not None
        )

        calibration_supplied = (
            self.calibration_percentage
            is not None
        )

        test_supplied = (
            self.test_percentage
            is not None
        )

        supplied_count = sum(
            (
                train_supplied,
                calibration_supplied,
                test_supplied,
            )
        )

        if supplied_count not in {
            0,
            3,
        }:
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


# ---------------------------------------------------------------------------
# Shared tuning controls
# ---------------------------------------------------------------------------


class XGBoostTuningConfig(BaseModel):
    """
    Define XGBoost validation-selection and early-stopping controls.

    These settings are shared by fixed-candidate tuning and Optuna tuning.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    primary_selection_metric: str = Field(
        default=DEFAULT_XGBOOST_PRIMARY_SELECTION_METRIC,
        min_length=1,
        description=(
            "Primary validation metric used to evaluate XGBoost "
            "hyperparameter configurations."
        ),
    )

    secondary_selection_metric: str = Field(
        default=DEFAULT_XGBOOST_SECONDARY_SELECTION_METRIC,
        min_length=1,
        description=(
            "Secondary validation metric available as a deterministic "
            "tie-break when primary performance is equal."
        ),
    )

    max_estimators: int = Field(
        default=DEFAULT_XGBOOST_MAX_ESTIMATORS,
        gt=0,
        description=(
            "Maximum number of boosting trees available while evaluating "
            "a tuning configuration with early stopping."
        ),
    )

    early_stopping_rounds: int = Field(
        default=DEFAULT_XGBOOST_EARLY_STOPPING_ROUNDS,
        gt=0,
        description=(
            "Validation rounds without improvement before temporary tuning "
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
        """Validate tuning-selection metrics against defaults.py."""

        normalized = (
            value
            .strip()
            .lower()
        )

        if normalized not in XGBOOST_ALLOWED_SELECTION_METRICS:
            raise ValueError(
                "Unsupported XGBoost hyperparameter selection metric. "
                f"Supported metrics: "
                f"{sorted(XGBOOST_ALLOWED_SELECTION_METRICS)}"
            )

        return normalized

    @model_validator(
        mode="after"
    )
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


# ---------------------------------------------------------------------------
# Residual calibration
# ---------------------------------------------------------------------------


class XGBoostCalibrationConfig(BaseModel):
    """
    Configure XGBoost residual-quantile calibration.

    ``residual_quantile`` represents the upper residual-tail quantile.

    For example, with ``residual_quantile=0.99``:

    - upper_only uses the 0.99 residual quantile;
    - lower_only uses the complementary 0.01 residual quantile; and
    - both uses both 0.01 and 0.99 residual quantiles.

    The actual direction is controlled by the shared
    ``CommonTrainingConfig.threshold_direction`` setting.
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
            "The lower-tail quantile is automatically "
            "1 - residual_quantile."
        ),
    )


# ---------------------------------------------------------------------------
# Model-specific dataset validation rules
# ---------------------------------------------------------------------------


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
    fields.
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

    excluded_part_c_feature_types: tuple[
        str,
        ...,
    ] = Field(
        default=DEFAULT_XGBOOST_EXCLUDED_PART_C_FEATURE_TYPES,
        description=(
            "Part C feature request families excluded from X because they "
            "directly contain the current KPI value used as y."
        ),
    )

    column_role_rules: dict[
        str,
        dict[
            str,
            object,
        ],
    ] = Field(
        default_factory=lambda: {
            role: dict(
                rule
            )
            for (
                role,
                rule,
            ) in DEFAULT_XGBOOST_COLUMN_ROLE_RULES.items()
        },
        description=(
            "Logical rules defining X features, the required y target, and "
            "non-feature structural/status/usability columns."
        ),
    )

    trusted_target_statuses: frozenset[
        int
    ] = Field(
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


# ---------------------------------------------------------------------------
# Complete XGBoost request
# ---------------------------------------------------------------------------


class XGBoostTrainingConfig(
    CommonTrainingConfig
):
    """
    Define the complete Part D XGBoost request.

    Six hyperparameter input modes are supported:

    Direct training
    ---------------
    1. tuning disabled + single configuration supplied;
    2. tuning disabled + no configuration -> internal single default.

    Fixed tuning
    ------------
    3. tuning enabled + method='fixed' + candidate list supplied;
    4. tuning enabled + method='fixed' + no candidate list
       -> internal default candidates.

    Optuna tuning
    -------------
    5. tuning enabled + method='optuna' + search space supplied;
    6. tuning enabled + method='optuna' + no search space
       -> internal default Optuna search space.

    Optuna controls such as n_trials and sampler_seed are optional and use
    internal defaults when omitted.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    model_type: Literal[
        "xgboost"
    ] = Field(
        default="xgboost",
    )

    split: XGBoostSplitConfig = Field(
        default_factory=XGBoostSplitConfig,
    )

    # ------------------------------------------------------------------
    # Direct training
    # ------------------------------------------------------------------

    hyperparameters: (
        XGBoostHyperparameterSet
        | None
    ) = Field(
        default=None,
        description=(
            "One complete configurable XGBoost hyperparameter set used when "
            "tuning is disabled. Omit to use the internal single default."
        ),
    )

    # ------------------------------------------------------------------
    # Fixed tuning
    # ------------------------------------------------------------------

    hyperparameter_candidates: (
        list[
            XGBoostHyperparameterSet
        ]
        | None
    ) = Field(
        default=None,
        min_length=1,
        description=(
            "Complete configurable XGBoost hyperparameter candidates used "
            "when tuning is enabled with method='fixed'. Omit to use the "
            "internal default candidate list."
        ),
    )

    # ------------------------------------------------------------------
    # Optuna tuning
    # ------------------------------------------------------------------

    optuna_search_space: (
        XGBoostOptunaSearchSpace
        | None
    ) = Field(
        default=None,
        description=(
            "Complete XGBoost hyperparameter search space used when tuning "
            "is enabled with method='optuna'. Omit to use the internal "
            "default Optuna search space."
        ),
    )

    optuna: (
        XGBoostOptunaConfig
        | None
    ) = Field(
        default=None,
        description=(
            "Optional Optuna-specific controls. Omit to use internal "
            "defaults for trial count and sampler seed."
        ),
    )

    # ------------------------------------------------------------------
    # Controls shared across tuning implementations
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Field validation
    # ------------------------------------------------------------------

    @field_validator(
        "hyperparameters",
    )
    @classmethod
    def validate_hyperparameters(
        cls,
        value: (
            XGBoostHyperparameterSet
            | None
        ),
    ) -> (
        XGBoostHyperparameterSet
        | None
    ):
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
        value: (
            list[
                XGBoostHyperparameterSet
            ]
            | None
        ),
    ) -> (
        list[
            XGBoostHyperparameterSet
        ]
        | None
    ):
        """Validate every explicitly supplied fixed tuning candidate."""

        if value is None:
            return None

        return [
            _validate_complete_hyperparameter_set(
                candidate,
                field_name=(
                    f"hyperparameter_candidates[{index}]"
                ),
            )
            for (
                index,
                candidate,
            ) in enumerate(
                value
            )
        ]

    @field_validator(
        "optuna_search_space",
    )
    @classmethod
    def validate_optuna_search_space(
        cls,
        value: (
            XGBoostOptunaSearchSpace
            | None
        ),
    ) -> (
        XGBoostOptunaSearchSpace
        | None
    ):
        """Validate an explicitly supplied complete Optuna search space."""

        if value is None:
            return None

        return _validate_complete_optuna_search_space(
            value,
            field_name="optuna_search_space",
        )

    # ------------------------------------------------------------------
    # Mode validation
    # ------------------------------------------------------------------

    @model_validator(
        mode="after"
    )
    def validate_hyperparameter_mode(
        self,
    ) -> XGBoostTrainingConfig:
        """
        Enforce direct-vs-fixed-vs-Optuna input exclusivity.

        Missing fixed candidates or missing Optuna ranges are valid because
        their respective internal defaults are resolved later.
        """

        # --------------------------------------------------------------
        # Tuning disabled
        # --------------------------------------------------------------

        if not self.hyperparameter_tuning_enabled:
            if self.hyperparameter_candidates is not None:
                raise ValueError(
                    "hyperparameter_candidates cannot be supplied when "
                    "hyperparameter_tuning_enabled=False."
                )

            if self.optuna_search_space is not None:
                raise ValueError(
                    "optuna_search_space cannot be supplied when "
                    "hyperparameter_tuning_enabled=False."
                )

            if self.optuna is not None:
                raise ValueError(
                    "optuna settings cannot be supplied when "
                    "hyperparameter_tuning_enabled=False."
                )

            return self

        # --------------------------------------------------------------
        # All tuning modes
        # --------------------------------------------------------------

        if self.hyperparameters is not None:
            raise ValueError(
                "hyperparameters cannot be supplied when "
                "hyperparameter_tuning_enabled=True."
            )

        # --------------------------------------------------------------
        # Fixed tuning
        # --------------------------------------------------------------

        if (
            self.hyperparameter_tuning_method
            == "fixed"
        ):
            if self.optuna_search_space is not None:
                raise ValueError(
                    "optuna_search_space cannot be supplied when "
                    "hyperparameter_tuning_method='fixed'."
                )

            if self.optuna is not None:
                raise ValueError(
                    "optuna settings cannot be supplied when "
                    "hyperparameter_tuning_method='fixed'."
                )

            return self

        # --------------------------------------------------------------
        # Optuna tuning
        # --------------------------------------------------------------

        if (
            self.hyperparameter_tuning_method
            == "optuna"
        ):
            if self.hyperparameter_candidates is not None:
                raise ValueError(
                    "hyperparameter_candidates cannot be supplied when "
                    "hyperparameter_tuning_method='optuna'."
                )

            return self

        raise ValueError(
            "Unsupported XGBoost hyperparameter_tuning_method: "
            f"{self.hyperparameter_tuning_method}"
        )

    # ------------------------------------------------------------------
    # Direct-training resolution
    # ------------------------------------------------------------------

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
            dict(
                parameters
            ),
            field_name="resolved hyperparameters",
        )

    # ------------------------------------------------------------------
    # Fixed-tuning resolution
    # ------------------------------------------------------------------

    def resolve_hyperparameter_candidates(
        self,
    ) -> tuple[
        XGBoostHyperparameterSet,
        ...,
    ]:
        """
        Return user-supplied or default fixed hyperparameter candidates.
        """

        if not self.hyperparameter_tuning_enabled:
            raise ValueError(
                "resolve_hyperparameter_candidates cannot be used while "
                "hyperparameter tuning is disabled."
            )

        if (
            self.hyperparameter_tuning_method
            != "fixed"
        ):
            raise ValueError(
                "resolve_hyperparameter_candidates requires "
                "hyperparameter_tuning_method='fixed'."
            )

        candidates = (
            self.hyperparameter_candidates
            if self.hyperparameter_candidates is not None
            else DEFAULT_XGBOOST_HYPERPARAMETER_CANDIDATES
        )

        if not candidates:
            raise ValueError(
                "The resolved XGBoost fixed hyperparameter candidate list "
                "cannot be empty."
            )

        return tuple(
            _validate_complete_hyperparameter_set(
                dict(
                    candidate
                ),
                field_name=(
                    f"resolved hyperparameter candidate[{index}]"
                ),
            )
            for (
                index,
                candidate,
            ) in enumerate(
                candidates
            )
        )

    # ------------------------------------------------------------------
    # Optuna resolution
    # ------------------------------------------------------------------

    def resolve_optuna_search_space(
        self,
    ) -> XGBoostOptunaSearchSpace:
        """
        Return user-supplied or default Optuna hyperparameter ranges.
        """

        if not self.hyperparameter_tuning_enabled:
            raise ValueError(
                "resolve_optuna_search_space cannot be used while "
                "hyperparameter tuning is disabled."
            )

        if (
            self.hyperparameter_tuning_method
            != "optuna"
        ):
            raise ValueError(
                "resolve_optuna_search_space requires "
                "hyperparameter_tuning_method='optuna'."
            )

        if self.optuna_search_space is not None:
            return _validate_complete_optuna_search_space(
                dict(
                    self.optuna_search_space
                ),
                field_name="resolved Optuna search space",
            )

        return _parse_default_optuna_search_space()

    def resolve_optuna_config(
        self,
    ) -> XGBoostOptunaConfig:
        """Return user-supplied or default Optuna execution controls."""

        if not self.hyperparameter_tuning_enabled:
            raise ValueError(
                "resolve_optuna_config cannot be used while "
                "hyperparameter tuning is disabled."
            )

        if (
            self.hyperparameter_tuning_method
            != "optuna"
        ):
            raise ValueError(
                "resolve_optuna_config requires "
                "hyperparameter_tuning_method='optuna'."
            )

        return (
            self.optuna
            if self.optuna is not None
            else XGBoostOptunaConfig()
        )

    # ------------------------------------------------------------------
    # Other resolved contracts
    # ------------------------------------------------------------------

    def resolve_split(
        self,
    ) -> ResolvedXGBoostSplitConfig:
        """Return split percentages appropriate for tuning ON/OFF."""

        return self.split.resolve_for_tuning(
            tuning_enabled=(
                self.hyperparameter_tuning_enabled
            ),
        )

    def get_validation_rules(
        self,
    ) -> XGBoostValidationRules:
        """Return fixed model-specific dataset-validation rules."""

        return XGBoostValidationRules()


__all__ = [
    "XGBoostHyperparameterSet",
    "XGBoostHyperparameterValue",
    "XGBoostOptunaConfig",
    "XGBoostOptunaParameterRange",
    "XGBoostOptunaSearchSpace",
    "XGBoostCalibrationConfig",
    "XGBoostSplitConfig",
    "XGBoostTrainingConfig",
    "XGBoostTuningConfig",
    "XGBoostValidationRules",
    "ResolvedXGBoostSplitConfig",
]