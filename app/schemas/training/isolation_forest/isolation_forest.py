"""
Pydantic schemas for Part D Isolation Forest training.

Intended project location::

    app/schemas/training/isolation_forest/isolation_forest.py

This module defines:

- the Isolation Forest fixed hyperparameter input contract;
- the Isolation Forest Optuna search-space input contract;
- direct, fixed-tuning, and Optuna-tuning hyperparameter modes;
- Isolation Forest-specific chronological split configuration;
- shared tuning controls;
- Optuna-specific tuning controls;
- model-specific dataset-validation rules; and
- resolution of user-provided or default settings.

History is not percentage-configured here. The Part D loader must reserve the
minimum history required by the resolved Part C feature-engineering
configuration before applying the model split percentages to the remaining
rows.

Isolation Forest does not use a separate calibration split. Its
``contamination`` hyperparameter defines the anomaly decision boundary.
The common ``threshold_direction`` setting is applied later by model logic
using the sign of the residual feature after Isolation Forest identifies a
point as anomalous.

Actual default values belong to::

    app/schemas/training/isolation_forest/defaults.py

This module does not load datasets, split dataframes, train Isolation Forest,
perform fixed hyperparameter tuning, perform Optuna optimization,
calculate anomaly scores, apply directional anomaly rules, test models,
or save outputs.
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
    DEFAULT_ISOLATION_FOREST_DIRECT_TEST_PERCENTAGE,
    DEFAULT_ISOLATION_FOREST_DIRECT_TRAIN_PERCENTAGE,
    DEFAULT_ISOLATION_FOREST_COLUMN_ROLE_RULES,
    DEFAULT_ISOLATION_FOREST_DIRECTION_FEATURE_RULE,
    DEFAULT_ISOLATION_FOREST_HYPERPARAMETER_CANDIDATES,
    DEFAULT_ISOLATION_FOREST_HYPERPARAMETERS,
    DEFAULT_ISOLATION_FOREST_MINIMUM_TEST_POINTS,
    DEFAULT_ISOLATION_FOREST_MINIMUM_TRAIN_POINTS,
    DEFAULT_ISOLATION_FOREST_MINIMUM_VALIDATION_POINTS,
    DEFAULT_ISOLATION_FOREST_OPTUNA_N_TRIALS,
    DEFAULT_ISOLATION_FOREST_OPTUNA_SAMPLER_SEED,
    DEFAULT_ISOLATION_FOREST_OPTUNA_SEARCH_SPACE,
    DEFAULT_ISOLATION_FOREST_REQUIRE_ALL_RESOLVED_FEATURES,
    DEFAULT_ISOLATION_FOREST_REQUIRE_COMPLETE_FEATURE_ROWS,
    DEFAULT_ISOLATION_FOREST_SELECTION_METRIC,
    DEFAULT_ISOLATION_FOREST_TRUSTED_TRAIN_STATUSES,
    DEFAULT_ISOLATION_FOREST_TUNING_TEST_PERCENTAGE,
    DEFAULT_ISOLATION_FOREST_TUNING_TRAIN_PERCENTAGE,
    DEFAULT_ISOLATION_FOREST_TUNING_VALIDATION_PERCENTAGE,
    DEFAULT_ISOLATION_FOREST_USE_ONLY_RESOLVED_FEATURE_COLUMNS,
    ISOLATION_FOREST_ALLOWED_SELECTION_METRICS,
    ISOLATION_FOREST_HYPERPARAMETER_RULES,
)


# ---------------------------------------------------------------------------
# Fixed/direct hyperparameter contracts
# ---------------------------------------------------------------------------

IsolationForestHyperparameterValue: TypeAlias = (
    str
    | int
    | float
    | bool
)

IsolationForestHyperparameterSet: TypeAlias = dict[
    str,
    IsolationForestHyperparameterValue,
]


def _validate_parameter_value(
    *,
    parameter_name: str,
    value: IsolationForestHyperparameterValue,
    field_name: str,
) -> None:
    """Validate one Isolation Forest hyperparameter from defaults.py rules."""

    rule = ISOLATION_FOREST_HYPERPARAMETER_RULES.get(
        parameter_name
    )

    if rule is None:
        raise ValueError(
            f"{field_name} contains unsupported Isolation Forest "
            f"hyperparameter '{parameter_name}'."
        )

    rule_type = rule["type"]

    if rule_type == "int":
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

        if value <= 0:
            raise ValueError(
                f"{field_name}.{parameter_name} must be greater than zero."
            )

        return

    if rule_type == "bool":
        if not isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} must be a boolean."
            )

        return

    if rule_type == "max_samples":
        if isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} cannot be boolean."
            )

        if isinstance(
            value,
            str,
        ):
            if value != "auto":
                raise ValueError(
                    f"{field_name}.{parameter_name} accepts only 'auto', "
                    "a positive integer, or a float in (0, 1]."
                )

            return

        if isinstance(
            value,
            int,
        ):
            if value <= 0:
                raise ValueError(
                    f"{field_name}.{parameter_name} must be greater than zero."
                )

            return

        if isinstance(
            value,
            float,
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"{field_name}.{parameter_name} as a float must lie "
                    "in (0, 1]."
                )

            return

        raise ValueError(
            f"{field_name}.{parameter_name} accepts only 'auto', "
            "a positive integer, or a float in (0, 1]."
        )

    if rule_type == "contamination":
        if isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} cannot be boolean."
            )

        if isinstance(
            value,
            str,
        ):
            if value != "auto":
                raise ValueError(
                    f"{field_name}.{parameter_name} accepts only 'auto' "
                    "or a float in (0, 0.5]."
                )

            return

        if isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            numeric_value = float(
                value
            )

            if not 0.0 < numeric_value <= 0.5:
                raise ValueError(
                    f"{field_name}.{parameter_name} must lie in (0, 0.5]."
                )

            return

        raise ValueError(
            f"{field_name}.{parameter_name} accepts only 'auto' "
            "or a float in (0, 0.5]."
        )

    if rule_type == "max_features":
        if isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"{field_name}.{parameter_name} cannot be boolean."
            )

        if isinstance(
            value,
            int,
        ):
            if value <= 0:
                raise ValueError(
                    f"{field_name}.{parameter_name} must be greater than zero."
                )

            return

        if isinstance(
            value,
            float,
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"{field_name}.{parameter_name} as a float must lie "
                    "in (0, 1]."
                )

            return

        raise ValueError(
            f"{field_name}.{parameter_name} must be a positive integer "
            "or a float in (0, 1]."
        )

    raise ValueError(
        f"Unsupported validation rule type '{rule_type}' for "
        f"Isolation Forest hyperparameter '{parameter_name}'."
    )


def _validate_complete_hyperparameter_set(
    parameters: IsolationForestHyperparameterSet,
    *,
    field_name: str,
) -> IsolationForestHyperparameterSet:
    """
    Validate one complete user-configurable Isolation Forest parameter set.

    The keys of DEFAULT_ISOLATION_FOREST_HYPERPARAMETERS define the complete
    configurable contract.
    """

    if not parameters:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    expected_parameters = set(
        DEFAULT_ISOLATION_FOREST_HYPERPARAMETERS
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
            f"{field_name} must provide every configured Isolation Forest "
            f"hyperparameter. Missing: {missing_parameters}"
        )

    if unexpected_parameters:
        raise ValueError(
            f"{field_name} contains unsupported Isolation Forest "
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
# Optuna search-space contracts
# ---------------------------------------------------------------------------

IsolationForestOptunaParameterType = Literal[
    "int",
    "float",
    "categorical",
]


class IsolationForestOptunaParameterConfig(BaseModel):
    """
    Define one Isolation Forest Optuna hyperparameter search specification.

    ``type='int'``:
        use low/high as integer boundaries.

    ``type='float'``:
        use low/high as floating-point boundaries.

    ``type='categorical'``:
        use explicit choices.

    Categorical search is required for settings such as ``bootstrap`` and
    parameters that may accept the string ``'auto'``.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    type: IsolationForestOptunaParameterType

    low: int | float | None = Field(
        default=None,
    )

    high: int | float | None = Field(
        default=None,
    )

    log: bool = Field(
        default=False,
    )

    choices: tuple[
        IsolationForestHyperparameterValue,
        ...,
    ] | None = Field(
        default=None,
        min_length=1,
    )

    @field_validator(
        "choices",
        mode="before",
    )
    @classmethod
    def normalize_choices(
        cls,
        value,
    ):
        """
        Convert JSON arrays to tuples before strict Pydantic validation.

        FastAPI parses JSON arrays as Python lists, while this schema keeps
        categorical choices internally as an immutable tuple.
        """

        if isinstance(
            value,
            list,
        ):
            return tuple(
                value
            )

        return value

    @model_validator(
        mode="after"
    )
    def validate_search_definition(
        self,
    ) -> IsolationForestOptunaParameterConfig:
        """Validate range-vs-categorical structure."""

        if self.type == "categorical":
            if self.choices is None:
                raise ValueError(
                    "Categorical Optuna parameters require choices."
                )

            if (
                self.low is not None
                or self.high is not None
            ):
                raise ValueError(
                    "Categorical Optuna parameters cannot define low/high."
                )

            if self.log:
                raise ValueError(
                    "Categorical Optuna parameters cannot use log=True."
                )

            return self

        # int / float range
        if (
            self.low is None
            or self.high is None
        ):
            raise ValueError(
                f"Optuna parameter type='{self.type}' requires low and high."
            )

        if self.choices is not None:
            raise ValueError(
                f"Optuna parameter type='{self.type}' cannot define choices."
            )

        if self.type == "int":
            if (
                isinstance(
                    self.low,
                    bool,
                )
                or not isinstance(
                    self.low,
                    int,
                )
            ):
                raise ValueError(
                    "Integer Optuna parameter low must be an integer."
                )

            if (
                isinstance(
                    self.high,
                    bool,
                )
                or not isinstance(
                    self.high,
                    int,
                )
            ):
                raise ValueError(
                    "Integer Optuna parameter high must be an integer."
                )

        if self.type == "float":
            if (
                isinstance(
                    self.low,
                    bool,
                )
                or not isinstance(
                    self.low,
                    (
                        int,
                        float,
                    ),
                )
            ):
                raise ValueError(
                    "Float Optuna parameter low must be numeric."
                )

            if (
                isinstance(
                    self.high,
                    bool,
                )
                or not isinstance(
                    self.high,
                    (
                        int,
                        float,
                    ),
                )
            ):
                raise ValueError(
                    "Float Optuna parameter high must be numeric."
                )

        if float(
            self.low
        ) >= float(
            self.high
        ):
            raise ValueError(
                "Optuna parameter requires low < high."
            )

        if (
            self.log
            and float(
                self.low
            ) <= 0.0
        ):
            raise ValueError(
                "Optuna logarithmic search requires low > 0."
            )

        return self


IsolationForestOptunaSearchSpace: TypeAlias = dict[
    str,
    IsolationForestOptunaParameterConfig,
]


def _validate_optuna_parameter_config(
    *,
    parameter_name: str,
    parameter_config: IsolationForestOptunaParameterConfig,
    field_name: str,
) -> IsolationForestOptunaParameterConfig:
    """
    Validate one Optuna search definition against the Isolation Forest
    hyperparameter contract.
    """

    rule = ISOLATION_FOREST_HYPERPARAMETER_RULES.get(
        parameter_name
    )

    if rule is None:
        raise ValueError(
            f"{field_name} contains unsupported Isolation Forest "
            f"hyperparameter '{parameter_name}'."
        )

    rule_type = rule[
        "type"
    ]

    # ------------------------------------------------------------------
    # Categorical parameters
    # ------------------------------------------------------------------

    if parameter_config.type == "categorical":
        choices = parameter_config.choices

        if choices is None:
            raise ValueError(
                f"{field_name}.{parameter_name} requires categorical choices."
            )

        for index, choice in enumerate(
            choices
        ):
            _validate_parameter_value(
                parameter_name=parameter_name,
                value=choice,
                field_name=(
                    f"{field_name}.{parameter_name}.choices[{index}]"
                ),
            )

        return parameter_config

    # ------------------------------------------------------------------
    # Numeric ranges
    # ------------------------------------------------------------------

    low = parameter_config.low
    high = parameter_config.high

    if (
        low is None
        or high is None
    ):
        raise ValueError(
            f"{field_name}.{parameter_name} requires low/high boundaries."
        )

    # Fixed integer-only parameter.
    if rule_type == "int":
        if parameter_config.type != "int":
            raise ValueError(
                f"{field_name}.{parameter_name} must use type='int' "
                "or type='categorical'."
            )

    # Boolean has no numeric range.
    elif rule_type == "bool":
        raise ValueError(
            f"{field_name}.{parameter_name} must use type='categorical'."
        )

    # max_samples supports integer counts or fractional floats.
    elif rule_type == "max_samples":
        if parameter_config.type not in {
            "int",
            "float",
        }:
            raise ValueError(
                f"{field_name}.{parameter_name} must use type='int', "
                "type='float', or type='categorical'."
            )

    # contamination is naturally searched as a float.
    elif rule_type == "contamination":
        if parameter_config.type != "float":
            raise ValueError(
                f"{field_name}.{parameter_name} must use type='float' "
                "or type='categorical'."
            )

    # max_features supports positive integer counts or fractional floats.
    elif rule_type == "max_features":
        if parameter_config.type not in {
            "int",
            "float",
        }:
            raise ValueError(
                f"{field_name}.{parameter_name} must use type='int', "
                "type='float', or type='categorical'."
            )

    else:
        raise ValueError(
            f"Unsupported Isolation Forest Optuna rule type '{rule_type}' "
            f"for parameter '{parameter_name}'."
        )

    # Reuse the existing model-specific parameter rules to validate both
    # boundaries.
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

    return parameter_config


def _validate_complete_optuna_search_space(
    search_space: IsolationForestOptunaSearchSpace,
    *,
    field_name: str,
) -> IsolationForestOptunaSearchSpace:
    """
    Validate one complete Isolation Forest Optuna search space.

    When a user supplies a search space, every configurable Isolation Forest
    hyperparameter must be represented.

    If the user omits the search space entirely, the internal default Optuna
    search space is used instead.
    """

    if not search_space:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    expected_parameters = set(
        DEFAULT_ISOLATION_FOREST_HYPERPARAMETERS
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
            f"{field_name} must provide a search definition for every "
            f"configurable Isolation Forest hyperparameter. "
            f"Missing: {missing_parameters}"
        )

    if unexpected_parameters:
        raise ValueError(
            f"{field_name} contains unsupported Isolation Forest "
            f"hyperparameters: {unexpected_parameters}"
        )

    validated: IsolationForestOptunaSearchSpace = {}

    for parameter_name, parameter_config in search_space.items():
        validated[
            parameter_name
        ] = _validate_optuna_parameter_config(
            parameter_name=parameter_name,
            parameter_config=parameter_config,
            field_name=field_name,
        )

    return validated


def _parse_default_optuna_search_space(
) -> IsolationForestOptunaSearchSpace:
    """
    Parse and validate the data-only default Isolation Forest Optuna space.
    """

    parsed: IsolationForestOptunaSearchSpace = {}

    for (
        parameter_name,
        raw_config,
    ) in DEFAULT_ISOLATION_FOREST_OPTUNA_SEARCH_SPACE.items():
        parsed[
            parameter_name
        ] = IsolationForestOptunaParameterConfig.model_validate(
            dict(
                raw_config
            )
        )

    return _validate_complete_optuna_search_space(
        parsed,
        field_name="default Optuna search space",
    )


class IsolationForestOptunaConfig(BaseModel):
    """Define Optuna-specific Isolation Forest optimization controls."""

    model_config = ConfigDict(
        extra="forbid",
    )

    n_trials: int = Field(
        default=DEFAULT_ISOLATION_FOREST_OPTUNA_N_TRIALS,
        gt=0,
        description=(
            "Maximum number of Optuna trials executed for one Isolation "
            "Forest site-series tuning run."
        ),
    )

    sampler_seed: int | None = Field(
        default=DEFAULT_ISOLATION_FOREST_OPTUNA_SAMPLER_SEED,
        description=(
            "Optional seed used by the Optuna sampler for reproducibility."
        ),
    )


# ---------------------------------------------------------------------------
# Chronological split configuration
# ---------------------------------------------------------------------------


class ResolvedIsolationForestSplitConfig(BaseModel):
    """Store the final split percentages selected for one training request."""

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

    test_percentage: float = Field(
        gt=0.0,
        lt=100.0,
    )

    uses_validation_split: bool


class IsolationForestSplitConfig(BaseModel):
    """
    Define optional post-history split overrides.

    When tuning is enabled, regardless of fixed or Optuna tuning, the
    resolved order is::

        train -> validation -> test

    When tuning is disabled the resolved order is::

        train -> test

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

    test_percentage: float | None = Field(
        default=None,
        gt=0.0,
        lt=100.0,
    )

    def resolve_for_tuning(
        self,
        *,
        tuning_enabled: bool,
    ) -> ResolvedIsolationForestSplitConfig:
        """Resolve split defaults and validate them against tuning mode."""

        if tuning_enabled:
            supplied = (
                self.train_percentage,
                self.validation_percentage,
                self.test_percentage,
            )

            supplied_count = sum(
                value is not None
                for value in supplied
            )

            if supplied_count not in {
                0,
                3,
            }:
                raise ValueError(
                    "When Isolation Forest tuning is enabled, either omit "
                    "all split percentages to use defaults or provide train, "
                    "validation, and test percentages together."
                )

            train_percentage = (
                self.train_percentage
                if self.train_percentage is not None
                else DEFAULT_ISOLATION_FOREST_TUNING_TRAIN_PERCENTAGE
            )

            validation_percentage = (
                self.validation_percentage
                if self.validation_percentage is not None
                else DEFAULT_ISOLATION_FOREST_TUNING_VALIDATION_PERCENTAGE
            )

            test_percentage = (
                self.test_percentage
                if self.test_percentage is not None
                else DEFAULT_ISOLATION_FOREST_TUNING_TEST_PERCENTAGE
            )

            if validation_percentage <= 0.0:
                raise ValueError(
                    "Isolation Forest tuning requires a positive "
                    "validation_percentage."
                )

            total = (
                train_percentage
                + validation_percentage
                + test_percentage
            )

            if not isclose(
                total,
                100.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "Isolation Forest train, validation, and test "
                    "percentages must sum to exactly 100 when tuning is "
                    f"enabled. Received {total}."
                )

            return ResolvedIsolationForestSplitConfig(
                train_percentage=train_percentage,
                validation_percentage=validation_percentage,
                test_percentage=test_percentage,
                uses_validation_split=True,
            )

        if (
            self.validation_percentage is not None
            and self.validation_percentage != 0.0
        ):
            raise ValueError(
                "validation_percentage is not used when Isolation Forest "
                "hyperparameter tuning is disabled."
            )

        train_supplied = (
            self.train_percentage
            is not None
        )

        test_supplied = (
            self.test_percentage
            is not None
        )

        if train_supplied != test_supplied:
            raise ValueError(
                "When Isolation Forest tuning is disabled, train_percentage "
                "and test_percentage must either both be omitted or both be "
                "provided."
            )

        train_percentage = (
            self.train_percentage
            if self.train_percentage is not None
            else DEFAULT_ISOLATION_FOREST_DIRECT_TRAIN_PERCENTAGE
        )

        test_percentage = (
            self.test_percentage
            if self.test_percentage is not None
            else DEFAULT_ISOLATION_FOREST_DIRECT_TEST_PERCENTAGE
        )

        total = (
            train_percentage
            + test_percentage
        )

        if not isclose(
            total,
            100.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Isolation Forest train and test percentages must sum to "
                "exactly 100 when tuning is disabled. "
                f"Received {total}."
            )

        return ResolvedIsolationForestSplitConfig(
            train_percentage=train_percentage,
            validation_percentage=0.0,
            test_percentage=test_percentage,
            uses_validation_split=False,
        )


# ---------------------------------------------------------------------------
# Tuning controls shared by fixed and Optuna
# ---------------------------------------------------------------------------


class IsolationForestTuningConfig(BaseModel):
    """
    Define how Isolation Forest tuning configurations are evaluated.

    This metric is shared by fixed-candidate tuning and Optuna tuning.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    selection_metric: str = Field(
        default=DEFAULT_ISOLATION_FOREST_SELECTION_METRIC,
        min_length=1,
        description=(
            "Validation metric used to evaluate Isolation Forest "
            "hyperparameter configurations."
        ),
    )

    @field_validator(
        "selection_metric",
    )
    @classmethod
    def validate_selection_metric(
        cls,
        value: str,
    ) -> str:
        """Validate tuning-selection metric against defaults.py."""

        normalized = (
            value
            .strip()
            .lower()
        )

        if normalized not in ISOLATION_FOREST_ALLOWED_SELECTION_METRICS:
            raise ValueError(
                "Unsupported Isolation Forest selection metric. "
                f"Supported metrics: "
                f"{sorted(ISOLATION_FOREST_ALLOWED_SELECTION_METRICS)}"
            )

        return normalized


# ---------------------------------------------------------------------------
# Model-specific dataset validation rules
# ---------------------------------------------------------------------------


class IsolationForestValidationRules(BaseModel):
    """
    Store fixed Isolation Forest dataset-validation and column-role rules.

    Isolation Forest is unsupervised in this pipeline:

    - X consists only of the resolved Part C feature columns;
    - there is no y target;
    - the original dataset KPI value is not automatically a feature; and
    - structural/status/usability columns are compulsory metadata but are
      never model features.

    For ``lower_only`` or ``upper_only`` requests, the configured Part C
    residual feature is additionally required so model logic can apply the
    requested direction after Isolation Forest identifies an anomaly.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    use_only_resolved_feature_columns: bool = Field(
        default=DEFAULT_ISOLATION_FOREST_USE_ONLY_RESOLVED_FEATURE_COLUMNS,
    )

    require_all_resolved_features: bool = Field(
        default=DEFAULT_ISOLATION_FOREST_REQUIRE_ALL_RESOLVED_FEATURES,
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
            ) in DEFAULT_ISOLATION_FOREST_COLUMN_ROLE_RULES.items()
        },
        description=(
            "Logical rules defining X features and compulsory non-feature "
            "structural/status/usability columns. Isolation Forest has no y."
        ),
    )

    direction_feature_rule: dict[
        str,
        object,
    ] = Field(
        default_factory=lambda: dict(
            DEFAULT_ISOLATION_FOREST_DIRECTION_FEATURE_RULE
        ),
        description=(
            "Conditional Part C feature requirement used to apply one-sided "
            "lower/upper anomaly direction after Isolation Forest scoring."
        ),
    )

    trusted_train_statuses: frozenset[
        int
    ] = Field(
        default=DEFAULT_ISOLATION_FOREST_TRUSTED_TRAIN_STATUSES,
    )

    require_complete_feature_rows: bool = Field(
        default=DEFAULT_ISOLATION_FOREST_REQUIRE_COMPLETE_FEATURE_ROWS,
    )

    minimum_train_points: int = Field(
        default=DEFAULT_ISOLATION_FOREST_MINIMUM_TRAIN_POINTS,
        gt=0,
    )

    minimum_validation_points: int = Field(
        default=DEFAULT_ISOLATION_FOREST_MINIMUM_VALIDATION_POINTS,
        gt=0,
    )

    minimum_test_points: int = Field(
        default=DEFAULT_ISOLATION_FOREST_MINIMUM_TEST_POINTS,
        gt=0,
    )


# ---------------------------------------------------------------------------
# Complete Isolation Forest request
# ---------------------------------------------------------------------------


class IsolationForestTrainingConfig(
    CommonTrainingConfig
):
    """
    Define the complete Part D Isolation Forest request.

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
        "isolation_forest"
    ] = Field(
        default="isolation_forest",
    )

    split: IsolationForestSplitConfig = Field(
        default_factory=IsolationForestSplitConfig,
    )

    # ------------------------------------------------------------------
    # Direct training
    # ------------------------------------------------------------------

    hyperparameters: (
        IsolationForestHyperparameterSet
        | None
    ) = Field(
        default=None,
        description=(
            "One complete configurable Isolation Forest hyperparameter set "
            "used when tuning is disabled. Omit to use the internal single "
            "default."
        ),
    )

    # ------------------------------------------------------------------
    # Fixed tuning
    # ------------------------------------------------------------------

    hyperparameter_candidates: (
        list[
            IsolationForestHyperparameterSet
        ]
        | None
    ) = Field(
        default=None,
        min_length=1,
        description=(
            "Complete configurable Isolation Forest hyperparameter "
            "candidates used when tuning is enabled with method='fixed'. "
            "Omit to use the internal default candidates."
        ),
    )

    # ------------------------------------------------------------------
    # Optuna tuning
    # ------------------------------------------------------------------

    optuna_search_space: (
        IsolationForestOptunaSearchSpace
        | None
    ) = Field(
        default=None,
        description=(
            "Complete Isolation Forest Optuna search space used when tuning "
            "is enabled with method='optuna'. Omit to use the internal "
            "default Optuna search space."
        ),
    )

    optuna: (
        IsolationForestOptunaConfig
        | None
    ) = Field(
        default=None,
        description=(
            "Optional Optuna-specific controls. Omit to use internal "
            "defaults for trial count and sampler seed."
        ),
    )

    # ------------------------------------------------------------------
    # Controls shared by fixed and Optuna tuning
    # ------------------------------------------------------------------

    tuning: IsolationForestTuningConfig = Field(
        default_factory=IsolationForestTuningConfig,
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
            IsolationForestHyperparameterSet
            | None
        ),
    ) -> (
        IsolationForestHyperparameterSet
        | None
    ):
        """Validate an explicitly supplied direct configuration."""

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
                IsolationForestHyperparameterSet
            ]
            | None
        ),
    ) -> (
        list[
            IsolationForestHyperparameterSet
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
            IsolationForestOptunaSearchSpace
            | None
        ),
    ) -> (
        IsolationForestOptunaSearchSpace
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
    ) -> IsolationForestTrainingConfig:
        """
        Enforce direct-vs-fixed-vs-Optuna input exclusivity.

        Missing fixed candidates or missing Optuna search definitions are
        valid because internal defaults are resolved later.
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

            self.split.resolve_for_tuning(
                tuning_enabled=False,
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

        # --------------------------------------------------------------
        # Optuna tuning
        # --------------------------------------------------------------

        elif (
            self.hyperparameter_tuning_method
            == "optuna"
        ):
            if self.hyperparameter_candidates is not None:
                raise ValueError(
                    "hyperparameter_candidates cannot be supplied when "
                    "hyperparameter_tuning_method='optuna'."
                )

        else:
            raise ValueError(
                "Unsupported Isolation Forest "
                "hyperparameter_tuning_method: "
                f"{self.hyperparameter_tuning_method}"
            )

        self.split.resolve_for_tuning(
            tuning_enabled=True,
        )

        return self

    # ------------------------------------------------------------------
    # Direct-training resolution
    # ------------------------------------------------------------------

    def resolve_hyperparameters(
        self,
    ) -> IsolationForestHyperparameterSet:
        """Return the single configuration used when tuning is disabled."""

        if self.hyperparameter_tuning_enabled:
            raise ValueError(
                "resolve_hyperparameters cannot be used while "
                "hyperparameter tuning is enabled."
            )

        parameters = (
            self.hyperparameters
            if self.hyperparameters is not None
            else DEFAULT_ISOLATION_FOREST_HYPERPARAMETERS
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
        IsolationForestHyperparameterSet,
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
            else DEFAULT_ISOLATION_FOREST_HYPERPARAMETER_CANDIDATES
        )

        if not candidates:
            raise ValueError(
                "The resolved Isolation Forest fixed hyperparameter "
                "candidate list cannot be empty."
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
    ) -> IsolationForestOptunaSearchSpace:
        """
        Return user-supplied or default Isolation Forest Optuna search space.
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
    ) -> IsolationForestOptunaConfig:
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
            else IsolationForestOptunaConfig()
        )

    # ------------------------------------------------------------------
    # Other resolved contracts
    # ------------------------------------------------------------------

    def resolve_split(
        self,
    ) -> ResolvedIsolationForestSplitConfig:
        """Return split percentages appropriate for tuning ON/OFF."""

        return self.split.resolve_for_tuning(
            tuning_enabled=(
                self.hyperparameter_tuning_enabled
            ),
        )

    def get_validation_rules(
        self,
    ) -> IsolationForestValidationRules:
        """Return fixed model-specific dataset-validation rules."""

        return IsolationForestValidationRules()


__all__ = [
    "IsolationForestHyperparameterSet",
    "IsolationForestHyperparameterValue",
    "IsolationForestOptunaConfig",
    "IsolationForestOptunaParameterConfig",
    "IsolationForestOptunaParameterType",
    "IsolationForestOptunaSearchSpace",
    "IsolationForestSplitConfig",
    "IsolationForestTrainingConfig",
    "IsolationForestTuningConfig",
    "IsolationForestValidationRules",
    "ResolvedIsolationForestSplitConfig",
]