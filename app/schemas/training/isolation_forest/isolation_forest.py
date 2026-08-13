"""
Pydantic schemas for Part D Isolation Forest training.

Intended project location::

    app/schemas/training/isolation_forest/isolation_forest.py

This module defines:

- the Isolation Forest hyperparameter input contract;
- the same four hyperparameter modes used by XGBoost;
- Isolation Forest-specific chronological split configuration;
- tuning controls;
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
perform hyperparameter tuning, calculate anomaly scores, apply directional
anomaly rules, test models, or save outputs.
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
    DEFAULT_ISOLATION_FOREST_REQUIRE_ALL_RESOLVED_FEATURES,
    DEFAULT_ISOLATION_FOREST_REQUIRE_COMPLETE_FEATURE_ROWS,
    DEFAULT_ISOLATION_FOREST_USE_ONLY_RESOLVED_FEATURE_COLUMNS,
    DEFAULT_ISOLATION_FOREST_SELECTION_METRIC,
    DEFAULT_ISOLATION_FOREST_TRUSTED_TRAIN_STATUSES,
    DEFAULT_ISOLATION_FOREST_TUNING_TEST_PERCENTAGE,
    DEFAULT_ISOLATION_FOREST_TUNING_TRAIN_PERCENTAGE,
    DEFAULT_ISOLATION_FOREST_TUNING_VALIDATION_PERCENTAGE,
    ISOLATION_FOREST_ALLOWED_SELECTION_METRICS,
    ISOLATION_FOREST_HYPERPARAMETER_RULES,
)


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
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{field_name}.{parameter_name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{field_name}.{parameter_name} must be greater than zero."
            )

        return

    if rule_type == "bool":
        if not isinstance(value, bool):
            raise ValueError(
                f"{field_name}.{parameter_name} must be a boolean."
            )

        return

    if rule_type == "max_samples":
        if isinstance(value, bool):
            raise ValueError(
                f"{field_name}.{parameter_name} cannot be boolean."
            )

        if isinstance(value, str):
            if value != "auto":
                raise ValueError(
                    f"{field_name}.{parameter_name} accepts only 'auto', "
                    "a positive integer, or a float in (0, 1]."
                )
            return

        if isinstance(value, int):
            if value <= 0:
                raise ValueError(
                    f"{field_name}.{parameter_name} must be greater than zero."
                )
            return

        if isinstance(value, float):
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
        if isinstance(value, bool):
            raise ValueError(
                f"{field_name}.{parameter_name} cannot be boolean."
            )

        if isinstance(value, str):
            if value != "auto":
                raise ValueError(
                    f"{field_name}.{parameter_name} accepts only 'auto' "
                    "or a float in (0, 0.5]."
                )
            return

        if isinstance(value, (int, float)):
            numeric_value = float(value)

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
        if isinstance(value, bool):
            raise ValueError(
                f"{field_name}.{parameter_name} cannot be boolean."
            )

        if isinstance(value, int):
            if value <= 0:
                raise ValueError(
                    f"{field_name}.{parameter_name} must be greater than zero."
                )
            return

        if isinstance(value, float):
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
    configurable contract. Adding or removing a configurable parameter
    therefore requires changes only in defaults.py.
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

    return dict(parameters)


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

    When tuning is enabled the resolved order is::

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

            if supplied_count not in {0, 3}:
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

        train_supplied = self.train_percentage is not None
        test_supplied = self.test_percentage is not None

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


class IsolationForestTuningConfig(BaseModel):
    """Define how complete Isolation Forest candidates are compared."""

    model_config = ConfigDict(
        extra="forbid",
    )

    selection_metric: str = Field(
        default=DEFAULT_ISOLATION_FOREST_SELECTION_METRIC,
        min_length=1,
        description=(
            "Validation metric used to select the best complete Isolation "
            "Forest hyperparameter candidate."
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
        """Validate candidate-selection metric against defaults.py."""

        normalized = value.strip().lower()

        if normalized not in ISOLATION_FOREST_ALLOWED_SELECTION_METRICS:
            raise ValueError(
                "Unsupported Isolation Forest selection metric. "
                f"Supported metrics: "
                f"{sorted(ISOLATION_FOREST_ALLOWED_SELECTION_METRICS)}"
            )

        return normalized


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

    column_role_rules: dict[str, dict[str, object]] = Field(
        default_factory=lambda: {
            role: dict(rule)
            for role, rule in DEFAULT_ISOLATION_FOREST_COLUMN_ROLE_RULES.items()
        },
        description=(
            "Logical rules defining X features and compulsory non-feature "
            "structural/status/usability columns. Isolation Forest has no y."
        ),
    )

    direction_feature_rule: dict[str, object] = Field(
        default_factory=lambda: dict(
            DEFAULT_ISOLATION_FOREST_DIRECTION_FEATURE_RULE
        ),
        description=(
            "Conditional Part C feature requirement used to apply one-sided "
            "lower/upper anomaly direction after Isolation Forest scoring."
        ),
    )

    trusted_train_statuses: frozenset[int] = Field(
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


class IsolationForestTrainingConfig(CommonTrainingConfig):
    """
    Define the complete Part D Isolation Forest request.

    Four hyperparameter modes are supported:

    1. tuning enabled + candidate list supplied;
    2. tuning enabled + no candidate list -> internal default candidates;
    3. tuning disabled + single configuration supplied;
    4. tuning disabled + no configuration -> internal single default.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    model_type: Literal["isolation_forest"] = Field(
        default="isolation_forest",
    )

    split: IsolationForestSplitConfig = Field(
        default_factory=IsolationForestSplitConfig,
    )

    hyperparameters: IsolationForestHyperparameterSet | None = Field(
        default=None,
        description=(
            "One complete configurable Isolation Forest hyperparameter set "
            "used when tuning is disabled. Omit to use the internal single "
            "default."
        ),
    )

    hyperparameter_candidates: (
        list[IsolationForestHyperparameterSet]
        | None
    ) = Field(
        default=None,
        min_length=1,
        description=(
            "Complete configurable Isolation Forest hyperparameter "
            "candidates used when tuning is enabled. No maximum candidate "
            "count is imposed. Omit to use the internal default candidates."
        ),
    )

    tuning: IsolationForestTuningConfig = Field(
        default_factory=IsolationForestTuningConfig,
    )

    @field_validator(
        "hyperparameters",
    )
    @classmethod
    def validate_hyperparameters(
        cls,
        value: IsolationForestHyperparameterSet | None,
    ) -> IsolationForestHyperparameterSet | None:
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
        value: list[IsolationForestHyperparameterSet] | None,
    ) -> list[IsolationForestHyperparameterSet] | None:
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
    ) -> IsolationForestTrainingConfig:
        """Validate list-vs-single input and split mode."""

        if self.hyperparameter_tuning_enabled:
            if self.hyperparameters is not None:
                raise ValueError(
                    "hyperparameters cannot be supplied when "
                    "hyperparameter_tuning_enabled=True. Supply "
                    "hyperparameter_candidates instead, or omit candidates "
                    "to use the internal default candidate list."
                )

        elif self.hyperparameter_candidates is not None:
            raise ValueError(
                "hyperparameter_candidates cannot be supplied when "
                "hyperparameter_tuning_enabled=False. Supply "
                "hyperparameters instead, or omit them to use the internal "
                "single default configuration."
            )

        self.split.resolve_for_tuning(
            tuning_enabled=self.hyperparameter_tuning_enabled,
        )

        return self

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
            dict(parameters),
            field_name="resolved hyperparameters",
        )

    def resolve_hyperparameter_candidates(
        self,
    ) -> tuple[IsolationForestHyperparameterSet, ...]:
        """Return supplied candidates or the internal default candidate list."""

        if not self.hyperparameter_tuning_enabled:
            raise ValueError(
                "resolve_hyperparameter_candidates cannot be used while "
                "hyperparameter tuning is disabled."
            )

        candidates = (
            self.hyperparameter_candidates
            if self.hyperparameter_candidates is not None
            else DEFAULT_ISOLATION_FOREST_HYPERPARAMETER_CANDIDATES
        )

        if not candidates:
            raise ValueError(
                "The resolved Isolation Forest hyperparameter candidate "
                "list cannot be empty."
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
    ) -> ResolvedIsolationForestSplitConfig:
        """Return the split percentages appropriate for the tuning mode."""

        return self.split.resolve_for_tuning(
            tuning_enabled=self.hyperparameter_tuning_enabled,
        )

    def get_validation_rules(
        self,
    ) -> IsolationForestValidationRules:
        """Return fixed model-specific dataset-validation rules."""

        return IsolationForestValidationRules()


__all__ = [
    "IsolationForestHyperparameterSet",
    "IsolationForestHyperparameterValue",
    "IsolationForestSplitConfig",
    "IsolationForestTrainingConfig",
    "IsolationForestTuningConfig",
    "IsolationForestValidationRules",
    "ResolvedIsolationForestSplitConfig",
]