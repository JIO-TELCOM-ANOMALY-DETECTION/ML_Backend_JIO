"""
Part D dataset validation for model training.

Intended project location::

    app/ml/training/validation.py

This module validates the Part C engineered dataset before the Part D loader
creates chronological split boundaries.

Responsibilities
----------------
- dispatch to model-specific validation rules using ``model_type``;
- validate the engineered CSV structure without loading the complete dataset
  into memory;
- resolve logical dataset-column roles through ``DatasetColumnMapping``;
- verify the resolved Part C feature contract;
- determine the exact columns that may be used as model X features;
- enforce XGBoost target-only and Isolation Forest no-target rules; and
- return a compact validation result for the loader/runner.

This module does NOT:
- create train/validation/calibration/test splits;
- materialize X/y dataframes;
- train models;
- tune hyperparameters;
- calculate residual/anomaly thresholds; or
- save model outputs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from app.schemas.common.dataset import DatasetColumnMapping
from app.schemas.feature_engineering import ResolvedFeatureEngineeringConfig
from app.schemas.training.common import TrainingModelType
from app.schemas.training.isolation_forest.isolation_forest import (
    IsolationForestTrainingConfig,
    IsolationForestValidationRules,
)
from app.schemas.training.xgboost.xgboost import (
    XGBoostTrainingConfig,
    XGBoostValidationRules,
)


# Chunking is an execution detail rather than a user/model configuration.
# It keeps validation memory-bounded for large Part C CSV files.
VALIDATION_READ_CHUNK_SIZE = 250_000


TrainingConfig = (
    XGBoostTrainingConfig
    | IsolationForestTrainingConfig
)


@dataclass(
    frozen=True,
    slots=True,
)
class TrainingValidationResult:
    """
    Compact validated contract passed from validation to later Part D stages.

    No training rows are stored here. The loader uses the input path,
    timestamp/site columns, Part C history requirement, and model split config
    to calculate date ranges later.
    """

    input_file: Path
    model_type: TrainingModelType

    all_part_c_feature_columns: tuple[str, ...]
    model_feature_columns: tuple[str, ...]
    excluded_part_c_feature_columns: tuple[str, ...]

    target_column: str | None

    site_identity_columns: tuple[str, ...]
    timestamp_column: str
    status_column: str
    usability_column: str
    invalid_reason_column: str

    minimum_history_points: int

    trusted_statuses: frozenset[int]
    require_complete_feature_rows: bool
    require_non_null_target: bool

    minimum_train_points: int
    minimum_validation_points: int
    minimum_calibration_points: int
    minimum_test_points: int

    total_rows: int


def _normalise_input_file(
    input_file: str | Path,
) -> Path:
    """Normalize and validate the Part C engineered CSV path."""

    path = Path(
        input_file
    ).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            "Part C engineered dataset does not exist:\n"
            f"  {path}"
        )

    if not path.is_file():
        raise ValueError(
            "Part C engineered dataset path must point to a file:\n"
            f"  {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Part D training currently expects a CSV produced by Part C. "
            f"Received:\n  {path}"
        )

    return path


def _read_csv_header(
    path: Path,
) -> tuple[str, ...]:
    """
    Read the physical CSV header without allowing duplicate names to hide.

    pandas renames duplicate headers while parsing, so the raw ``csv`` module
    is used first to detect exact duplicate column names.
    """

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.reader(
            handle
        )

        try:
            header = next(
                reader
            )
        except StopIteration as error:
            raise ValueError(
                "Part C engineered dataset is empty."
            ) from error

    if not header:
        raise ValueError(
            "Part C engineered dataset has no header columns."
        )

    normalized_header = tuple(
        str(column).strip()
        for column in header
    )

    if any(
        not column
        for column in normalized_header
    ):
        raise ValueError(
            "Part C engineered dataset contains an empty column name."
        )

    duplicates = sorted(
        {
            column
            for column in normalized_header
            if normalized_header.count(column) > 1
        }
    )

    if duplicates:
        raise ValueError(
            "Part C engineered dataset contains duplicate column names: "
            f"{duplicates}"
        )

    return normalized_header


def _resolve_rule_columns(
    *,
    column_role_rules: dict[str, dict[str, object]],
    dataset_columns: DatasetColumnMapping,
) -> dict[str, tuple[str, ...]]:
    """
    Resolve logical schema rules into physical dataset column names.

    ``dataset_mapping`` names are attributes of DatasetColumnMapping.
    ``fixed_output`` names are stable Part B/Part C output columns.
    """

    resolved: dict[str, tuple[str, ...]] = {}

    for role_name, rule in column_role_rules.items():
        source = rule.get(
            "source"
        )

        if source == "dataset_mapping":
            logical_name = rule.get(
                "logical_name"
            )
            logical_names = rule.get(
                "logical_names"
            )

            if logical_name is not None:
                if not isinstance(
                    logical_name,
                    str,
                ):
                    raise TypeError(
                        f"Invalid logical_name for role '{role_name}'."
                    )

                if not hasattr(
                    dataset_columns,
                    logical_name,
                ):
                    raise ValueError(
                        f"DatasetColumnMapping has no logical field "
                        f"'{logical_name}' required by role '{role_name}'."
                    )

                resolved[
                    role_name
                ] = (
                    str(
                        getattr(
                            dataset_columns,
                            logical_name,
                        )
                    ),
                )

            elif logical_names is not None:
                if not isinstance(
                    logical_names,
                    (tuple, list),
                ):
                    raise TypeError(
                        f"Invalid logical_names for role '{role_name}'."
                    )

                physical_columns = []

                for name in logical_names:
                    if not isinstance(
                        name,
                        str,
                    ):
                        raise TypeError(
                            f"Invalid logical field in role '{role_name}'."
                        )

                    if not hasattr(
                        dataset_columns,
                        name,
                    ):
                        raise ValueError(
                            f"DatasetColumnMapping has no logical field "
                            f"'{name}' required by role '{role_name}'."
                        )

                    physical_columns.append(
                        str(
                            getattr(
                                dataset_columns,
                                name,
                            )
                        )
                    )

                resolved[
                    role_name
                ] = tuple(
                    physical_columns
                )

            else:
                resolved[
                    role_name
                ] = tuple()

        elif source == "fixed_output":
            column_name = rule.get(
                "column_name"
            )

            if not isinstance(
                column_name,
                str,
            ) or not column_name.strip():
                raise ValueError(
                    f"Role '{role_name}' has an invalid fixed output column."
                )

            resolved[
                role_name
            ] = (
                column_name,
            )

        else:
            resolved[
                role_name
            ] = tuple()

    return resolved


def _required_role_columns(
    *,
    column_role_rules: dict[str, dict[str, object]],
    resolved_role_columns: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return every compulsory physical metadata/target column."""

    required: list[str] = []

    for role_name, rule in column_role_rules.items():
        if not bool(
            rule.get(
                "required",
                False,
            )
        ):
            continue

        if rule.get(
            "source"
        ) == "part_c_resolved_features":
            continue

        required.extend(
            resolved_role_columns.get(
                role_name,
                tuple(),
            )
        )

    return tuple(
        dict.fromkeys(
            required
        )
    )


def _validate_required_columns(
    *,
    available_columns: tuple[str, ...],
    required_columns: Iterable[str],
) -> None:
    """Fail when any required physical column is absent."""

    available = set(
        available_columns
    )

    missing = [
        column
        for column in required_columns
        if column not in available
    ]

    if missing:
        raise ValueError(
            "Part C engineered dataset is missing columns required for "
            f"training: {missing}"
        )


def _validate_feature_contract(
    *,
    feature_config: ResolvedFeatureEngineeringConfig,
) -> None:
    """Defensively validate the resolved Part C feature contract."""

    if not feature_config.feature_columns:
        raise ValueError(
            "Resolved Part C feature configuration contains no features."
        )

    if (
        len(
            feature_config.feature_columns
        )
        != len(
            set(
                feature_config.feature_columns
            )
        )
    ):
        raise ValueError(
            "Resolved Part C feature configuration contains duplicate "
            "feature column names."
        )

    if (
        len(
            feature_config.features
        )
        != len(
            feature_config.feature_column_groups
        )
    ):
        raise ValueError(
            "Resolved Part C feature_column_groups does not align with "
            "the feature request list."
        )

    flattened = tuple(
        column
        for group in feature_config.feature_column_groups
        for column in group
    )

    if flattened != tuple(
        feature_config.feature_columns
    ):
        raise ValueError(
            "Resolved Part C feature_column_groups does not flatten exactly "
            "to feature_columns."
        )


def _resolve_xgboost_feature_columns(
    *,
    feature_config: ResolvedFeatureEngineeringConfig,
    rules: XGBoostValidationRules,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    """
    Resolve XGBoost X columns while removing target-leaking feature families.

    Excluded Part C features may remain physically present in the engineered
    dataset; they are simply never included in X.
    """

    excluded_types = set(
        rules.excluded_part_c_feature_types
    )

    model_features: list[str] = []
    excluded_columns: list[str] = []

    for request, output_group in zip(
        feature_config.features,
        feature_config.feature_column_groups,
        strict=True,
    ):
        request_type = str(
            getattr(
                request,
                "type",
                "",
            )
        )

        if request_type in excluded_types:
            excluded_columns.extend(
                output_group
            )
        else:
            model_features.extend(
                output_group
            )

    if not model_features:
        raise ValueError(
            "XGBoost has no usable X features after applying its fixed "
            "target-leakage exclusions."
        )

    return (
        tuple(
            model_features
        ),
        tuple(
            excluded_columns
        ),
    )


def _validate_non_feature_roles_not_in_x(
    *,
    model_feature_columns: tuple[str, ...],
    resolved_role_columns: dict[str, tuple[str, ...]],
    column_role_rules: dict[str, dict[str, object]],
) -> None:
    """Ensure target/metadata columns cannot accidentally enter X."""

    x_columns = set(
        model_feature_columns
    )

    collisions: dict[str, list[str]] = {}

    for role_name, role_columns in resolved_role_columns.items():
        rule = column_role_rules.get(
            role_name,
            {}
        )

        if bool(
            rule.get(
                "use_as_x",
                False,
            )
        ):
            continue

        overlap = sorted(
            x_columns
            & set(
                role_columns
            )
        )

        if overlap:
            collisions[
                role_name
            ] = overlap

    if collisions:
        raise ValueError(
            "Model feature columns collide with columns that are fixed as "
            f"non-features: {collisions}"
        )


def _validate_direction_feature(
    *,
    training_config: IsolationForestTrainingConfig,
    feature_config: ResolvedFeatureEngineeringConfig,
    rules: IsolationForestValidationRules,
) -> None:
    """
    Require the configured directional feature only for one-sided IF output.

    Isolation Forest's own anomaly score has no upper/lower KPI direction.
    """

    direction_rule = rules.direction_feature_rule

    feature_name = direction_rule.get(
        "feature_name"
    )

    required_directions = direction_rule.get(
        "required_for_threshold_directions",
        tuple(),
    )

    if not isinstance(
        feature_name,
        str,
    ) or not feature_name:
        raise ValueError(
            "Isolation Forest direction feature rule is invalid."
        )

    if training_config.threshold_direction not in set(
        required_directions
    ):
        return

    if feature_name not in set(
        feature_config.feature_columns
    ):
        raise ValueError(
            "Isolation Forest threshold_direction="
            f"'{training_config.threshold_direction}' requires the Part C "
            f"feature '{feature_name}', but it is not present in the "
            "resolved feature configuration."
        )


def _scan_dataset_types(
    *,
    path: Path,
    required_columns: tuple[str, ...],
    numeric_feature_columns: tuple[str, ...],
    target_column: str | None,
    timestamp_column: str,
    site_identity_columns: tuple[str, ...],
    status_column: str,
) -> int:
    """
    Perform a memory-bounded CSV scan for basic data-type integrity.

    NaN feature/target values are allowed here because history warm-up and
    unresolved rows may legitimately contain them. Later model stages apply
    complete-row/trusted-status policies. Invalid non-numeric text and
    infinities are rejected.
    """

    use_columns = tuple(
        dict.fromkeys(
            required_columns
        )
    )

    total_rows = 0

    reader = pd.read_csv(
        path,
        usecols=list(
            use_columns
        ),
        chunksize=VALIDATION_READ_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        total_rows += len(
            chunk
        )

        if chunk.empty:
            continue

        # --------------------------------------------------------------
        # Site identity
        # --------------------------------------------------------------

        missing_identity = chunk[
            list(
                site_identity_columns
            )
        ].isna().any(axis=1)

        if bool(
            missing_identity.any()
        ):
            raise ValueError(
                "Part C engineered dataset contains missing site identity "
                f"values in validation chunk {chunk_number}."
            )

        # --------------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------------

        parsed_time = pd.to_datetime(
            chunk[
                timestamp_column
            ],
            errors="coerce",
        )

        if bool(
            parsed_time.isna().any()
        ):
            raise ValueError(
                "Part C engineered dataset contains invalid or missing "
                f"timestamps in validation chunk {chunk_number}."
            )

        # --------------------------------------------------------------
        # Model features + optional target
        # --------------------------------------------------------------

        numeric_columns = list(
            numeric_feature_columns
        )

        if target_column is not None:
            numeric_columns.append(
                target_column
            )

        numeric_columns.append(
            status_column
        )

        numeric_columns = list(
            dict.fromkeys(
                numeric_columns
            )
        )

        for column in numeric_columns:
            raw = chunk[
                column
            ]

            converted = pd.to_numeric(
                raw,
                errors="coerce",
            )

            invalid_text = (
                converted.isna()
                & raw.notna()
            )

            if bool(
                invalid_text.any()
            ):
                raise ValueError(
                    f"Column '{column}' contains non-numeric values in "
                    f"validation chunk {chunk_number}."
                )

            finite_values = converted.dropna()

            if (
                not finite_values.empty
                and bool(
                    np.isinf(
                        finite_values.to_numpy(
                            dtype=np.float64
                        )
                    ).any()
                )
            ):
                raise ValueError(
                    f"Column '{column}' contains infinite values in "
                    f"validation chunk {chunk_number}."
                )

        # Status is compulsory row metadata and should exist for every row.
        numeric_status = pd.to_numeric(
            chunk[
                status_column
            ],
            errors="coerce",
        )

        if bool(
            numeric_status.isna().any()
        ):
            raise ValueError(
                f"Status column '{status_column}' contains missing values "
                f"in validation chunk {chunk_number}."
            )

        fractional_status = (
            numeric_status
            % 1
            != 0
        )

        if bool(
            fractional_status.any()
        ):
            raise ValueError(
                f"Status column '{status_column}' must contain integer "
                "status labels."
            )

    if total_rows <= 0:
        raise ValueError(
            "Part C engineered dataset contains no data rows."
        )

    return total_rows


def _build_common_required_columns(
    *,
    feature_config: ResolvedFeatureEngineeringConfig,
    role_required_columns: tuple[str, ...],
) -> tuple[str, ...]:
    """Combine Part C feature columns with required target/metadata columns."""

    return tuple(
        dict.fromkeys(
            (
                *feature_config.feature_columns,
                *role_required_columns,
            )
        )
    )


def _validate_xgboost(
    *,
    path: Path,
    available_columns: tuple[str, ...],
    dataset_columns: DatasetColumnMapping,
    feature_config: ResolvedFeatureEngineeringConfig,
    training_config: XGBoostTrainingConfig,
) -> TrainingValidationResult:
    """Validate the Part C dataset against XGBoost-specific rules."""

    rules = training_config.get_validation_rules()

    resolved_roles = _resolve_rule_columns(
        column_role_rules=rules.column_role_rules,
        dataset_columns=dataset_columns,
    )

    role_required_columns = _required_role_columns(
        column_role_rules=rules.column_role_rules,
        resolved_role_columns=resolved_roles,
    )

    required_columns = _build_common_required_columns(
        feature_config=feature_config,
        role_required_columns=role_required_columns,
    )

    _validate_required_columns(
        available_columns=available_columns,
        required_columns=required_columns,
    )

    model_features, excluded_features = (
        _resolve_xgboost_feature_columns(
            feature_config=feature_config,
            rules=rules,
        )
    )

    target_columns = resolved_roles.get(
        "target",
        tuple(),
    )

    if len(
        target_columns
    ) != 1:
        raise ValueError(
            "XGBoost requires exactly one resolved target column."
        )

    target_column = target_columns[
        0
    ]

    site_identity_columns = resolved_roles.get(
        "site_identity",
        tuple(),
    )

    timestamp_columns = resolved_roles.get(
        "timestamp",
        tuple(),
    )

    status_columns = resolved_roles.get(
        "status",
        tuple(),
    )

    usability_columns = resolved_roles.get(
        "usability",
        tuple(),
    )

    invalid_reason_columns = resolved_roles.get(
        "invalid_reason",
        tuple(),
    )

    if len(
        site_identity_columns
    ) != 3:
        raise ValueError(
            "XGBoost requires exactly three resolved site identity columns."
        )

    if (
        len(timestamp_columns) != 1
        or len(status_columns) != 1
        or len(usability_columns) != 1
        or len(invalid_reason_columns) != 1
    ):
        raise ValueError(
            "XGBoost metadata column-role resolution is invalid."
        )

    _validate_non_feature_roles_not_in_x(
        model_feature_columns=model_features,
        resolved_role_columns=resolved_roles,
        column_role_rules=rules.column_role_rules,
    )

    total_rows = _scan_dataset_types(
        path=path,
        required_columns=required_columns,
        numeric_feature_columns=tuple(
            feature_config.feature_columns
        ),
        target_column=target_column,
        timestamp_column=timestamp_columns[0],
        site_identity_columns=site_identity_columns,
        status_column=status_columns[0],
    )

    return TrainingValidationResult(
        input_file=path,
        model_type="xgboost",
        all_part_c_feature_columns=tuple(
            feature_config.feature_columns
        ),
        model_feature_columns=model_features,
        excluded_part_c_feature_columns=excluded_features,
        target_column=target_column,
        site_identity_columns=site_identity_columns,
        timestamp_column=timestamp_columns[0],
        status_column=status_columns[0],
        usability_column=usability_columns[0],
        invalid_reason_column=invalid_reason_columns[0],
        minimum_history_points=feature_config.minimum_history_points,
        trusted_statuses=rules.trusted_target_statuses,
        require_complete_feature_rows=rules.require_complete_feature_rows,
        require_non_null_target=rules.require_non_null_target,
        minimum_train_points=rules.minimum_train_points,
        minimum_validation_points=rules.minimum_validation_points,
        minimum_calibration_points=rules.minimum_calibration_points,
        minimum_test_points=rules.minimum_test_points,
        total_rows=total_rows,
    )


def _validate_isolation_forest(
    *,
    path: Path,
    available_columns: tuple[str, ...],
    dataset_columns: DatasetColumnMapping,
    feature_config: ResolvedFeatureEngineeringConfig,
    training_config: IsolationForestTrainingConfig,
) -> TrainingValidationResult:
    """Validate the Part C dataset against Isolation Forest-specific rules."""

    rules = training_config.get_validation_rules()

    resolved_roles = _resolve_rule_columns(
        column_role_rules=rules.column_role_rules,
        dataset_columns=dataset_columns,
    )

    role_required_columns = _required_role_columns(
        column_role_rules=rules.column_role_rules,
        resolved_role_columns=resolved_roles,
    )

    required_columns = _build_common_required_columns(
        feature_config=feature_config,
        role_required_columns=role_required_columns,
    )

    _validate_required_columns(
        available_columns=available_columns,
        required_columns=required_columns,
    )

    model_features = tuple(
        feature_config.feature_columns
    )

    if not model_features:
        raise ValueError(
            "Isolation Forest requires at least one resolved Part C feature."
        )

    _validate_direction_feature(
        training_config=training_config,
        feature_config=feature_config,
        rules=rules,
    )

    site_identity_columns = resolved_roles.get(
        "site_identity",
        tuple(),
    )

    timestamp_columns = resolved_roles.get(
        "timestamp",
        tuple(),
    )

    status_columns = resolved_roles.get(
        "status",
        tuple(),
    )

    usability_columns = resolved_roles.get(
        "usability",
        tuple(),
    )

    invalid_reason_columns = resolved_roles.get(
        "invalid_reason",
        tuple(),
    )

    if len(
        site_identity_columns
    ) != 3:
        raise ValueError(
            "Isolation Forest requires exactly three resolved site identity "
            "columns."
        )

    if (
        len(timestamp_columns) != 1
        or len(status_columns) != 1
        or len(usability_columns) != 1
        or len(invalid_reason_columns) != 1
    ):
        raise ValueError(
            "Isolation Forest metadata column-role resolution is invalid."
        )

    _validate_non_feature_roles_not_in_x(
        model_feature_columns=model_features,
        resolved_role_columns=resolved_roles,
        column_role_rules=rules.column_role_rules,
    )

    total_rows = _scan_dataset_types(
        path=path,
        required_columns=required_columns,
        numeric_feature_columns=model_features,
        target_column=None,
        timestamp_column=timestamp_columns[0],
        site_identity_columns=site_identity_columns,
        status_column=status_columns[0],
    )

    minimum_validation_points = (
        rules.minimum_validation_points
        if training_config.hyperparameter_tuning_enabled
        else 0
    )

    return TrainingValidationResult(
        input_file=path,
        model_type="isolation_forest",
        all_part_c_feature_columns=tuple(
            feature_config.feature_columns
        ),
        model_feature_columns=model_features,
        excluded_part_c_feature_columns=tuple(),
        target_column=None,
        site_identity_columns=site_identity_columns,
        timestamp_column=timestamp_columns[0],
        status_column=status_columns[0],
        usability_column=usability_columns[0],
        invalid_reason_column=invalid_reason_columns[0],
        minimum_history_points=feature_config.minimum_history_points,
        trusted_statuses=rules.trusted_train_statuses,
        require_complete_feature_rows=rules.require_complete_feature_rows,
        require_non_null_target=False,
        minimum_train_points=rules.minimum_train_points,
        minimum_validation_points=minimum_validation_points,
        minimum_calibration_points=0,
        minimum_test_points=rules.minimum_test_points,
        total_rows=total_rows,
    )


def validate_training_dataset(
    *,
    model_type: TrainingModelType,
    input_file: str | Path,
    dataset_columns: DatasetColumnMapping,
    feature_config: ResolvedFeatureEngineeringConfig,
    training_config: TrainingConfig,
) -> TrainingValidationResult:
    """
    Validate one Part C engineered dataset for the selected Part D model.

    The explicit model switch is intentional. Adding a future algorithm
    requires adding its schema/trainer and a new dispatch branch here.
    """

    path = _normalise_input_file(
        input_file
    )

    _validate_feature_contract(
        feature_config=feature_config
    )

    available_columns = _read_csv_header(
        path
    )

    if model_type == "xgboost":
        if not isinstance(
            training_config,
            XGBoostTrainingConfig,
        ):
            raise TypeError(
                "model_type='xgboost' requires XGBoostTrainingConfig."
            )

        return _validate_xgboost(
            path=path,
            available_columns=available_columns,
            dataset_columns=dataset_columns,
            feature_config=feature_config,
            training_config=training_config,
        )

    if model_type == "isolation_forest":
        if not isinstance(
            training_config,
            IsolationForestTrainingConfig,
        ):
            raise TypeError(
                "model_type='isolation_forest' requires "
                "IsolationForestTrainingConfig."
            )

        return _validate_isolation_forest(
            path=path,
            available_columns=available_columns,
            dataset_columns=dataset_columns,
            feature_config=feature_config,
            training_config=training_config,
        )

    raise ValueError(
        f"Unsupported training model_type: {model_type}"
    )


__all__ = [
    "TrainingValidationResult",
    "validate_training_dataset",
]