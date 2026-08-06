"""Validation for Part C feature engineering.

Revision: feature-only output validation fixed.

Intended project location::

    app/ml/feature_engineering/feature_validation.py

This module validates the Part B dataset before feature generation and verifies
that the final feature-enriched dataset preserves the agreed contract.

It does not choose features, calculate features, clean KPI history, or save
files. Those responsibilities belong to the schema, feature modules, source
preparation, and runner respectively.

Validation policy
-----------------

- Parts A and B own timeline correction and missing-value processing.
- Part C may normalize CSV dtypes, but it never repairs timestamps, duplicate
  rows, irregular timelines, labels, or KPI values.
- Sites whose ``kpi_usability`` is ``invalid`` are retained in the output but
  skipped for feature generation.
- Valid sites must have unique timestamps and exact resolved bucket spacing.
- Insufficient history is reported per site; it is not a validation failure.
  Valid sites are still passed to feature functions, which may naturally produce
  warm-up NaN values when history is unavailable.
- Warm-up ``NaN`` values in generated features are valid. Infinite values are
  never valid feature outputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from .common import (
    FeatureInputColumns,
    iter_site_groups,
    require_columns,
    resolve_site_usability,
)


DEFAULT_USABILITY_COLUMN = "kpi_usability"
DEFAULT_STATUS_COLUMN = "Current_Status_Label"

ALLOWED_USABILITY_VALUES = frozenset({"valid", "invalid"})
ALLOWED_STATUS_VALUES = frozenset({-2, -1, 0, 1, 2, 3, 4})
UNRESOLVED_STATUS_VALUES = frozenset({-2, -1, 0})
VALUE_PRESENT_STATUS_VALUES = frozenset({1, 2, 3, 4})


class ResolvedFeatureConfigProtocol(Protocol):
    """Resolved schema fields required by feature validation."""

    bucket_minutes: int
    minimum_history_points: int
    feature_columns: Sequence[str]
    requires_status_column: bool
    retain_original_columns: bool


@dataclass(frozen=True, slots=True)
class SiteFeatureValidationResult:
    """Validation and feature-readiness result for one site."""

    composite_id: str
    row_count: int
    usability: str
    minimum_history_points: int
    has_sufficient_history: bool
    can_generate_features: bool
    skip_reason: str | None
    minimum_timestamp: pd.Timestamp | None
    maximum_timestamp: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class FeatureDatasetValidationResult:
    """Aggregate result returned to the feature runner."""

    row_count: int
    total_sites: int
    valid_sites: int
    invalid_sites: int
    processable_sites: int
    history_sufficient_sites: int
    insufficient_history_sites: int
    minimum_timestamp: pd.Timestamp
    maximum_timestamp: pd.Timestamp
    site_results: tuple[SiteFeatureValidationResult, ...]


# ---------------------------------------------------------------------------
# Input type normalization
# ---------------------------------------------------------------------------


def _nonblank_input_mask(series: pd.Series) -> pd.Series:
    """Return rows containing a supplied value rather than a blank/missing one."""

    text = series.astype("string")
    return series.notna() & text.str.strip().ne("")


def _normalize_timestamp_column(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str,
) -> None:
    """Parse one timestamp column in place without repairing invalid values."""

    original = dataframe[timestamp_column]
    parsed = pd.to_datetime(original, errors="coerce")

    invalid_mask = _nonblank_input_mask(original) & parsed.isna()
    missing_mask = ~_nonblank_input_mask(original)

    invalid_count = int(invalid_mask.sum())
    missing_count = int(missing_mask.sum())

    if invalid_count or missing_count:
        raise ValueError(
            f"Timestamp column '{timestamp_column}' contains "
            f"{invalid_count:,} invalid and {missing_count:,} missing values. "
            "Part C validates the timeline but does not repair timestamps."
        )

    if not is_datetime64_any_dtype(parsed.dtype):
        raise ValueError(
            f"Timestamp column '{timestamp_column}' could not be normalized "
            "to one consistent pandas datetime dtype. Mixed time zones or "
            "incompatible timestamp formats may be present."
        )

    dataframe[timestamp_column] = parsed


def _normalize_numeric_column(
    dataframe: pd.DataFrame,
    *,
    column: str,
    allow_missing: bool,
) -> None:
    """Convert a numeric column in place and reject supplied non-numeric text."""

    original = dataframe[column]
    numeric = pd.to_numeric(original, errors="coerce")

    invalid_mask = _nonblank_input_mask(original) & numeric.isna()
    invalid_count = int(invalid_mask.sum())

    if invalid_count:
        examples = original.loc[invalid_mask].head(5).tolist()
        raise ValueError(
            f"Column '{column}' contains {invalid_count:,} non-numeric values. "
            f"Examples: {examples}"
        )

    if not allow_missing and numeric.isna().any():
        raise ValueError(
            f"Column '{column}' contains {int(numeric.isna().sum()):,} missing "
            "values, but this column cannot be missing."
        )

    finite_values = numeric.dropna().to_numpy(dtype="float64")
    if finite_values.size and not np.isfinite(finite_values).all():
        raise ValueError(
            f"Column '{column}' contains positive or negative infinity."
        )

    dataframe[column] = numeric.astype("float64")


def _normalize_status_column(
    dataframe: pd.DataFrame,
    *,
    status_column: str,
) -> None:
    """Normalize and validate the Part B current-status label column."""

    original = dataframe[status_column]
    numeric = pd.to_numeric(original, errors="coerce")

    invalid_mask = _nonblank_input_mask(original) & numeric.isna()
    if invalid_mask.any():
        examples = original.loc[invalid_mask].head(5).tolist()
        raise ValueError(
            f"Status column '{status_column}' contains "
            f"{int(invalid_mask.sum()):,} non-numeric values. Examples: "
            f"{examples}"
        )

    if numeric.isna().any():
        raise ValueError(
            f"Status column '{status_column}' contains "
            f"{int(numeric.isna().sum()):,} missing labels."
        )

    fractional_mask = numeric.mod(1).ne(0)
    if fractional_mask.any():
        examples = numeric.loc[fractional_mask].head(5).tolist()
        raise ValueError(
            f"Status column '{status_column}' must contain integer labels. "
            f"Examples: {examples}"
        )

    integer_status = numeric.astype("int16")
    unsupported = sorted(
        set(integer_status.unique().tolist()) - ALLOWED_STATUS_VALUES
    )

    if unsupported:
        raise ValueError(
            f"Status column '{status_column}' contains unsupported labels: "
            f"{unsupported}. Allowed labels: {sorted(ALLOWED_STATUS_VALUES)}"
        )

    dataframe[status_column] = integer_status.astype("Int8")


def normalize_feature_input_dataframe(
    dataframe: pd.DataFrame,
    *,
    input_columns: FeatureInputColumns,
    status_column: str = DEFAULT_STATUS_COLUMN,
    status_required: bool,
) -> pd.DataFrame:
    """Return a copied dataframe with calculation-safe Part C dtypes.

    This performs representation normalization only. It does not fill values,
    change labels, remove duplicates, insert timestamps, or alter site IDs.
    """

    normalized = dataframe.copy()

    _normalize_timestamp_column(
        normalized,
        timestamp_column=input_columns.timestamp_column,
    )
    _normalize_numeric_column(
        normalized,
        column=input_columns.value_column,
        allow_missing=True,
    )

    if status_required or status_column in normalized.columns:
        _normalize_status_column(
            normalized,
            status_column=status_column,
        )

    return normalized


# ---------------------------------------------------------------------------
# Dataset and site validation
# ---------------------------------------------------------------------------


def _validate_dataframe_columns(
    dataframe: pd.DataFrame,
    *,
    input_columns: FeatureInputColumns,
    resolved_config: ResolvedFeatureConfigProtocol,
    usability_column: str,
    status_column: str,
) -> None:
    """Validate required inputs and reject output-column collisions early."""

    if dataframe.empty:
        raise ValueError("Feature-engineering input dataframe is empty.")

    if dataframe.columns.has_duplicates:
        duplicates = sorted(
            set(
                dataframe.columns[
                    dataframe.columns.duplicated()
                ].tolist()
            )
        )
        raise ValueError(
            f"Input dataframe contains duplicate column names: {duplicates}"
        )

    required = [
        *input_columns.required_columns,
        usability_column,
    ]

    if resolved_config.requires_status_column:
        required.append(status_column)

    require_columns(
        dataframe,
        required,
        context="feature-engineering input validation",
    )

    collisions = sorted(
        set(resolved_config.feature_columns) & set(dataframe.columns)
    )

    if collisions:
        raise ValueError(
            "Requested feature columns already exist in the Part B dataset: "
            f"{collisions}. Part C will not overwrite existing columns."
        )


def _validate_usability_column(
    dataframe: pd.DataFrame,
    *,
    usability_column: str,
) -> None:
    """Require one supported usability label on every input row."""

    normalized = (
        dataframe[usability_column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    if normalized.isna().any() or normalized.eq("").any():
        raise ValueError(
            f"Column '{usability_column}' contains missing or blank values."
        )

    unsupported = sorted(
        set(normalized.unique().tolist()) - ALLOWED_USABILITY_VALUES
    )

    if unsupported:
        raise ValueError(
            f"Column '{usability_column}' contains unsupported values: "
            f"{unsupported}. Allowed values: "
            f"{sorted(ALLOWED_USABILITY_VALUES)}"
        )

    dataframe[usability_column] = normalized


def _validate_status_value_consistency(
    dataframe: pd.DataFrame,
    *,
    value_column: str,
    status_column: str,
) -> None:
    """Check the Part B value/status contract used by feature cleaning."""

    values = dataframe[value_column]
    statuses = dataframe[status_column]

    missing_required_value = (
        statuses.isin(VALUE_PRESENT_STATUS_VALUES)
        & values.isna()
    )

    if missing_required_value.any():
        examples = dataframe.loc[
            missing_required_value,
            [status_column, value_column],
        ].head(5).to_dict("records")
        raise ValueError(
            f"{int(missing_required_value.sum()):,} rows have status in "
            f"{sorted(VALUE_PRESENT_STATUS_VALUES)} but no KPI value. "
            f"Examples: {examples}"
        )

    unexpected_value = (
        statuses.isin(UNRESOLVED_STATUS_VALUES)
        & values.notna()
    )

    if unexpected_value.any():
        examples = dataframe.loc[
            unexpected_value,
            [status_column, value_column],
        ].head(5).to_dict("records")
        raise ValueError(
            f"{int(unexpected_value.sum()):,} rows have unresolved status in "
            f"{sorted(UNRESOLVED_STATUS_VALUES)} but still contain a KPI "
            f"value. Examples: {examples}"
        )


def validate_site_timeline(
    site_dataframe: pd.DataFrame,
    *,
    composite_id: str,
    timestamp_column: str,
    bucket_minutes: int,
) -> None:
    """Validate one valid site's unique, regular, chronological timeline."""

    timestamps = site_dataframe[timestamp_column]

    duplicate_count = int(timestamps.duplicated().sum())
    if duplicate_count:
        duplicate_examples = (
            timestamps.loc[timestamps.duplicated(keep=False)]
            .head(5)
            .astype("string")
            .tolist()
        )
        raise ValueError(
            f"Valid site '{composite_id}' contains {duplicate_count:,} "
            f"duplicate timestamps. Examples: {duplicate_examples}"
        )

    if len(site_dataframe) <= 1:
        return

    expected = pd.Timedelta(minutes=bucket_minutes)
    differences = timestamps.diff().dropna()
    invalid = differences.ne(expected)

    if invalid.any():
        example_indexes = differences.index[invalid][:5]
        examples = [
            {
                "time": str(timestamps.loc[index]),
                "difference": str(differences.loc[index]),
            }
            for index in example_indexes
        ]
        raise ValueError(
            f"Valid site '{composite_id}' has an irregular timeline. "
            f"Expected every difference to equal {expected}. Examples: "
            f"{examples}"
        )


def validate_site_for_feature_generation(
    site_dataframe: pd.DataFrame,
    *,
    composite_id: str,
    timestamp_column: str,
    usability_column: str,
    bucket_minutes: int,
    minimum_history_points: int,
) -> SiteFeatureValidationResult:
    """Validate one site and report whether the runner should process it."""

    usability = resolve_site_usability(
        site_dataframe,
        usability_column=usability_column,
    )

    timestamps = site_dataframe[timestamp_column]
    minimum_timestamp = (
        timestamps.min()
        if not timestamps.empty
        else None
    )
    maximum_timestamp = (
        timestamps.max()
        if not timestamps.empty
        else None
    )

    if usability == "invalid":
        return SiteFeatureValidationResult(
            composite_id=composite_id,
            row_count=len(site_dataframe),
            usability=usability,
            minimum_history_points=minimum_history_points,
            has_sufficient_history=(
                len(site_dataframe) >= minimum_history_points
            ),
            can_generate_features=False,
            skip_reason="invalid_site",
            minimum_timestamp=minimum_timestamp,
            maximum_timestamp=maximum_timestamp,
        )

    validate_site_timeline(
        site_dataframe,
        composite_id=composite_id,
        timestamp_column=timestamp_column,
        bucket_minutes=bucket_minutes,
    )

    has_sufficient_history = (
        len(site_dataframe) >= minimum_history_points
    )

    return SiteFeatureValidationResult(
        composite_id=composite_id,
        row_count=len(site_dataframe),
        usability=usability,
        minimum_history_points=minimum_history_points,
        has_sufficient_history=has_sufficient_history,
        can_generate_features=True,
        skip_reason=None,
        minimum_timestamp=minimum_timestamp,
        maximum_timestamp=maximum_timestamp,
    )


def prepare_and_validate_feature_dataframe(
    dataframe: pd.DataFrame,
    *,
    input_columns: FeatureInputColumns,
    resolved_config: ResolvedFeatureConfigProtocol,
    usability_column: str = DEFAULT_USABILITY_COLUMN,
    status_column: str = DEFAULT_STATUS_COLUMN,
) -> tuple[pd.DataFrame, FeatureDatasetValidationResult]:
    """Normalize and validate the complete Part B input for the runner."""

    _validate_dataframe_columns(
        dataframe,
        input_columns=input_columns,
        resolved_config=resolved_config,
        usability_column=usability_column,
        status_column=status_column,
    )

    normalized = normalize_feature_input_dataframe(
        dataframe,
        input_columns=input_columns,
        status_column=status_column,
        status_required=resolved_config.requires_status_column,
    )

    _validate_usability_column(
        normalized,
        usability_column=usability_column,
    )

    if status_column in normalized.columns:
        _validate_status_value_consistency(
            normalized,
            value_column=input_columns.value_column,
            status_column=status_column,
        )

    site_results: list[SiteFeatureValidationResult] = []

    for site_group in iter_site_groups(
        normalized,
        site_identity_columns=input_columns.site_identity_columns,
        timestamp_column=input_columns.timestamp_column,
    ):
        site_results.append(
            validate_site_for_feature_generation(
                site_group.dataframe,
                composite_id=site_group.composite_id,
                timestamp_column=input_columns.timestamp_column,
                usability_column=usability_column,
                bucket_minutes=resolved_config.bucket_minutes,
                minimum_history_points=(
                    resolved_config.minimum_history_points
                ),
            )
        )

    if not site_results:
        raise ValueError(
            "Feature-engineering input contains no site groups."
        )

    valid_sites = sum(
        result.usability == "valid"
        for result in site_results
    )
    invalid_sites = sum(
        result.usability == "invalid"
        for result in site_results
    )
    processable_sites = sum(
        result.can_generate_features
        for result in site_results
    )
    history_sufficient_sites = sum(
        result.usability == "valid" and result.has_sufficient_history
        for result in site_results
    )
    insufficient_history_sites = sum(
        result.usability == "valid" and not result.has_sufficient_history
        for result in site_results
    )

    result = FeatureDatasetValidationResult(
        row_count=len(normalized),
        total_sites=len(site_results),
        valid_sites=valid_sites,
        invalid_sites=invalid_sites,
        processable_sites=processable_sites,
        history_sufficient_sites=history_sufficient_sites,
        insufficient_history_sites=insufficient_history_sites,
        minimum_timestamp=normalized[
            input_columns.timestamp_column
        ].min(),
        maximum_timestamp=normalized[
            input_columns.timestamp_column
        ].max(),
        site_results=tuple(site_results),
    )

    return normalized, result


# ---------------------------------------------------------------------------
# Post-generation validation
# ---------------------------------------------------------------------------


def validate_generated_feature_dataframe(
    *,
    source_dataframe: pd.DataFrame,
    feature_dataframe: pd.DataFrame,
    input_columns: FeatureInputColumns,
    resolved_config: ResolvedFeatureConfigProtocol,
    usability_column: str = DEFAULT_USABILITY_COLUMN,
) -> None:
    """Validate the runner's completed feature-enriched dataframe."""

    if len(feature_dataframe) != len(source_dataframe):
        raise ValueError(
            "Feature generation changed the dataset row count: "
            f"source={len(source_dataframe):,}, "
            f"output={len(feature_dataframe):,}."
        )

    if feature_dataframe.columns.has_duplicates:
        duplicates = sorted(
            set(
                feature_dataframe.columns[
                    feature_dataframe.columns.duplicated()
                ].tolist()
            )
        )
        raise ValueError(
            f"Feature output contains duplicate columns: {duplicates}"
        )

    require_columns(
        feature_dataframe,
        resolved_config.feature_columns,
        context="generated feature output",
    )

    if resolved_config.retain_original_columns:
        require_columns(
            feature_dataframe,
            source_dataframe.columns.tolist(),
            context="retained Part B output",
        )

        source_sorted = source_dataframe.sort_values(
            [
                *input_columns.site_identity_columns,
                input_columns.timestamp_column,
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        output_sorted = feature_dataframe.sort_values(
            [
                *input_columns.site_identity_columns,
                input_columns.timestamp_column,
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        try:
            pd.testing.assert_frame_equal(
                output_sorted[source_dataframe.columns],
                source_sorted[source_dataframe.columns],
                check_dtype=False,
                check_like=False,
            )
        except AssertionError as error:
            raise ValueError(
                "Feature generation modified one or more original Part B "
                "columns. Original columns must remain unchanged."
            ) from error

    for feature_column in resolved_config.feature_columns:
        numeric = pd.to_numeric(
            feature_dataframe[feature_column],
            errors="coerce",
        )

        invalid_numeric = (
            _nonblank_input_mask(feature_dataframe[feature_column])
            & numeric.isna()
        )
        if invalid_numeric.any():
            examples = (
                feature_dataframe.loc[
                    invalid_numeric,
                    feature_column,
                ]
                .head(5)
                .tolist()
            )
            raise ValueError(
                f"Generated feature '{feature_column}' contains "
                f"{int(invalid_numeric.sum()):,} non-numeric values. "
                f"Examples: {examples}"
            )

        finite_values = numeric.dropna().to_numpy(dtype="float64")
        if finite_values.size and not np.isfinite(finite_values).all():
            raise ValueError(
                f"Generated feature '{feature_column}' contains positive or "
                "negative infinity. Division-by-zero outputs must become NaN."
            )

    if usability_column in feature_dataframe.columns:
        normalized_usability = (
            feature_dataframe[usability_column]
            .astype("string")
            .str.strip()
            .str.lower()
        )
    else:
        # Feature-only output is valid when retain_original_columns=False.
        # In that mode the output has no usability column, so use the source
        # labels. Exact index preservation is required to align those labels
        # safely with the generated feature rows.
        if resolved_config.retain_original_columns:
            raise ValueError(
                f"Feature output is missing required retained column "
                f"'{usability_column}'."
            )

        if not feature_dataframe.index.equals(source_dataframe.index):
            raise ValueError(
                "Feature-only output must preserve the source dataframe "
                "index so invalid-site rows can be validated safely when "
                "retain_original_columns=False."
            )

        normalized_usability = (
            source_dataframe[usability_column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    invalid_site_rows = normalized_usability.eq("invalid")

    if invalid_site_rows.any():
        populated_invalid_features = feature_dataframe.loc[
            invalid_site_rows,
            list(resolved_config.feature_columns),
        ].notna().any(axis=1)

        if populated_invalid_features.any():
            raise ValueError(
                "Feature values were generated for rows belonging to sites "
                "marked kpi_usability='invalid'. Invalid sites must be "
                "retained but skipped, leaving requested feature columns NaN."
            )


__all__ = [
    "ALLOWED_STATUS_VALUES",    
    "ALLOWED_USABILITY_VALUES",
    "DEFAULT_STATUS_COLUMN",
    "DEFAULT_USABILITY_COLUMN",
    "FeatureDatasetValidationResult",
    "ResolvedFeatureConfigProtocol",
    "SiteFeatureValidationResult",
    "normalize_feature_input_dataframe",
    "prepare_and_validate_feature_dataframe",
    "validate_generated_feature_dataframe",
    "validate_site_for_feature_generation",
    "validate_site_timeline",
]