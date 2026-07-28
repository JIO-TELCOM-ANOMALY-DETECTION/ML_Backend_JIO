"""
Fill original Phase 2 gaps and retry unresolved Phase 1 points.

This module runs after ``missing_value_phase_1.py``. It expects the dataset
shape produced by Phase 1 and uses the immutable ``Original_Gap_Phase`` column
created by ``missing_gap_analysis.py``.

Eligibility
-----------

Phase 2 processes:

- primary targets: unresolved rows with ``Original_Gap_Phase == 2``;
- retry targets: unresolved rows with ``Original_Gap_Phase == 1``.

Rows already filled safely by Phase 1 retain ``Current_Status_Label == 3`` and
are not filled again. Invalid site series are skipped.

Trusted references
------------------

A historical reference is trusted only when:

- it belongs to the same site;
- ``Current_Status_Label`` is ``1`` or ``3``;
- its working KPI value is finite; and
- its working KPI value lies inside the configured valid range.

Labels ``2``, ``4``, ``0``, ``-1``, and ``-2`` are never references.
Previous safe Phase 1 fills and earlier Phase 2 fills may be reused.

Seasonal references
-------------------

For every unresolved target timestamp ``T``:

1. Recent same-type day:
   search backward by calendar day up to the configured limit and select the
   nearest earlier timestamp having the same weekday/weekend type as ``T``.

2. Last-week reference:
   use exactly ``T - configured_last_week_days``.

3. Neighbour reference:
   inspect the configured offset before and after a seasonal reference. Use
   their mean when both are trusted, or the single trusted value when only one
   is available.

Filling ladder
--------------

1. recent exact + last-week exact;
2. recent neighbours + last-week exact;
3. recent exact + last-week neighbours;
4. recent neighbours + last-week neighbours;
5. recent exact only;
6. last-week exact only;
7. recent neighbours only;
8. last-week neighbours only;
9. failure: leave the point unchanged for Phase 3.

Successful fills are past-only and therefore receive status label ``3``.
This module never assigns label ``4``.

Only these fields may change:

- the configured KPI value column;
- ``Current_Status_Label``; and
- ``Fill_Method``.

All other input columns, row count, and row order are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
)
from app.ml.eda.data_integrity import (
    COMPOSITE_ID_COLUMN,
    COMPOSITE_ID_SEPARATOR,
    MISSING_ID_VALUE,
)
from app.schemas.common.dataset import DatasetColumnMapping
from app.schemas.preprocessing import (
    PreprocessingConfig,
    ResolvedFrequencyConfig,
)
from app.services.file_service import (
    ensure_directory,
    read_csv_file,
    sanitise_filename,
    write_csv_file,
)

from .missing_gap_analysis import (
    MISSING_ORIGINAL_LABELS,
    ORIGINAL_GAP_PHASE_COLUMN,
    PHASE_1,
    PHASE_2,
)
from .missing_value_phase_1 import FILL_METHOD_COLUMN
from .timeline_correction import (
    CURRENT_STATUS_LABEL_COLUMN,
    INVALID_USABILITY_VALUE,
    KPI_USABILITY_COLUMN,
    ORIGINAL_CONDITION_LABEL_COLUMN,
    VALID_USABILITY_VALUE,
)


METHOD_RECENT_EXACT_LASTWEEK_EXACT = (
    "phase2_recent_exact_lastweek_exact"
)
METHOD_RECENT_NEIGHBORS_LASTWEEK_EXACT = (
    "phase2_recent_neighbors_lastweek_exact"
)
METHOD_RECENT_EXACT_LASTWEEK_NEIGHBORS = (
    "phase2_recent_exact_lastweek_neighbors"
)
METHOD_RECENT_NEIGHBORS_LASTWEEK_NEIGHBORS = (
    "phase2_recent_neighbors_lastweek_neighbors"
)
METHOD_RECENT_EXACT_ONLY = "phase2_recent_exact_only"
METHOD_LASTWEEK_EXACT_ONLY = "phase2_lastweek_exact_only"
METHOD_RECENT_NEIGHBORS_ONLY = "phase2_recent_neighbors_only"
METHOD_LASTWEEK_NEIGHBORS_ONLY = (
    "phase2_lastweek_neighbors_only"
)

PHASE_2_METHODS = {
    METHOD_RECENT_EXACT_LASTWEEK_EXACT,
    METHOD_RECENT_NEIGHBORS_LASTWEEK_EXACT,
    METHOD_RECENT_EXACT_LASTWEEK_NEIGHBORS,
    METHOD_RECENT_NEIGHBORS_LASTWEEK_NEIGHBORS,
    METHOD_RECENT_EXACT_ONLY,
    METHOD_LASTWEEK_EXACT_ONLY,
    METHOD_RECENT_NEIGHBORS_ONLY,
    METHOD_LASTWEEK_NEIGHBORS_ONLY,
}

OUTPUT_FILE_SUFFIX = "_phase2_filled.csv"
SUMMARY_FILE_SUFFIX = "_phase2_summary.csv"

TRUSTED_CURRENT_LABELS = {1, 3}
UNRESOLVED_CURRENT_LABELS = {-2, -1, 0}

_ALLOWED_ORIGINAL_LABELS = {-2, -1, 0, 1}
_ALLOWED_CURRENT_LABELS = {-2, -1, 0, 1, 2, 3}
_ALLOWED_GAP_PHASES = {1, 2, 3, 4, 5}
_ALLOWED_USABILITY_VALUES = {
    VALID_USABILITY_VALUE,
    INVALID_USABILITY_VALUE,
}

_STATUS_PROCESSED = "processed"
_STATUS_SKIPPED_UNUSABLE = "skipped_unusable_series"
_STATUS_NO_TARGETS = "no_phase2_targets"

_INTERNAL_TIMESTAMP_COLUMN = "__phase2_timestamp__"
_INTERNAL_ROW_ORDER_COLUMN = "__phase2_row_order__"
_INTERNAL_GAP_NUMBER_COLUMN = "__phase2_gap_number__"


@dataclass(frozen=True, slots=True)
class Phase2ReferenceSet:
    """
    Store available seasonal references for one target point.

    Attributes:
        recent_exact:
            Trusted exact value from the nearest earlier same-type day.

        recent_neighbors:
            Trusted neighbour-derived value around the recent reference.

        last_week_exact:
            Trusted exact value from the configured last-week timestamp.

        last_week_neighbors:
            Trusted neighbour-derived value around the last-week reference.
    """

    recent_exact: float | None
    recent_neighbors: float | None
    last_week_exact: float | None
    last_week_neighbors: float | None


@dataclass(frozen=True, slots=True)
class Phase2PointOutcome:
    """
    Store one target-point filling result.

    Attributes:
        filled:
            Whether a Phase 2 method produced a value.

        value:
            Generated finite value, or ``None`` when unresolved.

        method:
            Applied Phase 2 method, or ``None`` when unresolved.
    """

    filled: bool
    value: float | None
    method: str | None


@dataclass(frozen=True, slots=True)
class Phase2FillingData:
    """
    Store the in-memory Phase 2 result.

    Attributes:
        dataframe:
            Updated dataset.

        summary:
            One-row-per-site Phase 2 summary.

        total_series:
            Number of unique site series.

        processed_series:
            Number of usable sites containing unresolved Phase 2 targets.

        skipped_series:
            Number of unusable sites containing unresolved Phase 2 targets.

        primary_gaps:
            Number of original Phase 2 gaps.

        primary_points:
            Number of rows belonging to original Phase 2 gaps.

        retry_gaps:
            Number of original Phase 1 gaps containing unresolved retry points.

        retry_points:
            Number of unresolved Phase 1 points retried.

        already_resolved_phase1_points:
            Original Phase 1 points already safely filled before Phase 2.

        eligible_points:
            Total unresolved primary and retry targets on usable sites.

        filled_points:
            Number of points filled by Phase 2.

        unresolved_points:
            Number of eligible points left unresolved for Phase 3.

        completely_filled_gaps:
            Number of target gaps with no unresolved target points after
            Phase 2.

        partially_filled_gaps:
            Number of target gaps with both filled and unresolved targets.

        unresolved_gaps:
            Number of target gaps with at least one unresolved target.

        skipped_unusable_points:
            Target points skipped because their site is invalid.

        method_counts:
            Counts for each of the eight Phase 2 methods.
    """

    dataframe: pd.DataFrame
    summary: pd.DataFrame
    total_series: int
    processed_series: int
    skipped_series: int
    primary_gaps: int
    primary_points: int
    retry_gaps: int
    retry_points: int
    already_resolved_phase1_points: int
    eligible_points: int
    filled_points: int
    unresolved_points: int
    completely_filled_gaps: int
    partially_filled_gaps: int
    unresolved_gaps: int
    skipped_unusable_points: int
    method_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class Phase2FillingResult:
    """
    Store saved Phase 2 outputs and headline statistics.

    Attributes:
        source_file:
            Phase 1 dataset supplied by the preprocessing runner.

        output_file:
            Newly written Phase 2 dataset.

        summary_file:
            Per-site Phase 2 summary CSV.

        total_rows:
            Number of rows preserved in the output.

        total_series:
            Number of unique site series.

        processed_series:
            Number of usable sites containing unresolved Phase 2 targets.

        skipped_series:
            Number of unusable sites containing unresolved Phase 2 targets.

        primary_gaps:
            Number of original Phase 2 gaps.

        primary_points:
            Number of original Phase 2 rows.

        retry_gaps:
            Number of original Phase 1 gaps retried.

        retry_points:
            Number of unresolved Phase 1 points retried.

        already_resolved_phase1_points:
            Phase 1 points already safely resolved before Phase 2.

        eligible_points:
            Total usable unresolved targets.

        filled_points:
            Successfully filled target points.

        unresolved_points:
            Eligible targets left unresolved.

        completely_filled_gaps:
            Target gaps fully resolved after Phase 2.

        partially_filled_gaps:
            Target gaps partially resolved after Phase 2.

        unresolved_gaps:
            Target gaps retaining at least one unresolved point.

        skipped_unusable_points:
            Targets skipped because their site is invalid.

        method_counts:
            Counts for each Phase 2 method.
    """

    source_file: Path
    output_file: Path
    summary_file: Path
    total_rows: int
    total_series: int
    processed_series: int
    skipped_series: int
    primary_gaps: int
    primary_points: int
    retry_gaps: int
    retry_points: int
    already_resolved_phase1_points: int
    eligible_points: int
    filled_points: int
    unresolved_points: int
    completely_filled_gaps: int
    partially_filled_gaps: int
    unresolved_gaps: int
    skipped_unusable_points: int
    method_counts: dict[str, int]


def add_composite_id(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """
    Return a dataframe copy containing the shared composite site identifier.

    Args:
        dataframe:
            Dataset containing configured site identity columns.

        columns:
            Shared dataset column mapping.

    Returns:
        pd.DataFrame:
            Copy containing ``Composite_ID``.

    Raises:
        KeyError:
            If a configured site identity column is absent.
    """

    result = dataframe.copy()
    site_columns = list(
        columns.get_site_identity_columns()
    )

    identity_values = (
        result[site_columns]
        .astype("string")
        .fillna(MISSING_ID_VALUE)
    )

    result[COMPOSITE_ID_COLUMN] = (
        identity_values.agg(
            COMPOSITE_ID_SEPARATOR.join,
            axis=1,
        )
    )

    return result


def parse_integer_column(
    dataframe: pd.DataFrame,
    column_name: str,
    allowed_values: set[int],
) -> pd.Series:
    """
    Parse and validate a required integral state column.

    Args:
        dataframe:
            Dataset containing the state column.

        column_name:
            Column to parse.

        allowed_values:
            Accepted integer values.

    Returns:
        pd.Series:
            Parsed ``int8`` values.

    Raises:
        InputValidationError:
            If values are missing, non-integral, or unsupported.
    """

    numeric = pd.to_numeric(
        dataframe[column_name],
        errors="coerce",
    )

    invalid_mask = (
        numeric.isna()
        | ~np.isfinite(numeric)
        | numeric.mod(1).ne(0)
        | ~numeric.isin(allowed_values)
    )

    if invalid_mask.any():
        raise InputValidationError(
            f"{column_name} contains invalid Phase 2 values.",
            error_code="INVALID_PHASE2_INTEGER_COLUMN",
            details={
                "column_name": column_name,
                "allowed_values": sorted(allowed_values),
                "invalid_rows": int(invalid_mask.sum()),
                "sample_values": (
                    dataframe.loc[
                        invalid_mask,
                        column_name,
                    ]
                    .head(20)
                    .astype("string")
                    .tolist()
                ),
            },
        )

    return numeric.astype(np.int8)


def parse_gap_phase_column(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Parse and validate nullable original gap-phase values.

    Args:
        dataframe:
            Dataset containing ``Original_Gap_Phase``.

    Returns:
        pd.Series:
            Nullable ``Int8`` gap phases.

    Raises:
        InputValidationError:
            If non-null values are non-integral or outside 1 through 5.
    """

    raw = dataframe[ORIGINAL_GAP_PHASE_COLUMN]
    numeric = pd.to_numeric(
        raw,
        errors="coerce",
    )

    invalid_non_null = (
        raw.notna()
        & (
            numeric.isna()
            | ~np.isfinite(numeric)
            | numeric.mod(1).ne(0)
            | ~numeric.isin(_ALLOWED_GAP_PHASES)
        )
    )

    if invalid_non_null.any():
        raise InputValidationError(
            "Original_Gap_Phase contains invalid values.",
            error_code="INVALID_PHASE2_GAP_PHASE",
            details={
                "invalid_rows": int(invalid_non_null.sum()),
                "sample_values": (
                    raw.loc[invalid_non_null]
                    .head(20)
                    .astype("string")
                    .tolist()
                ),
            },
        )

    return numeric.astype("Int8")


def validate_required_columns(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> list[str]:
    """
    Validate the Phase 1 output schema required by Phase 2.

    Args:
        dataframe:
            Phase 1 dataset.

        columns:
            Shared dataset column mapping.

    Returns:
        list[str]:
            Input columns in their existing order.

    Raises:
        InputValidationError:
            If the dataframe is empty, required columns are absent, or
            reserved internal names are present.
    """

    if dataframe.empty:
        raise InputValidationError(
            "Phase 2 cannot process an empty dataframe.",
            error_code="EMPTY_PHASE2_DATASET",
        )

    required_columns = [
        *columns.get_site_identity_columns(),
        columns.timestamp_column,
        columns.value_column,
        ORIGINAL_CONDITION_LABEL_COLUMN,
        CURRENT_STATUS_LABEL_COLUMN,
        KPI_USABILITY_COLUMN,
        ORIGINAL_GAP_PHASE_COLUMN,
        FILL_METHOD_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "The Phase 1 dataset is missing columns required by Phase 2.",
            error_code="MISSING_PHASE2_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "required_columns": required_columns,
                "available_columns": list(dataframe.columns),
            },
        )

    reserved_columns = {
        COMPOSITE_ID_COLUMN,
        _INTERNAL_TIMESTAMP_COLUMN,
        _INTERNAL_ROW_ORDER_COLUMN,
        _INTERNAL_GAP_NUMBER_COLUMN,
    }
    conflicts = sorted(
        reserved_columns.intersection(dataframe.columns)
    )

    if conflicts:
        raise InputValidationError(
            "The Phase 1 dataset contains reserved Phase 2 columns.",
            error_code="RESERVED_PHASE2_COLUMN_CONFLICT",
            details={
                "conflicting_columns": conflicts,
            },
        )

    return list(dataframe.columns)


def validate_usability(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Normalize and validate site usability values.

    Args:
        dataframe:
            Working dataset containing ``Composite_ID``.

    Returns:
        pd.Series:
            Normalized usability values.

    Raises:
        InputValidationError:
            If values are missing, unsupported, or inconsistent within a site.
    """

    usability = (
        dataframe[KPI_USABILITY_COLUMN]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    invalid_mask = (
        usability.isna()
        | ~usability.isin(_ALLOWED_USABILITY_VALUES)
    )

    if invalid_mask.any():
        raise InputValidationError(
            "kpi_usability contains invalid Phase 2 values.",
            error_code="INVALID_PHASE2_USABILITY",
            details={
                "invalid_rows": int(invalid_mask.sum()),
                "sample_values": (
                    dataframe.loc[
                        invalid_mask,
                        KPI_USABILITY_COLUMN,
                    ]
                    .head(20)
                    .astype("string")
                    .tolist()
                ),
            },
        )

    per_site_counts = (
        pd.DataFrame(
            {
                COMPOSITE_ID_COLUMN: dataframe[COMPOSITE_ID_COLUMN],
                KPI_USABILITY_COLUMN: usability,
            }
        )
        .groupby(
            COMPOSITE_ID_COLUMN,
            dropna=False,
        )[KPI_USABILITY_COLUMN]
        .nunique(dropna=False)
    )

    inconsistent = per_site_counts.loc[
        per_site_counts != 1
    ]

    if not inconsistent.empty:
        raise InputValidationError(
            "kpi_usability is not constant within one or more sites.",
            error_code="INCONSISTENT_PHASE2_USABILITY",
            details={
                "sample_composite_ids": (
                    inconsistent.index
                    .astype(str)
                    .tolist()[:20]
                ),
            },
        )

    return usability


def validate_fill_methods(
    dataframe: pd.DataFrame,
    current_labels: pd.Series,
) -> pd.Series:
    """
    Validate Phase 1 fill-method state and protect against Phase 2 reruns.

    Args:
        dataframe:
            Phase 1 dataset.

        current_labels:
            Parsed current status labels.

    Returns:
        pd.Series:
            Normalized string fill methods.

    Raises:
        InputValidationError:
            If Phase 2 methods already exist or label-3 state lacks provenance.
    """

    methods = (
        dataframe[FILL_METHOD_COLUMN]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    phase2_existing = methods.str.startswith(
        "phase2_"
    )

    if phase2_existing.any():
        raise InputValidationError(
            "The dataset already contains Phase 2 fill methods.",
            error_code="PHASE2_ALREADY_APPLIED",
            details={
                "rows": int(phase2_existing.sum()),
                "sample_methods": (
                    methods.loc[phase2_existing]
                    .drop_duplicates()
                    .head(20)
                    .tolist()
                ),
            },
        )

    safe_without_method = (
        current_labels.eq(3)
        & methods.eq("")
    )

    if safe_without_method.any():
        raise InputValidationError(
            "Rows with Current_Status_Label 3 require Fill_Method provenance.",
            error_code="PHASE2_SAFE_FILL_WITHOUT_METHOD",
            details={
                "invalid_rows": int(safe_without_method.sum()),
            },
        )

    method_without_safe_label = (
        methods.ne("")
        & current_labels.ne(3)
    )

    if method_without_safe_label.any():
        raise InputValidationError(
            "Fill_Method is present on rows not labelled as safe fills.",
            error_code="PHASE2_METHOD_WITHOUT_SAFE_LABEL",
            details={
                "invalid_rows": int(method_without_safe_label.sum()),
            },
        )

    return methods


def prepare_phase2_dataframe(
    *,
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate and prepare one Phase 1 dataset for Phase 2 filling.

    Args:
        dataframe:
            Dataset returned by Phase 1.

        columns:
            Shared dataset column mapping.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Resolved frequency-dependent settings.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            Prepared working dataframe and original input column order.

    Raises:
        InputValidationError:
            If required schema, states, values, timelines, or configuration
            relationships are invalid.
    """

    input_columns = validate_required_columns(
        dataframe=dataframe,
        columns=columns,
    )

    working = add_composite_id(
        dataframe=dataframe,
        columns=columns,
    )
    working[_INTERNAL_ROW_ORDER_COLUMN] = np.arange(
        len(working),
        dtype=np.int64,
    )

    parsed_timestamps = pd.to_datetime(
        working[columns.timestamp_column],
        errors="coerce",
    )

    invalid_timestamps = parsed_timestamps.isna()

    if invalid_timestamps.any():
        raise InputValidationError(
            "The Phase 1 dataset contains invalid timestamps.",
            error_code="INVALID_PHASE2_TIMESTAMPS",
            details={
                "invalid_rows": int(invalid_timestamps.sum()),
            },
        )

    working[_INTERNAL_TIMESTAMP_COLUMN] = parsed_timestamps

    original_labels = parse_integer_column(
        dataframe=working,
        column_name=ORIGINAL_CONDITION_LABEL_COLUMN,
        allowed_values=_ALLOWED_ORIGINAL_LABELS,
    )
    current_labels = parse_integer_column(
        dataframe=working,
        column_name=CURRENT_STATUS_LABEL_COLUMN,
        allowed_values=_ALLOWED_CURRENT_LABELS,
    )
    gap_phases = parse_gap_phase_column(
        dataframe=working,
    )

    working[ORIGINAL_CONDITION_LABEL_COLUMN] = original_labels
    working[CURRENT_STATUS_LABEL_COLUMN] = current_labels
    working[ORIGINAL_GAP_PHASE_COLUMN] = gap_phases
    working[KPI_USABILITY_COLUMN] = validate_usability(
        working
    )
    working[FILL_METHOD_COLUMN] = validate_fill_methods(
        dataframe=working,
        current_labels=current_labels,
    )

    missing_origin = original_labels.isin(
        MISSING_ORIGINAL_LABELS
    )
    valid_origin = original_labels.eq(1)

    invalid_transition = (
        (
            missing_origin
            & ~current_labels.isin(
                {-2, -1, 0, 3}
            )
        )
        | (
            valid_origin
            & ~current_labels.isin({1, 2})
        )
    )

    if invalid_transition.any():
        raise InputValidationError(
            "Original and current labels contain invalid pre-Phase-2 states.",
            error_code="INVALID_PHASE2_LABEL_TRANSITION",
            details={
                "invalid_rows": int(invalid_transition.sum()),
            },
        )

    missing_without_phase = (
        missing_origin
        & gap_phases.isna()
    )
    nonmissing_with_phase = (
        ~missing_origin
        & gap_phases.notna()
    )

    if missing_without_phase.any() or nonmissing_with_phase.any():
        raise InputValidationError(
            "Original_Gap_Phase does not match immutable original labels.",
            error_code="INVALID_PHASE2_GAP_PHASE_STATE",
            details={
                "missing_without_phase": int(
                    missing_without_phase.sum()
                ),
                "nonmissing_with_phase": int(
                    nonmissing_with_phase.sum()
                ),
            },
        )

    numeric_values = pd.to_numeric(
        working[columns.value_column],
        errors="coerce",
    )

    unresolved = current_labels.isin(
        UNRESOLVED_CURRENT_LABELS
    )
    resolved = current_labels.isin(
        TRUSTED_CURRENT_LABELS | {2}
    )

    unresolved_with_value = (
        unresolved
        & numeric_values.notna()
    )
    resolved_without_value = (
        resolved
        & (
            numeric_values.isna()
            | ~np.isfinite(numeric_values)
        )
    )

    if unresolved_with_value.any() or resolved_without_value.any():
        raise InputValidationError(
            "KPI values do not match current Phase 2 status labels.",
            error_code="INVALID_PHASE2_VALUE_STATE",
            details={
                "unresolved_with_value": int(
                    unresolved_with_value.sum()
                ),
                "resolved_without_value": int(
                    resolved_without_value.sum()
                ),
            },
        )

    lower_bound = config.valid_values.lower_bound
    upper_bound = config.valid_values.upper_bound

    present_outside_range = (
        numeric_values.notna()
        & (
            numeric_values.lt(lower_bound)
            | numeric_values.gt(upper_bound)
            | ~np.isfinite(numeric_values)
        )
    )

    if present_outside_range.any():
        raise InputValidationError(
            "The Phase 1 dataset contains present KPI values outside the "
            "configured valid range.",
            error_code="PHASE2_VALUE_OUTSIDE_VALID_RANGE",
            details={
                "invalid_rows": int(present_outside_range.sum()),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )

    phase2_already_filled = (
        gap_phases.eq(PHASE_2)
        & current_labels.eq(3)
    )

    if phase2_already_filled.any():
        raise InputValidationError(
            "Original Phase 2 rows are already labelled as safe fills before "
            "Phase 2 begins.",
            error_code="PHASE2_TARGET_ALREADY_FILLED",
            details={
                "invalid_rows": int(phase2_already_filled.sum()),
            },
        )

    duplicate_mask = working.duplicated(
        subset=[
            COMPOSITE_ID_COLUMN,
            _INTERNAL_TIMESTAMP_COLUMN,
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise InputValidationError(
            "Duplicate site-timestamp rows remain before Phase 2.",
            error_code="DUPLICATE_TIMESTAMPS_BEFORE_PHASE2",
            details={
                "duplicate_rows": int(duplicate_mask.sum()),
            },
        )

    expected_frequency = pd.Timedelta(
        minutes=resolved_frequency.bucket_minutes
    )

    if expected_frequency <= pd.Timedelta(0):
        raise InputValidationError(
            "The resolved Phase 2 frequency must be positive.",
            error_code="NON_POSITIVE_PHASE2_FREQUENCY",
        )

    spacing_errors: list[dict[str, str]] = []

    for composite_id, site_dataframe in working.groupby(
        COMPOSITE_ID_COLUMN,
        sort=False,
        dropna=False,
    ):
        timestamps = (
            site_dataframe[_INTERNAL_TIMESTAMP_COLUMN]
            .sort_values(kind="stable")
        )
        differences = timestamps.diff()
        bad_spacing = (
            differences.notna()
            & differences.ne(expected_frequency)
        )

        if bad_spacing.any():
            bad_index = bad_spacing.loc[
                bad_spacing
            ].index[0]
            ordered_indices = list(timestamps.index)
            position = ordered_indices.index(bad_index)
            previous_index = ordered_indices[position - 1]

            spacing_errors.append(
                {
                    COMPOSITE_ID_COLUMN: str(composite_id),
                    "previous_timestamp": str(
                        working.at[
                            previous_index,
                            _INTERNAL_TIMESTAMP_COLUMN,
                        ]
                    ),
                    "current_timestamp": str(
                        working.at[
                            bad_index,
                            _INTERNAL_TIMESTAMP_COLUMN,
                        ]
                    ),
                }
            )

            if len(spacing_errors) >= 20:
                break

    if spacing_errors:
        raise InputValidationError(
            "One or more site timelines are incomplete before Phase 2.",
            error_code="INCOMPLETE_TIMELINE_BEFORE_PHASE2",
            details={
                "sample": spacing_errors,
            },
        )

    expected_neighbor_points = (
        resolved_frequency.phase_2_neighbor_offset_points
    )

    if expected_neighbor_points <= 0:
        raise InputValidationError(
            "The resolved Phase 2 neighbour offset must be positive.",
            error_code="NON_POSITIVE_PHASE2_NEIGHBOR_OFFSET",
        )

    return working, input_columns


def is_weekend(
    timestamp: pd.Timestamp,
) -> bool:
    """
    Determine whether a timestamp belongs to Saturday or Sunday.

    Args:
        timestamp:
            Parsed timestamp.

    Returns:
        bool:
            True for weekend timestamps, otherwise False.

    Raises:
        None.
    """

    return bool(timestamp.weekday() >= 5)


def is_trusted_position(
    *,
    position: int,
    values: np.ndarray,
    labels: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> bool:
    """
    Determine whether one site-array position is a trusted reference.

    Args:
        position:
            Zero-based site-array position.

        values:
            Current working values.

        labels:
            Current status labels.

        lower_bound:
            Inclusive valid lower bound.

        upper_bound:
            Inclusive valid upper bound.

    Returns:
        bool:
            True when the position is a safe trusted reference.

    Raises:
        None.
    """

    if position < 0 or position >= len(values):
        return False

    if int(labels[position]) not in TRUSTED_CURRENT_LABELS:
        return False

    value = values[position]

    return bool(
        np.isfinite(value)
        and lower_bound <= value <= upper_bound
    )


def get_exact_reference(
    *,
    reference_timestamp: pd.Timestamp | None,
    timestamp_to_position: dict[pd.Timestamp, int],
    values: np.ndarray,
    labels: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> float | None:
    """
    Return a trusted exact historical value.

    Args:
        reference_timestamp:
            Historical timestamp, or ``None`` when unavailable.

        timestamp_to_position:
            Site timestamp-position lookup.

        values:
            Current working values.

        labels:
            Current status labels.

        lower_bound:
            Inclusive valid lower bound.

        upper_bound:
            Inclusive valid upper bound.

    Returns:
        float | None:
            Trusted value or ``None``.

    Raises:
        None.
    """

    if reference_timestamp is None:
        return None

    position = timestamp_to_position.get(
        pd.Timestamp(reference_timestamp)
    )

    if position is None:
        return None

    if not is_trusted_position(
        position=position,
        values=values,
        labels=labels,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    ):
        return None

    return float(values[position])


def get_neighbor_reference(
    *,
    reference_timestamp: pd.Timestamp | None,
    neighbor_offset: pd.Timedelta,
    timestamp_to_position: dict[pd.Timestamp, int],
    values: np.ndarray,
    labels: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> float | None:
    """
    Build a trusted neighbour reference around one historical timestamp.

    The mean is used when both neighbouring points are trusted. A single
    trusted neighbour is used when only one is available.

    Args:
        reference_timestamp:
            Historical centre timestamp.

        neighbor_offset:
            Configured offset on either side.

        timestamp_to_position:
            Site timestamp-position lookup.

        values:
            Current working values.

        labels:
            Current status labels.

        lower_bound:
            Inclusive valid lower bound.

        upper_bound:
            Inclusive valid upper bound.

    Returns:
        float | None:
            Neighbour-derived value or ``None``.

    Raises:
        None.
    """

    if reference_timestamp is None:
        return None

    candidate_values: list[float] = []

    for candidate_timestamp in (
        reference_timestamp - neighbor_offset,
        reference_timestamp + neighbor_offset,
    ):
        candidate_position = timestamp_to_position.get(
            pd.Timestamp(candidate_timestamp)
        )

        if candidate_position is None:
            continue

        if is_trusted_position(
            position=candidate_position,
            values=values,
            labels=labels,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ):
            candidate_values.append(
                float(values[candidate_position])
            )

    if not candidate_values:
        return None

    return float(np.mean(candidate_values))


def find_recent_same_type_timestamp(
    *,
    target_timestamp: pd.Timestamp,
    timestamp_to_position: dict[pd.Timestamp, int],
    maximum_lookback_days: int,
) -> pd.Timestamp | None:
    """
    Find the nearest earlier timestamp having the same broad day type.

    Day type means weekday versus weekend. The same time of day is preserved.
    Selection depends only on timeline presence, not on whether the selected
    exact value is trusted.

    Args:
        target_timestamp:
            Missing target timestamp.

        timestamp_to_position:
            Complete site timeline lookup.

        maximum_lookback_days:
            Maximum backward calendar-day search.

    Returns:
        pd.Timestamp | None:
            Nearest earlier same-type timestamp, or ``None``.

    Raises:
        None.
    """

    target_is_weekend = is_weekend(
        target_timestamp
    )

    for day_offset in range(
        1,
        maximum_lookback_days + 1,
    ):
        candidate = (
            target_timestamp
            - pd.Timedelta(days=day_offset)
        )

        if is_weekend(candidate) != target_is_weekend:
            continue

        candidate = pd.Timestamp(candidate)

        if candidate in timestamp_to_position:
            return candidate

    return None


def collect_reference_set(
    *,
    target_timestamp: pd.Timestamp,
    timestamp_to_position: dict[pd.Timestamp, int],
    values: np.ndarray,
    labels: np.ndarray,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> Phase2ReferenceSet:
    """
    Collect exact and neighbour seasonal references for one target.

    Args:
        target_timestamp:
            Unresolved target timestamp.

        timestamp_to_position:
            Complete site timeline lookup.

        values:
            Current working values, including earlier safe fills.

        labels:
            Current status labels.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Resolved Phase 2 neighbour offset.

    Returns:
        Phase2ReferenceSet:
            Four optional seasonal references.

    Raises:
        None.
    """

    phase2_config = config.phase_2
    lower_bound = config.valid_values.lower_bound
    upper_bound = config.valid_values.upper_bound

    recent_timestamp = find_recent_same_type_timestamp(
        target_timestamp=target_timestamp,
        timestamp_to_position=timestamp_to_position,
        maximum_lookback_days=(
            phase2_config.recent_same_type_lookback_days
        ),
    )

    last_week_timestamp = pd.Timestamp(
        target_timestamp
        - pd.Timedelta(
            days=phase2_config.last_week_lookback_days
        )
    )

    if last_week_timestamp not in timestamp_to_position:
        last_week_timestamp = None

    neighbor_offset = pd.Timedelta(
        minutes=(
            resolved_frequency.bucket_minutes
            * resolved_frequency.phase_2_neighbor_offset_points
        )
    )

    return Phase2ReferenceSet(
        recent_exact=get_exact_reference(
            reference_timestamp=recent_timestamp,
            timestamp_to_position=timestamp_to_position,
            values=values,
            labels=labels,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ),
        recent_neighbors=get_neighbor_reference(
            reference_timestamp=recent_timestamp,
            neighbor_offset=neighbor_offset,
            timestamp_to_position=timestamp_to_position,
            values=values,
            labels=labels,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ),
        last_week_exact=get_exact_reference(
            reference_timestamp=last_week_timestamp,
            timestamp_to_position=timestamp_to_position,
            values=values,
            labels=labels,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ),
        last_week_neighbors=get_neighbor_reference(
            reference_timestamp=last_week_timestamp,
            neighbor_offset=neighbor_offset,
            timestamp_to_position=timestamp_to_position,
            values=values,
            labels=labels,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ),
    )


def combine_references(
    *,
    recent_value: float,
    last_week_value: float,
    config: PreprocessingConfig,
) -> float:
    """
    Combine recent and last-week references using configured weights.

    Args:
        recent_value:
            Available recent seasonal reference.

        last_week_value:
            Available last-week seasonal reference.

        config:
            Complete preprocessing configuration.

    Returns:
        float:
            Weighted generated value.

    Raises:
        None.
    """

    return float(
        config.phase_2.recent_weight
        * recent_value
        + config.phase_2.last_week_weight
        * last_week_value
    )


def select_phase2_method(
    *,
    references: Phase2ReferenceSet,
    config: PreprocessingConfig,
) -> Phase2PointOutcome:
    """
    Apply the ordered eight-level Phase 2 filling ladder.

    Args:
        references:
            Available exact and neighbour seasonal references.

        config:
            Complete preprocessing configuration.

    Returns:
        Phase2PointOutcome:
            Selected generated value and method, or unresolved outcome.

    Raises:
        None.
    """

    recent_exact = references.recent_exact
    recent_neighbors = references.recent_neighbors
    last_week_exact = references.last_week_exact
    last_week_neighbors = references.last_week_neighbors

    if (
        recent_exact is not None
        and last_week_exact is not None
    ):
        return Phase2PointOutcome(
            filled=True,
            value=combine_references(
                recent_value=recent_exact,
                last_week_value=last_week_exact,
                config=config,
            ),
            method=METHOD_RECENT_EXACT_LASTWEEK_EXACT,
        )

    if (
        recent_neighbors is not None
        and last_week_exact is not None
    ):
        return Phase2PointOutcome(
            filled=True,
            value=combine_references(
                recent_value=recent_neighbors,
                last_week_value=last_week_exact,
                config=config,
            ),
            method=(
                METHOD_RECENT_NEIGHBORS_LASTWEEK_EXACT
            ),
        )

    if (
        recent_exact is not None
        and last_week_neighbors is not None
    ):
        return Phase2PointOutcome(
            filled=True,
            value=combine_references(
                recent_value=recent_exact,
                last_week_value=last_week_neighbors,
                config=config,
            ),
            method=(
                METHOD_RECENT_EXACT_LASTWEEK_NEIGHBORS
            ),
        )

    if (
        recent_neighbors is not None
        and last_week_neighbors is not None
    ):
        return Phase2PointOutcome(
            filled=True,
            value=combine_references(
                recent_value=recent_neighbors,
                last_week_value=last_week_neighbors,
                config=config,
            ),
            method=(
                METHOD_RECENT_NEIGHBORS_LASTWEEK_NEIGHBORS
            ),
        )

    if recent_exact is not None:
        return Phase2PointOutcome(
            filled=True,
            value=float(recent_exact),
            method=METHOD_RECENT_EXACT_ONLY,
        )

    if last_week_exact is not None:
        return Phase2PointOutcome(
            filled=True,
            value=float(last_week_exact),
            method=METHOD_LASTWEEK_EXACT_ONLY,
        )

    if recent_neighbors is not None:
        return Phase2PointOutcome(
            filled=True,
            value=float(recent_neighbors),
            method=METHOD_RECENT_NEIGHBORS_ONLY,
        )

    if last_week_neighbors is not None:
        return Phase2PointOutcome(
            filled=True,
            value=float(last_week_neighbors),
            method=METHOD_LASTWEEK_NEIGHBORS_ONLY,
        )

    return Phase2PointOutcome(
        filled=False,
        value=None,
        method=None,
    )


def normalize_generated_value(
    *,
    value: float | None,
    config: PreprocessingConfig,
) -> float | None:
    """
    Validate and optionally clip one generated Phase 2 value.

    Args:
        value:
            Raw generated value.

        config:
            Valid-value configuration.

    Returns:
        float | None:
            Finite accepted value, or ``None``.

    Raises:
        None.
    """

    if value is None or not np.isfinite(value):
        return None

    lower_bound = config.valid_values.lower_bound
    upper_bound = config.valid_values.upper_bound
    generated = float(value)

    if config.valid_values.clip_generated_values:
        generated = float(
            np.clip(
                generated,
                lower_bound,
                upper_bound,
            )
        )

    if not lower_bound <= generated <= upper_bound:
        return None

    return generated


def assign_original_gap_numbers(
    ordered: pd.DataFrame,
) -> pd.Series:
    """
    Assign stable one-based original gap numbers within one site.

    Original gaps are continuous runs where ``Original_Gap_Phase`` is present.
    The number is used only for Phase 2 accounting and is never written to the
    output dataset.

    Args:
        ordered:
            Timestamp-sorted site dataframe.

    Returns:
        pd.Series:
            Nullable integer gap number aligned with ``ordered``.

    Raises:
        None.
    """

    gap_present = ordered[
        ORIGINAL_GAP_PHASE_COLUMN
    ].notna()

    new_gap = (
        gap_present
        & ~gap_present.shift(
            1,
            fill_value=False,
        )
    )

    gap_numbers = new_gap.cumsum().astype("Int64")
    return gap_numbers.where(
        gap_present,
        pd.NA,
    )


def summarize_target_gaps(
    *,
    gap_numbers: pd.Series,
    gap_phases: np.ndarray,
    before_labels: np.ndarray,
    after_labels: np.ndarray,
    usability: str,
) -> dict[str, int]:
    """
    Calculate gap-level Phase 2 results for one site.

    Args:
        gap_numbers:
            Original gap number per row.

        gap_phases:
            Original phase values.

        before_labels:
            Current labels before Phase 2.

        after_labels:
            Current labels after Phase 2.

        usability:
            Site usability.

    Returns:
        dict[str, int]:
            Primary/retry and resolution counts.

    Raises:
        None.
    """

    statistics = {
        "primary_gaps": 0,
        "primary_points": 0,
        "retry_gaps": 0,
        "retry_points": 0,
        "already_resolved_phase1_points": 0,
        "eligible_points": 0,
        "filled_points": 0,
        "unresolved_points": 0,
        "completely_filled_gaps": 0,
        "partially_filled_gaps": 0,
        "unresolved_gaps": 0,
        "skipped_unusable_points": 0,
    }

    phase1_mask = gap_phases == PHASE_1
    phase2_mask = gap_phases == PHASE_2

    statistics["primary_points"] = int(
        phase2_mask.sum()
    )
    statistics["already_resolved_phase1_points"] = int(
        (
            phase1_mask
            & np.isin(before_labels, [3])
        ).sum()
    )

    retry_target_mask = (
        phase1_mask
        & np.isin(
            before_labels,
            list(UNRESOLVED_CURRENT_LABELS),
        )
    )
    primary_target_mask = (
        phase2_mask
        & np.isin(
            before_labels,
            list(UNRESOLVED_CURRENT_LABELS),
        )
    )
    target_mask = (
        retry_target_mask
        | primary_target_mask
    )

    statistics["retry_points"] = int(
        retry_target_mask.sum()
    )

    gap_number_array = gap_numbers.to_numpy(
        dtype=object
    )

    primary_gap_ids = {
        int(gap_number_array[position])
        for position in np.flatnonzero(phase2_mask)
        if gap_number_array[position] is not pd.NA
    }
    retry_gap_ids = {
        int(gap_number_array[position])
        for position in np.flatnonzero(retry_target_mask)
        if gap_number_array[position] is not pd.NA
    }

    statistics["primary_gaps"] = len(
        primary_gap_ids
    )
    statistics["retry_gaps"] = len(
        retry_gap_ids
    )

    if usability == INVALID_USABILITY_VALUE:
        statistics["skipped_unusable_points"] = int(
            target_mask.sum()
        )
        return statistics

    statistics["eligible_points"] = int(
        target_mask.sum()
    )

    filled_target_mask = (
        target_mask
        & (after_labels == 3)
    )
    unresolved_target_mask = (
        target_mask
        & np.isin(
            after_labels,
            list(UNRESOLVED_CURRENT_LABELS),
        )
    )

    statistics["filled_points"] = int(
        filled_target_mask.sum()
    )
    statistics["unresolved_points"] = int(
        unresolved_target_mask.sum()
    )

    target_gap_ids = sorted(
        primary_gap_ids | retry_gap_ids
    )

    for gap_id in target_gap_ids:
        gap_mask = (
            np.array(
                [
                    value is not pd.NA
                    and int(value) == gap_id
                    for value in gap_number_array
                ],
                dtype=bool,
            )
            & target_mask
        )

        if not gap_mask.any():
            continue

        gap_filled = int(
            filled_target_mask[gap_mask].sum()
        )
        gap_unresolved = int(
            unresolved_target_mask[gap_mask].sum()
        )

        if gap_unresolved == 0:
            statistics[
                "completely_filled_gaps"
            ] += 1
        else:
            statistics["unresolved_gaps"] += 1

            if gap_filled > 0:
                statistics[
                    "partially_filled_gaps"
                ] += 1

    return statistics


def process_one_site(
    *,
    site_dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply Phase 2 filling to one site series.

    Targets are processed chronologically. Successful fills are immediately
    available as trusted references for later target timestamps.

    Args:
        site_dataframe:
            Prepared rows for one site.

        columns:
            Shared dataset column mapping.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Resolved Phase 2 settings.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]:
            Updated site dataframe and one summary record.

    Raises:
        InputValidationError:
            If a generated target state becomes inconsistent.
    """

    ordered = site_dataframe.sort_values(
        _INTERNAL_TIMESTAMP_COLUMN,
        kind="stable",
    ).copy()

    timestamps = ordered[
        _INTERNAL_TIMESTAMP_COLUMN
    ].tolist()
    timestamp_to_position = {
        pd.Timestamp(timestamp): position
        for position, timestamp in enumerate(
            timestamps
        )
    }

    values = pd.to_numeric(
        ordered[columns.value_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    labels = pd.to_numeric(
        ordered[CURRENT_STATUS_LABEL_COLUMN],
        errors="raise",
    ).to_numpy(dtype=np.int8)
    original_labels = pd.to_numeric(
        ordered[ORIGINAL_CONDITION_LABEL_COLUMN],
        errors="raise",
    ).to_numpy(dtype=np.int8)
    gap_phases = (
        ordered[ORIGINAL_GAP_PHASE_COLUMN]
        .astype("Int8")
        .to_numpy(dtype=float, na_value=np.nan)
    )
    fill_methods = (
        ordered[FILL_METHOD_COLUMN]
        .astype("string")
        .fillna("")
        .to_numpy(dtype=object)
    )

    before_labels = labels.copy()
    gap_numbers = assign_original_gap_numbers(
        ordered
    )
    usability = str(
        ordered[KPI_USABILITY_COLUMN].iloc[0]
    )

    method_counts = {
        method: 0
        for method in sorted(PHASE_2_METHODS)
    }

    target_mask = (
        np.isin(
            gap_phases,
            [PHASE_1, PHASE_2],
        )
        & np.isin(
            labels,
            list(UNRESOLVED_CURRENT_LABELS),
        )
    )

    if (
        usability == VALID_USABILITY_VALUE
        and target_mask.any()
    ):
        for position in np.flatnonzero(
            target_mask
        ):
            if (
                int(original_labels[position])
                not in MISSING_ORIGINAL_LABELS
            ):
                raise InputValidationError(
                    "A Phase 2 target does not have a missing-origin label.",
                    error_code="INVALID_PHASE2_TARGET_ORIGIN",
                    details={
                        "composite_id": str(
                            ordered[
                                COMPOSITE_ID_COLUMN
                            ].iloc[0]
                        ),
                        "timestamp": str(
                            timestamps[position]
                        ),
                    },
                )

            if (
                int(labels[position])
                not in UNRESOLVED_CURRENT_LABELS
                or np.isfinite(values[position])
            ):
                continue

            references = collect_reference_set(
                target_timestamp=pd.Timestamp(
                    timestamps[position]
                ),
                timestamp_to_position=timestamp_to_position,
                values=values,
                labels=labels,
                config=config,
                resolved_frequency=resolved_frequency,
            )
            outcome = select_phase2_method(
                references=references,
                config=config,
            )
            generated = normalize_generated_value(
                value=outcome.value,
                config=config,
            )

            if (
                not outcome.filled
                or generated is None
                or outcome.method is None
            ):
                continue

            values[position] = generated
            labels[position] = 3
            fill_methods[position] = outcome.method
            method_counts[outcome.method] += 1

    ordered[columns.value_column] = values
    ordered[CURRENT_STATUS_LABEL_COLUMN] = labels
    ordered[FILL_METHOD_COLUMN] = fill_methods

    statistics = summarize_target_gaps(
        gap_numbers=gap_numbers,
        gap_phases=gap_phases,
        before_labels=before_labels,
        after_labels=labels,
        usability=usability,
    )

    if not target_mask.any():
        status = _STATUS_NO_TARGETS
    elif usability == INVALID_USABILITY_VALUE:
        status = _STATUS_SKIPPED_UNUSABLE
    else:
        status = _STATUS_PROCESSED

    summary_row: dict[str, Any] = {
        COMPOSITE_ID_COLUMN: str(
            ordered[COMPOSITE_ID_COLUMN].iloc[0]
        ),
        KPI_USABILITY_COLUMN: usability,
        "status": status,
        **statistics,
    }

    for method in sorted(PHASE_2_METHODS):
        summary_row[
            f"{method}_points"
        ] = method_counts[method]

    return ordered, summary_row


def validate_phase2_output(
    *,
    input_dataframe: pd.DataFrame,
    output_dataframe: pd.DataFrame,
    input_columns: list[str],
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
) -> None:
    """
    Validate Phase 2 output invariants.

    Args:
        input_dataframe:
            Phase 1 input dataset.

        output_dataframe:
            Phase 2 result.

        input_columns:
            Original input column order.

        columns:
            Shared dataset column mapping.

        config:
            Complete preprocessing configuration.

    Returns:
        None.

    Raises:
        InputValidationError:
            If schema, row count, protected fields, or fill states changed
            incorrectly.
    """

    if list(output_dataframe.columns) != input_columns:
        raise InputValidationError(
            "Phase 2 changed the dataset schema or column order.",
            error_code="INVALID_PHASE2_OUTPUT_COLUMNS",
            details={
                "expected_columns": input_columns,
                "actual_columns": list(output_dataframe.columns),
            },
        )

    if len(output_dataframe) != len(input_dataframe):
        raise InputValidationError(
            "Phase 2 changed the dataset row count.",
            error_code="PHASE2_ROW_COUNT_CHANGED",
            details={
                "input_rows": len(input_dataframe),
                "output_rows": len(output_dataframe),
            },
        )

    mutable_columns = {
        columns.value_column,
        CURRENT_STATUS_LABEL_COLUMN,
        FILL_METHOD_COLUMN,
    }

    changed_protected = [
        column
        for column in input_columns
        if (
            column not in mutable_columns
            and not output_dataframe[column].equals(
                input_dataframe[column]
            )
        )
    ]

    if changed_protected:
        raise InputValidationError(
            "Phase 2 modified protected input columns.",
            error_code="PHASE2_MODIFIED_PROTECTED_COLUMNS",
            details={
                "changed_columns": changed_protected,
            },
        )

    input_labels = pd.to_numeric(
        input_dataframe[CURRENT_STATUS_LABEL_COLUMN],
        errors="coerce",
    )
    output_labels = pd.to_numeric(
        output_dataframe[CURRENT_STATUS_LABEL_COLUMN],
        errors="coerce",
    )
    input_values = pd.to_numeric(
        input_dataframe[columns.value_column],
        errors="coerce",
    )
    output_values = pd.to_numeric(
        output_dataframe[columns.value_column],
        errors="coerce",
    )
    output_methods = (
        output_dataframe[FILL_METHOD_COLUMN]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    newly_filled = (
        input_labels.isin(UNRESOLVED_CURRENT_LABELS)
        & output_labels.eq(3)
    )

    illegal_label_change = (
        ~newly_filled
        & output_labels.ne(input_labels)
    )

    if illegal_label_change.any():
        raise InputValidationError(
            "Phase 2 changed current labels outside successful targets.",
            error_code="ILLEGAL_PHASE2_LABEL_CHANGE",
            details={
                "invalid_rows": int(illegal_label_change.sum()),
            },
        )

    changed_existing_values = (
        ~newly_filled
        & ~(
            output_values.eq(input_values)
            | (
                output_values.isna()
                & input_values.isna()
            )
        )
    )

    if changed_existing_values.any():
        raise InputValidationError(
            "Phase 2 changed values outside newly filled targets.",
            error_code="ILLEGAL_PHASE2_VALUE_CHANGE",
            details={
                "invalid_rows": int(
                    changed_existing_values.sum()
                ),
            },
        )

    new_method_mask = output_methods.isin(
        PHASE_2_METHODS
    )

    if not new_method_mask.equals(newly_filled):
        raise InputValidationError(
            "Phase 2 method provenance does not match newly filled rows.",
            error_code="INVALID_PHASE2_METHOD_ASSIGNMENT",
            details={
                "method_rows": int(new_method_mask.sum()),
                "newly_filled_rows": int(newly_filled.sum()),
            },
        )

    filled_values_invalid = (
        newly_filled
        & (
            output_values.isna()
            | ~np.isfinite(output_values)
            | output_values.lt(
                config.valid_values.lower_bound
            )
            | output_values.gt(
                config.valid_values.upper_bound
            )
        )
    )

    if filled_values_invalid.any():
        raise InputValidationError(
            "Phase 2 generated invalid filled values.",
            error_code="INVALID_PHASE2_GENERATED_VALUE",
            details={
                "invalid_rows": int(
                    filled_values_invalid.sum()
                ),
            },
        )

    unresolved_input = input_labels.isin(
        UNRESOLVED_CURRENT_LABELS
    )
    still_unresolved = (
        unresolved_input
        & output_labels.isin(
            UNRESOLVED_CURRENT_LABELS
        )
    )

    unresolved_state_invalid = (
        still_unresolved
        & (
            output_values.notna()
            | output_methods.ne(
                input_dataframe[
                    FILL_METHOD_COLUMN
                ]
                .astype("string")
                .fillna("")
                .str.strip()
            )
        )
    )

    if unresolved_state_invalid.any():
        raise InputValidationError(
            "Unresolved Phase 2 targets were not left unchanged.",
            error_code="INVALID_UNRESOLVED_PHASE2_STATE",
            details={
                "invalid_rows": int(
                    unresolved_state_invalid.sum()
                ),
            },
        )


def fill_phase_2_missing_values(
    *,
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
) -> Phase2FillingData:
    """
    Fill primary Phase 2 gaps and retry unresolved Phase 1 points.

    Args:
        dataframe:
            Dataset returned by Phase 1.

        columns:
            Shared dataset column mapping.

        config:
            Complete validated preprocessing configuration.

        resolved_frequency:
            Frequency-dependent Phase 2 settings.

        logger:
            Run-specific logger.

    Returns:
        Phase2FillingData:
            Updated dataframe, per-site summary, and aggregate counts.

    Raises:
        InputValidationError:
            If validation, filling, or output invariants fail.
    """

    working, input_columns = prepare_phase2_dataframe(
        dataframe=dataframe,
        columns=columns,
        config=config,
        resolved_frequency=resolved_frequency,
    )

    updated = working.copy()
    summary_rows: list[dict[str, Any]] = []

    grouped_indices = updated.groupby(
        COMPOSITE_ID_COLUMN,
        sort=False,
        dropna=False,
    ).groups

    for _, indices in grouped_indices.items():
        site_dataframe = updated.loc[
            indices
        ].copy()

        processed_site, summary_row = process_one_site(
            site_dataframe=site_dataframe,
            columns=columns,
            config=config,
            resolved_frequency=resolved_frequency,
        )

        source_indices = processed_site.index

        updated.loc[
            source_indices,
            columns.value_column,
        ] = processed_site[
            columns.value_column
        ].to_numpy()
        updated.loc[
            source_indices,
            CURRENT_STATUS_LABEL_COLUMN,
        ] = processed_site[
            CURRENT_STATUS_LABEL_COLUMN
        ].to_numpy()
        updated.loc[
            source_indices,
            FILL_METHOD_COLUMN,
        ] = processed_site[
            FILL_METHOD_COLUMN
        ].to_numpy()

        summary_rows.append(summary_row)

    aligned = updated.reindex(
        dataframe.index
    )

    output_dataframe = dataframe.copy()
    output_dataframe[columns.value_column] = (
        aligned[columns.value_column].to_numpy()
    )
    output_dataframe[CURRENT_STATUS_LABEL_COLUMN] = (
        aligned[CURRENT_STATUS_LABEL_COLUMN]
        .to_numpy()
    )
    output_dataframe[FILL_METHOD_COLUMN] = (
        aligned[FILL_METHOD_COLUMN].to_numpy()
    )

    validate_phase2_output(
        input_dataframe=dataframe,
        output_dataframe=output_dataframe,
        input_columns=input_columns,
        columns=columns,
        config=config,
    )

    summary_columns = [
        COMPOSITE_ID_COLUMN,
        KPI_USABILITY_COLUMN,
        "status",
        "primary_gaps",
        "primary_points",
        "retry_gaps",
        "retry_points",
        "already_resolved_phase1_points",
        "eligible_points",
        "filled_points",
        "unresolved_points",
        "completely_filled_gaps",
        "partially_filled_gaps",
        "unresolved_gaps",
        "skipped_unusable_points",
        *[
            f"{method}_points"
            for method in sorted(PHASE_2_METHODS)
        ],
    ]

    summary = pd.DataFrame(
        summary_rows,
        columns=summary_columns,
    )

    aggregate_columns = [
        "primary_gaps",
        "primary_points",
        "retry_gaps",
        "retry_points",
        "already_resolved_phase1_points",
        "eligible_points",
        "filled_points",
        "unresolved_points",
        "completely_filled_gaps",
        "partially_filled_gaps",
        "unresolved_gaps",
        "skipped_unusable_points",
    ]

    totals = {
        column: int(
            pd.to_numeric(
                summary[column],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        for column in aggregate_columns
    }

    method_counts = {
        method: int(
            pd.to_numeric(
                summary[f"{method}_points"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        for method in sorted(PHASE_2_METHODS)
    }

    processed_series = int(
        summary["status"].eq(
            _STATUS_PROCESSED
        ).sum()
    )
    skipped_series = int(
        summary["status"].eq(
            _STATUS_SKIPPED_UNUSABLE
        ).sum()
    )

    logger.info(
        "Phase 2 filling completed: primary_gaps=%d, primary_points=%d, "
        "retry_gaps=%d, retry_points=%d, eligible_points=%d, "
        "filled_points=%d, unresolved_points=%d, fully_filled_gaps=%d, "
        "partially_filled_gaps=%d, unresolved_gaps=%d, "
        "skipped_unusable_points=%d, processed_series=%d, "
        "skipped_series=%d.",
        totals["primary_gaps"],
        totals["primary_points"],
        totals["retry_gaps"],
        totals["retry_points"],
        totals["eligible_points"],
        totals["filled_points"],
        totals["unresolved_points"],
        totals["completely_filled_gaps"],
        totals["partially_filled_gaps"],
        totals["unresolved_gaps"],
        totals["skipped_unusable_points"],
        processed_series,
        skipped_series,
    )

    return Phase2FillingData(
        dataframe=output_dataframe,
        summary=summary,
        total_series=len(summary),
        processed_series=processed_series,
        skipped_series=skipped_series,
        primary_gaps=totals["primary_gaps"],
        primary_points=totals["primary_points"],
        retry_gaps=totals["retry_gaps"],
        retry_points=totals["retry_points"],
        already_resolved_phase1_points=(
            totals[
                "already_resolved_phase1_points"
            ]
        ),
        eligible_points=totals["eligible_points"],
        filled_points=totals["filled_points"],
        unresolved_points=totals["unresolved_points"],
        completely_filled_gaps=(
            totals["completely_filled_gaps"]
        ),
        partially_filled_gaps=(
            totals["partially_filled_gaps"]
        ),
        unresolved_gaps=totals["unresolved_gaps"],
        skipped_unusable_points=(
            totals["skipped_unusable_points"]
        ),
        method_counts=method_counts,
    )


def save_phase_2_outputs(
    *,
    filling_data: Phase2FillingData,
    source_file: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    """
    Save the Phase 2 dataset and per-site summary.

    Args:
        filling_data:
            In-memory Phase 2 result.

        source_file:
            Phase 1 source path used for output naming.

        output_directory:
            Destination directory supplied by the preprocessing runner.

    Returns:
        tuple[Path, Path]:
            Dataset path followed by summary path.

    Raises:
        FileProcessingError:
            If output files cannot be created.
    """

    resolved_output_directory = ensure_directory(
        output_directory
    )
    safe_stem = sanitise_filename(
        source_file.stem
    )

    output_file = (
        resolved_output_directory
        / f"{safe_stem}{OUTPUT_FILE_SUFFIX}"
    )
    summary_file = (
        resolved_output_directory
        / f"{safe_stem}{SUMMARY_FILE_SUFFIX}"
    )

    write_csv_file(
        filling_data.dataframe,
        output_file,
    )
    write_csv_file(
        filling_data.summary,
        summary_file,
    )

    return output_file, summary_file


def create_phase_2_filled_dataset(
    *,
    source_file: str | Path,
    output_directory: str | Path,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
) -> Phase2FillingResult:
    """
    Read, fill, and save one Phase 2 KPI dataset.

    The preprocessing runner should pass ``phase_1_result.output_file`` as
    ``source_file``. The runner should resolve
    ``PreprocessingConfig.resolve_for_bucket`` once and reuse the returned
    configuration.

    Args:
        source_file:
            Exact Phase 1 output dataset.

        output_directory:
            KPI preprocessing directory selected by the runner.

        columns:
            Shared dataset column mapping.

        config:
            Complete validated preprocessing configuration.

        resolved_frequency:
            Frequency-dependent Phase 2 settings.

        logger:
            Run-specific logger.

    Returns:
        Phase2FillingResult:
            Saved paths and Phase 2 statistics.

    Raises:
        FileProcessingError:
            If the source or output files cannot be processed.

        InputValidationError:
            If Phase 2 input, filling, or output invariants fail.
    """

    resolved_source_file = Path(
        source_file
    ).expanduser().resolve()
    resolved_output_directory = Path(
        output_directory
    ).expanduser().resolve()

    if not resolved_source_file.is_file():
        raise FileProcessingError(
            "The Phase 1 dataset does not exist.",
            error_code="MISSING_PHASE2_SOURCE_FILE",
            details={
                "source_file": str(
                    resolved_source_file
                ),
            },
        )

    logger.info(
        "Starting Phase 2 missing-value filling for %s.",
        resolved_source_file.name,
    )

    dataframe = read_csv_file(
        resolved_source_file
    )

    filling_data = fill_phase_2_missing_values(
        dataframe=dataframe,
        columns=columns,
        config=config,
        resolved_frequency=resolved_frequency,
        logger=logger,
    )

    output_file, summary_file = save_phase_2_outputs(
        filling_data=filling_data,
        source_file=resolved_source_file,
        output_directory=resolved_output_directory,
    )

    result = Phase2FillingResult(
        source_file=resolved_source_file,
        output_file=output_file,
        summary_file=summary_file,
        total_rows=len(filling_data.dataframe),
        total_series=filling_data.total_series,
        processed_series=filling_data.processed_series,
        skipped_series=filling_data.skipped_series,
        primary_gaps=filling_data.primary_gaps,
        primary_points=filling_data.primary_points,
        retry_gaps=filling_data.retry_gaps,
        retry_points=filling_data.retry_points,
        already_resolved_phase1_points=(
            filling_data.already_resolved_phase1_points
        ),
        eligible_points=filling_data.eligible_points,
        filled_points=filling_data.filled_points,
        unresolved_points=filling_data.unresolved_points,
        completely_filled_gaps=(
            filling_data.completely_filled_gaps
        ),
        partially_filled_gaps=(
            filling_data.partially_filled_gaps
        ),
        unresolved_gaps=filling_data.unresolved_gaps,
        skipped_unusable_points=(
            filling_data.skipped_unusable_points
        ),
        method_counts=dict(
            filling_data.method_counts
        ),
    )

    logger.info(
        "Phase 2 dataset saved to %s.",
        result.output_file,
    )
    logger.info(
        "Phase 2 summary saved to %s.",
        result.summary_file,
    )

    return result


__all__ = [
    "METHOD_LASTWEEK_EXACT_ONLY",
    "METHOD_LASTWEEK_NEIGHBORS_ONLY",
    "METHOD_RECENT_EXACT_LASTWEEK_EXACT",
    "METHOD_RECENT_EXACT_LASTWEEK_NEIGHBORS",
    "METHOD_RECENT_EXACT_ONLY",
    "METHOD_RECENT_NEIGHBORS_LASTWEEK_EXACT",
    "METHOD_RECENT_NEIGHBORS_LASTWEEK_NEIGHBORS",
    "METHOD_RECENT_NEIGHBORS_ONLY",
    "PHASE_2_METHODS",
    "Phase2FillingData",
    "Phase2FillingResult",
    "Phase2PointOutcome",
    "Phase2ReferenceSet",
    "create_phase_2_filled_dataset",
    "fill_phase_2_missing_values",
    "save_phase_2_outputs",
]