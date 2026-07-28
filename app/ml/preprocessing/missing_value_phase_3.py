"""
Fill original Phase 3 gaps and retry unresolved earlier-phase points.

This module runs after ``missing_value_phase_2.py``. It expects the immutable
``Original_Gap_Phase`` classifications produced by
``missing_gap_analysis.py`` and the working-value state produced by Phases 1
and 2.

Eligibility
-----------

Phase 3 processes:

- primary targets: unresolved rows with ``Original_Gap_Phase == 3``;
- retry targets: unresolved rows with ``Original_Gap_Phase`` equal to ``1``
  or ``2``.

Rows already filled safely by earlier phases retain
``Current_Status_Label == 3`` and are never overwritten. Invalid site series
are skipped.

Trusted references
------------------

A historical value is trusted only when:

- it belongs to the same site;
- ``Current_Status_Label`` is ``1`` or ``3``;
- its working KPI value is finite; and
- its working KPI value lies inside the configured valid range.

Labels ``2``, ``4``, ``0``, ``-1``, and ``-2`` are never references.
Previous safe fills from Phases 1 and 2, and earlier Phase 3 fills, may be
reused.

Filling ladder
--------------

For each unresolved target timestamp ``T``:

1. Level 1 same-type seasonal median:
   collect trusted values from the configured narrow window around up to the
   configured number of previous weekday/weekend-matching days. Require the
   resolved Level 1 minimum clean-value count.

2. Level 2 same-type seasonal median:
   collect trusted values from the configured wider window around the same
   previous same-type days. Require the configured Level 2 minimum.

3. Same-type exact-time median:
   collect trusted exact-time values from the same previous same-type days.
   Require at least one value.

4. Previous-day-any-type median:
   walk backward one calendar day at a time and use the first day having the
   configured minimum trusted values in the Level 1 window.

5. Failure:
   leave the point unchanged for Phase 4.

Every reference timestamp is earlier than the target. Therefore every
successful Phase 3 fill receives ``Current_Status_Label == 3``. This module
never assigns status label ``4``.

Only these fields may change:

- the configured KPI value column;
- ``Current_Status_Label``; and
- ``Fill_Method``.

All other columns, row count, and row order are preserved.
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
    PHASE_3,
)
from .missing_value_phase_1 import FILL_METHOD_COLUMN
from .timeline_correction import (
    CURRENT_STATUS_LABEL_COLUMN,
    INVALID_USABILITY_VALUE,
    KPI_USABILITY_COLUMN,
    ORIGINAL_CONDITION_LABEL_COLUMN,
    VALID_USABILITY_VALUE,
)


METHOD_LEVEL_1_SAME_TYPE_MEDIAN = (
    "phase3_same_type_median_level_1"
)
METHOD_LEVEL_2_SAME_TYPE_MEDIAN = (
    "phase3_same_type_median_level_2"
)
METHOD_SAME_TYPE_EXACT_MEDIAN = (
    "phase3_same_type_exact_median"
)
METHOD_PREVIOUS_DAY_ANY_TYPE_MEDIAN = (
    "phase3_previous_day_any_type_median"
)

PHASE_3_METHODS = {
    METHOD_LEVEL_1_SAME_TYPE_MEDIAN,
    METHOD_LEVEL_2_SAME_TYPE_MEDIAN,
    METHOD_SAME_TYPE_EXACT_MEDIAN,
    METHOD_PREVIOUS_DAY_ANY_TYPE_MEDIAN,
}

OUTPUT_FILE_SUFFIX = "_phase3_filled.csv"
SUMMARY_FILE_SUFFIX = "_phase3_summary.csv"

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
_STATUS_NO_TARGETS = "no_phase3_targets"

_INTERNAL_TIMESTAMP_COLUMN = "__phase3_timestamp__"
_INTERNAL_ROW_ORDER_COLUMN = "__phase3_row_order__"


@dataclass(frozen=True, slots=True)
class Phase3ReferencePool:
    """
    Store Phase 3 historical reference pools for one target timestamp.

    Attributes:
        level_1_values:
            Trusted values from the narrow windows around previous same-type
            days.

        level_2_values:
            Trusted values from the wider windows around previous same-type
            days.

        exact_values:
            Trusted exact-time values from previous same-type days.
    """

    level_1_values: tuple[float, ...]
    level_2_values: tuple[float, ...]
    exact_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Phase3PointOutcome:
    """
    Store one target-point Phase 3 result.

    Attributes:
        filled:
            Whether a Phase 3 method produced an accepted value.

        value:
            Generated value, or ``None`` when unresolved.

        method:
            Applied fill method, or ``None`` when unresolved.
    """

    filled: bool
    value: float | None
    method: str | None


@dataclass(frozen=True, slots=True)
class Phase3FillingData:
    """
    Store the in-memory Phase 3 dataset, summary, and statistics.

    Attributes:
        dataframe:
            Updated dataset.

        summary:
            One-row-per-site Phase 3 summary.

        total_series:
            Total number of site series.

        processed_series:
            Usable sites containing unresolved Phase 3 targets.

        skipped_series:
            Invalid sites containing unresolved targets.

        primary_gaps:
            Original Phase 3 gaps.

        primary_points:
            Rows belonging to original Phase 3 gaps.

        retry_gaps:
            Original Phase 1 or 2 gaps containing unresolved retry points.

        retry_points:
            Unresolved earlier-phase points retried by Phase 3.

        already_resolved_earlier_points:
            Original Phase 1 or 2 points already safely filled before Phase 3.

        eligible_points:
            Unresolved targets on usable sites.

        filled_points:
            Points successfully filled by Phase 3.

        unresolved_points:
            Eligible points left unresolved for Phase 4.

        completely_filled_gaps:
            Target gaps with no remaining target points after Phase 3.

        partially_filled_gaps:
            Target gaps containing both filled and unresolved target points.

        unresolved_gaps:
            Target gaps retaining at least one unresolved target.

        skipped_unusable_points:
            Targets skipped because the site is invalid.

        method_counts:
            Counts for each Phase 3 method.
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
    already_resolved_earlier_points: int
    eligible_points: int
    filled_points: int
    unresolved_points: int
    completely_filled_gaps: int
    partially_filled_gaps: int
    unresolved_gaps: int
    skipped_unusable_points: int
    method_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class Phase3FillingResult:
    """
    Store saved Phase 3 output paths and headline statistics.

    Attributes:
        source_file:
            Exact Phase 2 dataset supplied by the preprocessing runner.

        output_file:
            Newly written Phase 3 dataset.

        summary_file:
            Per-site Phase 3 summary CSV.

        total_rows:
            Number of rows preserved in the output.

        total_series:
            Total number of site series.

        processed_series:
            Usable sites containing unresolved Phase 3 targets.

        skipped_series:
            Invalid sites containing unresolved targets.

        primary_gaps:
            Original Phase 3 gaps.

        primary_points:
            Original Phase 3 points.

        retry_gaps:
            Earlier-phase gaps retried.

        retry_points:
            Unresolved earlier-phase points retried.

        already_resolved_earlier_points:
            Earlier-phase points already safely filled before Phase 3.

        eligible_points:
            Total usable unresolved targets.

        filled_points:
            Successfully filled targets.

        unresolved_points:
            Eligible targets left for Phase 4.

        completely_filled_gaps:
            Target gaps fully resolved by Phase 3.

        partially_filled_gaps:
            Target gaps partially resolved by Phase 3.

        unresolved_gaps:
            Target gaps retaining unresolved targets.

        skipped_unusable_points:
            Targets skipped because their site is invalid.

        method_counts:
            Counts for each Phase 3 method.
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
    already_resolved_earlier_points: int
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
            Dataset containing the configured site identity columns.

        columns:
            Shared physical dataset-column mapping.

    Returns:
        pd.DataFrame:
            Copy containing ``Composite_ID``.

    Raises:
        KeyError:
            If a configured site identity column is unavailable.
    """

    result = dataframe.copy()
    identity_columns = list(
        columns.get_site_identity_columns()
    )

    identity_values = (
        result[identity_columns]
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
    Parse and validate one required integer state column.

    Args:
        dataframe:
            Dataset containing the state column.

        column_name:
            Column to validate.

        allowed_values:
            Complete accepted integer set.

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
            f"{column_name} contains invalid Phase 3 values.",
            error_code="INVALID_PHASE3_INTEGER_COLUMN",
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
            Nullable ``Int8`` phase values.

    Raises:
        InputValidationError:
            If a non-null phase is non-integral or outside 1 through 5.
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
            "Original_Gap_Phase contains invalid Phase 3 values.",
            error_code="INVALID_PHASE3_GAP_PHASE",
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
    Validate the Phase 2 output schema required by Phase 3.

    Args:
        dataframe:
            Dataset returned by Phase 2.

        columns:
            Shared dataset column mapping.

    Returns:
        list[str]:
            Input columns in their existing order.

    Raises:
        InputValidationError:
            If the dataset is empty, required columns are missing, or an
            internal Phase 3 column is already present.
    """

    if dataframe.empty:
        raise InputValidationError(
            "Phase 3 cannot process an empty dataframe.",
            error_code="EMPTY_PHASE3_DATASET",
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
            "The Phase 2 dataset is missing columns required by Phase 3.",
            error_code="MISSING_PHASE3_COLUMNS",
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
    }
    conflicts = sorted(
        reserved_columns.intersection(
            dataframe.columns
        )
    )

    if conflicts:
        raise InputValidationError(
            "The Phase 2 dataset contains reserved Phase 3 columns.",
            error_code="RESERVED_PHASE3_COLUMN_CONFLICT",
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
            Normalized usability strings.

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
        | ~usability.isin(
            _ALLOWED_USABILITY_VALUES
        )
    )

    if invalid_mask.any():
        raise InputValidationError(
            "kpi_usability contains invalid Phase 3 values.",
            error_code="INVALID_PHASE3_USABILITY",
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
                COMPOSITE_ID_COLUMN: dataframe[
                    COMPOSITE_ID_COLUMN
                ],
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
            error_code="INCONSISTENT_PHASE3_USABILITY",
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
    Validate fill-method provenance and reject accidental Phase 3 reruns.

    Args:
        dataframe:
            Phase 2 dataset.

        current_labels:
            Parsed current labels.

    Returns:
        pd.Series:
            Normalized fill-method strings.

    Raises:
        InputValidationError:
            If Phase 3 methods already exist or provenance is inconsistent.
    """

    methods = (
        dataframe[FILL_METHOD_COLUMN]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    existing_phase3 = methods.str.startswith(
        "phase3_"
    )

    if existing_phase3.any():
        raise InputValidationError(
            "The dataset already contains Phase 3 fill methods.",
            error_code="PHASE3_ALREADY_APPLIED",
            details={
                "rows": int(existing_phase3.sum()),
                "sample_methods": (
                    methods.loc[existing_phase3]
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
            error_code="PHASE3_SAFE_FILL_WITHOUT_METHOD",
            details={
                "invalid_rows": int(
                    safe_without_method.sum()
                ),
            },
        )

    method_without_safe_label = (
        methods.ne("")
        & current_labels.ne(3)
    )

    if method_without_safe_label.any():
        raise InputValidationError(
            "Fill_Method is present on rows not labelled as safe fills.",
            error_code="PHASE3_METHOD_WITHOUT_SAFE_LABEL",
            details={
                "invalid_rows": int(
                    method_without_safe_label.sum()
                ),
            },
        )

    return methods


def prepare_phase3_dataframe(
    *,
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate and prepare one Phase 2 dataset for Phase 3 filling.

    Args:
        dataframe:
            Dataset returned by Phase 2.

        columns:
            Shared dataset column mapping.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Frequency-dependent Phase 3 settings.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            Prepared working dataframe and original input-column order.

    Raises:
        InputValidationError:
            If schema, labels, values, timeline, or resolved settings are
            inconsistent.
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
            "The Phase 2 dataset contains invalid timestamps.",
            error_code="INVALID_PHASE3_TIMESTAMPS",
            details={
                "invalid_rows": int(
                    invalid_timestamps.sum()
                ),
            },
        )

    working[_INTERNAL_TIMESTAMP_COLUMN] = (
        parsed_timestamps
    )

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

    working[ORIGINAL_CONDITION_LABEL_COLUMN] = (
        original_labels
    )
    working[CURRENT_STATUS_LABEL_COLUMN] = (
        current_labels
    )
    working[ORIGINAL_GAP_PHASE_COLUMN] = (
        gap_phases
    )
    working[KPI_USABILITY_COLUMN] = (
        validate_usability(
            working
        )
    )
    working[FILL_METHOD_COLUMN] = (
        validate_fill_methods(
            dataframe=working,
            current_labels=current_labels,
        )
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
            & ~current_labels.isin(
                {1, 2}
            )
        )
    )

    if invalid_transition.any():
        raise InputValidationError(
            "Original and current labels contain invalid pre-Phase-3 states.",
            error_code="INVALID_PHASE3_LABEL_TRANSITION",
            details={
                "invalid_rows": int(
                    invalid_transition.sum()
                ),
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

    if (
        missing_without_phase.any()
        or nonmissing_with_phase.any()
    ):
        raise InputValidationError(
            "Original_Gap_Phase does not match immutable original labels.",
            error_code="INVALID_PHASE3_GAP_PHASE_STATE",
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
    present_state = current_labels.isin(
        TRUSTED_CURRENT_LABELS | {2}
    )

    unresolved_with_value = (
        unresolved
        & numeric_values.notna()
    )
    present_state_without_value = (
        present_state
        & (
            numeric_values.isna()
            | ~np.isfinite(numeric_values)
        )
    )

    if (
        unresolved_with_value.any()
        or present_state_without_value.any()
    ):
        raise InputValidationError(
            "KPI values do not match current Phase 3 status labels.",
            error_code="INVALID_PHASE3_VALUE_STATE",
            details={
                "unresolved_with_value": int(
                    unresolved_with_value.sum()
                ),
                "present_state_without_value": int(
                    present_state_without_value.sum()
                ),
            },
        )

    lower_bound = config.valid_values.lower_bound
    upper_bound = config.valid_values.upper_bound

    present_outside_range = (
        numeric_values.notna()
        & (
            ~np.isfinite(numeric_values)
            | numeric_values.lt(lower_bound)
            | numeric_values.gt(upper_bound)
        )
    )

    if present_outside_range.any():
        raise InputValidationError(
            "The Phase 2 dataset contains present KPI values outside the "
            "configured valid range.",
            error_code="PHASE3_VALUE_OUTSIDE_VALID_RANGE",
            details={
                "invalid_rows": int(
                    present_outside_range.sum()
                ),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )

    primary_already_filled = (
        gap_phases.eq(PHASE_3)
        & current_labels.eq(3)
    )

    if primary_already_filled.any():
        raise InputValidationError(
            "Original Phase 3 rows are already safely filled before Phase 3.",
            error_code="PHASE3_PRIMARY_TARGET_ALREADY_FILLED",
            details={
                "invalid_rows": int(
                    primary_already_filled.sum()
                ),
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
            "Duplicate site-timestamp rows remain before Phase 3.",
            error_code="DUPLICATE_TIMESTAMPS_BEFORE_PHASE3",
            details={
                "duplicate_rows": int(
                    duplicate_mask.sum()
                ),
            },
        )

    expected_frequency = pd.Timedelta(
        minutes=resolved_frequency.bucket_minutes
    )

    if expected_frequency <= pd.Timedelta(0):
        raise InputValidationError(
            "The resolved Phase 3 frequency must be positive.",
            error_code="NON_POSITIVE_PHASE3_FREQUENCY",
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
            ordered_indices = list(
                timestamps.index
            )
            position = ordered_indices.index(
                bad_index
            )
            previous_index = ordered_indices[
                position - 1
            ]

            spacing_errors.append(
                {
                    COMPOSITE_ID_COLUMN: str(
                        composite_id
                    ),
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
            "One or more site timelines are incomplete before Phase 3.",
            error_code="INCOMPLETE_TIMELINE_BEFORE_PHASE3",
            details={
                "sample": spacing_errors,
            },
        )

    phase3_config = config.phase_3

    if (
        phase3_config.maximum_same_type_days <= 0
        or phase3_config.maximum_lookback_days <= 0
    ):
        raise InputValidationError(
            "Phase 3 historical lookback settings must be positive.",
            error_code="INVALID_PHASE3_LOOKBACK_SETTINGS",
        )

    if (
        phase3_config.maximum_lookback_days
        < phase3_config.maximum_same_type_days
    ):
        raise InputValidationError(
            "Phase 3 maximum lookback cannot be lower than the requested "
            "same-type day count.",
            error_code="INVALID_PHASE3_LOOKBACK_RELATIONSHIP",
        )

    resolved_values = {
        "phase_3_level_1_half_window_points": (
            resolved_frequency
            .phase_3_level_1_half_window_points
        ),
        "phase_3_level_2_half_window_points": (
            resolved_frequency
            .phase_3_level_2_half_window_points
        ),
        "phase_3_level_1_min_clean_values": (
            resolved_frequency
            .phase_3_level_1_min_clean_values
        ),
        "phase_3_level_2_min_clean_values": (
            phase3_config.level_2_min_clean_values
        ),
        "phase_3_level_4_min_clean_values": (
            phase3_config.level_4_min_clean_values
        ),
    }

    non_positive_resolved = {
        name: value
        for name, value in resolved_values.items()
        if value <= 0
    }

    if non_positive_resolved:
        raise InputValidationError(
            "Resolved Phase 3 point settings must be positive.",
            error_code="NON_POSITIVE_PHASE3_RESOLVED_SETTING",
            details={
                "settings": non_positive_resolved,
            },
        )

    if (
        resolved_frequency
        .phase_3_level_2_half_window_points
        <
        resolved_frequency
        .phase_3_level_1_half_window_points
    ):
        raise InputValidationError(
            "Resolved Phase 3 Level 2 window cannot be narrower than Level 1.",
            error_code="INVALID_PHASE3_RESOLVED_WINDOWS",
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

    return bool(
        timestamp.weekday() >= 5
    )


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
            Zero-based position in the site arrays.

        values:
            Current working KPI values.

        labels:
            Current status labels.

        lower_bound:
            Inclusive valid lower bound.

        upper_bound:
            Inclusive valid upper bound.

    Returns:
        bool:
            True when the value is a trusted reference.

    Raises:
        None.
    """

    if position < 0 or position >= len(
        values
    ):
        return False

    if int(labels[position]) not in (
        TRUSTED_CURRENT_LABELS
    ):
        return False

    value = values[position]

    return bool(
        np.isfinite(value)
        and lower_bound <= value <= upper_bound
    )


def get_trusted_value(
    *,
    timestamp: pd.Timestamp,
    timestamp_to_position: dict[pd.Timestamp, int],
    values: np.ndarray,
    labels: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> float | None:
    """
    Return a trusted value at one exact site timestamp.

    Args:
        timestamp:
            Historical timestamp.

        timestamp_to_position:
            Site timestamp-position lookup.

        values:
            Current working KPI values.

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

    position = timestamp_to_position.get(
        pd.Timestamp(timestamp)
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

    return float(
        values[position]
    )


def collect_window_values(
    *,
    anchor_timestamp: pd.Timestamp,
    half_window_points: int,
    bucket_step: pd.Timedelta,
    timestamp_to_position: dict[pd.Timestamp, int],
    values: np.ndarray,
    labels: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> list[float]:
    """
    Collect trusted values from an inclusive historical time window.

    Args:
        anchor_timestamp:
            Historical exact-time centre.

        half_window_points:
            Bucket steps included on either side.

        bucket_step:
            Resolved bucket duration.

        timestamp_to_position:
            Site timestamp-position lookup.

        values:
            Current working KPI values.

        labels:
            Current status labels.

        lower_bound:
            Inclusive valid lower bound.

        upper_bound:
            Inclusive valid upper bound.

    Returns:
        list[float]:
            Trusted values found in the window.

    Raises:
        None.
    """

    collected: list[float] = []

    for offset in range(
        -half_window_points,
        half_window_points + 1,
    ):
        candidate_timestamp = (
            anchor_timestamp
            + offset * bucket_step
        )

        trusted_value = get_trusted_value(
            timestamp=pd.Timestamp(
                candidate_timestamp
            ),
            timestamp_to_position=timestamp_to_position,
            values=values,
            labels=labels,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        if trusted_value is not None:
            collected.append(
                trusted_value
            )

    return collected


def get_previous_same_type_timestamps(
    *,
    target_timestamp: pd.Timestamp,
    timestamp_to_position: dict[pd.Timestamp, int],
    maximum_same_type_days: int,
    maximum_lookback_days: int,
) -> list[pd.Timestamp]:
    """
    Collect nearest earlier weekday/weekend-matching timestamps.

    Candidate timestamps preserve the target time of day and must exist in the
    site timeline. Their values may still be missing or untrusted.

    Args:
        target_timestamp:
            Unresolved target timestamp.

        timestamp_to_position:
            Complete site timeline lookup.

        maximum_same_type_days:
            Maximum matching days to collect.

        maximum_lookback_days:
            Maximum backward calendar-day search.

    Returns:
        list[pd.Timestamp]:
            Earlier matching timestamps ordered nearest to farthest.

    Raises:
        None.
    """

    target_is_weekend = is_weekend(
        target_timestamp
    )
    found: list[pd.Timestamp] = []

    for days_back in range(
        1,
        maximum_lookback_days + 1,
    ):
        candidate = pd.Timestamp(
            target_timestamp
            - pd.Timedelta(
                days=days_back
            )
        )

        if (
            is_weekend(candidate)
            != target_is_weekend
        ):
            continue

        if candidate not in (
            timestamp_to_position
        ):
            continue

        found.append(candidate)

        if len(found) >= (
            maximum_same_type_days
        ):
            break

    return found


def collect_same_type_reference_pool(
    *,
    target_timestamp: pd.Timestamp,
    timestamp_to_position: dict[pd.Timestamp, int],
    values: np.ndarray,
    labels: np.ndarray,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> Phase3ReferencePool:
    """
    Collect all same-type reference pools for one target timestamp.

    Args:
        target_timestamp:
            Unresolved target timestamp.

        timestamp_to_position:
            Complete site timeline lookup.

        values:
            Current working values.

        labels:
            Current status labels.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Resolved Phase 3 windows.

    Returns:
        Phase3ReferencePool:
            Narrow-window, wide-window, and exact-time values.

    Raises:
        None.
    """

    phase3_config = config.phase_3
    lower_bound = (
        config.valid_values.lower_bound
    )
    upper_bound = (
        config.valid_values.upper_bound
    )
    bucket_step = pd.Timedelta(
        minutes=resolved_frequency.bucket_minutes
    )

    same_type_timestamps = (
        get_previous_same_type_timestamps(
            target_timestamp=target_timestamp,
            timestamp_to_position=timestamp_to_position,
            maximum_same_type_days=(
                phase3_config.maximum_same_type_days
            ),
            maximum_lookback_days=(
                phase3_config.maximum_lookback_days
            ),
        )
    )

    level_1_values: list[float] = []
    level_2_values: list[float] = []
    exact_values: list[float] = []

    for anchor_timestamp in (
        same_type_timestamps
    ):
        level_1_values.extend(
            collect_window_values(
                anchor_timestamp=anchor_timestamp,
                half_window_points=(
                    resolved_frequency
                    .phase_3_level_1_half_window_points
                ),
                bucket_step=bucket_step,
                timestamp_to_position=timestamp_to_position,
                values=values,
                labels=labels,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        )

        level_2_values.extend(
            collect_window_values(
                anchor_timestamp=anchor_timestamp,
                half_window_points=(
                    resolved_frequency
                    .phase_3_level_2_half_window_points
                ),
                bucket_step=bucket_step,
                timestamp_to_position=timestamp_to_position,
                values=values,
                labels=labels,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        )

        exact_value = get_trusted_value(
            timestamp=anchor_timestamp,
            timestamp_to_position=timestamp_to_position,
            values=values,
            labels=labels,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        if exact_value is not None:
            exact_values.append(
                exact_value
            )

    return Phase3ReferencePool(
        level_1_values=tuple(
            level_1_values
        ),
        level_2_values=tuple(
            level_2_values
        ),
        exact_values=tuple(
            exact_values
        ),
    )


def find_previous_day_any_type_values(
    *,
    target_timestamp: pd.Timestamp,
    timestamp_to_position: dict[pd.Timestamp, int],
    values: np.ndarray,
    labels: np.ndarray,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> list[float]:
    """
    Find the first previous day of any type with enough trusted values.

    Args:
        target_timestamp:
            Unresolved target timestamp.

        timestamp_to_position:
            Complete site timeline lookup.

        values:
            Current working values.

        labels:
            Current status labels.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Resolved bucket and Level 1 window.

    Returns:
        list[float]:
            Trusted values from the first qualifying day, or an empty list.

    Raises:
        None.
    """

    phase3_config = config.phase_3
    bucket_step = pd.Timedelta(
        minutes=resolved_frequency.bucket_minutes
    )
    lower_bound = (
        config.valid_values.lower_bound
    )
    upper_bound = (
        config.valid_values.upper_bound
    )

    for days_back in range(
        1,
        phase3_config.maximum_lookback_days + 1,
    ):
        anchor_timestamp = pd.Timestamp(
            target_timestamp
            - pd.Timedelta(
                days=days_back
            )
        )

        if anchor_timestamp not in (
            timestamp_to_position
        ):
            continue

        candidate_values = collect_window_values(
            anchor_timestamp=anchor_timestamp,
            half_window_points=(
                resolved_frequency
                .phase_3_level_1_half_window_points
            ),
            bucket_step=bucket_step,
            timestamp_to_position=timestamp_to_position,
            values=values,
            labels=labels,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        if len(candidate_values) >= (
            phase3_config
            .level_4_min_clean_values
        ):
            return candidate_values

    return []


def normalize_generated_value(
    *,
    value: float | None,
    config: PreprocessingConfig,
) -> float | None:
    """
    Validate and optionally clip one generated Phase 3 value.

    Args:
        value:
            Raw generated median.

        config:
            Complete preprocessing configuration.

    Returns:
        float | None:
            Finite accepted value, or ``None``.

    Raises:
        None.
    """

    if (
        value is None
        or not np.isfinite(value)
    ):
        return None

    generated = float(value)
    lower_bound = (
        config.valid_values.lower_bound
    )
    upper_bound = (
        config.valid_values.upper_bound
    )

    if (
        config.valid_values
        .clip_generated_values
    ):
        generated = float(
            np.clip(
                generated,
                lower_bound,
                upper_bound,
            )
        )

    if not (
        lower_bound
        <= generated
        <= upper_bound
    ):
        return None

    return generated


def select_phase3_fill(
    *,
    target_timestamp: pd.Timestamp,
    timestamp_to_position: dict[pd.Timestamp, int],
    values: np.ndarray,
    labels: np.ndarray,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
) -> Phase3PointOutcome:
    """
    Apply the ordered four-level Phase 3 median ladder.

    Args:
        target_timestamp:
            Unresolved target timestamp.

        timestamp_to_position:
            Complete site timeline lookup.

        values:
            Current working values.

        labels:
            Current status labels.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Resolved Phase 3 windows and minimum.

    Returns:
        Phase3PointOutcome:
            Selected generated value and method, or unresolved result.

    Raises:
        None.
    """

    reference_pool = (
        collect_same_type_reference_pool(
            target_timestamp=target_timestamp,
            timestamp_to_position=timestamp_to_position,
            values=values,
            labels=labels,
            config=config,
            resolved_frequency=resolved_frequency,
        )
    )

    if len(
        reference_pool.level_1_values
    ) >= (
        resolved_frequency
        .phase_3_level_1_min_clean_values
    ):
        generated = normalize_generated_value(
            value=float(
                np.median(
                    reference_pool
                    .level_1_values
                )
            ),
            config=config,
        )

        if generated is not None:
            return Phase3PointOutcome(
                filled=True,
                value=generated,
                method=(
                    METHOD_LEVEL_1_SAME_TYPE_MEDIAN
                ),
            )

    if len(
        reference_pool.level_2_values
    ) >= (
        config.phase_3
        .level_2_min_clean_values
    ):
        generated = normalize_generated_value(
            value=float(
                np.median(
                    reference_pool
                    .level_2_values
                )
            ),
            config=config,
        )

        if generated is not None:
            return Phase3PointOutcome(
                filled=True,
                value=generated,
                method=(
                    METHOD_LEVEL_2_SAME_TYPE_MEDIAN
                ),
            )

    if reference_pool.exact_values:
        generated = normalize_generated_value(
            value=float(
                np.median(
                    reference_pool.exact_values
                )
            ),
            config=config,
        )

        if generated is not None:
            return Phase3PointOutcome(
                filled=True,
                value=generated,
                method=(
                    METHOD_SAME_TYPE_EXACT_MEDIAN
                ),
            )

    any_type_values = (
        find_previous_day_any_type_values(
            target_timestamp=target_timestamp,
            timestamp_to_position=timestamp_to_position,
            values=values,
            labels=labels,
            config=config,
            resolved_frequency=resolved_frequency,
        )
    )

    if any_type_values:
        generated = normalize_generated_value(
            value=float(
                np.median(
                    any_type_values
                )
            ),
            config=config,
        )

        if generated is not None:
            return Phase3PointOutcome(
                filled=True,
                value=generated,
                method=(
                    METHOD_PREVIOUS_DAY_ANY_TYPE_MEDIAN
                ),
            )

    return Phase3PointOutcome(
        filled=False,
        value=None,
        method=None,
    )


def assign_original_gap_numbers(
    ordered: pd.DataFrame,
) -> pd.Series:
    """
    Assign one-based original gap numbers within a timestamp-sorted site.

    Args:
        ordered:
            Prepared site dataframe ordered by timestamp.

    Returns:
        pd.Series:
            Nullable gap numbers aligned with ``ordered``.

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

    gap_numbers = (
        new_gap.cumsum().astype("Int64")
    )

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
    Calculate site-level Phase 3 target and gap-resolution statistics.

    Args:
        gap_numbers:
            Original gap number per row.

        gap_phases:
            Original gap-phase values.

        before_labels:
            Current labels before Phase 3.

        after_labels:
            Current labels after Phase 3.

        usability:
            Site usability value.

    Returns:
        dict[str, int]:
            Site target, fill, and gap-resolution counts.

    Raises:
        None.
    """

    statistics = {
        "primary_gaps": 0,
        "primary_points": 0,
        "retry_gaps": 0,
        "retry_points": 0,
        "already_resolved_earlier_points": 0,
        "eligible_points": 0,
        "filled_points": 0,
        "unresolved_points": 0,
        "completely_filled_gaps": 0,
        "partially_filled_gaps": 0,
        "unresolved_gaps": 0,
        "skipped_unusable_points": 0,
    }

    phase1_or_2_mask = np.isin(
        gap_phases,
        [PHASE_1, PHASE_2],
    )
    phase3_mask = gap_phases == PHASE_3

    statistics["primary_points"] = int(
        phase3_mask.sum()
    )
    statistics[
        "already_resolved_earlier_points"
    ] = int(
        (
            phase1_or_2_mask
            & (before_labels == 3)
        ).sum()
    )

    retry_target_mask = (
        phase1_or_2_mask
        & np.isin(
            before_labels,
            list(
                UNRESOLVED_CURRENT_LABELS
            ),
        )
    )
    primary_target_mask = (
        phase3_mask
        & np.isin(
            before_labels,
            list(
                UNRESOLVED_CURRENT_LABELS
            ),
        )
    )
    target_mask = (
        retry_target_mask
        | primary_target_mask
    )

    statistics["retry_points"] = int(
        retry_target_mask.sum()
    )

    gap_number_array = (
        gap_numbers.to_numpy(
            dtype=object
        )
    )

    def gap_ids_for_mask(
        mask: np.ndarray,
    ) -> set[int]:
        """
        Return concrete original gap identifiers selected by a row mask.

        Args:
            mask:
                Boolean row mask aligned with the site arrays.

        Returns:
            set[int]:
                Selected gap identifiers.

        Raises:
            None.
        """

        result: set[int] = set()

        for position in np.flatnonzero(
            mask
        ):
            raw_gap_id = (
                gap_number_array[position]
            )

            if pd.isna(raw_gap_id):
                continue

            result.add(
                int(raw_gap_id)
            )

        return result

    primary_gap_ids = gap_ids_for_mask(
        phase3_mask
    )
    retry_gap_ids = gap_ids_for_mask(
        retry_target_mask
    )

    statistics["primary_gaps"] = len(
        primary_gap_ids
    )
    statistics["retry_gaps"] = len(
        retry_gap_ids
    )

    if usability == (
        INVALID_USABILITY_VALUE
    ):
        statistics[
            "skipped_unusable_points"
        ] = int(
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
            list(
                UNRESOLVED_CURRENT_LABELS
            ),
        )
    )

    statistics["filled_points"] = int(
        filled_target_mask.sum()
    )
    statistics["unresolved_points"] = int(
        unresolved_target_mask.sum()
    )

    target_gap_ids = sorted(
        primary_gap_ids
        | retry_gap_ids
    )

    for gap_id in target_gap_ids:
        gap_mask = np.array(
            [
                (
                    not pd.isna(raw_gap_id)
                    and int(raw_gap_id)
                    == gap_id
                )
                for raw_gap_id in (
                    gap_number_array
                )
            ],
            dtype=bool,
        ) & target_mask

        if not gap_mask.any():
            continue

        gap_filled = int(
            filled_target_mask[
                gap_mask
            ].sum()
        )
        gap_unresolved = int(
            unresolved_target_mask[
                gap_mask
            ].sum()
        )

        if gap_unresolved == 0:
            statistics[
                "completely_filled_gaps"
            ] += 1
        else:
            statistics[
                "unresolved_gaps"
            ] += 1

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
    Apply Phase 3 filling to one site series.

    Targets are processed chronologically. Successful safe fills become
    available to later target timestamps.

    Args:
        site_dataframe:
            Prepared rows for one site.

        columns:
            Shared dataset column mapping.

        config:
            Complete preprocessing configuration.

        resolved_frequency:
            Resolved Phase 3 settings.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]:
            Updated site dataframe and one per-site summary row.

    Raises:
        InputValidationError:
            If a target has an invalid original state.
    """

    ordered = site_dataframe.sort_values(
        _INTERNAL_TIMESTAMP_COLUMN,
        kind="stable",
    ).copy()

    timestamps = [
        pd.Timestamp(timestamp)
        for timestamp in ordered[
            _INTERNAL_TIMESTAMP_COLUMN
        ].tolist()
    ]
    timestamp_to_position = {
        timestamp: position
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
        .to_numpy(
            dtype=float,
            na_value=np.nan,
        )
    )
    fill_methods = (
        ordered[FILL_METHOD_COLUMN]
        .astype("string")
        .fillna("")
        .to_numpy(dtype=object)
    )

    before_labels = labels.copy()
    gap_numbers = (
        assign_original_gap_numbers(
            ordered
        )
    )
    usability = str(
        ordered[
            KPI_USABILITY_COLUMN
        ].iloc[0]
    )

    method_counts = {
        method: 0
        for method in sorted(
            PHASE_3_METHODS
        )
    }

    target_mask = (
        np.isin(
            gap_phases,
            [
                PHASE_1,
                PHASE_2,
                PHASE_3,
            ],
        )
        & np.isin(
            labels,
            list(
                UNRESOLVED_CURRENT_LABELS
            ),
        )
    )

    if (
        usability
        == VALID_USABILITY_VALUE
        and target_mask.any()
    ):
        for position in np.flatnonzero(
            target_mask
        ):
            if (
                int(
                    original_labels[position]
                )
                not in MISSING_ORIGINAL_LABELS
            ):
                raise InputValidationError(
                    "A Phase 3 target does not have a missing-origin label.",
                    error_code="INVALID_PHASE3_TARGET_ORIGIN",
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
                not in (
                    UNRESOLVED_CURRENT_LABELS
                )
                or np.isfinite(
                    values[position]
                )
            ):
                continue

            outcome = select_phase3_fill(
                target_timestamp=(
                    timestamps[position]
                ),
                timestamp_to_position=timestamp_to_position,
                values=values,
                labels=labels,
                config=config,
                resolved_frequency=resolved_frequency,
            )

            if (
                not outcome.filled
                or outcome.value is None
                or outcome.method is None
            ):
                continue

            values[position] = (
                outcome.value
            )
            labels[position] = 3
            fill_methods[position] = (
                outcome.method
            )
            method_counts[
                outcome.method
            ] += 1

    ordered[columns.value_column] = (
        values
    )
    ordered[CURRENT_STATUS_LABEL_COLUMN] = (
        labels
    )
    ordered[FILL_METHOD_COLUMN] = (
        fill_methods
    )

    statistics = summarize_target_gaps(
        gap_numbers=gap_numbers,
        gap_phases=gap_phases,
        before_labels=before_labels,
        after_labels=labels,
        usability=usability,
    )

    if not target_mask.any():
        status = _STATUS_NO_TARGETS
    elif usability == (
        INVALID_USABILITY_VALUE
    ):
        status = _STATUS_SKIPPED_UNUSABLE
    else:
        status = _STATUS_PROCESSED

    summary_row: dict[str, Any] = {
        COMPOSITE_ID_COLUMN: str(
            ordered[
                COMPOSITE_ID_COLUMN
            ].iloc[0]
        ),
        KPI_USABILITY_COLUMN: usability,
        "status": status,
        **statistics,
    }

    for method in sorted(
        PHASE_3_METHODS
    ):
        summary_row[
            f"{method}_points"
        ] = method_counts[method]

    return ordered, summary_row


def validate_phase3_output(
    *,
    input_dataframe: pd.DataFrame,
    output_dataframe: pd.DataFrame,
    input_columns: list[str],
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
) -> None:
    """
    Validate Phase 3 output invariants.

    Args:
        input_dataframe:
            Phase 2 input dataset.

        output_dataframe:
            Phase 3 result.

        input_columns:
            Original input-column order.

        columns:
            Shared dataset column mapping.

        config:
            Complete preprocessing configuration.

    Returns:
        None.

    Raises:
        InputValidationError:
            If schema, row count, protected fields, labels, values, or method
            provenance changed incorrectly.
    """

    if (
        list(output_dataframe.columns)
        != input_columns
    ):
        raise InputValidationError(
            "Phase 3 changed the dataset schema or column order.",
            error_code="INVALID_PHASE3_OUTPUT_COLUMNS",
            details={
                "expected_columns": input_columns,
                "actual_columns": list(
                    output_dataframe.columns
                ),
            },
        )

    if len(output_dataframe) != len(
        input_dataframe
    ):
        raise InputValidationError(
            "Phase 3 changed the dataset row count.",
            error_code="PHASE3_ROW_COUNT_CHANGED",
            details={
                "input_rows": len(
                    input_dataframe
                ),
                "output_rows": len(
                    output_dataframe
                ),
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
            and not output_dataframe[
                column
            ].equals(
                input_dataframe[column]
            )
        )
    ]

    if changed_protected:
        raise InputValidationError(
            "Phase 3 modified protected input columns.",
            error_code="PHASE3_MODIFIED_PROTECTED_COLUMNS",
            details={
                "changed_columns": (
                    changed_protected
                ),
            },
        )

    input_labels = pd.to_numeric(
        input_dataframe[
            CURRENT_STATUS_LABEL_COLUMN
        ],
        errors="coerce",
    )
    output_labels = pd.to_numeric(
        output_dataframe[
            CURRENT_STATUS_LABEL_COLUMN
        ],
        errors="coerce",
    )
    input_values = pd.to_numeric(
        input_dataframe[
            columns.value_column
        ],
        errors="coerce",
    )
    output_values = pd.to_numeric(
        output_dataframe[
            columns.value_column
        ],
        errors="coerce",
    )
    input_methods = (
        input_dataframe[
            FILL_METHOD_COLUMN
        ]
        .astype("string")
        .fillna("")
        .str.strip()
    )
    output_methods = (
        output_dataframe[
            FILL_METHOD_COLUMN
        ]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    newly_filled = (
        input_labels.isin(
            UNRESOLVED_CURRENT_LABELS
        )
        & output_labels.eq(3)
    )

    illegal_label_change = (
        ~newly_filled
        & output_labels.ne(
            input_labels
        )
    )

    if illegal_label_change.any():
        raise InputValidationError(
            "Phase 3 changed current labels outside successful targets.",
            error_code="ILLEGAL_PHASE3_LABEL_CHANGE",
            details={
                "invalid_rows": int(
                    illegal_label_change.sum()
                ),
            },
        )

    unchanged_row_value_changed = (
        ~newly_filled
        & ~(
            output_values.eq(
                input_values
            )
            | (
                output_values.isna()
                & input_values.isna()
            )
        )
    )

    if (
        unchanged_row_value_changed.any()
    ):
        raise InputValidationError(
            "Phase 3 changed KPI values outside newly filled targets.",
            error_code="ILLEGAL_PHASE3_VALUE_CHANGE",
            details={
                "invalid_rows": int(
                    unchanged_row_value_changed
                    .sum()
                ),
            },
        )

    new_method_mask = (
        output_methods.isin(
            PHASE_3_METHODS
        )
    )

    if not new_method_mask.equals(
        newly_filled
    ):
        raise InputValidationError(
            "Phase 3 method provenance does not match newly filled rows.",
            error_code="INVALID_PHASE3_METHOD_ASSIGNMENT",
            details={
                "method_rows": int(
                    new_method_mask.sum()
                ),
                "newly_filled_rows": int(
                    newly_filled.sum()
                ),
            },
        )

    earlier_methods_changed = (
        ~newly_filled
        & output_methods.ne(
            input_methods
        )
    )

    if earlier_methods_changed.any():
        raise InputValidationError(
            "Phase 3 changed earlier fill-method provenance.",
            error_code="PHASE3_CHANGED_EARLIER_METHODS",
            details={
                "invalid_rows": int(
                    earlier_methods_changed.sum()
                ),
            },
        )

    filled_values_invalid = (
        newly_filled
        & (
            output_values.isna()
            | ~np.isfinite(
                output_values
            )
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
            "Phase 3 generated invalid filled values.",
            error_code="INVALID_PHASE3_GENERATED_VALUE",
            details={
                "invalid_rows": int(
                    filled_values_invalid.sum()
                ),
            },
        )

    still_unresolved = (
        input_labels.isin(
            UNRESOLVED_CURRENT_LABELS
        )
        & output_labels.isin(
            UNRESOLVED_CURRENT_LABELS
        )
    )

    unresolved_state_invalid = (
        still_unresolved
        & (
            output_values.notna()
            | output_methods.ne(
                input_methods
            )
        )
    )

    if unresolved_state_invalid.any():
        raise InputValidationError(
            "Unresolved Phase 3 targets were not left unchanged.",
            error_code="INVALID_UNRESOLVED_PHASE3_STATE",
            details={
                "invalid_rows": int(
                    unresolved_state_invalid.sum()
                ),
            },
        )


def fill_phase_3_missing_values(
    *,
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
) -> Phase3FillingData:
    """
    Fill original Phase 3 gaps and retry unresolved Phase 1/2 points.

    Args:
        dataframe:
            Dataset returned by Phase 2.

        columns:
            Shared dataset column mapping.

        config:
            Complete validated preprocessing configuration.

        resolved_frequency:
            Frequency-dependent Phase 3 settings.

        logger:
            Run-specific logger.

    Returns:
        Phase3FillingData:
            Updated dataframe, per-site summary, and aggregate counts.

    Raises:
        InputValidationError:
            If validation, filling, or output invariants fail.
    """

    working, input_columns = (
        prepare_phase3_dataframe(
            dataframe=dataframe,
            columns=columns,
            config=config,
            resolved_frequency=resolved_frequency,
        )
    )

    updated = working.copy()
    summary_rows: list[
        dict[str, Any]
    ] = []

    grouped_indices = updated.groupby(
        COMPOSITE_ID_COLUMN,
        sort=False,
        dropna=False,
    ).groups

    for _, indices in (
        grouped_indices.items()
    ):
        site_dataframe = updated.loc[
            indices
        ].copy()

        (
            processed_site,
            summary_row,
        ) = process_one_site(
            site_dataframe=site_dataframe,
            columns=columns,
            config=config,
            resolved_frequency=resolved_frequency,
        )

        source_indices = (
            processed_site.index
        )

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

        summary_rows.append(
            summary_row
        )

    aligned = updated.reindex(
        dataframe.index
    )

    output_dataframe = dataframe.copy()
    output_dataframe[
        columns.value_column
    ] = aligned[
        columns.value_column
    ].to_numpy()
    output_dataframe[
        CURRENT_STATUS_LABEL_COLUMN
    ] = aligned[
        CURRENT_STATUS_LABEL_COLUMN
    ].to_numpy()
    output_dataframe[
        FILL_METHOD_COLUMN
    ] = aligned[
        FILL_METHOD_COLUMN
    ].to_numpy()

    validate_phase3_output(
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
        "already_resolved_earlier_points",
        "eligible_points",
        "filled_points",
        "unresolved_points",
        "completely_filled_gaps",
        "partially_filled_gaps",
        "unresolved_gaps",
        "skipped_unusable_points",
        *[
            f"{method}_points"
            for method in sorted(
                PHASE_3_METHODS
            )
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
        "already_resolved_earlier_points",
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
        for column in (
            aggregate_columns
        )
    }

    method_counts = {
        method: int(
            pd.to_numeric(
                summary[
                    f"{method}_points"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )
        for method in sorted(
            PHASE_3_METHODS
        )
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
        "Phase 3 filling completed: primary_gaps=%d, primary_points=%d, "
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

    return Phase3FillingData(
        dataframe=output_dataframe,
        summary=summary,
        total_series=len(summary),
        processed_series=processed_series,
        skipped_series=skipped_series,
        primary_gaps=totals[
            "primary_gaps"
        ],
        primary_points=totals[
            "primary_points"
        ],
        retry_gaps=totals[
            "retry_gaps"
        ],
        retry_points=totals[
            "retry_points"
        ],
        already_resolved_earlier_points=(
            totals[
                "already_resolved_earlier_points"
            ]
        ),
        eligible_points=totals[
            "eligible_points"
        ],
        filled_points=totals[
            "filled_points"
        ],
        unresolved_points=totals[
            "unresolved_points"
        ],
        completely_filled_gaps=(
            totals[
                "completely_filled_gaps"
            ]
        ),
        partially_filled_gaps=(
            totals[
                "partially_filled_gaps"
            ]
        ),
        unresolved_gaps=totals[
            "unresolved_gaps"
        ],
        skipped_unusable_points=(
            totals[
                "skipped_unusable_points"
            ]
        ),
        method_counts=method_counts,
    )


def save_phase_3_outputs(
    *,
    filling_data: Phase3FillingData,
    source_file: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    """
    Save the Phase 3 dataset and per-site summary.

    Args:
        filling_data:
            In-memory Phase 3 result.

        source_file:
            Phase 2 source path used for output naming.

        output_directory:
            Destination directory supplied by the preprocessing runner.

    Returns:
        tuple[Path, Path]:
            Dataset path followed by summary path.

    Raises:
        FileProcessingError:
            If either output cannot be created.
    """

    resolved_output_directory = (
        ensure_directory(
            output_directory
        )
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


def create_phase_3_filled_dataset(
    *,
    source_file: str | Path,
    output_directory: str | Path,
    columns: DatasetColumnMapping,
    config: PreprocessingConfig,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
) -> Phase3FillingResult:
    """
    Read, fill, and save one Phase 3 KPI dataset.

    The preprocessing runner should pass ``phase_2_result.output_file`` as
    ``source_file`` and reuse the one ``ResolvedFrequencyConfig`` created for
    the KPI run.

    Args:
        source_file:
            Exact Phase 2 output dataset.

        output_directory:
            KPI preprocessing directory selected by the runner.

        columns:
            Shared dataset column mapping.

        config:
            Complete validated preprocessing configuration.

        resolved_frequency:
            Frequency-dependent Phase 3 settings.

        logger:
            Run-specific logger.

    Returns:
        Phase3FillingResult:
            Saved output paths and Phase 3 statistics.

    Raises:
        FileProcessingError:
            If source or output files cannot be processed.

        InputValidationError:
            If Phase 3 input, filling, or output invariants fail.
    """

    resolved_source_file = Path(
        source_file
    ).expanduser().resolve()
    resolved_output_directory = Path(
        output_directory
    ).expanduser().resolve()

    if not resolved_source_file.is_file():
        raise FileProcessingError(
            "The Phase 2 dataset does not exist.",
            error_code="MISSING_PHASE3_SOURCE_FILE",
            details={
                "source_file": str(
                    resolved_source_file
                ),
            },
        )

    logger.info(
        "Starting Phase 3 missing-value filling for %s.",
        resolved_source_file.name,
    )

    dataframe = read_csv_file(
        resolved_source_file
    )

    filling_data = (
        fill_phase_3_missing_values(
            dataframe=dataframe,
            columns=columns,
            config=config,
            resolved_frequency=resolved_frequency,
            logger=logger,
        )
    )

    (
        output_file,
        summary_file,
    ) = save_phase_3_outputs(
        filling_data=filling_data,
        source_file=resolved_source_file,
        output_directory=(
            resolved_output_directory
        ),
    )

    result = Phase3FillingResult(
        source_file=resolved_source_file,
        output_file=output_file,
        summary_file=summary_file,
        total_rows=len(
            filling_data.dataframe
        ),
        total_series=(
            filling_data.total_series
        ),
        processed_series=(
            filling_data.processed_series
        ),
        skipped_series=(
            filling_data.skipped_series
        ),
        primary_gaps=(
            filling_data.primary_gaps
        ),
        primary_points=(
            filling_data.primary_points
        ),
        retry_gaps=(
            filling_data.retry_gaps
        ),
        retry_points=(
            filling_data.retry_points
        ),
        already_resolved_earlier_points=(
            filling_data
            .already_resolved_earlier_points
        ),
        eligible_points=(
            filling_data.eligible_points
        ),
        filled_points=(
            filling_data.filled_points
        ),
        unresolved_points=(
            filling_data.unresolved_points
        ),
        completely_filled_gaps=(
            filling_data
            .completely_filled_gaps
        ),
        partially_filled_gaps=(
            filling_data
            .partially_filled_gaps
        ),
        unresolved_gaps=(
            filling_data.unresolved_gaps
        ),
        skipped_unusable_points=(
            filling_data
            .skipped_unusable_points
        ),
        method_counts=dict(
            filling_data.method_counts
        ),
    )

    logger.info(
        "Phase 3 dataset saved to %s.",
        result.output_file,
    )
    logger.info(
        "Phase 3 summary saved to %s.",
        result.summary_file,
    )

    return result


__all__ = [
    "METHOD_LEVEL_1_SAME_TYPE_MEDIAN",
    "METHOD_LEVEL_2_SAME_TYPE_MEDIAN",
    "METHOD_PREVIOUS_DAY_ANY_TYPE_MEDIAN",
    "METHOD_SAME_TYPE_EXACT_MEDIAN",
    "PHASE_3_METHODS",
    "Phase3FillingData",
    "Phase3FillingResult",
    "Phase3PointOutcome",
    "Phase3ReferencePool",
    "create_phase_3_filled_dataset",
    "fill_phase_3_missing_values",
    "save_phase_3_outputs",
]