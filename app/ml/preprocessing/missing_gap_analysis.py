"""
Classify original missing-value gaps into preprocessing phases.

This module runs after ``outlier_labelling.py`` and before any missing-value
filling stage. It expects the dataset shape produced by the earlier Part B
stages and adds one new column:

    Original_Gap_Phase

The phase column is assigned only to rows that belonged to an original missing
gap. Missing rows are identified exclusively from the immutable
``Original_Condition_Label``:

- ``-2``: timestamp inserted during timeline correction;
- ``-1``: original invalid value converted to ``NaN``; and
- ``0``: originally missing value.

Outlier rows are not treated as missing because they retain
``Original_Condition_Label == 1`` even when ``Current_Status_Label == 2``.

Phase values:

- ``1``: Phase 1;
- ``2``: Phase 2;
- ``3``: Phase 3;
- ``4``: Phase 4 Pass 1;
- ``5``: Phase 4 Pass 2.

Non-missing rows receive a blank nullable value in ``Original_Gap_Phase``.

The complete gap profile is received through ``ResolvedFrequencyConfig``.
Therefore, built-in 5-minute or 15-minute ranges and user-supplied custom
ranges are handled through the same implementation.

This module does not fill values, modify status labels, change KPI values, or
skip invalid site series. It classifies every original gap in the dataset.
Later filling stages decide whether a site is eligible for filling.
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
    GapProfileConfig,
    GapRangeConfig,
    ResolvedFrequencyConfig,
)
from app.services.file_service import (
    ensure_directory,
    read_csv_file,
    sanitise_filename,
    write_csv_file,
)

from .timeline_correction import (
    CURRENT_STATUS_LABEL_COLUMN,
    INVALID_USABILITY_VALUE,
    KPI_USABILITY_COLUMN,
    ORIGINAL_CONDITION_LABEL_COLUMN,
    VALID_USABILITY_VALUE,
)


ORIGINAL_GAP_PHASE_COLUMN = "Original_Gap_Phase"

OUTPUT_FILE_SUFFIX = "_missing_gap_classified.csv"

PHASE_1 = 1
PHASE_2 = 2
PHASE_3 = 3
PHASE_4_PASS_1 = 4
PHASE_4_PASS_2 = 5

MISSING_ORIGINAL_LABELS = {-2, -1, 0}
_ALLOWED_ORIGINAL_LABELS = {-2, -1, 0, 1}
_ALLOWED_CURRENT_LABELS = {-2, -1, 0, 1, 2}
_ALLOWED_GAP_PHASES = {
    PHASE_1,
    PHASE_2,
    PHASE_3,
    PHASE_4_PASS_1,
    PHASE_4_PASS_2,
}

_INTERNAL_PARSED_TIMESTAMP_COLUMN = "__gap_analysis_timestamp__"
_INTERNAL_ROW_ORDER_COLUMN = "__gap_analysis_row_order__"


@dataclass(frozen=True, slots=True)
class MissingGapRecord:
    """
    Store one original continuous missing-value gap.

    Attributes:
        composite_id:
            Stable site identifier.

        gap_number:
            One-based gap number within the site, ordered by timestamp.

        gap_start:
            First timestamp in the original gap.

        gap_end:
            Last timestamp in the original gap.

        gap_size:
            Number of consecutive original-missing points.

        gap_phase:
            Assigned phase number from 1 to 5.
    """

    composite_id: str
    gap_number: int
    gap_start: pd.Timestamp
    gap_end: pd.Timestamp
    gap_size: int
    gap_phase: int


@dataclass(frozen=True, slots=True)
class MissingGapAnalysisData:
    """
    Store the in-memory missing-gap classification result.

    Attributes:
        dataframe:
            Input dataframe copy with ``Original_Gap_Phase`` appended.

        gap_summary:
            One row per original continuous missing-value gap.

        total_gaps:
            Total number of identified gaps.

        total_missing_points:
            Number of rows carrying original labels ``-2``, ``-1``, or ``0``.

        phase_1_gaps:
            Number of gaps assigned to Phase 1.

        phase_2_gaps:
            Number of gaps assigned to Phase 2.

        phase_3_gaps:
            Number of gaps assigned to Phase 3.

        phase_4_pass_1_gaps:
            Number of gaps assigned to Phase 4 Pass 1.

        phase_4_pass_2_gaps:
            Number of gaps assigned to Phase 4 Pass 2.

        phase_1_points:
            Number of missing rows assigned to Phase 1.

        phase_2_points:
            Number of missing rows assigned to Phase 2.

        phase_3_points:
            Number of missing rows assigned to Phase 3.

        phase_4_pass_1_points:
            Number of missing rows assigned to Phase 4 Pass 1.

        phase_4_pass_2_points:
            Number of missing rows assigned to Phase 4 Pass 2.
    """

    dataframe: pd.DataFrame
    gap_summary: pd.DataFrame
    total_gaps: int
    total_missing_points: int
    phase_1_gaps: int
    phase_2_gaps: int
    phase_3_gaps: int
    phase_4_pass_1_gaps: int
    phase_4_pass_2_gaps: int
    phase_1_points: int
    phase_2_points: int
    phase_3_points: int
    phase_4_pass_1_points: int
    phase_4_pass_2_points: int


@dataclass(frozen=True, slots=True)
class MissingGapAnalysisResult:
    """
    Store the saved missing-gap dataset and headline statistics.

    Attributes:
        source_file:
            Outlier-labelled dataset supplied by the preprocessing runner.

        output_file:
            New dataset containing ``Original_Gap_Phase``.

        total_rows:
            Number of input rows preserved in the output.

        total_series:
            Number of unique site series.

        total_gaps:
            Number of original continuous missing-value gaps.

        total_missing_points:
            Number of original missing rows classified.

        phase_1_gaps:
            Number of Phase 1 gaps.

        phase_2_gaps:
            Number of Phase 2 gaps.

        phase_3_gaps:
            Number of Phase 3 gaps.

        phase_4_pass_1_gaps:
            Number of Phase 4 Pass 1 gaps.

        phase_4_pass_2_gaps:
            Number of Phase 4 Pass 2 gaps.

        phase_1_points:
            Number of rows belonging to Phase 1 gaps.

        phase_2_points:
            Number of rows belonging to Phase 2 gaps.

        phase_3_points:
            Number of rows belonging to Phase 3 gaps.

        phase_4_pass_1_points:
            Number of rows belonging to Phase 4 Pass 1 gaps.

        phase_4_pass_2_points:
            Number of rows belonging to Phase 4 Pass 2 gaps.
    """

    source_file: Path
    output_file: Path
    total_rows: int
    total_series: int
    total_gaps: int
    total_missing_points: int
    phase_1_gaps: int
    phase_2_gaps: int
    phase_3_gaps: int
    phase_4_pass_1_gaps: int
    phase_4_pass_2_gaps: int
    phase_1_points: int
    phase_2_points: int
    phase_3_points: int
    phase_4_pass_1_points: int
    phase_4_pass_2_points: int


def add_composite_id(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """
    Return a dataframe copy containing the Part A composite site identifier.

    Args:
        dataframe:
            Dataset containing the configured site identity columns.

        columns:
            Shared physical column mapping.

    Returns:
        pd.DataFrame:
            Copy containing ``Composite_ID``.

    Raises:
        KeyError:
            If a configured identity column is unavailable.
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


def validate_integer_label_column(
    dataframe: pd.DataFrame,
    column_name: str,
    allowed_values: set[int],
) -> pd.Series:
    """
    Parse and validate one required integer label column.

    Args:
        dataframe:
            Dataset containing the label column.

        column_name:
            Physical label column to validate.

        allowed_values:
            Complete accepted integer set.

    Returns:
        pd.Series:
            Parsed ``int8`` label values aligned with the dataframe.

    Raises:
        InputValidationError:
            If values are missing, non-integral, or outside the allowed set.
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
            f"{column_name} contains invalid values for missing-gap analysis.",
            error_code="INVALID_GAP_ANALYSIS_LABEL_COLUMN",
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


def validate_required_columns(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> list[str]:
    """
    Validate the outlier-labelled dataset schema.

    Args:
        dataframe:
            Outlier-labelled dataset.

        columns:
            Shared dataset column mapping.

    Returns:
        list[str]:
            Input columns in their existing order.

    Raises:
        InputValidationError:
            If the dataframe is empty, a required column is absent, or the
            output phase column already exists.
    """

    if dataframe.empty:
        raise InputValidationError(
            "Missing-gap analysis cannot process an empty dataframe.",
            error_code="EMPTY_MISSING_GAP_DATASET",
        )

    required_columns = [
        *columns.get_site_identity_columns(),
        columns.timestamp_column,
        columns.value_column,
        ORIGINAL_CONDITION_LABEL_COLUMN,
        CURRENT_STATUS_LABEL_COLUMN,
        KPI_USABILITY_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "The outlier-labelled dataset is missing columns required for "
            "missing-gap analysis.",
            error_code="MISSING_GAP_ANALYSIS_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "required_columns": required_columns,
                "available_columns": list(dataframe.columns),
            },
        )

    if ORIGINAL_GAP_PHASE_COLUMN in dataframe.columns:
        raise InputValidationError(
            "The dataset already contains Original_Gap_Phase. Missing-gap "
            "analysis will not overwrite an earlier classification.",
            error_code="GAP_PHASE_COLUMN_ALREADY_EXISTS",
        )

    reserved_conflicts = [
        column
        for column in (
            COMPOSITE_ID_COLUMN,
            _INTERNAL_PARSED_TIMESTAMP_COLUMN,
            _INTERNAL_ROW_ORDER_COLUMN,
        )
        if column in dataframe.columns
    ]

    if reserved_conflicts:
        raise InputValidationError(
            "The dataset contains reserved missing-gap analysis columns.",
            error_code="RESERVED_GAP_ANALYSIS_COLUMN_CONFLICT",
            details={
                "conflicting_columns": reserved_conflicts,
            },
        )

    return list(dataframe.columns)


def validate_usability_values(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Validate site usability values and site-level consistency.

    Both valid and invalid site series are classified. The usability column is
    validated only to ensure that the input is the expected output of timeline
    correction and outlier labelling.

    Args:
        dataframe:
            Dataset containing ``kpi_usability`` and ``Composite_ID``.

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

    allowed_values = {
        VALID_USABILITY_VALUE,
        INVALID_USABILITY_VALUE,
    }

    invalid_mask = (
        usability.isna()
        | ~usability.isin(allowed_values)
    )

    if invalid_mask.any():
        raise InputValidationError(
            "kpi_usability contains invalid values.",
            error_code="INVALID_KPI_USABILITY_FOR_GAP_ANALYSIS",
            details={
                "allowed_values": sorted(allowed_values),
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

    usability_count_by_site = (
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

    inconsistent_sites = usability_count_by_site.loc[
        usability_count_by_site != 1
    ]

    if not inconsistent_sites.empty:
        raise InputValidationError(
            "kpi_usability is not constant within one or more site series.",
            error_code="INCONSISTENT_KPI_USABILITY_FOR_GAP_ANALYSIS",
            details={
                "sample_composite_ids": (
                    inconsistent_sites.index
                    .astype(str)
                    .tolist()[:20]
                ),
            },
        )

    return usability


def prepare_gap_analysis_dataframe(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    resolved_frequency: ResolvedFrequencyConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate and prepare the outlier-labelled dataset for gap classification.

    This function validates:

    - required columns;
    - timestamp parsing;
    - original and current status labels;
    - allowed label transitions;
    - value-to-label consistency;
    - site usability;
    - duplicate site-timestamp keys; and
    - complete bucket spacing within every site.

    Args:
        dataframe:
            Outlier-labelled dataset.

        columns:
            Shared dataset column mapping.

        resolved_frequency:
            Frequency-dependent settings resolved by the preprocessing runner.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            Prepared working dataframe and original input column order.

    Raises:
        InputValidationError:
            If any dataset invariant required by gap analysis fails.
    """

    input_columns = validate_required_columns(
        dataframe=dataframe,
        columns=columns,
    )

    result = add_composite_id(
        dataframe=dataframe,
        columns=columns,
    )

    parsed_timestamps = pd.to_datetime(
        result[columns.timestamp_column],
        errors="coerce",
    )

    invalid_timestamp_mask = parsed_timestamps.isna()

    if invalid_timestamp_mask.any():
        raise InputValidationError(
            "The outlier-labelled dataset contains missing or invalid "
            "timestamps.",
            error_code="INVALID_TIMESTAMPS_FOR_GAP_ANALYSIS",
            details={
                "invalid_rows": int(invalid_timestamp_mask.sum()),
                "sample_values": (
                    result.loc[
                        invalid_timestamp_mask,
                        columns.timestamp_column,
                    ]
                    .head(20)
                    .astype("string")
                    .tolist()
                ),
            },
        )

    result[_INTERNAL_PARSED_TIMESTAMP_COLUMN] = parsed_timestamps
    result[_INTERNAL_ROW_ORDER_COLUMN] = np.arange(
        len(result),
        dtype=np.int64,
    )

    original_labels = validate_integer_label_column(
        dataframe=result,
        column_name=ORIGINAL_CONDITION_LABEL_COLUMN,
        allowed_values=_ALLOWED_ORIGINAL_LABELS,
    )
    current_labels = validate_integer_label_column(
        dataframe=result,
        column_name=CURRENT_STATUS_LABEL_COLUMN,
        allowed_values=_ALLOWED_CURRENT_LABELS,
    )

    result[ORIGINAL_CONDITION_LABEL_COLUMN] = original_labels
    result[CURRENT_STATUS_LABEL_COLUMN] = current_labels

    invalid_transition_mask = (
        (
            original_labels.isin(MISSING_ORIGINAL_LABELS)
            & current_labels.ne(original_labels)
        )
        | (
            original_labels.eq(1)
            & ~current_labels.isin({1, 2})
        )
    )

    if invalid_transition_mask.any():
        raise InputValidationError(
            "Original and current status labels contain transitions that are "
            "invalid before missing-value filling.",
            error_code="INVALID_STATUS_TRANSITION_FOR_GAP_ANALYSIS",
            details={
                "invalid_rows": int(invalid_transition_mask.sum()),
                "sample": (
                    result.loc[
                        invalid_transition_mask,
                        [
                            COMPOSITE_ID_COLUMN,
                            columns.timestamp_column,
                            ORIGINAL_CONDITION_LABEL_COLUMN,
                            CURRENT_STATUS_LABEL_COLUMN,
                        ],
                    ]
                    .head(20)
                    .to_dict(orient="records")
                ),
            },
        )

    numeric_values = pd.to_numeric(
        result[columns.value_column],
        errors="coerce",
    )

    missing_origin_mask = original_labels.isin(
        MISSING_ORIGINAL_LABELS
    )
    valid_origin_mask = original_labels.eq(1)

    missing_label_with_value_mask = (
        missing_origin_mask
        & numeric_values.notna()
    )
    valid_label_without_value_mask = (
        valid_origin_mask
        & (
            numeric_values.isna()
            | ~np.isfinite(numeric_values)
        )
    )

    if missing_label_with_value_mask.any():
        raise InputValidationError(
            "Rows with original labels -2, -1, or 0 must have missing KPI "
            "values before gap analysis.",
            error_code="MISSING_LABEL_WITH_PRESENT_VALUE",
            details={
                "invalid_rows": int(
                    missing_label_with_value_mask.sum()
                ),
            },
        )

    if valid_label_without_value_mask.any():
        raise InputValidationError(
            "Rows with Original_Condition_Label 1 must contain finite KPI "
            "values before gap analysis.",
            error_code="VALID_LABEL_WITH_MISSING_VALUE",
            details={
                "invalid_rows": int(
                    valid_label_without_value_mask.sum()
                ),
            },
        )

    result[KPI_USABILITY_COLUMN] = validate_usability_values(
        result
    )

    duplicate_mask = result.duplicated(
        subset=[
            COMPOSITE_ID_COLUMN,
            _INTERNAL_PARSED_TIMESTAMP_COLUMN,
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise InputValidationError(
            "Duplicate site-timestamp rows remain before missing-gap analysis.",
            error_code="DUPLICATE_TIMESTAMPS_FOR_GAP_ANALYSIS",
            details={
                "duplicate_rows": int(duplicate_mask.sum()),
                "sample": (
                    result.loc[
                        duplicate_mask,
                        [
                            COMPOSITE_ID_COLUMN,
                            columns.timestamp_column,
                        ],
                    ]
                    .head(20)
                    .to_dict(orient="records")
                ),
            },
        )

    expected_frequency = pd.Timedelta(
        minutes=resolved_frequency.bucket_minutes
    )

    if expected_frequency <= pd.Timedelta(0):
        raise InputValidationError(
            "The resolved bucket frequency must be positive.",
            error_code="NON_POSITIVE_GAP_ANALYSIS_FREQUENCY",
            details={
                "bucket_minutes": resolved_frequency.bucket_minutes,
            },
        )

    spacing_errors: list[dict[str, Any]] = []

    for composite_id, site_dataframe in result.groupby(
        COMPOSITE_ID_COLUMN,
        sort=False,
        dropna=False,
    ):
        ordered_timestamps = (
            site_dataframe[_INTERNAL_PARSED_TIMESTAMP_COLUMN]
            .sort_values(kind="stable")
        )
        timestamp_differences = ordered_timestamps.diff()
        invalid_spacing = (
            timestamp_differences.notna()
            & timestamp_differences.ne(expected_frequency)
        )

        if invalid_spacing.any():
            first_bad_index = invalid_spacing.loc[
                invalid_spacing
            ].index[0]

            ordered_positions = list(ordered_timestamps.index)
            current_position = ordered_positions.index(first_bad_index)
            previous_index = ordered_positions[current_position - 1]

            spacing_errors.append(
                {
                    COMPOSITE_ID_COLUMN: str(composite_id),
                    "previous_timestamp": str(
                        result.at[
                            previous_index,
                            _INTERNAL_PARSED_TIMESTAMP_COLUMN,
                        ]
                    ),
                    "current_timestamp": str(
                        result.at[
                            first_bad_index,
                            _INTERNAL_PARSED_TIMESTAMP_COLUMN,
                        ]
                    ),
                    "actual_difference": str(
                        result.at[
                            first_bad_index,
                            _INTERNAL_PARSED_TIMESTAMP_COLUMN,
                        ]
                        - result.at[
                            previous_index,
                            _INTERNAL_PARSED_TIMESTAMP_COLUMN,
                        ]
                    ),
                    "expected_difference": str(expected_frequency),
                }
            )

            if len(spacing_errors) >= 20:
                break

    if spacing_errors:
        raise InputValidationError(
            "One or more site timelines are not complete at the resolved "
            "bucket frequency. Missing-gap analysis expects the output of "
            "timeline correction.",
            error_code="INCOMPLETE_TIMELINE_FOR_GAP_ANALYSIS",
            details={
                "sample": spacing_errors,
            },
        )

    return result, input_columns


def gap_range_contains(
    gap_range: GapRangeConfig,
    gap_size: int,
) -> bool:
    """
    Determine whether one gap size belongs to an inclusive configured range.

    Args:
        gap_range:
            Validated minimum and optional maximum point range.

        gap_size:
            Positive original gap size.

    Returns:
        bool:
            True when the size belongs to the range.

    Raises:
        None.
    """

    if gap_size < gap_range.minimum_points:
        return False

    if gap_range.maximum_points is None:
        return True

    return gap_size <= gap_range.maximum_points


def classify_gap_phase(
    gap_size: int,
    gap_profile: GapProfileConfig,
) -> int:
    """
    Map one positive original gap size to exactly one phase number.

    Args:
        gap_size:
            Number of consecutive original missing points.

        gap_profile:
            Complete validated user or built-in gap profile.

    Returns:
        int:
            Phase number 1, 2, 3, 4, or 5.

    Raises:
        InputValidationError:
            If the size does not match exactly one configured phase.
    """

    if gap_size <= 0:
        raise InputValidationError(
            "An original missing gap must contain at least one point.",
            error_code="NON_POSITIVE_ORIGINAL_GAP_SIZE",
            details={
                "gap_size": gap_size,
            },
        )

    phase_ranges = (
        (PHASE_1, gap_profile.phase_1),
        (PHASE_2, gap_profile.phase_2),
        (PHASE_3, gap_profile.phase_3),
        (PHASE_4_PASS_1, gap_profile.phase_4_pass_1),
        (PHASE_4_PASS_2, gap_profile.phase_4_pass_2),
    )

    matching_phases = [
        phase
        for phase, gap_range in phase_ranges
        if gap_range_contains(
            gap_range=gap_range,
            gap_size=gap_size,
        )
    ]

    if len(matching_phases) != 1:
        raise InputValidationError(
            "The original gap size does not map to exactly one configured "
            "phase.",
            error_code="AMBIGUOUS_GAP_PHASE_ASSIGNMENT",
            details={
                "gap_size": gap_size,
                "matching_phases": matching_phases,
            },
        )

    return matching_phases[0]


def classify_site_gaps(
    site_dataframe: pd.DataFrame,
    gap_profile: GapProfileConfig,
) -> tuple[pd.Series, list[MissingGapRecord]]:
    """
    Classify every continuous original missing gap in one site series.

    The site dataframe is ordered by parsed timestamp only for classification.
    Returned phase values remain indexed by the original dataframe row index so
    the complete dataset's row order can be preserved.

    Args:
        site_dataframe:
            Prepared rows belonging to one unique site.

        gap_profile:
            Complete resolved gap profile.

    Returns:
        tuple[pd.Series, list[MissingGapRecord]]:
            Nullable row-aligned phase assignments and gap records.

    Raises:
        InputValidationError:
            If a gap cannot be mapped to exactly one phase.
    """

    ordered = site_dataframe.sort_values(
        _INTERNAL_PARSED_TIMESTAMP_COLUMN,
        kind="stable",
    )

    original_labels = ordered[
        ORIGINAL_CONDITION_LABEL_COLUMN
    ]
    missing_mask = original_labels.isin(
        MISSING_ORIGINAL_LABELS
    )

    phase_values = pd.Series(
        pd.array(
            [pd.NA] * len(ordered),
            dtype="Int8",
        ),
        index=ordered.index,
        name=ORIGINAL_GAP_PHASE_COLUMN,
    )

    if not missing_mask.any():
        return phase_values, []

    new_gap_mask = (
        missing_mask
        & ~missing_mask.shift(
            1,
            fill_value=False,
        )
    )
    gap_ids = new_gap_mask.cumsum()

    records: list[MissingGapRecord] = []
    composite_id = str(
        ordered[COMPOSITE_ID_COLUMN].iloc[0]
    )
    gap_number = 0

    missing_rows = ordered.loc[
        missing_mask
    ].copy()
    missing_rows["__original_gap_id__"] = (
        gap_ids.loc[missing_mask]
    )

    for _, gap_dataframe in missing_rows.groupby(
        "__original_gap_id__",
        sort=True,
        dropna=False,
    ):
        gap_number += 1
        gap_size = int(len(gap_dataframe))
        gap_phase = classify_gap_phase(
            gap_size=gap_size,
            gap_profile=gap_profile,
        )

        phase_values.loc[
            gap_dataframe.index
        ] = gap_phase

        records.append(
            MissingGapRecord(
                composite_id=composite_id,
                gap_number=gap_number,
                gap_start=pd.Timestamp(
                    gap_dataframe[
                        _INTERNAL_PARSED_TIMESTAMP_COLUMN
                    ].min()
                ),
                gap_end=pd.Timestamp(
                    gap_dataframe[
                        _INTERNAL_PARSED_TIMESTAMP_COLUMN
                    ].max()
                ),
                gap_size=gap_size,
                gap_phase=gap_phase,
            )
        )

    return phase_values, records


def build_gap_summary(
    records: list[MissingGapRecord],
) -> pd.DataFrame:
    """
    Convert gap records into a one-row-per-gap dataframe.

    The summary is returned in memory for logging, tests, and future runner
    use. This module saves only the classified dataset.

    Args:
        records:
            Gap records from all site series.

    Returns:
        pd.DataFrame:
            Gap summary with stable column order.

    Raises:
        None.
    """

    columns = [
        COMPOSITE_ID_COLUMN,
        "Original_Gap_Number",
        "Original_Gap_Start",
        "Original_Gap_End",
        "Original_Gap_Size",
        ORIGINAL_GAP_PHASE_COLUMN,
    ]

    if not records:
        return pd.DataFrame(
            columns=columns
        )

    return pd.DataFrame(
        [
            {
                COMPOSITE_ID_COLUMN: record.composite_id,
                "Original_Gap_Number": record.gap_number,
                "Original_Gap_Start": record.gap_start,
                "Original_Gap_End": record.gap_end,
                "Original_Gap_Size": record.gap_size,
                ORIGINAL_GAP_PHASE_COLUMN: record.gap_phase,
            }
            for record in records
        ],
        columns=columns,
    )


def calculate_phase_statistics(
    gap_summary: pd.DataFrame,
) -> dict[str, int]:
    """
    Calculate gap and point counts for every phase.

    Args:
        gap_summary:
            One-row-per-gap summary.

    Returns:
        dict[str, int]:
            Total and phase-specific gap and point counts.

    Raises:
        None.
    """

    statistics = {
        "total_gaps": int(len(gap_summary)),
        "total_missing_points": 0,
        "phase_1_gaps": 0,
        "phase_2_gaps": 0,
        "phase_3_gaps": 0,
        "phase_4_pass_1_gaps": 0,
        "phase_4_pass_2_gaps": 0,
        "phase_1_points": 0,
        "phase_2_points": 0,
        "phase_3_points": 0,
        "phase_4_pass_1_points": 0,
        "phase_4_pass_2_points": 0,
    }

    if gap_summary.empty:
        return statistics

    gap_sizes = pd.to_numeric(
        gap_summary["Original_Gap_Size"],
        errors="coerce",
    )
    phases = pd.to_numeric(
        gap_summary[ORIGINAL_GAP_PHASE_COLUMN],
        errors="coerce",
    )

    statistics["total_missing_points"] = int(
        gap_sizes.sum()
    )

    mapping = {
        PHASE_1: (
            "phase_1_gaps",
            "phase_1_points",
        ),
        PHASE_2: (
            "phase_2_gaps",
            "phase_2_points",
        ),
        PHASE_3: (
            "phase_3_gaps",
            "phase_3_points",
        ),
        PHASE_4_PASS_1: (
            "phase_4_pass_1_gaps",
            "phase_4_pass_1_points",
        ),
        PHASE_4_PASS_2: (
            "phase_4_pass_2_gaps",
            "phase_4_pass_2_points",
        ),
    }

    for phase, (
        gap_count_key,
        point_count_key,
    ) in mapping.items():
        phase_mask = phases.eq(phase)
        statistics[gap_count_key] = int(
            phase_mask.sum()
        )
        statistics[point_count_key] = int(
            gap_sizes.loc[phase_mask].sum()
        )

    return statistics


def validate_gap_analysis_output(
    input_dataframe: pd.DataFrame,
    output_dataframe: pd.DataFrame,
    input_columns: list[str],
) -> None:
    """
    Validate that classification changed only by appending the phase column.

    Args:
        input_dataframe:
            Original outlier-labelled input.

        output_dataframe:
            Classified output.

        input_columns:
            Input columns in original order.

    Returns:
        None.

    Raises:
        InputValidationError:
            If rows, original columns, labels, values, or phase assignments
            were unexpectedly changed.
    """

    expected_columns = [
        *input_columns,
        ORIGINAL_GAP_PHASE_COLUMN,
    ]

    if list(output_dataframe.columns) != expected_columns:
        raise InputValidationError(
            "Missing-gap analysis produced an unexpected output schema.",
            error_code="INVALID_GAP_ANALYSIS_OUTPUT_COLUMNS",
            details={
                "expected_columns": expected_columns,
                "actual_columns": list(output_dataframe.columns),
            },
        )

    if len(output_dataframe) != len(input_dataframe):
        raise InputValidationError(
            "Missing-gap analysis changed the dataset row count.",
            error_code="GAP_ANALYSIS_ROW_COUNT_CHANGED",
            details={
                "input_rows": len(input_dataframe),
                "output_rows": len(output_dataframe),
            },
        )

    if not output_dataframe[input_columns].equals(
        input_dataframe[input_columns]
    ):
        changed_columns = [
            column
            for column in input_columns
            if not output_dataframe[column].equals(
                input_dataframe[column]
            )
        ]

        raise InputValidationError(
            "Missing-gap analysis modified protected input columns.",
            error_code="GAP_ANALYSIS_MODIFIED_INPUT_COLUMNS",
            details={
                "changed_columns": changed_columns,
            },
        )

    original_labels = pd.to_numeric(
        output_dataframe[
            ORIGINAL_CONDITION_LABEL_COLUMN
        ],
        errors="coerce",
    )
    missing_mask = original_labels.isin(
        MISSING_ORIGINAL_LABELS
    )
    phase_values = output_dataframe[
        ORIGINAL_GAP_PHASE_COLUMN
    ]

    missing_without_phase = (
        missing_mask
        & phase_values.isna()
    )
    non_missing_with_phase = (
        ~missing_mask
        & phase_values.notna()
    )
    invalid_phase = (
        phase_values.notna()
        & ~phase_values.isin(_ALLOWED_GAP_PHASES)
    )

    if (
        missing_without_phase.any()
        or non_missing_with_phase.any()
        or invalid_phase.any()
    ):
        raise InputValidationError(
            "Original_Gap_Phase assignments do not match immutable original "
            "missing labels.",
            error_code="INVALID_ORIGINAL_GAP_PHASE_ASSIGNMENTS",
            details={
                "missing_without_phase": int(
                    missing_without_phase.sum()
                ),
                "non_missing_with_phase": int(
                    non_missing_with_phase.sum()
                ),
                "invalid_phase_rows": int(
                    invalid_phase.sum()
                ),
            },
        )


def analyse_missing_gaps(
    *,
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
) -> MissingGapAnalysisData:
    """
    Classify every original missing-value gap in an outlier-labelled dataset.

    Args:
        dataframe:
            Dataset returned by ``outlier_labelling.py``.

        columns:
            Shared dataset column mapping.

        resolved_frequency:
            Result of ``PreprocessingConfig.resolve_for_bucket``. Its
            ``gap_profile`` may be a built-in profile or a complete custom
            profile supplied by the user.

        logger:
            Run-specific logger.

    Returns:
        MissingGapAnalysisData:
            Classified dataframe, in-memory gap summary, and counts.

    Raises:
        InputValidationError:
            If the input dataset or classification result violates required
            invariants.
    """

    working, input_columns = (
        prepare_gap_analysis_dataframe(
            dataframe=dataframe,
            columns=columns,
            resolved_frequency=resolved_frequency,
        )
    )

    phase_assignments = pd.Series(
        pd.array(
            [pd.NA] * len(working),
            dtype="Int8",
        ),
        index=working.index,
        name=ORIGINAL_GAP_PHASE_COLUMN,
    )
    all_records: list[MissingGapRecord] = []

    for _, site_dataframe in working.groupby(
        COMPOSITE_ID_COLUMN,
        sort=False,
        dropna=False,
    ):
        site_phases, site_records = (
            classify_site_gaps(
                site_dataframe=site_dataframe,
                gap_profile=(
                    resolved_frequency.gap_profile
                ),
            )
        )

        phase_assignments.loc[
            site_phases.index
        ] = site_phases
        all_records.extend(site_records)

    classified = dataframe.copy()
    classified[ORIGINAL_GAP_PHASE_COLUMN] = (
        phase_assignments.reindex(
            classified.index
        ).astype("Int8")
    )

    gap_summary = build_gap_summary(
        records=all_records,
    )
    statistics = calculate_phase_statistics(
        gap_summary=gap_summary,
    )

    validate_gap_analysis_output(
        input_dataframe=dataframe,
        output_dataframe=classified,
        input_columns=input_columns,
    )

    labelled_missing_points = int(
        classified[ORIGINAL_GAP_PHASE_COLUMN]
        .notna()
        .sum()
    )

    if (
        labelled_missing_points
        != statistics["total_missing_points"]
    ):
        raise InputValidationError(
            "Gap-summary point counts do not match row-level phase labels.",
            error_code="GAP_ANALYSIS_POINT_ACCOUNTING_MISMATCH",
            details={
                "row_level_missing_points": labelled_missing_points,
                "summary_missing_points": (
                    statistics["total_missing_points"]
                ),
            },
        )

    logger.info(
        "Missing-gap analysis completed: gaps=%d, missing_points=%d, "
        "phase1_gaps=%d, phase2_gaps=%d, phase3_gaps=%d, "
        "phase4_pass1_gaps=%d, phase4_pass2_gaps=%d.",
        statistics["total_gaps"],
        statistics["total_missing_points"],
        statistics["phase_1_gaps"],
        statistics["phase_2_gaps"],
        statistics["phase_3_gaps"],
        statistics["phase_4_pass_1_gaps"],
        statistics["phase_4_pass_2_gaps"],
    )

    return MissingGapAnalysisData(
        dataframe=classified,
        gap_summary=gap_summary,
        total_gaps=statistics["total_gaps"],
        total_missing_points=(
            statistics["total_missing_points"]
        ),
        phase_1_gaps=statistics["phase_1_gaps"],
        phase_2_gaps=statistics["phase_2_gaps"],
        phase_3_gaps=statistics["phase_3_gaps"],
        phase_4_pass_1_gaps=(
            statistics["phase_4_pass_1_gaps"]
        ),
        phase_4_pass_2_gaps=(
            statistics["phase_4_pass_2_gaps"]
        ),
        phase_1_points=statistics["phase_1_points"],
        phase_2_points=statistics["phase_2_points"],
        phase_3_points=statistics["phase_3_points"],
        phase_4_pass_1_points=(
            statistics["phase_4_pass_1_points"]
        ),
        phase_4_pass_2_points=(
            statistics["phase_4_pass_2_points"]
        ),
    )


def save_missing_gap_dataset(
    *,
    analysed_data: MissingGapAnalysisData,
    source_file: Path,
    output_directory: Path,
) -> Path:
    """
    Save the classified dataset to the runner-supplied directory.

    Args:
        analysed_data:
            In-memory missing-gap classification result.

        source_file:
            Outlier-labelled source path used for output naming.

        output_directory:
            Destination directory supplied by the preprocessing runner.

    Returns:
        Path:
            Written classified dataset path.

    Raises:
        FileProcessingError:
            If output creation fails.
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

    write_csv_file(
        analysed_data.dataframe,
        output_file,
    )

    return output_file


def create_missing_gap_analysed_dataset(
    *,
    source_file: str | Path,
    output_directory: str | Path,
    columns: DatasetColumnMapping,
    resolved_frequency: ResolvedFrequencyConfig,
    logger: Logger,
) -> MissingGapAnalysisResult:
    """
    Read, classify, and save one outlier-labelled KPI dataset.

    The future preprocessing runner should provide
    ``outlier_result.output_file`` as ``source_file`` and the already resolved
    frequency configuration produced once from the user's preprocessing
    settings.

    Args:
        source_file:
            Exact outlier-labelled dataset path returned by the previous stage.

        output_directory:
            KPI preprocessing directory selected by the runner.

        columns:
            Shared dataset column mapping.

        resolved_frequency:
            Frequency-dependent settings containing the complete gap profile.

        logger:
            Run-specific logger.

    Returns:
        MissingGapAnalysisResult:
            Saved output path and phase-specific counts.

    Raises:
        FileProcessingError:
            If the source file is missing or the output cannot be written.

        InputValidationError:
            If input validation or classification accounting fails.
    """

    resolved_source_file = Path(
        source_file
    ).expanduser().resolve()
    resolved_output_directory = Path(
        output_directory
    ).expanduser().resolve()

    if not resolved_source_file.is_file():
        raise FileProcessingError(
            "The outlier-labelled dataset does not exist.",
            error_code="MISSING_OUTLIER_LABELLED_DATASET",
            details={
                "source_file": str(resolved_source_file),
            },
        )

    logger.info(
        "Starting missing-gap analysis for %s.",
        resolved_source_file.name,
    )

    dataframe = read_csv_file(
        resolved_source_file
    )

    analysed_data = analyse_missing_gaps(
        dataframe=dataframe,
        columns=columns,
        resolved_frequency=resolved_frequency,
        logger=logger,
    )

    output_file = save_missing_gap_dataset(
        analysed_data=analysed_data,
        source_file=resolved_source_file,
        output_directory=resolved_output_directory,
    )

    total_series = int(
        add_composite_id(
            dataframe=dataframe,
            columns=columns,
        )[COMPOSITE_ID_COLUMN]
        .nunique(dropna=False)
    )

    result = MissingGapAnalysisResult(
        source_file=resolved_source_file,
        output_file=output_file,
        total_rows=len(analysed_data.dataframe),
        total_series=total_series,
        total_gaps=analysed_data.total_gaps,
        total_missing_points=(
            analysed_data.total_missing_points
        ),
        phase_1_gaps=analysed_data.phase_1_gaps,
        phase_2_gaps=analysed_data.phase_2_gaps,
        phase_3_gaps=analysed_data.phase_3_gaps,
        phase_4_pass_1_gaps=(
            analysed_data.phase_4_pass_1_gaps
        ),
        phase_4_pass_2_gaps=(
            analysed_data.phase_4_pass_2_gaps
        ),
        phase_1_points=analysed_data.phase_1_points,
        phase_2_points=analysed_data.phase_2_points,
        phase_3_points=analysed_data.phase_3_points,
        phase_4_pass_1_points=(
            analysed_data.phase_4_pass_1_points
        ),
        phase_4_pass_2_points=(
            analysed_data.phase_4_pass_2_points
        ),
    )

    logger.info(
        "Missing-gap-classified dataset saved to %s.",
        result.output_file,
    )

    return result


__all__ = [
    "MISSING_ORIGINAL_LABELS",
    "MissingGapAnalysisData",
    "MissingGapAnalysisResult",
    "MissingGapRecord",
    "ORIGINAL_GAP_PHASE_COLUMN",
    "PHASE_1",
    "PHASE_2",
    "PHASE_3",
    "PHASE_4_PASS_1",
    "PHASE_4_PASS_2",
    "analyse_missing_gaps",
    "classify_gap_phase",
    "classify_site_gaps",
    "create_missing_gap_analysed_dataset",
    "save_missing_gap_dataset",
]