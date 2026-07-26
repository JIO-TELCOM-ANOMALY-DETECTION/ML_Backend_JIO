"""
Analyse timestamp integrity for one KPI dataset.

This module reads the unique-site summary produced by ``site_analysis.py``
and audits timestamp completeness for every listed site.

The expected timeline is supplied using exactly one of these methods:

1. ``expected_start`` and ``expected_end``;
2. ``num_days``, counted from the earliest valid timestamp present in the
   KPI dataframe after filtering to the listed sites; or
3. ``global_min_max``, using the earliest and latest valid timestamps in
   that prepared KPI dataframe after filtering to the listed sites.

The module is read-only with respect to the source dataset. It does not insert
missing rows, fill values, remove duplicates, or modify KPI observations.

Generated files:

- summary.txt
- series_summary.csv
- gap_details.csv
- global_missing_timestamps.csv
- duplicate_timestamp_details.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
)
from app.schemas.common.dataset import DatasetColumnMapping
from app.services.file_service import (
    ensure_directory,
    read_csv_file,
    sanitise_filename,
    write_csv_file,
)


COMPOSITE_ID_COLUMN = "Composite_ID"
COMPOSITE_ID_SEPARATOR = " | "
MISSING_ID_VALUE = "<MISSING>"

GAP_BIN_1 = "Gap_1_15min"
GAP_BIN_2 = "Gap_2_4_30_60min"
GAP_BIN_3 = "Gap_5_24_75min_6hr"
GAP_BIN_4 = "Gap_25_96_6hr_24hr"
GAP_BIN_5 = "Gap_GT_96_GT_24hr"

SUMMARY_FILE_NAME = "summary.txt"
SERIES_SUMMARY_FILE_NAME = "series_summary.csv"
GAP_DETAILS_FILE_NAME = "gap_details.csv"
GLOBAL_MISSING_FILE_NAME = "global_missing_timestamps.csv"
DUPLICATE_DETAILS_FILE_NAME = "duplicate_timestamp_details.csv"

EXPECTED_TIMESTAMPS_COLUMN = "Expected_Timestamps"
OBSERVED_TIMESTAMPS_COLUMN = "Observed_Timestamps"
MISSING_TIMESTAMPS_COLUMN = "Missing_Timestamps"
COVERAGE_PERCENT_COLUMN = "Coverage_Percent"
GAP_EVENTS_COLUMN = "Gap_Events"
LONGEST_GAP_COLUMN = "Longest_Gap"
NAN_VALUES_COLUMN = "NaN_Values"
DUPLICATE_TIMESTAMPS_COLUMN = "Duplicate_Timestamps"
OFF_GRID_TIMESTAMPS_COLUMN = "Off_Grid_Timestamps"


@dataclass(frozen=True, slots=True)
class ExpectedTimeline:
    """
    Store the resolved expected timeline used by integrity analysis.

    Attributes:
        method:
            Timeline-selection method: ``date_range``, ``num_days``, or ``global_min_max``.

        start:
            First expected timestamp.

        end:
            Final expected timestamp, inclusive.

        frequency:
            Expected interval between consecutive timestamps.

        index:
            Complete expected timestamp index.
    """

    method: str
    start: pd.Timestamp
    end: pd.Timestamp
    frequency: pd.Timedelta
    index: pd.DatetimeIndex


@dataclass(frozen=True, slots=True)
class DataIntegrityResult:
    """
    Store output paths and headline timestamp-integrity statistics.

    Attributes:
        source_file:
            KPI-specific source file associated with the dataframe.

        site_summary_file:
            Site-analysis CSV used to define the complete site list.

        output_directory:
            Directory containing the generated integrity reports.

        summary_file:
            Human-readable dataset-level summary text file.

        series_summary_file:
            CSV containing one row per unique site.

        gap_details_file:
            CSV containing every continuous missing-timestamp gap.

        global_missing_file:
            CSV containing timestamps absent from the complete dataset.

        duplicate_details_file:
            CSV containing duplicate site-timestamp combinations.

        total_sites:
            Number of sites analysed from the site-analysis summary.

        expected_timestamps_per_site:
            Number of expected timestamps in the resolved timeline.

        total_missing_timestamps:
            Sum of missing timestamps across all sites.

        total_gap_events:
            Number of continuous missing-timestamp blocks.

        rows_with_invalid_timestamp:
            Number of source rows whose timestamp could not be parsed.

        rows_outside_expected_timeline:
            Number of valid-timestamp rows before or after the timeline.
    """

    source_file: Path
    site_summary_file: Path
    output_directory: Path
    summary_file: Path
    series_summary_file: Path
    gap_details_file: Path
    global_missing_file: Path
    duplicate_details_file: Path
    total_sites: int
    expected_timestamps_per_site: int
    total_missing_timestamps: int
    total_gap_events: int
    rows_with_invalid_timestamp: int
    rows_outside_expected_timeline: int


def validate_data_integrity_inputs(
    dataframe: pd.DataFrame,
    site_summary: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> None:
    """
    Validate source and site-summary dataframes before analysis.

    Args:
        dataframe:
            KPI-specific source dataframe.

        site_summary:
            Site-analysis output containing every unique site.

        columns:
            Dataset column mapping used to locate required columns.

    Returns:
        None.

    Raises:
        InputValidationError:
            If either dataframe is empty or required columns are missing.
    """

    if dataframe.empty:
        raise InputValidationError(
            "Data-integrity analysis cannot process an empty dataframe.",
            error_code="EMPTY_DATA_INTEGRITY_DATASET",
        )

    if site_summary.empty:
        raise InputValidationError(
            "The site-analysis summary does not contain any sites.",
            error_code="EMPTY_SITE_ANALYSIS_SUMMARY",
        )

    site_columns = list(
        columns.get_site_identity_columns()
    )

    required_dataset_columns = [
        *site_columns,
        columns.timestamp_column,
        columns.value_column,
    ]

    missing_dataset_columns = [
        column
        for column in required_dataset_columns
        if column not in dataframe.columns
    ]

    if missing_dataset_columns:
        raise InputValidationError(
            "The dataframe is missing columns required for timestamp "
            "integrity analysis.",
            error_code="MISSING_DATA_INTEGRITY_COLUMNS",
            details={
                "missing_columns": missing_dataset_columns,
                "required_columns": required_dataset_columns,
                "available_columns": list(dataframe.columns),
            },
        )

    missing_site_summary_columns = [
        column
        for column in site_columns
        if column not in site_summary.columns
    ]

    if missing_site_summary_columns:
        raise InputValidationError(
            "The site-analysis summary is missing required site columns.",
            error_code="MISSING_SITE_SUMMARY_COLUMNS",
            details={
                "missing_columns": missing_site_summary_columns,
                "required_columns": site_columns,
                "available_columns": list(site_summary.columns),
            },
        )


def normalise_expected_frequency(
    expected_frequency: str | pd.Timedelta,
) -> pd.Timedelta:
    """
    Convert the configured expected frequency into a positive Timedelta.

    Args:
        expected_frequency:
            Pandas-compatible frequency such as ``15min`` or a Timedelta.

    Returns:
        pd.Timedelta:
            Positive expected interval between timestamps.

    Raises:
        InputValidationError:
            If the value cannot be parsed or is not positive.
    """

    try:
        frequency = pd.Timedelta(
            expected_frequency
        )
    except (TypeError, ValueError) as error:
        raise InputValidationError(
            "The expected timestamp frequency is invalid.",
            error_code="INVALID_EXPECTED_FREQUENCY",
            details={
                "expected_frequency": str(expected_frequency),
            },
        ) from error

    if frequency <= pd.Timedelta(0):
        raise InputValidationError(
            "The expected timestamp frequency must be greater than zero.",
            error_code="NON_POSITIVE_EXPECTED_FREQUENCY",
            details={
                "expected_frequency": str(expected_frequency),
            },
        )

    return frequency


def parse_timeline_boundary(
    value: datetime | str | pd.Timestamp,
    boundary_name: str,
) -> pd.Timestamp:
    """
    Parse one expected-timeline boundary into a valid Timestamp.

    Args:
        value:
            Raw datetime-like boundary.

        boundary_name:
            Name used in validation errors.

    Returns:
        pd.Timestamp:
            Parsed timestamp boundary.

    Raises:
        InputValidationError:
            If the supplied boundary cannot be parsed.
    """

    try:
        parsed_value = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise InputValidationError(
            f"{boundary_name} is not a valid timestamp.",
            error_code="INVALID_TIMELINE_BOUNDARY",
            details={
                "boundary_name": boundary_name,
                "received_value": str(value),
            },
        ) from error

    if pd.isna(parsed_value):
        raise InputValidationError(
            f"{boundary_name} is not a valid timestamp.",
            error_code="INVALID_TIMELINE_BOUNDARY",
            details={
                "boundary_name": boundary_name,
                "received_value": str(value),
            },
        )

    return parsed_value


def resolve_expected_timeline(
    valid_timestamps: pd.Series,
    *,
    expected_frequency: str | pd.Timedelta = "15min",
    expected_start: datetime | str | pd.Timestamp | None = None,
    expected_end: datetime | str | pd.Timestamp | None = None,
    num_days: int | None = None,
    use_global_min_max: bool = False,
) -> ExpectedTimeline:
    """
    Resolve the complete expected timeline using exactly one input method.

    ``date_range`` uses explicit inclusive boundaries. ``num_days`` starts
    from the earliest valid timestamp and creates the requested number of
    complete 24-hour periods. ``global_min_max`` starts at the earliest valid
    timestamp and uses the latest valid timestamp as the analysis boundary.

    Args:
        valid_timestamps:
            Parsed, non-null timestamps belonging to recognised sites.

        expected_frequency:
            Expected interval between consecutive timestamps.

        expected_start:
            Optional explicit first timestamp.

        expected_end:
            Optional explicit final timestamp, inclusive.

        num_days:
            Optional positive number of complete days counted from the
            earliest valid source timestamp.

        use_global_min_max:
            Whether to derive boundaries from the earliest and latest valid
            timestamps in the prepared KPI dataframe.

    Returns:
        ExpectedTimeline:
            Resolved timeline metadata and DatetimeIndex.

    Raises:
        InputValidationError:
            If the timeline selection is contradictory, boundaries are
            invalid, no usable timestamps exist, ``num_days`` is invalid, or
            the frequency cannot represent a complete day exactly.
    """

    frequency = normalise_expected_frequency(
        expected_frequency
    )

    if not isinstance(use_global_min_max, bool):
        raise InputValidationError(
            "use_global_min_max must be a boolean value.",
            error_code="INVALID_GLOBAL_MIN_MAX_FLAG",
            details={
                "use_global_min_max": use_global_min_max,
            },
        )

    has_start = expected_start is not None
    has_end = expected_end is not None
    has_date_range = has_start or has_end
    has_num_days = num_days is not None

    if has_date_range and not (has_start and has_end):
        raise InputValidationError(
            "expected_start and expected_end must be supplied together.",
            error_code="INCOMPLETE_EXPECTED_DATE_RANGE",
        )

    selected_method_count = sum(
        (
            bool(has_start and has_end),
            bool(has_num_days),
            use_global_min_max,
        )
    )

    if selected_method_count != 1:
        raise InputValidationError(
            "Provide exactly one timeline method: expected_start with "
            "expected_end, num_days, or global_min_max.",
            error_code="INVALID_TIMELINE_SELECTION",
            details={
                "expected_start_supplied": has_start,
                "expected_end_supplied": has_end,
                "num_days_supplied": has_num_days,
                "global_min_max_selected": use_global_min_max,
            },
        )

    if has_start and has_end:
        start = parse_timeline_boundary(
            expected_start,
            "expected_start",
        )
        end = parse_timeline_boundary(
            expected_end,
            "expected_end",
        )

        if end <= start:
            raise InputValidationError(
                "expected_end must be later than expected_start.",
                error_code="INVALID_EXPECTED_TIMELINE_ORDER",
                details={
                    "expected_start": str(start),
                    "expected_end": str(end),
                },
            )

        duration = end - start

        if duration % frequency != pd.Timedelta(0):
            raise InputValidationError(
                "The explicit timeline boundaries must align exactly with "
                "the expected frequency.",
                error_code="MISALIGNED_EXPECTED_TIMELINE",
                details={
                    "expected_start": str(start),
                    "expected_end": str(end),
                    "expected_frequency": str(frequency),
                },
            )

        timeline_index = pd.date_range(
            start=start,
            end=end,
            freq=frequency,
        )

        return ExpectedTimeline(
            method="date_range",
            start=start,
            end=end,
            frequency=frequency,
            index=timeline_index,
        )

    available_timestamps = valid_timestamps.dropna()

    if available_timestamps.empty:
        raise InputValidationError(
            "The expected timeline cannot be resolved because no valid "
            "timestamp is available after site filtering.",
            error_code="NO_VALID_TIMESTAMP_FOR_TIMELINE",
        )

    start = pd.Timestamp(
        available_timestamps.min()
    )

    if use_global_min_max:
        end = pd.Timestamp(
            available_timestamps.max()
        )

        if end < start:
            raise InputValidationError(
                "The resolved maximum timestamp is earlier than the "
                "resolved minimum timestamp.",
                error_code="INVALID_GLOBAL_TIMESTAMP_ORDER",
                details={
                    "global_min_timestamp": str(start),
                    "global_max_timestamp": str(end),
                },
            )

        timeline_index = pd.date_range(
            start=start,
            end=end,
            freq=frequency,
        )

        if timeline_index.empty:
            raise InputValidationError(
                "The global minimum and maximum could not produce an "
                "expected timestamp timeline.",
                error_code="EMPTY_GLOBAL_MIN_MAX_TIMELINE",
                details={
                    "global_min_timestamp": str(start),
                    "global_max_timestamp": str(end),
                    "expected_frequency": str(frequency),
                },
            )

        return ExpectedTimeline(
            method="global_min_max",
            start=start,
            end=end,
            frequency=frequency,
            index=timeline_index,
        )

    if not isinstance(num_days, int) or isinstance(num_days, bool):
        raise InputValidationError(
            "num_days must be supplied as a positive integer.",
            error_code="INVALID_NUM_DAYS_TYPE",
            details={
                "num_days": num_days,
            },
        )

    if num_days <= 0:
        raise InputValidationError(
            "num_days must be greater than zero.",
            error_code="NON_POSITIVE_NUM_DAYS",
            details={
                "num_days": num_days,
            },
        )

    one_day = pd.Timedelta(days=1)

    if one_day % frequency != pd.Timedelta(0):
        raise InputValidationError(
            "The expected frequency must divide one complete day exactly "
            "when num_days is used.",
            error_code="FREQUENCY_DOES_NOT_DIVIDE_DAY",
            details={
                "expected_frequency": str(frequency),
            },
        )

    timestamps_per_day = int(
        one_day / frequency
    )
    expected_periods = num_days * timestamps_per_day
    end = start + (
        expected_periods - 1
    ) * frequency

    timeline_index = pd.date_range(
        start=start,
        periods=expected_periods,
        freq=frequency,
    )

    return ExpectedTimeline(
        method="num_days",
        start=start,
        end=end,
        frequency=frequency,
        index=timeline_index,
    )



def add_composite_id(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """
    Return a dataframe copy containing a stable composite site identifier.

    Args:
        dataframe:
            Dataframe containing the configured site identity columns.

        columns:
            Dataset column mapping used to locate the identity columns.

    Returns:
        pd.DataFrame:
            Copy containing ``Composite_ID``.

    Raises:
        KeyError:
            If a required site identity column is unavailable.
    """

    result = dataframe.copy()
    site_columns = list(
        columns.get_site_identity_columns()
    )

    identity_values = result[
        site_columns
    ].astype("string").fillna(
        MISSING_ID_VALUE
    )

    result[COMPOSITE_ID_COLUMN] = (
        identity_values
        .agg(
            COMPOSITE_ID_SEPARATOR.join,
            axis=1,
        )
    )

    return result


def prepare_dataframes(
    dataframe: pd.DataFrame,
    site_summary: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Parse source values and filter rows to sites from site analysis.

    Args:
        dataframe:
            KPI-specific source dataframe.

        site_summary:
            Site-analysis output containing the authoritative site list.

        columns:
            Dataset column mapping.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, int, int]:
            Prepared source dataframe, prepared unique-site dataframe,
            invalid timestamp row count, and rows excluded because their site
            identity was not present in the site summary.

    Raises:
        InputValidationError:
            If filtering leaves no source rows.
    """

    site_columns = list(
        columns.get_site_identity_columns()
    )
    timestamp_column = columns.timestamp_column
    value_column = columns.value_column

    unique_sites = (
        site_summary[site_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    unique_sites = add_composite_id(
        unique_sites,
        columns,
    )

    working_dataframe = dataframe[
        [
            *site_columns,
            timestamp_column,
            value_column,
        ]
    ].copy()

    working_dataframe[timestamp_column] = pd.to_datetime(
        working_dataframe[timestamp_column],
        errors="coerce",
    )

    invalid_timestamp_rows = int(
        working_dataframe[timestamp_column]
        .isna()
        .sum()
    )

    working_dataframe = working_dataframe.dropna(
        subset=[timestamp_column]
    ).copy()

    working_dataframe[value_column] = pd.to_numeric(
        working_dataframe[value_column],
        errors="coerce",
    )

    working_dataframe = add_composite_id(
        working_dataframe,
        columns,
    )

    recognised_site_ids = set(
        unique_sites[COMPOSITE_ID_COLUMN]
        .tolist()
    )

    recognised_mask = working_dataframe[
        COMPOSITE_ID_COLUMN
    ].isin(
        recognised_site_ids
    )

    excluded_site_rows = int(
        (~recognised_mask).sum()
    )

    working_dataframe = working_dataframe.loc[
        recognised_mask
    ].copy()

    if working_dataframe.empty:
        raise InputValidationError(
            "No source rows remain after filtering to sites from the "
            "site-analysis summary.",
            error_code="NO_ROWS_AFTER_SITE_FILTERING",
        )

    return (
        working_dataframe,
        unique_sites,
        invalid_timestamp_rows,
        excluded_site_rows,
    )


def create_global_missing_timestamps(
    dataframe: pd.DataFrame,
    timeline: ExpectedTimeline,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Identify expected timestamps absent from the complete dataset.

    Args:
        dataframe:
            Prepared rows inside the expected timeline.

        timeline:
            Resolved expected timestamp index.

        timestamp_column:
            Physical timestamp column name.

    Returns:
        pd.DataFrame:
            One ``Missing_Timestamp`` column containing globally absent times.

    Raises:
        None.
    """

    observed_index = pd.DatetimeIndex(
        dataframe[timestamp_column]
        .drop_duplicates()
        .sort_values()
    )

    missing_index = timeline.index.difference(
        observed_index
    )

    return pd.DataFrame(
        {
            "Missing_Timestamp": missing_index,
        }
    )


def create_duplicate_timestamp_details(
    dataframe: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Create duplicate timestamp details for every site.

    Args:
        dataframe:
            Prepared rows inside the expected timeline.

        timestamp_column:
            Physical timestamp column name.

    Returns:
        pd.DataFrame:
            Composite site, duplicate timestamp, and occurrence count.

    Raises:
        None.
    """

    duplicate_counts = (
        dataframe
        .groupby(
            [
                COMPOSITE_ID_COLUMN,
                timestamp_column,
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="Occurrences"
        )
    )

    duplicate_details = duplicate_counts.loc[
        duplicate_counts["Occurrences"] > 1
    ].copy()

    duplicate_details = duplicate_details.rename(
        columns={
            timestamp_column: "Timestamp",
        }
    )

    return (
        duplicate_details[
            [
                COMPOSITE_ID_COLUMN,
                "Timestamp",
                "Occurrences",
            ]
        ]
        .sort_values(
            [
                COMPOSITE_ID_COLUMN,
                "Timestamp",
            ]
        )
        .reset_index(drop=True)
    )


def classify_gap(
    missing_timestamp_count: int,
) -> str:
    """
    Classify one continuous gap using the established point-count bins.

    Args:
        missing_timestamp_count:
            Number of consecutive expected timestamps absent from a site.

    Returns:
        str:
            Gap category name.

    Raises:
        InputValidationError:
            If the supplied count is not positive.
    """

    if missing_timestamp_count <= 0:
        raise InputValidationError(
            "A missing-timestamp gap must contain at least one point.",
            error_code="INVALID_GAP_SIZE",
            details={
                "missing_timestamp_count": missing_timestamp_count,
            },
        )

    if missing_timestamp_count == 1:
        return GAP_BIN_1

    if missing_timestamp_count <= 4:
        return GAP_BIN_2

    if missing_timestamp_count <= 24:
        return GAP_BIN_3

    if missing_timestamp_count <= 96:
        return GAP_BIN_4

    return GAP_BIN_5


def create_gap_blocks(
    composite_id: str,
    missing_index: pd.DatetimeIndex,
    frequency: pd.Timedelta,
) -> list[dict[str, Any]]:
    """
    Convert missing timestamps into continuous gap-detail records.

    Args:
        composite_id:
            Site identifier associated with the missing timestamps.

        missing_index:
            Sorted timestamps missing from the site.

        frequency:
            Expected interval between timestamps.

    Returns:
        list[dict[str, Any]]:
            One dictionary per continuous missing block.

    Raises:
        None.
    """

    if len(missing_index) == 0:
        return []

    missing_series = pd.Series(
        missing_index,
        name="Missing_Timestamp",
    )

    new_gap = (
        missing_series.diff().isna()
        | missing_series.diff().ne(frequency)
    )

    gap_ids = new_gap.cumsum()
    gap_records: list[dict[str, Any]] = []

    for _, timestamps in missing_series.groupby(
        gap_ids
    ):
        missing_count = int(
            len(timestamps)
        )

        gap_records.append(
            {
                COMPOSITE_ID_COLUMN: composite_id,
                "Gap_Start": timestamps.min(),
                "Gap_End": timestamps.max(),
                MISSING_TIMESTAMPS_COLUMN: missing_count,
                "Duration": str(
                    frequency * missing_count
                ),
                "Category": classify_gap(
                    missing_count
                ),
            }
        )

    return gap_records


def create_series_and_gap_summaries(
    dataframe: pd.DataFrame,
    unique_sites: pd.DataFrame,
    duplicate_details: pd.DataFrame,
    timeline: ExpectedTimeline,
    columns: DatasetColumnMapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-site coverage statistics and continuous gap details.

    Args:
        dataframe:
            Prepared rows inside the expected timeline.

        unique_sites:
            Complete site list read from the site-analysis output.

        duplicate_details:
            Duplicate site-timestamp combinations.

        timeline:
            Resolved expected timeline.

        columns:
            Dataset column mapping.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Per-site summary and gap-details dataframes.

    Raises:
        None.
    """

    site_columns = list(
        columns.get_site_identity_columns()
    )
    timestamp_column = columns.timestamp_column
    value_column = columns.value_column
    expected_count = len(
        timeline.index
    )
    expected_timestamp_set = set(
        timeline.index
    )

    duplicate_counts_by_site = (
        duplicate_details
        .groupby(COMPOSITE_ID_COLUMN)
        .size()
        .to_dict()
    )

    grouped_dataframes = {
        composite_id: group
        for composite_id, group in dataframe.groupby(
            COMPOSITE_ID_COLUMN,
            sort=False,
            dropna=False,
        )
    }

    series_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []

    for _, site_row in unique_sites.iterrows():
        composite_id = str(
            site_row[COMPOSITE_ID_COLUMN]
        )

        series_dataframe = grouped_dataframes.get(
            composite_id,
            dataframe.iloc[0:0],
        )

        observed_all_index = pd.DatetimeIndex(
            series_dataframe[timestamp_column]
            .drop_duplicates()
            .sort_values()
        )

        observed_expected_index = pd.DatetimeIndex(
            [
                timestamp
                for timestamp in observed_all_index
                if timestamp in expected_timestamp_set
            ]
        )

        missing_index = timeline.index.difference(
            observed_expected_index
        )

        off_grid_count = int(
            len(observed_all_index)
            - len(observed_expected_index)
        )
        observed_count = int(
            len(observed_expected_index)
        )
        missing_count = int(
            len(missing_index)
        )

        coverage_percent = (
            observed_count / expected_count * 100
            if expected_count > 0
            else 0.0
        )

        site_gap_rows = create_gap_blocks(
            composite_id=composite_id,
            missing_index=missing_index,
            frequency=timeline.frequency,
        )
        gap_rows.extend(
            site_gap_rows
        )

        category_counts = {
            GAP_BIN_1: 0,
            GAP_BIN_2: 0,
            GAP_BIN_3: 0,
            GAP_BIN_4: 0,
            GAP_BIN_5: 0,
        }

        for gap_row in site_gap_rows:
            category_counts[
                gap_row["Category"]
            ] += 1

        longest_gap = max(
            (
                int(row[MISSING_TIMESTAMPS_COLUMN])
                for row in site_gap_rows
            ),
            default=0,
        )

        series_rows.append(
            {
                COMPOSITE_ID_COLUMN: composite_id,
                **{
                    column: site_row[column]
                    for column in site_columns
                },
                EXPECTED_TIMESTAMPS_COLUMN: expected_count,
                OBSERVED_TIMESTAMPS_COLUMN: observed_count,
                MISSING_TIMESTAMPS_COLUMN: missing_count,
                COVERAGE_PERCENT_COLUMN: round(
                    coverage_percent,
                    4,
                ),
                GAP_EVENTS_COLUMN: len(
                    site_gap_rows
                ),
                GAP_BIN_1: category_counts[GAP_BIN_1],
                GAP_BIN_2: category_counts[GAP_BIN_2],
                GAP_BIN_3: category_counts[GAP_BIN_3],
                GAP_BIN_4: category_counts[GAP_BIN_4],
                GAP_BIN_5: category_counts[GAP_BIN_5],
                LONGEST_GAP_COLUMN: longest_gap,
                NAN_VALUES_COLUMN: int(
                    series_dataframe[value_column]
                    .isna()
                    .sum()
                ),
                DUPLICATE_TIMESTAMPS_COLUMN: int(
                    duplicate_counts_by_site.get(
                        composite_id,
                        0,
                    )
                ),
                OFF_GRID_TIMESTAMPS_COLUMN: off_grid_count,
            }
        )

    series_summary = pd.DataFrame(
        series_rows
    )

    series_summary = series_summary[
        [
            COMPOSITE_ID_COLUMN,
            *site_columns,
            EXPECTED_TIMESTAMPS_COLUMN,
            OBSERVED_TIMESTAMPS_COLUMN,
            MISSING_TIMESTAMPS_COLUMN,
            COVERAGE_PERCENT_COLUMN,
            GAP_EVENTS_COLUMN,
            GAP_BIN_1,
            GAP_BIN_2,
            GAP_BIN_3,
            GAP_BIN_4,
            GAP_BIN_5,
            LONGEST_GAP_COLUMN,
            NAN_VALUES_COLUMN,
            DUPLICATE_TIMESTAMPS_COLUMN,
            OFF_GRID_TIMESTAMPS_COLUMN,
        ]
    ]

    gap_details = pd.DataFrame(
        gap_rows,
        columns=[
            COMPOSITE_ID_COLUMN,
            "Gap_Start",
            "Gap_End",
            MISSING_TIMESTAMPS_COLUMN,
            "Duration",
            "Category",
        ],
    )

    if not gap_details.empty:
        gap_details = gap_details.sort_values(
            [
                COMPOSITE_ID_COLUMN,
                "Gap_Start",
            ]
        ).reset_index(drop=True)

    return (
        series_summary,
        gap_details,
    )


def calculate_summary_statistics(
    series_summary: pd.DataFrame,
    duplicate_details: pd.DataFrame,
    global_missing_timestamps: pd.DataFrame,
    timeline: ExpectedTimeline,
) -> dict[str, Any]:
    """
    Calculate dataset-level statistics used by ``summary.txt``.

    Args:
        series_summary:
            Per-site integrity summary.

        duplicate_details:
            Duplicate timestamp details.

        global_missing_timestamps:
            Globally absent expected timestamps.

        timeline:
            Resolved expected timeline.

    Returns:
        dict[str, Any]:
            Dataset-level summary values.

    Raises:
        None.
    """

    total_sites = int(
        len(series_summary)
    )
    expected_per_site = int(
        len(timeline.index)
    )
    total_expected = expected_per_site * total_sites
    total_observed = int(
        series_summary[OBSERVED_TIMESTAMPS_COLUMN]
        .sum()
    )
    total_missing = int(
        series_summary[MISSING_TIMESTAMPS_COLUMN]
        .sum()
    )

    overall_coverage = (
        total_observed / total_expected * 100
        if total_expected > 0
        else 0.0
    )

    longest_gap_points = int(
        series_summary[LONGEST_GAP_COLUMN]
        .max()
    )

    if longest_gap_points > 0:
        longest_gap_row = series_summary.loc[
            series_summary[LONGEST_GAP_COLUMN]
            == longest_gap_points
        ].iloc[0]
        longest_gap_composite_id = longest_gap_row[
            COMPOSITE_ID_COLUMN
        ]
    else:
        longest_gap_composite_id = "None"

    if not duplicate_details.empty:
        duplicate_occurrences = int(
            duplicate_details["Occurrences"]
            .sum()
        )
        extra_duplicate_rows = int(
            (
                duplicate_details["Occurrences"]
                - 1
            ).sum()
        )
    else:
        duplicate_occurrences = 0
        extra_duplicate_rows = 0

    return {
        "total_sites": total_sites,
        "expected_per_site": expected_per_site,
        "total_expected": total_expected,
        "total_observed": total_observed,
        "total_missing": total_missing,
        "overall_coverage": overall_coverage,
        "global_missing": int(
            len(global_missing_timestamps)
        ),
        "coverage_min": float(
            series_summary[COVERAGE_PERCENT_COLUMN]
            .min()
        ),
        "coverage_max": float(
            series_summary[COVERAGE_PERCENT_COLUMN]
            .max()
        ),
        "coverage_mean": float(
            series_summary[COVERAGE_PERCENT_COLUMN]
            .mean()
        ),
        "coverage_median": float(
            series_summary[COVERAGE_PERCENT_COLUMN]
            .median()
        ),
        "missing_min": int(
            series_summary[MISSING_TIMESTAMPS_COLUMN]
            .min()
        ),
        "missing_max": int(
            series_summary[MISSING_TIMESTAMPS_COLUMN]
            .max()
        ),
        "missing_mean": float(
            series_summary[MISSING_TIMESTAMPS_COLUMN]
            .mean()
        ),
        "missing_median": float(
            series_summary[MISSING_TIMESTAMPS_COLUMN]
            .median()
        ),
        "total_gap_events": int(
            series_summary[GAP_EVENTS_COLUMN]
            .sum()
        ),
        "total_gap_1": int(
            series_summary[GAP_BIN_1]
            .sum()
        ),
        "total_gap_2_4": int(
            series_summary[GAP_BIN_2]
            .sum()
        ),
        "total_gap_5_24": int(
            series_summary[GAP_BIN_3]
            .sum()
        ),
        "total_gap_25_96": int(
            series_summary[GAP_BIN_4]
            .sum()
        ),
        "total_gap_gt_96": int(
            series_summary[GAP_BIN_5]
            .sum()
        ),
        "longest_gap_points": longest_gap_points,
        "longest_gap_duration": str(
            timeline.frequency * longest_gap_points
        ),
        "longest_gap_composite_id": longest_gap_composite_id,
        "total_nan_values": int(
            series_summary[NAN_VALUES_COLUMN]
            .sum()
        ),
        "series_with_nan_values": int(
            (
                series_summary[NAN_VALUES_COLUMN]
                > 0
            ).sum()
        ),
        "nan_min": int(
            series_summary[NAN_VALUES_COLUMN]
            .min()
        ),
        "nan_max": int(
            series_summary[NAN_VALUES_COLUMN]
            .max()
        ),
        "nan_mean": float(
            series_summary[NAN_VALUES_COLUMN]
            .mean()
        ),
        "nan_median": float(
            series_summary[NAN_VALUES_COLUMN]
            .median()
        ),
        "duplicate_timestamp_groups": int(
            len(duplicate_details)
        ),
        "duplicate_occurrences": duplicate_occurrences,
        "extra_duplicate_rows": extra_duplicate_rows,
        "series_with_duplicates": int(
            (
                series_summary[DUPLICATE_TIMESTAMPS_COLUMN]
                > 0
            ).sum()
        ),
        "total_off_grid_timestamps": int(
            series_summary[OFF_GRID_TIMESTAMPS_COLUMN]
            .sum()
        ),
        "series_with_off_grid_timestamps": int(
            (
                series_summary[OFF_GRID_TIMESTAMPS_COLUMN]
                > 0
            ).sum()
        ),
    }


def write_integrity_summary(
    *,
    output_path: Path,
    source_file: Path,
    site_summary_file: Path,
    output_directory: Path,
    timeline: ExpectedTimeline,
    statistics: dict[str, Any],
    original_row_count: int,
    filtered_row_count: int,
    in_timeline_row_count: int,
    rows_with_invalid_timestamp: int,
    rows_excluded_by_site_filter: int,
    rows_before_timeline: int,
    rows_after_timeline: int,
    global_observed_timestamp_count: int,
    series_summary_file: Path,
    gap_details_file: Path,
    global_missing_file: Path,
    duplicate_details_file: Path,
    columns: DatasetColumnMapping,
) -> Path:
    """
    Save the human-readable timestamp-integrity summary.

    Args:
        output_path:
            Destination ``summary.txt`` path.

        source_file:
            KPI-specific source file.

        site_summary_file:
            Site-analysis CSV used as the site list.

        output_directory:
            Directory containing all generated reports.

        timeline:
            Resolved expected timeline.

        statistics:
            Dataset-level calculated statistics.

        original_row_count:
            Number of rows received before timestamp parsing.

        filtered_row_count:
            Valid-timestamp rows retained after site filtering.

        in_timeline_row_count:
            Rows inside the expected start and end boundaries.

        rows_with_invalid_timestamp:
            Rows whose timestamp could not be parsed.

        rows_excluded_by_site_filter:
            Valid-timestamp rows not represented in the site summary.

        rows_before_timeline:
            Rows earlier than the expected start.

        rows_after_timeline:
            Rows later than the expected end.

        global_observed_timestamp_count:
            Unique expected timestamps observed anywhere in the dataset.

        series_summary_file:
            Saved per-site summary CSV.

        gap_details_file:
            Saved gap-details CSV.

        global_missing_file:
            Saved globally missing timestamp CSV.

        duplicate_details_file:
            Saved duplicate-details CSV.

        columns:
            Dataset column mapping used during analysis.

    Returns:
        Path:
            Saved summary path.

    Raises:
        FileProcessingError:
            If the summary file cannot be written.
    """

    site_columns = list(
        columns.get_site_identity_columns()
    )

    lines = [
        "=" * 100,
        "TELECOM KPI DATA INTEGRITY ANALYSIS",
        "=" * 100,
        "",
        "DATASET INFORMATION",
        "-" * 100,
        f"Dataset name                    : {source_file.stem}",
        f"Dataset file                    : {source_file}",
        f"Site-analysis summary           : {site_summary_file}",
        f"Output folder                   : {output_directory}",
        f"Original dataset rows           : {original_row_count}",
        f"Rows with invalid timestamp     : {rows_with_invalid_timestamp}",
        f"Rows excluded by site filter    : {rows_excluded_by_site_filter}",
        f"Rows after site filtering       : {filtered_row_count}",
        f"Rows before expected timeline   : {rows_before_timeline}",
        f"Rows after expected timeline    : {rows_after_timeline}",
        f"Rows inside expected timeline   : {in_timeline_row_count}",
        f"Unique site series analysed     : {statistics['total_sites']}",
        f"Composite ID columns            : {', '.join(site_columns)}",
        f"Value column                    : {columns.value_column}",
        f"Timestamp column                : {columns.timestamp_column}",
        "",
        "TIMELINE INFORMATION",
        "-" * 100,
        f"Timeline method                 : {timeline.method}",
        f"Expected frequency              : {timeline.frequency}",
        f"Expected start                  : {timeline.start}",
        f"Expected end                    : {timeline.end}",
        f"Expected timestamps per series  : {statistics['expected_per_site']}",
        f"Total expected timestamps       : {statistics['total_expected']}",
        "",
        "GLOBAL MISSING TIMESTAMP STATISTICS",
        "-" * 100,
        f"Global observed timestamps      : {global_observed_timestamp_count}",
        f"Global missing timestamps       : {statistics['global_missing']}",
        f"Saved file                      : {global_missing_file}",
        "",
        "PER-SERIES COVERAGE STATISTICS",
        "-" * 100,
        f"Total observed timestamps       : {statistics['total_observed']}",
        f"Total missing timestamps        : {statistics['total_missing']}",
        f"Overall coverage percent        : {statistics['overall_coverage']:.4f}",
        f"Minimum coverage percent        : {statistics['coverage_min']:.4f}",
        f"Maximum coverage percent        : {statistics['coverage_max']:.4f}",
        f"Mean coverage percent           : {statistics['coverage_mean']:.4f}",
        f"Median coverage percent         : {statistics['coverage_median']:.4f}",
        "",
        "PER-SERIES MISSING TIMESTAMP STATISTICS",
        "-" * 100,
        f"Minimum missing timestamps      : {statistics['missing_min']}",
        f"Maximum missing timestamps      : {statistics['missing_max']}",
        f"Mean missing timestamps         : {statistics['missing_mean']:.4f}",
        f"Median missing timestamps       : {statistics['missing_median']:.4f}",
        "",
        "GAP STATISTICS",
        "-" * 100,
        f"Total gap events                : {statistics['total_gap_events']}",
        f"{GAP_BIN_1:<32}: {statistics['total_gap_1']}",
        f"{GAP_BIN_2:<32}: {statistics['total_gap_2_4']}",
        f"{GAP_BIN_3:<32}: {statistics['total_gap_5_24']}",
        f"{GAP_BIN_4:<32}: {statistics['total_gap_25_96']}",
        f"{GAP_BIN_5:<32}: {statistics['total_gap_gt_96']}",
        f"Gap details saved file          : {gap_details_file}",
        "",
        "LONGEST GAP SUMMARY",
        "-" * 100,
        f"Longest gap points              : {statistics['longest_gap_points']}",
        f"Longest gap duration            : {statistics['longest_gap_duration']}",
        f"Longest gap Composite_ID        : {statistics['longest_gap_composite_id']}",
        "",
        "NaN STATISTICS",
        "-" * 100,
        f"Total NaN values                : {statistics['total_nan_values']}",
        f"Series with NaN values          : {statistics['series_with_nan_values']}",
        f"Minimum NaN values per series   : {statistics['nan_min']}",
        f"Maximum NaN values per series   : {statistics['nan_max']}",
        f"Mean NaN values per series      : {statistics['nan_mean']:.4f}",
        f"Median NaN values per series    : {statistics['nan_median']:.4f}",
        "",
        "DUPLICATE TIMESTAMP STATISTICS",
        "-" * 100,
        f"Duplicate timestamp groups      : {statistics['duplicate_timestamp_groups']}",
        f"Duplicate timestamp occurrences : {statistics['duplicate_occurrences']}",
        f"Extra duplicate rows            : {statistics['extra_duplicate_rows']}",
        f"Series with duplicates          : {statistics['series_with_duplicates']}",
        f"Duplicate details saved file    : {duplicate_details_file}",
        "",
        "OFF-GRID TIMESTAMP STATISTICS",
        "-" * 100,
        f"Total off-grid timestamps       : {statistics['total_off_grid_timestamps']}",
        f"Series with off-grid timestamps : {statistics['series_with_off_grid_timestamps']}",
        "",
        "OUTPUT FILES",
        "-" * 100,
        f"summary.txt                     : {output_path}",
        f"series_summary.csv              : {series_summary_file}",
        f"gap_details.csv                 : {gap_details_file}",
        f"global_missing_timestamps.csv   : {global_missing_file}",
        f"duplicate_timestamp_details.csv : {duplicate_details_file}",
        "",
        "=" * 100,
        "END OF SUMMARY",
        "=" * 100,
        "",
    ]

    try:
        output_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
    except OSError as error:
        raise FileProcessingError(
            "Unable to write the data-integrity summary.",
            error_code="DATA_INTEGRITY_SUMMARY_WRITE_FAILED",
            details={
                "output_path": str(output_path),
                "original_error": str(error),
            },
        ) from error

    return output_path


def analyse_data_integrity(
    dataframe: pd.DataFrame,
    source_file: str | Path,
    site_summary_file: str | Path,
    output_directory: str | Path,
    columns: DatasetColumnMapping,
    logger: Logger,
    *,
    expected_frequency: str | pd.Timedelta = "15min",
    expected_start: datetime | str | pd.Timestamp | None = None,
    expected_end: datetime | str | pd.Timestamp | None = None,
    num_days: int | None = None,
    use_global_min_max: bool = False,
) -> DataIntegrityResult:
    """
    Run timestamp-integrity analysis for every site from site analysis.

    This is the main orchestration function for ``data_integrity.py``. The EDA
    runner supplies the loaded KPI dataframe, source path, site-analysis output
    path, destination directory, resolved bucket frequency, and exactly one
    timeline method.

    Args:
        dataframe:
            Loaded dataframe belonging to one KPI.

        source_file:
            KPI-specific source CSV associated with the dataframe.

        site_summary_file:
            CSV generated by ``analyse_sites`` containing every unique site.

        output_directory:
            Directory receiving all integrity-analysis reports.

        columns:
            Dataset column mapping.

        logger:
            Run-specific logger used to record progress and results.

        expected_frequency:
            Expected interval between timestamps. Defaults to ``15min``.

        expected_start:
            Optional explicit first expected timestamp.

        expected_end:
            Optional explicit final expected timestamp, inclusive.

        num_days:
            Optional number of complete days counted from the earliest valid
            timestamp after site filtering.

        use_global_min_max:
            Whether to use the earliest and latest valid timestamps found in
            this prepared KPI dataframe after site filtering.

    Returns:
        DataIntegrityResult:
            Structured output paths and headline statistics.

    Raises:
        InputValidationError:
            If required columns, site inputs, or timeline settings are invalid.

        FileProcessingError:
            If an output file cannot be created.

        TypeError:
            If a supplied path cannot be converted to ``Path``.
    """

    resolved_source_file = Path(
        source_file
    ).expanduser().resolve()
    resolved_site_summary_file = Path(
        site_summary_file
    ).expanduser().resolve()
    resolved_output_directory = ensure_directory(
        output_directory
    )

    logger.info(
        "Starting data-integrity analysis for %s.",
        resolved_source_file.name,
    )

    site_summary = read_csv_file(
        resolved_site_summary_file
    )

    validate_data_integrity_inputs(
        dataframe=dataframe,
        site_summary=site_summary,
        columns=columns,
    )

    original_row_count = int(
        len(dataframe)
    )

    (
        prepared_dataframe,
        unique_sites,
        invalid_timestamp_rows,
        excluded_site_rows,
    ) = prepare_dataframes(
        dataframe=dataframe,
        site_summary=site_summary,
        columns=columns,
    )

    timestamp_column = columns.timestamp_column

    timeline = resolve_expected_timeline(
        valid_timestamps=prepared_dataframe[
            timestamp_column
        ],
        expected_frequency=expected_frequency,
        expected_start=expected_start,
        expected_end=expected_end,
        num_days=num_days,
        use_global_min_max=use_global_min_max,
    )

    rows_before_timeline = int(
        (
            prepared_dataframe[timestamp_column]
            < timeline.start
        ).sum()
    )
    rows_after_timeline = int(
        (
            prepared_dataframe[timestamp_column]
            > timeline.end
        ).sum()
    )

    timeline_dataframe = prepared_dataframe.loc[
        prepared_dataframe[timestamp_column].between(
            timeline.start,
            timeline.end,
            inclusive="both",
        )
    ].copy()

    global_missing_timestamps = (
        create_global_missing_timestamps(
            dataframe=timeline_dataframe,
            timeline=timeline,
            timestamp_column=timestamp_column,
        )
    )

    duplicate_details = (
        create_duplicate_timestamp_details(
            dataframe=timeline_dataframe,
            timestamp_column=timestamp_column,
        )
    )

    (
        series_summary,
        gap_details,
    ) = create_series_and_gap_summaries(
        dataframe=timeline_dataframe,
        unique_sites=unique_sites,
        duplicate_details=duplicate_details,
        timeline=timeline,
        columns=columns,
    )

    output_prefix = sanitise_filename(
        resolved_source_file.stem
    )

    summary_file = (
        resolved_output_directory
        / SUMMARY_FILE_NAME
    )
    series_summary_file = (
        resolved_output_directory
        / SERIES_SUMMARY_FILE_NAME
    )
    gap_details_file = (
        resolved_output_directory
        / GAP_DETAILS_FILE_NAME
    )
    global_missing_file = (
        resolved_output_directory
        / GLOBAL_MISSING_FILE_NAME
    )
    duplicate_details_file = (
        resolved_output_directory
        / DUPLICATE_DETAILS_FILE_NAME
    )

    logger.debug(
        "Resolved data-integrity output prefix for %s as %s.",
        resolved_source_file.name,
        output_prefix,
    )

    write_csv_file(
        series_summary,
        series_summary_file,
    )
    write_csv_file(
        gap_details,
        gap_details_file,
    )
    write_csv_file(
        global_missing_timestamps,
        global_missing_file,
    )
    write_csv_file(
        duplicate_details,
        duplicate_details_file,
    )

    statistics = calculate_summary_statistics(
        series_summary=series_summary,
        duplicate_details=duplicate_details,
        global_missing_timestamps=(
            global_missing_timestamps
        ),
        timeline=timeline,
    )

    global_observed_timestamp_count = int(
        len(timeline.index)
        - len(global_missing_timestamps)
    )

    write_integrity_summary(
        output_path=summary_file,
        source_file=resolved_source_file,
        site_summary_file=(
            resolved_site_summary_file
        ),
        output_directory=(
            resolved_output_directory
        ),
        timeline=timeline,
        statistics=statistics,
        original_row_count=original_row_count,
        filtered_row_count=len(
            prepared_dataframe
        ),
        in_timeline_row_count=len(
            timeline_dataframe
        ),
        rows_with_invalid_timestamp=(
            invalid_timestamp_rows
        ),
        rows_excluded_by_site_filter=(
            excluded_site_rows
        ),
        rows_before_timeline=rows_before_timeline,
        rows_after_timeline=rows_after_timeline,
        global_observed_timestamp_count=(
            global_observed_timestamp_count
        ),
        series_summary_file=(
            series_summary_file
        ),
        gap_details_file=gap_details_file,
        global_missing_file=global_missing_file,
        duplicate_details_file=(
            duplicate_details_file
        ),
        columns=columns,
    )

    result = DataIntegrityResult(
        source_file=resolved_source_file,
        site_summary_file=(
            resolved_site_summary_file
        ),
        output_directory=(
            resolved_output_directory
        ),
        summary_file=summary_file,
        series_summary_file=(
            series_summary_file
        ),
        gap_details_file=gap_details_file,
        global_missing_file=global_missing_file,
        duplicate_details_file=(
            duplicate_details_file
        ),
        total_sites=statistics[
            "total_sites"
        ],
        expected_timestamps_per_site=(
            statistics["expected_per_site"]
        ),
        total_missing_timestamps=(
            statistics["total_missing"]
        ),
        total_gap_events=(
            statistics["total_gap_events"]
        ),
        rows_with_invalid_timestamp=(
            invalid_timestamp_rows
        ),
        rows_outside_expected_timeline=(
            rows_before_timeline
            + rows_after_timeline
        ),
    )

    logger.info(
        "Data-integrity analysis completed for %s: sites=%d, "
        "expected_timestamps_per_site=%d, missing_timestamps=%d, "
        "gap_events=%d.",
        resolved_source_file.name,
        result.total_sites,
        result.expected_timestamps_per_site,
        result.total_missing_timestamps,
        result.total_gap_events,
    )

    logger.info(
        "Data-integrity reports saved in %s.",
        result.output_directory,
    )

    return result
