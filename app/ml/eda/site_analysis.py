"""
Identify unique telecom sites and calculate row counts per site.

This module performs the first two checks in Part A of the telecom
anomaly-detection pipeline:

1. Identify all unique sites.
2. Calculate the number of rows belonging to each site.

A unique site is defined by the combination of:

- Jio circle
- Jio cluster
- AMF site name

The module processes one KPI dataframe at a time. Looping through
multiple KPI files is handled by the EDA runner.

This module does not perform server validation, timestamp analysis,
NaN analysis, duplicate detection, preprocessing, feature generation,
or model execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path

import pandas as pd

from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
)
from app.schemas.common.dataset import DatasetColumnMapping
from app.services.file_service import (
    ensure_directory,
    sanitise_filename,
    write_csv_file,
)


ROW_COUNT_COLUMN = "Row_Count"


@dataclass(frozen=True, slots=True)
class SiteAnalysisResult:
    """
    Store the result of site identification and row-count analysis.

    Attributes:
        source_file:
            KPI-specific CSV associated with the analysis.

        summary_file:
            CSV containing one row for every unique site.

        total_rows:
            Total number of rows analysed.

        total_unique_sites:
            Number of unique site combinations identified.

        minimum_rows_per_site:
            Lowest row count found for one site.

        maximum_rows_per_site:
            Highest row count found for one site.

        average_rows_per_site:
            Average number of rows across all identified sites.

        rows_with_missing_site_identity:
            Number of rows where at least one site-identifying value
            is missing.
    """

    source_file: Path
    summary_file: Path
    total_rows: int
    total_unique_sites: int
    minimum_rows_per_site: int
    maximum_rows_per_site: int
    average_rows_per_site: float
    rows_with_missing_site_identity: int


def validate_site_analysis_input(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> None:
    """
    Validate that the dataframe can be used for site analysis.

    Args:
        dataframe:
            KPI-specific dataframe to analyse.

        columns:
            Dataset column mapping containing the physical site-column
            names.

    Returns:
        None.

    Raises:
        InputValidationError:
            If the dataframe is empty or one or more site-identifying
            columns are missing.
    """

    if dataframe.empty:
        raise InputValidationError(
            "Site analysis cannot process an empty dataframe.",
            error_code="EMPTY_SITE_ANALYSIS_DATASET",
        )

    site_columns = (
        columns.get_site_identity_columns()
    )

    missing_columns = [
        column
        for column in site_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "The dataframe is missing columns required for site "
            "analysis.",
            error_code="MISSING_SITE_IDENTITY_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "required_columns": list(
                    site_columns
                ),
                "available_columns": list(
                    dataframe.columns
                ),
            },
        )


def count_rows_with_missing_site_identity(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> int:
    """
    Count rows with incomplete site-identifying information.

    A row is considered incomplete when its circle, cluster, or site
    name is missing.

    Args:
        dataframe:
            KPI-specific dataframe being analysed.

        columns:
            Dataset column mapping containing the site-column names.

    Returns:
        int:
            Number of rows containing at least one missing site value.

    Raises:
        KeyError:
            If a required site column is unavailable. This should not
            occur when input validation has already completed.
    """

    site_columns = list(
        columns.get_site_identity_columns()
    )

    missing_identity_mask = (
        dataframe[site_columns]
        .isna()
        .any(axis=1)
    )

    return int(
        missing_identity_mask.sum()
    )


def create_site_row_count_summary(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """
    Identify unique sites and calculate row counts per site.

    Rows with missing site-identifying values are retained in the
    summary by using ``dropna=False`` during grouping.

    Args:
        dataframe:
            Validated KPI-specific dataframe.

        columns:
            Dataset column mapping containing the circle, cluster, and
            site-name columns.

    Returns:
        pd.DataFrame:
            A dataframe containing one row per unique site combination
            and its corresponding ``Row_Count``.

    Raises:
        InputValidationError:
            If no site groups can be generated from the dataframe.
    """

    site_columns = list(
        columns.get_site_identity_columns()
    )

    summary = (
        dataframe
        .groupby(
            site_columns,
            dropna=False,
        )
        .size()
        .reset_index(
            name=ROW_COUNT_COLUMN
        )
        .sort_values(
            by=[
                ROW_COUNT_COLUMN,
                *site_columns,
            ],
            ascending=True,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    if summary.empty:
        raise InputValidationError(
            "No unique sites could be identified.",
            error_code="NO_UNIQUE_SITES_IDENTIFIED",
        )

    return summary


def save_site_row_count_summary(
    summary: pd.DataFrame,
    source_file: Path,
    output_directory: Path,
) -> Path:
    """
    Save the unique-site row-count summary as a CSV.

    Args:
        summary:
            Site-level row-count dataframe to save.

        source_file:
            KPI-specific source file used to create the output name.

        output_directory:
            Directory where the summary CSV should be stored.

    Returns:
        Path:
            Path of the generated site-analysis CSV.

    Raises:
        FileProcessingError:
            If the output directory cannot be created or the CSV cannot
            be written.
    """

    resolved_output_directory = ensure_directory(
        output_directory
    )

    output_filename = sanitise_filename(
        f"{source_file.stem}_unique_site_row_counts.csv"
    )

    output_path = (
        resolved_output_directory
        / output_filename
    )

    write_csv_file(
        summary,
        output_path,
    )

    return output_path


def analyse_sites(
    dataframe: pd.DataFrame,
    source_file: str | Path,
    output_directory: str | Path,
    columns: DatasetColumnMapping,
    logger: Logger,
) -> SiteAnalysisResult:
    """
    Run unique-site identification and row-count analysis.

    This is the main calculation function for the first two steps of
    Part A. It validates the dataframe, identifies unique sites,
    calculates row counts, saves the summary, records key statistics,
    and returns a structured result.

    Args:
        dataframe:
            Loaded dataframe belonging to one KPI.

        source_file:
            KPI-specific source CSV associated with the dataframe.

        output_directory:
            Directory where the site-analysis summary should be saved.

        columns:
            Dataset column mapping used to locate the site identity
            columns.

        logger:
            Run-specific logger used to record analysis progress and
            summary information.

    Returns:
        SiteAnalysisResult:
            Structured site-analysis statistics and output-file path.

    Raises:
        InputValidationError:
            If the dataframe is empty, required columns are missing, or
            no unique sites can be identified.

        FileProcessingError:
            If the output directory or summary CSV cannot be created.

        TypeError:
            If source_file or output_directory cannot be converted to
            Path objects.
    """

    resolved_source_file = Path(
        source_file
    ).expanduser().resolve()

    resolved_output_directory = Path(
        output_directory
    ).expanduser().resolve()

    logger.info(
        "Starting site analysis for %s.",
        resolved_source_file.name,
    )

    validate_site_analysis_input(
        dataframe=dataframe,
        columns=columns,
    )

    missing_identity_rows = (
        count_rows_with_missing_site_identity(
            dataframe=dataframe,
            columns=columns,
        )
    )

    if missing_identity_rows > 0:
        logger.warning(
            "%d row(s) contain incomplete site-identifying values in "
            "%s.",
            missing_identity_rows,
            resolved_source_file.name,
        )

    summary = create_site_row_count_summary(
        dataframe=dataframe,
        columns=columns,
    )

    summary_file = save_site_row_count_summary(
        summary=summary,
        source_file=resolved_source_file,
        output_directory=resolved_output_directory,
    )

    total_unique_sites = len(
        summary
    )

    minimum_rows = int(
        summary[ROW_COUNT_COLUMN].min()
    )

    maximum_rows = int(
        summary[ROW_COUNT_COLUMN].max()
    )

    average_rows = float(
        summary[ROW_COUNT_COLUMN].mean()
    )

    result = SiteAnalysisResult(
        source_file=resolved_source_file,
        summary_file=summary_file,
        total_rows=len(
            dataframe
        ),
        total_unique_sites=total_unique_sites,
        minimum_rows_per_site=minimum_rows,
        maximum_rows_per_site=maximum_rows,
        average_rows_per_site=average_rows,
        rows_with_missing_site_identity=(
            missing_identity_rows
        ),
    )

    logger.info(
        "Site analysis completed for %s: total_rows=%d, "
        "unique_sites=%d, minimum_rows=%d, maximum_rows=%d, "
        "average_rows=%.2f.",
        resolved_source_file.name,
        result.total_rows,
        result.total_unique_sites,
        result.minimum_rows_per_site,
        result.maximum_rows_per_site,
        result.average_rows_per_site,
    )

    logger.info(
        "Site row-count summary saved to %s.",
        result.summary_file,
    )

    return result