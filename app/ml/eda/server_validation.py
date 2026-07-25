"""
Validate server-type consistency for each unique telecom site.

This module performs the server check from Part A of the telecom
anomaly-detection pipeline.

For every unique site, it checks whether all available non-blank
server values refer to the same server type.

A unique site is defined by:

- Jio circle
- Jio cluster
- AMF site name

Rules:

- If the server column is absent, the check is skipped and logged.
- Blank server values are logged separately.
- Multiple server types for one site are logged as warnings.
- Server problems do not stop pipeline execution.
- No output CSV is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path

import pandas as pd

from app.core.exceptions import InputValidationError
from app.schemas.common.dataset import DatasetColumnMapping


UNIQUE_SERVER_TYPES_COLUMN = "Unique_Server_Types"
SERVER_TYPES_COLUMN = "Server_Types"
MISSING_SERVER_ROWS_COLUMN = "Missing_Server_Rows"


@dataclass(frozen=True, slots=True)
class ServerValidationResult:
    """
    Store the summary returned by server validation.

    Attributes:
        source_file:
            KPI-specific source file associated with the dataframe.

        check_skipped:
            Whether validation was skipped because the server column
            was unavailable.

        total_sites:
            Total number of unique sites checked.

        consistent_site_count:
            Sites containing exactly one non-blank server type.

        inconsistent_site_count:
            Sites containing multiple non-blank server types.

        unresolved_site_count:
            Sites containing no usable server value.

        rows_with_missing_server:
            Total rows containing a missing or blank server value.

        sites_with_missing_server:
            Total sites containing at least one missing server value.
    """

    source_file: Path
    check_skipped: bool
    total_sites: int
    consistent_site_count: int
    inconsistent_site_count: int
    unresolved_site_count: int
    rows_with_missing_server: int
    sites_with_missing_server: int


def validate_server_check_input(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> None:
    """
    Validate the dataframe before running the server check.

    The server column itself is optional and is therefore not validated
    here. Its absence causes the server check to be skipped.

    Args:
        dataframe:
            KPI-specific dataframe to inspect.

        columns:
            Dataset column mapping containing the site identity column
            names.

    Returns:
        None.

    Raises:
        InputValidationError:
            If the dataframe is empty or one or more site identity
            columns are missing.
    """

    if dataframe.empty:
        raise InputValidationError(
            "Server validation cannot process an empty dataframe.",
            error_code="EMPTY_SERVER_VALIDATION_DATASET",
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
            "The dataframe is missing columns required for server "
            "validation.",
            error_code="MISSING_SERVER_VALIDATION_COLUMNS",
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


def normalise_server_values(
    server_values: pd.Series,
) -> pd.Series:
    """
    Normalize server values before consistency analysis.

    Values are converted to trimmed strings. Empty strings are treated
    as missing values.

    Args:
        server_values:
            Source server-type column.

    Returns:
        pd.Series:
            Normalized server values with blanks represented as
            missing values.

    Raises:
        None.
    """

    normalized_values = (
        server_values
        .astype("string")
        .str.strip()
    )

    return normalized_values.mask(
        normalized_values.eq("")
    )


def create_server_summary(
    dataframe: pd.DataFrame,
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """
    Create a site-level server consistency summary.

    Missing server values are excluded when counting unique server
    types but are counted separately for logging.

    Args:
        dataframe:
            KPI-specific dataframe containing the server column.

        columns:
            Dataset column mapping containing site and server column
            names.

    Returns:
        pd.DataFrame:
            One row per unique site containing:

            - number of non-blank server types;
            - comma-separated server types;
            - number of rows with missing server values.

    Raises:
        KeyError:
            If required columns are unavailable. This should not occur
            after input validation.
    """

    site_columns = list(
        columns.get_site_identity_columns()
    )

    server_column = (
        columns.server_type_column
    )

    working_dataframe = dataframe[
        site_columns + [server_column]
    ].copy()

    normalized_server_column = (
        "_Normalized_Server_Type"
    )

    working_dataframe[
        normalized_server_column
    ] = normalise_server_values(
        working_dataframe[server_column]
    )

    summary = (
        working_dataframe
        .groupby(
            site_columns,
            dropna=False,
        )[normalized_server_column]
        .agg(
            **{
                UNIQUE_SERVER_TYPES_COLUMN: (
                    lambda values: int(
                        values.dropna().nunique()
                    )
                ),
                SERVER_TYPES_COLUMN: (
                    lambda values: ", ".join(
                        sorted(
                            values
                            .dropna()
                            .unique()
                            .tolist()
                        )
                    )
                ),
                MISSING_SERVER_ROWS_COLUMN: (
                    lambda values: int(
                        values.isna().sum()
                    )
                ),
            }
        )
        .reset_index()
    )

    return summary


def log_missing_server_values(
    summary: pd.DataFrame,
    columns: DatasetColumnMapping,
    logger: Logger,
) -> None:
    """
    Log sites containing missing or blank server values.

    Args:
        summary:
            Site-level server summary.

        columns:
            Dataset column mapping containing site identity columns.

        logger:
            Run-specific logger used to record warnings.

    Returns:
        None.

    Raises:
        None.
    """

    site_columns = list(
        columns.get_site_identity_columns()
    )

    affected_sites = summary.loc[
        summary[MISSING_SERVER_ROWS_COLUMN] > 0
    ]

    for _, row in affected_sites.iterrows():
        site_details = {
            column: row[column]
            for column in site_columns
        }

        logger.warning(
            "Missing server value detected: site=%s, "
            "missing_server_rows=%d.",
            site_details,
            int(
                row[MISSING_SERVER_ROWS_COLUMN]
            ),
        )


def log_inconsistent_servers(
    summary: pd.DataFrame,
    columns: DatasetColumnMapping,
    logger: Logger,
) -> None:
    """
    Log sites containing multiple non-blank server types.

    Args:
        summary:
            Site-level server summary.

        columns:
            Dataset column mapping containing site identity columns.

        logger:
            Run-specific logger used to record warnings.

    Returns:
        None.

    Raises:
        None.
    """

    site_columns = list(
        columns.get_site_identity_columns()
    )

    inconsistent_sites = summary.loc[
        summary[UNIQUE_SERVER_TYPES_COLUMN] > 1
    ]

    for _, row in inconsistent_sites.iterrows():
        site_details = {
            column: row[column]
            for column in site_columns
        }

        logger.warning(
            "Multiple server types detected: site=%s, "
            "server_type_count=%d, server_types=%s.",
            site_details,
            int(
                row[UNIQUE_SERVER_TYPES_COLUMN]
            ),
            row[SERVER_TYPES_COLUMN],
        )


def validate_server_consistency(
    dataframe: pd.DataFrame,
    source_file: str | Path,
    columns: DatasetColumnMapping,
    logger: Logger,
) -> ServerValidationResult:
    """
    Check server consistency for every unique site.

    This is the main server-validation function. It logs missing server
    values and sites containing multiple server types but does not stop
    pipeline execution for either condition.

    Args:
        dataframe:
            Loaded dataframe belonging to one KPI.

        source_file:
            KPI-specific CSV associated with the dataframe.

        columns:
            Dataset column mapping containing the site identity and
            server column names.

        logger:
            Run-specific logger used to record validation results.

    Returns:
        ServerValidationResult:
            Structured counts describing server consistency.

    Raises:
        InputValidationError:
            If the dataframe is empty or required site identity columns
            are missing.

        TypeError:
            If source_file cannot be converted into a Path.
    """

    resolved_source_file = Path(
        source_file
    ).expanduser().resolve()

    validate_server_check_input(
        dataframe=dataframe,
        columns=columns,
    )

    site_columns = list(
        columns.get_site_identity_columns()
    )

    total_sites = int(
        dataframe
        .groupby(
            site_columns,
            dropna=False,
        )
        .ngroups
    )

    server_column = (
        columns.server_type_column
    )

    if server_column not in dataframe.columns:
        logger.warning(
            "Server validation skipped for %s because optional column "
            "'%s' is unavailable.",
            resolved_source_file.name,
            server_column,
        )

        return ServerValidationResult(
            source_file=resolved_source_file,
            check_skipped=True,
            total_sites=total_sites,
            consistent_site_count=0,
            inconsistent_site_count=0,
            unresolved_site_count=0,
            rows_with_missing_server=0,
            sites_with_missing_server=0,
        )

    logger.info(
        "Starting server validation for %s.",
        resolved_source_file.name,
    )

    summary = create_server_summary(
        dataframe=dataframe,
        columns=columns,
    )

    rows_with_missing_server = int(
        summary[MISSING_SERVER_ROWS_COLUMN].sum()
    )

    sites_with_missing_server = int(
        (
            summary[MISSING_SERVER_ROWS_COLUMN]
            > 0
        ).sum()
    )

    consistent_site_count = int(
        (
            summary[UNIQUE_SERVER_TYPES_COLUMN]
            == 1
        ).sum()
    )

    inconsistent_site_count = int(
        (
            summary[UNIQUE_SERVER_TYPES_COLUMN]
            > 1
        ).sum()
    )

    unresolved_site_count = int(
        (
            summary[UNIQUE_SERVER_TYPES_COLUMN]
            == 0
        ).sum()
    )

    if rows_with_missing_server > 0:
        logger.warning(
            "Found %d missing server value(s) across %d site(s) in "
            "%s.",
            rows_with_missing_server,
            sites_with_missing_server,
            resolved_source_file.name,
        )

        log_missing_server_values(
            summary=summary,
            columns=columns,
            logger=logger,
        )

    if inconsistent_site_count > 0:
        logger.warning(
            "Found %d site(s) with multiple server types in %s.",
            inconsistent_site_count,
            resolved_source_file.name,
        )

        log_inconsistent_servers(
            summary=summary,
            columns=columns,
            logger=logger,
        )

    if unresolved_site_count > 0:
        logger.warning(
            "Found %d site(s) without any usable server value in %s.",
            unresolved_site_count,
            resolved_source_file.name,
        )

    logger.info(
        "Server validation completed for %s: total_sites=%d, "
        "consistent_sites=%d, inconsistent_sites=%d, "
        "unresolved_sites=%d.",
        resolved_source_file.name,
        total_sites,
        consistent_site_count,
        inconsistent_site_count,
        unresolved_site_count,
    )

    return ServerValidationResult(
        source_file=resolved_source_file,
        check_skipped=False,
        total_sites=total_sites,
        consistent_site_count=consistent_site_count,
        inconsistent_site_count=inconsistent_site_count,
        unresolved_site_count=unresolved_site_count,
        rows_with_missing_server=(
            rows_with_missing_server
        ),
        sites_with_missing_server=(
            sites_with_missing_server
        ),
    )