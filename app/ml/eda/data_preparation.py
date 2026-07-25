"""
Prepare an input CSV for the telecom anomaly-detection pipeline.

This module performs the common preparation required before EDA,
preprocessing, feature engineering, training, or inference.

The workflow should first copy the uploaded input file into the run's input
directory. This module then creates a sibling directory named:

    <input_file_stem>_dataset

and writes one normalized CSV per KPI into that directory.

Example:

    runs/<run_id>/input/amf_data.csv

becomes:

    runs/<run_id>/input/amf_data_dataset/
        KPI_A.csv
        KPI_B.csv

If the source contains only one KPI, one normalized KPI file is still written
so downstream stages always receive the same directory structure.

This module does not perform EDA checks, timeline correction, outlier
detection, missing-value filling, feature generation, model training, or
inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Sequence

import pandas as pd

from app.core.exceptions import InputValidationError
from app.schemas.common.dataset import (
    BucketConfig,
    DatasetInputConfig,
)
from app.services.file_service import (
    ensure_directory,
    read_csv_file,
    sanitise_filename,
    validate_input_file,
    write_csv_file,
)


@dataclass(frozen=True, slots=True)
class PreparedKPIDataset:
    """
    Store metadata for one prepared KPI-specific CSV.

    Attributes:
        kpi_name:
            KPI represented by the prepared file.
        file_path:
            Location of the prepared KPI-specific CSV.
        row_count:
            Number of rows written to the prepared file.
    """

    kpi_name: str
    file_path: Path
    row_count: int


@dataclass(frozen=True, slots=True)
class DatasetPreparationResult:
    """
    Store the complete result of dataset preparation.

    Attributes:
        source_file:
            Validated input CSV used by the preparation stage.
        dataset_directory:
            Folder containing the prepared KPI-specific files.
        bucket:
            Resolved bucket configuration used by the dataset.
        prepared_datasets:
            Metadata for each KPI-specific CSV created.
        total_input_rows:
            Number of rows read from the source CSV.
        was_split:
            True when the source contained multiple KPIs.
    """

    source_file: Path
    dataset_directory: Path
    bucket: BucketConfig
    prepared_datasets: tuple[PreparedKPIDataset, ...]
    total_input_rows: int
    was_split: bool


def validate_base_columns(
    dataframe: pd.DataFrame,
    config: DatasetInputConfig,
) -> None:
    """
    Validate columns required in every accepted telecom dataset.

    Args:
        dataframe:
            Loaded input dataset.
        config:
            Dataset configuration containing the physical column names.

    Returns:
        None.

    Raises:
        InputValidationError:
            If one or more mandatory base columns are absent.
    """

    required_columns = (
        config.columns.get_base_required_columns()
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "The input dataset is missing mandatory columns.",
            error_code="MISSING_BASE_DATASET_COLUMNS",
            details={
                "missing_columns": missing_columns,
                "required_columns": list(
                    required_columns
                ),
                "available_columns": list(
                    dataframe.columns
                ),
            },
        )


def resolve_kpi_column(
    dataframe: pd.DataFrame,
    config: DatasetInputConfig,
    logger: Logger,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """
    Resolve KPI names and create a complete KPI-name column.

    Rules:

    - When the KPI column is absent, single_kpi_name must be supplied.
    - When both sources are present, all non-empty CSV KPI values must
      match single_kpi_name.
    - Blank KPI values may be filled using single_kpi_name.
    - Without single_kpi_name, blank KPI values are rejected.
    - Multiple valid KPI names are retained for file separation.

    Args:
        dataframe:
            Loaded input dataset.
        config:
            Dataset configuration containing the KPI column mapping and
            optional single-KPI declaration.
        logger:
            Run-specific logger used to record KPI resolution.

    Returns:
        tuple[pd.DataFrame, tuple[str, ...]]:
            A dataframe with a complete normalized KPI column and the
            distinct KPI names in their original order.

    Raises:
        InputValidationError:
            If KPI identity is unavailable, contradictory, blank, or
            otherwise cannot be resolved safely.
    """

    resolved = dataframe.copy()

    kpi_column = (
        config.columns.kpi_name_column
    )
    declared_kpi = config.single_kpi_name

    if kpi_column not in resolved.columns:
        if declared_kpi is None:
            raise InputValidationError(
                "The KPI-name column is missing and no single KPI name "
                "was supplied.",
                error_code="KPI_IDENTITY_NOT_PROVIDED",
                details={
                    "kpi_name_column": kpi_column,
                },
            )

        resolved[kpi_column] = declared_kpi

        logger.info(
            "Added KPI column '%s' using declared KPI '%s'.",
            kpi_column,
            declared_kpi,
        )

    else:
        normalized_kpis = (
            resolved[kpi_column]
            .astype("string")
            .str.strip()
        )

        normalized_kpis = normalized_kpis.mask(
            normalized_kpis.eq("")
        )

        if declared_kpi is not None:
            conflicting_kpis = (
                normalized_kpis.dropna()
                .loc[
                    lambda values: values.ne(
                        declared_kpi
                    )
                ]
                .drop_duplicates()
                .tolist()
            )

            if conflicting_kpis:
                raise InputValidationError(
                    "The supplied single KPI name conflicts with KPI "
                    "names inside the source dataset.",
                    error_code="SINGLE_KPI_NAME_CONFLICT",
                    details={
                        "declared_kpi_name": declared_kpi,
                        "conflicting_kpi_names": (
                            conflicting_kpis
                        ),
                    },
                )

            blank_count = int(
                normalized_kpis.isna().sum()
            )

            if blank_count:
                normalized_kpis = (
                    normalized_kpis.fillna(
                        declared_kpi
                    )
                )

                logger.warning(
                    "Filled %d blank KPI-name values using '%s'.",
                    blank_count,
                    declared_kpi,
                )

        elif normalized_kpis.isna().any():
            raise InputValidationError(
                "The KPI-name column contains blank values that cannot "
                "be assigned to a KPI-specific dataset.",
                error_code="UNRESOLVED_KPI_NAMES",
                details={
                    "kpi_name_column": kpi_column,
                    "blank_kpi_row_count": int(
                        normalized_kpis.isna().sum()
                    ),
                },
            )

        resolved[kpi_column] = normalized_kpis

    kpi_names = tuple(
        dict.fromkeys(
            resolved[kpi_column]
            .astype(str)
            .tolist()
        )
    )

    if not kpi_names:
        raise InputValidationError(
            "No KPI names were resolved from the input dataset.",
            error_code="NO_KPI_NAMES_RESOLVED",
            details={
                "kpi_name_column": kpi_column,
            },
        )

    logger.info(
        "Resolved %d KPI name(s): %s.",
        len(kpi_names),
        list(kpi_names),
    )

    return resolved, kpi_names


def parse_bucket_sizes(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Parse and validate values from the bucket-size column.

    Missing values are retained temporarily because they may be filled
    using request-level bucket configuration.

    Args:
        series:
            Source bucket-size column.
        column_name:
            Physical bucket-size column name used in error details.

    Returns:
        pd.Series:
            Parsed numeric bucket-size values.

    Raises:
        InputValidationError:
            If a non-empty bucket size is non-numeric, non-positive, or
            not a whole number.
    """

    text_values = (
        series.astype("string")
        .str.strip()
    )

    blank_mask = (
        series.isna()
        | text_values.eq("")
    )

    numeric_values = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid_numeric = (
        numeric_values.isna()
        & ~blank_mask
    )

    if invalid_numeric.any():
        raise InputValidationError(
            "The bucket-size column contains non-numeric values.",
            error_code="INVALID_BUCKET_SIZE_VALUES",
            details={
                "bucket_size_column": column_name,
                "invalid_values": (
                    text_values.loc[
                        invalid_numeric
                    ]
                    .drop_duplicates()
                    .tolist()
                ),
            },
        )

    available_values = numeric_values.dropna()

    invalid_range = (
        available_values.le(0)
        | available_values.mod(1).ne(0)
    )

    if invalid_range.any():
        raise InputValidationError(
            "Bucket sizes must be positive whole numbers.",
            error_code="INVALID_BUCKET_SIZE_RANGE",
            details={
                "bucket_size_column": column_name,
                "invalid_values": (
                    available_values.loc[
                        invalid_range
                    ]
                    .drop_duplicates()
                    .tolist()
                ),
            },
        )

    return numeric_values


def resolve_bucket_config(
    dataframe: pd.DataFrame,
    config: DatasetInputConfig,
    logger: Logger,
) -> tuple[pd.DataFrame, BucketConfig]:
    """
    Resolve one bucket configuration for the complete dataset.

    Bucket information may come from the request, the source CSV, or
    both. When both are present, they must agree. The resolved values
    are written uniformly into the dataframe.

    Args:
        dataframe:
            Dataset whose bucket metadata must be resolved.
        config:
            Dataset configuration containing optional request-level
            bucket information and bucket column mappings.
        logger:
            Run-specific logger used to record bucket resolution.

    Returns:
        tuple[pd.DataFrame, BucketConfig]:
            A dataframe with complete uniform bucket columns and the
            resolved bucket configuration.

    Raises:
        InputValidationError:
            If bucket columns are incomplete, bucket information is
            unavailable, source values are invalid or non-uniform, or
            request and source values conflict.
    """

    resolved = dataframe.copy()

    type_column = (
        config.columns.bucket_type_column
    )
    size_column = (
        config.columns.bucket_size_column
    )

    type_present = (
        type_column in resolved.columns
    )
    size_present = (
        size_column in resolved.columns
    )

    if type_present != size_present:
        raise InputValidationError(
            "bucketType and bucketSize columns must appear together.",
            error_code="INCOMPLETE_BUCKET_COLUMN_PAIR",
            details={
                "bucket_type_column": type_column,
                "bucket_size_column": size_column,
            },
        )

    requested_bucket = config.bucket

    if not type_present:
        if requested_bucket is None:
            raise InputValidationError(
                "Bucket information is missing from both the request "
                "and the source dataset.",
                error_code="BUCKET_CONFIG_NOT_PROVIDED",
                details={
                    "bucket_type_column": type_column,
                    "bucket_size_column": size_column,
                },
            )

        resolved[type_column] = (
            requested_bucket.bucket_type
        )
        resolved[size_column] = (
            requested_bucket.bucket_size
        )

        logger.info(
            "Added request bucket configuration: %s %d.",
            requested_bucket.bucket_type,
            requested_bucket.bucket_size,
        )

        return resolved, requested_bucket

    bucket_types = (
        resolved[type_column]
        .astype("string")
        .str.strip()
    )

    bucket_types = bucket_types.mask(
        bucket_types.eq("")
    )

    bucket_sizes = parse_bucket_sizes(
        series=resolved[size_column],
        column_name=size_column,
    )

    available_types = bucket_types.dropna()

    normalized_type_keys = (
        available_types
        .str.casefold()
        .drop_duplicates()
        .tolist()
    )

    available_sizes = (
        bucket_sizes.dropna()
        .astype(int)
    )

    unique_sizes = (
        available_sizes
        .drop_duplicates()
        .tolist()
    )

    if len(normalized_type_keys) > 1:
        raise InputValidationError(
            "The source dataset contains multiple bucket types.",
            error_code="MULTIPLE_BUCKET_TYPES",
            details={
                "bucket_types": (
                    available_types
                    .drop_duplicates()
                    .tolist()
                ),
            },
        )

    if len(unique_sizes) > 1:
        raise InputValidationError(
            "The source dataset contains multiple bucket sizes.",
            error_code="MULTIPLE_BUCKET_SIZES",
            details={
                "bucket_sizes": unique_sizes,
            },
        )

    if requested_bucket is not None:
        if (
            normalized_type_keys
            and normalized_type_keys[0]
            != requested_bucket.bucket_type.casefold()
        ):
            raise InputValidationError(
                "The request bucket type conflicts with the source "
                "dataset.",
                error_code="BUCKET_TYPE_CONFLICT",
                details={
                    "request_bucket_type": (
                        requested_bucket.bucket_type
                    ),
                    "source_bucket_types": (
                        available_types
                        .drop_duplicates()
                        .tolist()
                    ),
                },
            )

        if (
            unique_sizes
            and unique_sizes[0]
            != requested_bucket.bucket_size
        ):
            raise InputValidationError(
                "The request bucket size conflicts with the source "
                "dataset.",
                error_code="BUCKET_SIZE_CONFLICT",
                details={
                    "request_bucket_size": (
                        requested_bucket.bucket_size
                    ),
                    "source_bucket_sizes": unique_sizes,
                },
            )

        resolved_bucket = requested_bucket

    else:
        if (
            bucket_types.isna().any()
            or bucket_sizes.isna().any()
            or not normalized_type_keys
            or not unique_sizes
        ):
            raise InputValidationError(
                "The source bucket columns contain missing values and "
                "no request fallback was supplied.",
                error_code="INCOMPLETE_BUCKET_VALUES",
                details={
                    "blank_bucket_type_count": int(
                        bucket_types.isna().sum()
                    ),
                    "blank_bucket_size_count": int(
                        bucket_sizes.isna().sum()
                    ),
                },
            )

        resolved_bucket = BucketConfig(
            bucket_type=str(
                available_types.iloc[0]
            ),
            bucket_size=int(
                unique_sizes[0]
            ),
        )

    resolved[type_column] = (
        resolved_bucket.bucket_type
    )
    resolved[size_column] = (
        resolved_bucket.bucket_size
    )

    logger.info(
        "Resolved bucket configuration: %s %d.",
        resolved_bucket.bucket_type,
        resolved_bucket.bucket_size,
    )

    return resolved, resolved_bucket


def build_unique_kpi_filename(
    kpi_name: str,
    used_filenames: set[str],
) -> str:
    """
    Build a safe CSV filename unique within the dataset folder.

    Args:
        kpi_name:
            Original KPI name represented by the file.
        used_filenames:
            Case-insensitive filenames already assigned.

    Returns:
        str:
            Sanitized unique CSV filename.

    Raises:
        InputValidationError:
            If a unique filename cannot be generated.
    """

    base_filename = sanitise_filename(
        f"{kpi_name}.csv"
    )

    candidate = base_filename
    suffix = 2

    while candidate.casefold() in used_filenames:
        candidate = sanitise_filename(
            f"{Path(base_filename).stem}_{suffix}.csv"
        )

        suffix += 1

        if suffix > 10_000:
            raise InputValidationError(
                "A unique KPI filename could not be generated.",
                error_code="KPI_FILENAME_COLLISION",
                details={
                    "kpi_name": kpi_name,
                    "base_filename": base_filename,
                },
            )

    used_filenames.add(
        candidate.casefold()
    )

    return candidate


def write_kpi_datasets(
    dataframe: pd.DataFrame,
    kpi_names: Sequence[str],
    kpi_column: str,
    dataset_directory: Path,
    logger: Logger,
) -> tuple[PreparedKPIDataset, ...]:
    """
    Write one normalized CSV for each resolved KPI.

    Args:
        dataframe:
            Prepared dataset containing complete KPI and bucket columns.
        kpi_names:
            KPI names to write in their original source order.
        kpi_column:
            Physical KPI-name column used to select rows.
        dataset_directory:
            Output folder for all KPI-specific files.
        logger:
            Run-specific logger used to record generated files.

    Returns:
        tuple[PreparedKPIDataset, ...]:
            Metadata for every KPI-specific CSV written.

    Raises:
        InputValidationError:
            If a resolved KPI unexpectedly contains no rows.
        FileProcessingError:
            If the file service cannot write an output CSV.
    """

    prepared_files: list[
        PreparedKPIDataset
    ] = []

    used_filenames: set[str] = set()

    for kpi_name in kpi_names:
        kpi_dataframe = (
            dataframe.loc[
                dataframe[kpi_column].eq(
                    kpi_name
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        if kpi_dataframe.empty:
            raise InputValidationError(
                "A resolved KPI unexpectedly contains no rows.",
                error_code="EMPTY_KPI_DATASET",
                details={
                    "kpi_name": kpi_name,
                },
            )

        filename = build_unique_kpi_filename(
            kpi_name=kpi_name,
            used_filenames=used_filenames,
        )

        output_path = (
            dataset_directory
            / filename
        )

        write_csv_file(
            kpi_dataframe,
            output_path,
        )

        prepared_files.append(
            PreparedKPIDataset(
                kpi_name=kpi_name,
                file_path=output_path,
                row_count=len(
                    kpi_dataframe
                ),
            )
        )

        logger.info(
            "Created KPI dataset '%s' with %d rows at %s.",
            kpi_name,
            len(kpi_dataframe),
            output_path,
        )

    return tuple(
        prepared_files
    )


def prepare_dataset(
    config: DatasetInputConfig,
    logger: Logger,
) -> DatasetPreparationResult:
    """
    Validate and prepare an input CSV for downstream ML stages.

    The function reuses the existing file service for input validation,
    CSV loading, directory creation, filename sanitization, and CSV
    writing.

    A folder named ``<input_file_stem>_dataset`` is created beside the
    source CSV. One normalized CSV is written per KPI, including when
    the source contains only one KPI.

    Args:
        config:
            Validated shared dataset input configuration.
        logger:
            Run-specific logger used to record preparation activity.

    Returns:
        DatasetPreparationResult:
            Structured details about the source, resolved bucket,
            output directory, and prepared KPI-specific files.

    Raises:
        InputValidationError:
            If the input contains no rows, mandatory columns are
            missing, KPI identity cannot be resolved, or bucket
            metadata is invalid or contradictory.
        FileProcessingError:
            If the file service cannot validate or read the source,
            create the output directory, or write an output file.
    """

    input_path = (
        config.input_file
        .expanduser()
        .resolve()
    )

    validate_input_file(
        input_path
    )

    logger.info(
        "Starting dataset preparation for %s.",
        input_path,
    )

    dataframe = read_csv_file(
        input_path
    )

    if dataframe.empty:
        raise InputValidationError(
            "The input dataset contains no rows.",
            error_code="EMPTY_INPUT_DATASET",
            details={
                "input_file": str(
                    input_path
                ),
            },
        )

    validate_base_columns(
        dataframe=dataframe,
        config=config,
    )

    dataframe, kpi_names = resolve_kpi_column(
        dataframe=dataframe,
        config=config,
        logger=logger,
    )

    dataframe, resolved_bucket = (
        resolve_bucket_config(
            dataframe=dataframe,
            config=config,
            logger=logger,
        )
    )

    folder_name = sanitise_filename(
        f"{input_path.stem}_dataset"
    )

    dataset_directory = ensure_directory(
        input_path.parent
        / folder_name
    )

    prepared_datasets = write_kpi_datasets(
        dataframe=dataframe,
        kpi_names=kpi_names,
        kpi_column=(
            config.columns.kpi_name_column
        ),
        dataset_directory=dataset_directory,
        logger=logger,
    )

    result = DatasetPreparationResult(
        source_file=input_path,
        dataset_directory=dataset_directory,
        bucket=resolved_bucket,
        prepared_datasets=prepared_datasets,
        total_input_rows=len(
            dataframe
        ),
        was_split=len(
            prepared_datasets
        ) > 1,
    )

    logger.info(
        "Dataset preparation completed with %d input rows and %d "
        "prepared KPI file(s). Output directory: %s.",
        result.total_input_rows,
        len(
            result.prepared_datasets
        ),
        result.dataset_directory,
    )

    return result