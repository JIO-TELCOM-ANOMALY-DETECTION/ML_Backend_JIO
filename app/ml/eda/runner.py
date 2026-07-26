"""
Run dataset preparation and Part A EDA for one pipeline input.

The runner performs orchestration only. It:

1. calls ``prepare_dataset``;
2. loops through every prepared KPI-specific CSV;
3. creates one summary directory beside each prepared KPI file;
4. runs site analysis;
5. runs server validation;
6. runs timestamp data-integrity analysis; and
7. returns structured metadata containing every generated path.

The runner does not reconstruct output filenames created by individual stages.
Instead, it passes exact paths returned by one stage into the next stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path

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
)

# These relative imports assume runner.py is stored in the same package as
# data_preparation.py, site_analysis.py, server_validation.py, and
# data_integrity.py.
from .data_integrity import (
    DataIntegrityResult,
    analyse_data_integrity,
)
from .data_preparation import (
    DatasetPreparationResult,
    PreparedKPIDataset,
    prepare_dataset,
)
from .server_validation import (
    ServerValidationResult,
    validate_server_consistency,
)
from .site_analysis import (
    SiteAnalysisResult,
    analyse_sites,
)


@dataclass(frozen=True, slots=True)
class KPIEDARunResult:
    """
    Store all Part A results for one prepared KPI dataset.

    Attributes:
        prepared_dataset:
            Metadata returned by dataset preparation for the KPI file.

        summary_directory:
            Folder containing the generated site and integrity reports.

        site_analysis:
            Structured result returned by site analysis.

        server_validation:
            Structured result returned by server validation.

        data_integrity:
            Structured result returned by timestamp-integrity analysis.
    """

    prepared_dataset: PreparedKPIDataset
    summary_directory: Path
    site_analysis: SiteAnalysisResult
    server_validation: ServerValidationResult
    data_integrity: DataIntegrityResult


@dataclass(frozen=True, slots=True)
class EDARunnerResult:
    """
    Store the complete result of dataset preparation and Part A EDA.

    Attributes:
        preparation:
            Complete metadata returned by dataset preparation.

        kpi_results:
            Part A results for every prepared KPI-specific dataset.
    """

    preparation: DatasetPreparationResult
    kpi_results: tuple[KPIEDARunResult, ...]


_BUCKET_TIME_UNITS: dict[str, str] = {
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "day": "D",
    "days": "D",
}


def convert_bucket_to_frequency(
    bucket: BucketConfig,
) -> pd.Timedelta:
    """
    Convert the resolved bucket configuration into a pandas frequency.

    Args:
        bucket:
            Uniform bucket configuration returned by dataset preparation.

    Returns:
        pd.Timedelta:
            Positive expected interval between timestamps.

    Raises:
        InputValidationError:
            If the bucket unit is unsupported or cannot be converted.
    """

    normalized_bucket_type = (
        bucket.bucket_type
        .strip()
        .casefold()
    )

    pandas_unit = _BUCKET_TIME_UNITS.get(
        normalized_bucket_type
    )

    if pandas_unit is None:
        raise InputValidationError(
            "The resolved bucket type cannot be converted into a "
            "timestamp frequency.",
            error_code="UNSUPPORTED_BUCKET_TYPE_FOR_EDA",
            details={
                "bucket_type": bucket.bucket_type,
                "supported_bucket_types": sorted(
                    _BUCKET_TIME_UNITS
                ),
            },
        )

    try:
        frequency = pd.to_timedelta(
            bucket.bucket_size,
            unit=pandas_unit,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise InputValidationError(
            "The resolved bucket configuration could not be converted "
            "into a timestamp frequency.",
            error_code="INVALID_BUCKET_FREQUENCY",
            details={
                "bucket_type": bucket.bucket_type,
                "bucket_size": bucket.bucket_size,
                "original_error": str(error),
            },
        ) from error

    if frequency <= pd.Timedelta(0):
        raise InputValidationError(
            "The expected timestamp frequency must be positive.",
            error_code="NON_POSITIVE_EDA_FREQUENCY",
            details={
                "frequency": str(frequency),
            },
        )

    return frequency


def resolve_integrity_timeline_inputs(
    config: DatasetInputConfig,
    num_days: int | None,
) -> tuple[
    pd.Timestamp | None,
    pd.Timestamp | None,
    int | None,
]:
    """
    Resolve the timeline method passed to data-integrity analysis.

    Exactly one method is required:

    - both ``expected_start`` and ``expected_end`` from the request; or
    - a positive ``num_days`` value supplied to this runner.

    Args:
        config:
            Validated dataset input configuration.

        num_days:
            Optional number of complete days counted from the earliest
            valid timestamp in each prepared KPI file.

    Returns:
        tuple[pd.Timestamp | None, pd.Timestamp | None, int | None]:
            Explicit start, explicit end, and number of days to pass to
            timestamp-integrity analysis.

    Raises:
        InputValidationError:
            If neither method, both methods, an incomplete date range,
            or a non-positive number of days is supplied.
    """

    expected_start = None
    expected_end = None

    if config.expected_timeline is not None:
        expected_start = (
            config.expected_timeline.expected_start
        )
        expected_end = (
            config.expected_timeline.expected_end
        )

    has_start = expected_start is not None
    has_end = expected_end is not None

    if has_start != has_end:
        raise InputValidationError(
            "expected_start and expected_end must be supplied together.",
            error_code="INCOMPLETE_EXPECTED_TIMELINE",
            details={
                "expected_start_supplied": has_start,
                "expected_end_supplied": has_end,
            },
        )

    has_explicit_range = has_start and has_end
    has_num_days = num_days is not None

    if has_explicit_range == has_num_days:
        raise InputValidationError(
            "Provide exactly one data-integrity timeline method: either "
            "expected_start with expected_end, or num_days.",
            error_code="INVALID_DATA_INTEGRITY_TIMELINE_METHOD",
            details={
                "explicit_range_supplied": has_explicit_range,
                "num_days_supplied": has_num_days,
            },
        )

    if num_days is not None and num_days <= 0:
        raise InputValidationError(
            "num_days must be greater than zero.",
            error_code="INVALID_DATA_INTEGRITY_NUM_DAYS",
            details={
                "num_days": num_days,
            },
        )

    return (
        pd.Timestamp(expected_start)
        if expected_start is not None
        else None,
        pd.Timestamp(expected_end)
        if expected_end is not None
        else None,
        num_days,
    )


def build_kpi_summary_directory(
    prepared_dataset: PreparedKPIDataset,
) -> Path:
    """
    Create the summary directory for one prepared KPI file.

    The folder is created beside the prepared KPI CSV using:

        <prepared_file_stem>_summary

    Args:
        prepared_dataset:
            Metadata for the prepared KPI-specific file.

    Returns:
        Path:
            Resolved summary-directory path.

    Raises:
        FileProcessingError:
            If the directory cannot be created.
    """

    summary_folder_name = sanitise_filename(
        f"{prepared_dataset.file_path.stem}_summary"
    )

    return ensure_directory(
        prepared_dataset.file_path.parent
        / summary_folder_name
    )


def run_eda_pipeline(
    config: DatasetInputConfig,
    logger: Logger,
    *,
    num_days: int | None = None,
) -> EDARunnerResult:
    """
    Run dataset preparation and Part A EDA for every resolved KPI.

    This is the main orchestration function. Exceptions are intentionally
    allowed to propagate to the higher workflow layer, which can update the
    run manifest and mark the complete run as failed.

    Args:
        config:
            Validated dataset input configuration for the source CSV.

        logger:
            Run-specific logger used by all called stages.

        num_days:
            Optional number of expected complete days. Supply this only
            when explicit expected timeline boundaries are absent from
            ``config``.

    Returns:
        EDARunnerResult:
            Dataset-preparation metadata and one structured EDA result
            for every prepared KPI file.

    Raises:
        InputValidationError:
            If timeline selection or any called validation stage fails.

        FileProcessingError:
            If input, preparation, or report files cannot be processed.
    """

    logger.info(
        "Starting dataset preparation and Part A EDA for %s.",
        config.input_file,
    )

    preparation_result = prepare_dataset(
        config=config,
        logger=logger,
    )

    expected_frequency = convert_bucket_to_frequency(
        preparation_result.bucket
    )

    (
        expected_start,
        expected_end,
        resolved_num_days,
    ) = resolve_integrity_timeline_inputs(
        config=config,
        num_days=num_days,
    )

    kpi_results: list[KPIEDARunResult] = []

    for prepared_dataset in (
        preparation_result.prepared_datasets
    ):
        logger.info(
            "Starting Part A EDA for KPI '%s' using %s.",
            prepared_dataset.kpi_name,
            prepared_dataset.file_path,
        )

        dataframe = read_csv_file(
            prepared_dataset.file_path
        )

        summary_directory = (
            build_kpi_summary_directory(
                prepared_dataset
            )
        )

        site_result = analyse_sites(
            dataframe=dataframe,
            source_file=prepared_dataset.file_path,
            output_directory=summary_directory,
            columns=config.columns,
            logger=logger,
        )

        server_result = validate_server_consistency(
            dataframe=dataframe,
            source_file=prepared_dataset.file_path,
            columns=config.columns,
            logger=logger,
        )

        integrity_result = analyse_data_integrity(
            dataframe=dataframe,
            source_file=prepared_dataset.file_path,
            site_summary_file=(
                site_result.summary_file
            ),
            output_directory=summary_directory,
            columns=config.columns,
            logger=logger,
            expected_frequency=expected_frequency,
            expected_start=expected_start,
            expected_end=expected_end,
            num_days=resolved_num_days,
        )

        kpi_result = KPIEDARunResult(
            prepared_dataset=prepared_dataset,
            summary_directory=summary_directory,
            site_analysis=site_result,
            server_validation=server_result,
            data_integrity=integrity_result,
        )

        kpi_results.append(
            kpi_result
        )

        logger.info(
            "Part A EDA completed for KPI '%s'. Summary directory: %s.",
            prepared_dataset.kpi_name,
            summary_directory,
        )

    result = EDARunnerResult(
        preparation=preparation_result,
        kpi_results=tuple(
            kpi_results
        ),
    )

    logger.info(
        "Dataset preparation and Part A EDA completed for %d KPI "
        "dataset(s).",
        len(result.kpi_results),
    )

    return result