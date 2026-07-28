"""
Run dataset preparation and Part A EDA for one pipeline input.

The caller gives this runner one root output directory. The runner owns the
complete EDA directory layout:

    <output_directory>/
        prepared_datasets/
            KPI_A.csv
            KPI_B.csv
        summaries/
            KPI_A/
                ...site and integrity reports...
            KPI_B/
                ...site and integrity reports...

The runner performs orchestration only. It:

1. creates the runner-controlled output directories;
2. calls ``prepare_dataset`` with the exact prepared-dataset directory;
3. loops through every prepared KPI-specific CSV;
4. creates one KPI-specific directory under ``summaries``;
5. runs site analysis;
6. runs server validation;
7. runs timestamp data-integrity analysis; and
8. returns structured metadata containing every generated path.

Individual stages may construct filenames inside directories supplied by the
runner, but they no longer decide where those directories are located.
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
        output_directory:
            Root output directory supplied to the runner.

        prepared_datasets_directory:
            Runner-created directory containing normalized KPI CSV files.

        summaries_directory:
            Runner-created parent directory containing one report directory
            per KPI.

        preparation:
            Complete metadata returned by dataset preparation.

        kpi_results:
            Part A results for every prepared KPI-specific dataset.
    """

    output_directory: Path
    prepared_datasets_directory: Path
    summaries_directory: Path
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
) -> tuple[
    pd.Timestamp | None,
    pd.Timestamp | None,
    int | None,
    bool,
]:
    """
    Resolve schema timeline settings for data-integrity analysis.

    The request schema has already validated the selected mode and its
    associated values. This function only translates that validated schema
    into the keyword arguments expected by ``analyse_data_integrity``.

    Args:
        config:
            Validated dataset input configuration containing one of the
            ``date_range``, ``num_days``, or ``global_min_max`` modes.

    Returns:
        tuple[pd.Timestamp | None, pd.Timestamp | None, int | None, bool]:
            Explicit start, explicit end, number of days, and whether the
            timeline must use the KPI dataset's global minimum and maximum.

    Raises:
        InputValidationError:
            If an unsupported timeline mode reaches the runner. This should
            not occur after successful Pydantic schema validation.
    """

    timeline_config = config.expected_timeline
    timeline_mode = timeline_config.timeline_mode

    if timeline_mode == "date_range":
        return (
            pd.Timestamp(
                timeline_config.expected_start
            ),
            pd.Timestamp(
                timeline_config.expected_end
            ),
            None,
            False,
        )

    if timeline_mode == "num_days":
        return (
            None,
            None,
            timeline_config.num_days,
            False,
        )

    if timeline_mode == "global_min_max":
        return (
            None,
            None,
            None,
            True,
        )

    raise InputValidationError(
        "The data-integrity timeline mode is unsupported.",
        error_code="UNSUPPORTED_DATA_INTEGRITY_TIMELINE_MODE",
        details={
            "timeline_mode": timeline_mode,
        },
    )



def build_kpi_summary_directory(
    prepared_dataset: PreparedKPIDataset,
    summaries_directory: str | Path,
) -> Path:
    """
    Create the runner-controlled summary directory for one KPI file.

    The folder is created under the supplied summaries parent using the
    prepared KPI filename stem:

        <summaries_directory>/<prepared_file_stem>/

    Args:
        prepared_dataset:
            Metadata for the prepared KPI-specific file.

        summaries_directory:
            Runner-controlled parent directory for all KPI summaries.

    Returns:
        Path:
            Resolved KPI summary-directory path.

    Raises:
        FileProcessingError:
            If the directory cannot be created.
    """

    summary_folder_name = sanitise_filename(
        prepared_dataset.file_path.stem
    )

    return ensure_directory(
        Path(summaries_directory)
        / summary_folder_name
    )


def run_eda_pipeline(
    config: DatasetInputConfig,
    logger: Logger,
    *,
    output_directory: str | Path,
) -> EDARunnerResult:
    """
    Run dataset preparation and Part A EDA for every resolved KPI.

    The caller supplies one root output directory. This runner creates and
    controls the following structure inside it:

        prepared_datasets/
        summaries/<prepared_kpi_file_stem>/

    Exceptions are intentionally allowed to propagate to the higher workflow
    layer, which can update the run manifest and mark the complete run as
    failed.

    Timeline selection is read entirely from
    ``config.expected_timeline``. The runner therefore has one validated
    source of truth for ``date_range``, ``num_days``, and
    ``global_min_max`` modes.

    Args:
        config:
            Validated dataset input configuration for the source CSV.

        logger:
            Run-specific logger used by all called stages.

        output_directory:
            Root directory where this EDA run must save all prepared datasets
            and reports.

    Returns:
        EDARunnerResult:
            Runner-controlled directories, dataset-preparation metadata, and
            one structured EDA result for every prepared KPI file.

    Raises:
        InputValidationError:
            If timeline conversion or any called validation stage fails.

        FileProcessingError:
            If input, preparation, or report files cannot be processed.
    """

    resolved_output_directory = ensure_directory(
        output_directory
    )

    prepared_datasets_directory = ensure_directory(
        resolved_output_directory
        / "prepared_datasets"
    )

    summaries_directory = ensure_directory(
        resolved_output_directory
        / "summaries"
    )

    logger.info(
        "Starting dataset preparation and Part A EDA for %s. Root "
        "output directory: %s.",
        config.input_file,
        resolved_output_directory,
    )

    preparation_result = prepare_dataset(
        config=config,
        logger=logger,
        dataset_directory=(
            prepared_datasets_directory
        ),
    )

    expected_frequency = convert_bucket_to_frequency(
        preparation_result.bucket
    )

    (
        expected_start,
        expected_end,
        resolved_num_days,
        use_global_min_max,
    ) = resolve_integrity_timeline_inputs(
        config=config,
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
                prepared_dataset=(
                    prepared_dataset
                ),
                summaries_directory=(
                    summaries_directory
                ),
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
            use_global_min_max=use_global_min_max,
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
        output_directory=(
            resolved_output_directory
        ),
        prepared_datasets_directory=(
            prepared_datasets_directory
        ),
        summaries_directory=(
            summaries_directory
        ),
        preparation=preparation_result,
        kpi_results=tuple(
            kpi_results
        ),
    )

    logger.info(
        "Dataset preparation and Part A EDA completed for %d KPI "
        "dataset(s). Root output directory: %s.",
        len(result.kpi_results),
        result.output_directory,
    )

    return result