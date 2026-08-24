"""
Load inference-time history inputs.

Intended project location::

    app/ml/inference/history/loader.py

This module is the data-loading layer of the inference history subsystem.

The parent inference runner resolves the saved run ID, validates
``inference_config.json``, and passes the already-resolved paths/settings to
the history subsystem. This loader therefore does not:

- query the run database;
- parse inference configuration JSON;
- decide how much history is required;
- validate timestamp continuity or gap limits;
- detect late-arriving real data;
- combine old and new data;
- run Parts A, B, or C; or
- update persistent history files.

For normal inference it loads only:

1. the newly supplied raw input file;
2. the Z-day raw cache; and
3. the Z-day processed cache for every trained KPI.

Full raw and processed history are intentionally not loaded during normal
inference.

For reconstruction only, ``load_reconstruction_raw_history_slice(...)`` may
read the saved full raw history and return exactly the older ``cache_days``
window immediately before the current raw-cache start. This keeps full-history
access off the normal inference path while giving reconstruction enough older
source context to rebuild even a late point near the beginning of the current
cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
)
from app.services.file_service import read_csv_file


@dataclass(frozen=True, slots=True)
class LoadedKPIProcessedCache:
    """
    Store one KPI's loaded Z-day processed cache.

    Attributes:
        kpi_name:
            Saved KPI identity from the completed training run.

        source_file:
            Exact processed-cache file supplied by the parent inference layer.

        dataframe:
            Loaded processed-cache dataframe. No history-semantic validation is
            performed in this module.
    """

    kpi_name: str
    source_file: Path
    dataframe: pd.DataFrame


@dataclass(frozen=True, slots=True)
class HistoryLoadResult:
    """
    Store all data loaded for one history-processing run.

    The result contains only the working caches used by normal inference.
    Full-history datasets are not loaded by this module.

    Attributes:
        incoming_file:
            Exact new raw input file.

        incoming_dataframe:
            Loaded new raw data.

        raw_cache_file:
            Exact Z-day raw-history cache file.

        raw_cache_dataframe:
            Loaded Z-day raw-history cache.

        processed_caches:
            One loaded Z-day processed cache per trained KPI, retained in the
            same deterministic KPI order supplied by ``expected_kpis``.
    """

    incoming_file: Path
    incoming_dataframe: pd.DataFrame

    raw_cache_file: Path
    raw_cache_dataframe: pd.DataFrame

    processed_caches: tuple[LoadedKPIProcessedCache, ...]

    def get_processed_cache(
        self,
        kpi_name: str,
    ) -> LoadedKPIProcessedCache:
        """
        Return the loaded processed cache for one KPI.

        Raises:
            KeyError:
                If the requested KPI was not part of the saved training run.
        """

        for cache in self.processed_caches:
            if cache.kpi_name == kpi_name:
                return cache

        raise KeyError(
            f"No processed history cache was loaded for KPI {kpi_name!r}."
        )

    @property
    def processed_cache_by_kpi(
        self,
    ) -> dict[str, pd.DataFrame]:
        """Return a convenient KPI-to-dataframe mapping."""

        return {
            cache.kpi_name: cache.dataframe
            for cache in self.processed_caches
        }


@dataclass(frozen=True, slots=True)
class LoadedReconstructionRawHistory:
    """
    Older raw-history slice loaded only for reconstruction.

    ``dataframe`` contains rows in the interval::

        [requested_start, cache_start)

    where ``requested_start`` is exactly ``cache_days`` before the earliest
    timestamp in the current raw Z-day cache.
    """

    source_file: Path
    requested_start: pd.Timestamp
    cache_start: pd.Timestamp
    dataframe: pd.DataFrame

    @property
    def row_count(self) -> int:
        return len(self.dataframe)



def _resolve_existing_file(
    path: str | Path,
    *,
    role: str,
) -> Path:
    """
    Normalize one required input path and require an existing regular file.

    The loader performs only filesystem/readability checks. Dataset structure
    and timestamp semantics belong to ``history/validation.py``.
    """

    resolved = Path(
        path
    ).expanduser().resolve()

    if not resolved.is_file():
        raise FileProcessingError(
            f"Required inference history file does not exist for {role}.",
            error_code="MISSING_INFERENCE_HISTORY_FILE",
            details={
                "role": role,
                "path": str(resolved),
            },
        )

    return resolved


def _load_required_csv(
    path: str | Path,
    *,
    role: str,
) -> tuple[Path, pd.DataFrame]:
    """Resolve and load one required CSV using the shared file service."""

    resolved = _resolve_existing_file(
        path,
        role=role,
    )

    dataframe = read_csv_file(
        resolved
    )

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise FileProcessingError(
            "The shared CSV reader did not return a pandas DataFrame.",
            error_code="INVALID_INFERENCE_HISTORY_LOAD_RESULT",
            details={
                "role": role,
                "path": str(resolved),
                "received_type": type(dataframe).__name__,
            },
        )

    return resolved, dataframe


def _validate_processed_cache_mapping(
    *,
    expected_kpis: tuple[str, ...],
    processed_cache_by_kpi: Mapping[str, str | Path],
) -> None:
    """
    Validate loader-level KPI/path mapping preconditions.

    The saved inference schema should already guarantee these relationships.
    This defensive check prevents the history subsystem from silently loading
    the wrong cache if it is called incorrectly from another internal caller.
    """

    if not expected_kpis:
        raise InputValidationError(
            "At least one trained KPI is required for history loading.",
            error_code="EMPTY_INFERENCE_HISTORY_KPI_LIST",
        )

    if len(expected_kpis) != len(set(expected_kpis)):
        raise InputValidationError(
            "Expected KPI names cannot contain duplicates.",
            error_code="DUPLICATE_INFERENCE_HISTORY_KPIS",
            details={
                "expected_kpis": list(expected_kpis),
            },
        )

    supplied_kpis = set(
        processed_cache_by_kpi
    )
    expected_kpi_set = set(
        expected_kpis
    )

    if supplied_kpis != expected_kpi_set:
        raise InputValidationError(
            "Processed history cache paths must contain exactly the trained "
            "KPI names.",
            error_code="INFERENCE_PROCESSED_CACHE_KPI_MISMATCH",
            details={
                "expected_kpis": sorted(expected_kpi_set),
                "supplied_kpis": sorted(supplied_kpis),
            },
        )


def load_reconstruction_raw_history_slice(
    *,
    raw_history_file: str | Path,
    raw_cache_dataframe: pd.DataFrame,
    timestamp_column: str,
    cache_days: int,
) -> LoadedReconstructionRawHistory:
    """
    Load the extra older raw-history window needed only for reconstruction.

    The returned slice is::

        [raw_cache_start - cache_days, raw_cache_start)

    Normal inference never needs to call this function.
    """

    if cache_days < 1:
        raise InputValidationError(
            "cache_days must be at least one for reconstruction loading.",
            error_code="INVALID_RECONSTRUCTION_CACHE_DAYS",
            details={"cache_days": cache_days},
        )

    timestamp_column = str(timestamp_column).strip()
    if not timestamp_column:
        raise InputValidationError(
            "A timestamp column is required for reconstruction loading.",
            error_code="MISSING_RECONSTRUCTION_TIMESTAMP_COLUMN",
        )

    if timestamp_column not in raw_cache_dataframe.columns:
        raise InputValidationError(
            "The current raw Z-day cache is missing the timestamp column "
            "required for reconstruction loading.",
            error_code="MISSING_RECONSTRUCTION_CACHE_TIMESTAMP_COLUMN",
            details={"timestamp_column": timestamp_column},
        )

    if raw_cache_dataframe.empty:
        raise InputValidationError(
            "The current raw Z-day cache is empty.",
            error_code="EMPTY_RECONSTRUCTION_RAW_CACHE",
        )

    cache_timestamps = pd.to_datetime(
        raw_cache_dataframe[timestamp_column],
        errors="coerce",
        utc=True,
    )
    invalid_cache_timestamps = cache_timestamps.isna()

    if bool(invalid_cache_timestamps.any()):
        raise InputValidationError(
            "The current raw Z-day cache contains invalid timestamps.",
            error_code="INVALID_RECONSTRUCTION_CACHE_TIMESTAMPS",
            details={
                "timestamp_column": timestamp_column,
                "invalid_rows": int(invalid_cache_timestamps.sum()),
            },
        )

    cache_start = pd.Timestamp(cache_timestamps.min())
    requested_start = cache_start - pd.Timedelta(days=cache_days)

    resolved_raw_history_file = _resolve_existing_file(
        raw_history_file,
        role="full raw history for reconstruction",
    )

    # Full raw history can become much larger than the rolling cache. Scan it
    # in bounded chunks and retain only the older reconstruction window instead
    # of materializing the complete history dataframe in memory.
    matching_chunks: list[pd.DataFrame] = []
    full_history_start: pd.Timestamp | None = None
    saw_history_rows = False

    try:
        chunk_iterator = pd.read_csv(
            resolved_raw_history_file,
            chunksize=100_000,
        )

        for chunk in chunk_iterator:
            if timestamp_column not in chunk.columns:
                raise InputValidationError(
                    "The saved full raw history is missing the timestamp "
                    "column required for reconstruction loading.",
                    error_code=(
                        "MISSING_RECONSTRUCTION_HISTORY_TIMESTAMP_COLUMN"
                    ),
                    details={
                        "timestamp_column": timestamp_column,
                        "raw_history_file": str(
                            resolved_raw_history_file
                        ),
                    },
                )

            if chunk.empty:
                continue

            saw_history_rows = True

            chunk_timestamps = pd.to_datetime(
                chunk[timestamp_column],
                errors="coerce",
                utc=True,
            )

            invalid_chunk_timestamps = chunk_timestamps.isna()

            if bool(invalid_chunk_timestamps.any()):
                raise InputValidationError(
                    "The saved full raw history contains invalid timestamps.",
                    error_code=(
                        "INVALID_RECONSTRUCTION_RAW_HISTORY_TIMESTAMPS"
                    ),
                    details={
                        "timestamp_column": timestamp_column,
                        "invalid_rows": int(
                            invalid_chunk_timestamps.sum()
                        ),
                        "raw_history_file": str(
                            resolved_raw_history_file
                        ),
                    },
                )

            chunk_start = pd.Timestamp(
                chunk_timestamps.min()
            )

            if (
                full_history_start is None
                or chunk_start < full_history_start
            ):
                full_history_start = chunk_start

            mask = (
                chunk_timestamps.ge(
                    requested_start
                )
                & chunk_timestamps.lt(
                    cache_start
                )
            )

            if bool(mask.any()):
                matching_chunks.append(
                    chunk.loc[
                        mask
                    ].copy()
                )

    except InputValidationError:
        raise
    except Exception as error:
        raise FileProcessingError(
            "The saved full raw history could not be read for "
            "reconstruction.",
            error_code="RECONSTRUCTION_RAW_HISTORY_READ_FAILED",
            details={
                "raw_history_file": str(
                    resolved_raw_history_file
                ),
                "reason": str(error),
            },
        ) from error

    if (
        not saw_history_rows
        or full_history_start is None
    ):
        raise InputValidationError(
            "The saved full raw history is empty.",
            error_code="EMPTY_RECONSTRUCTION_RAW_HISTORY",
            details={
                "raw_history_file": str(
                    resolved_raw_history_file
                ),
            },
        )

    if full_history_start > requested_start:
        raise InputValidationError(
            "The saved full raw history does not extend far enough back to "
            "provide the requested reconstruction history window.",
            error_code="INSUFFICIENT_FULL_RAW_HISTORY_FOR_RECONSTRUCTION",
            details={
                "requested_start": str(
                    requested_start
                ),
                "available_start": str(
                    full_history_start
                ),
                "cache_start": str(
                    cache_start
                ),
                "cache_days": cache_days,
                "raw_history_file": str(
                    resolved_raw_history_file
                ),
            },
        )

    if not matching_chunks:
        raise InputValidationError(
            "The requested older reconstruction window contains no raw "
            "history rows.",
            error_code="EMPTY_RECONSTRUCTION_RAW_HISTORY_SLICE",
            details={
                "requested_start": str(
                    requested_start
                ),
                "cache_start": str(
                    cache_start
                ),
                "cache_days": cache_days,
                "raw_history_file": str(
                    resolved_raw_history_file
                ),
            },
        )

    older_slice = (
        pd.concat(
            matching_chunks,
            ignore_index=True,
        )
        .reset_index(
            drop=True
        )
    )

    return LoadedReconstructionRawHistory(
        source_file=resolved_raw_history_file,
        requested_start=requested_start,
        cache_start=cache_start,
        dataframe=older_slice,
    )



def load_history_inputs(
    *,
    incoming_file: str | Path,
    raw_cache_file: str | Path,
    processed_cache_by_kpi: Mapping[str, str | Path],
    expected_kpis: tuple[str, ...],
) -> HistoryLoadResult:
    """
    Load the working data required by the inference history subsystem.

    Args:
        incoming_file:
            New raw data file supplied for the current inference call.

        raw_cache_file:
            Z-day raw-history cache path resolved from the saved run contract.

        processed_cache_by_kpi:
            Mapping from every trained KPI name to its Z-day processed-cache
            path.

        expected_kpis:
            Deterministic trained KPI sequence from the saved dataset contract.

    Returns:
        HistoryLoadResult:
            Loaded incoming raw data, raw cache, and one processed cache per
            trained KPI.

    Raises:
        FileProcessingError:
            If a required file is unavailable or cannot be loaded correctly.

        InputValidationError:
            If the internal KPI-to-cache mapping does not match the saved
            trained KPI contract.

    Notes:
        Empty dataframes, columns, timestamps, continuity, overlap, gaps,
        required-history sufficiency, and late-arriving values are deliberately
        not validated here. Those responsibilities belong to the next history
        layer: ``validation.py``.
    """

    _validate_processed_cache_mapping(
        expected_kpis=expected_kpis,
        processed_cache_by_kpi=processed_cache_by_kpi,
    )

    (
        resolved_incoming_file,
        incoming_dataframe,
    ) = _load_required_csv(
        incoming_file,
        role="incoming raw inference data",
    )

    (
        resolved_raw_cache_file,
        raw_cache_dataframe,
    ) = _load_required_csv(
        raw_cache_file,
        role="Z-day raw history cache",
    )

    loaded_processed_caches: list[
        LoadedKPIProcessedCache
    ] = []

    seen_paths: dict[
        Path,
        str,
    ] = {}

    for kpi_name in expected_kpis:
        source_path = processed_cache_by_kpi[
            kpi_name
        ]

        (
            resolved_source_file,
            dataframe,
        ) = _load_required_csv(
            source_path,
            role=(
                "Z-day processed history cache "
                f"for KPI {kpi_name!r}"
            ),
        )

        previous_kpi = seen_paths.get(
            resolved_source_file
        )

        if previous_kpi is not None:
            raise InputValidationError(
                "One processed history cache file cannot be assigned to "
                "multiple trained KPIs.",
                error_code="DUPLICATE_INFERENCE_PROCESSED_CACHE_PATH",
                details={
                    "path": str(resolved_source_file),
                    "first_kpi": previous_kpi,
                    "duplicate_kpi": kpi_name,
                },
            )

        seen_paths[
            resolved_source_file
        ] = kpi_name

        loaded_processed_caches.append(
            LoadedKPIProcessedCache(
                kpi_name=kpi_name,
                source_file=resolved_source_file,
                dataframe=dataframe,
            )
        )

    return HistoryLoadResult(
        incoming_file=resolved_incoming_file,
        incoming_dataframe=incoming_dataframe,
        raw_cache_file=resolved_raw_cache_file,
        raw_cache_dataframe=raw_cache_dataframe,
        processed_caches=tuple(
            loaded_processed_caches
        ),
    )


__all__ = [
    "HistoryLoadResult",
    "LoadedKPIProcessedCache",
    "LoadedReconstructionRawHistory",
    "load_history_inputs",
    "load_reconstruction_raw_history_slice",
]