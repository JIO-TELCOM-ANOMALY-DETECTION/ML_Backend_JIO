"""
Prepare working datasets for inference-time history processing.

Intended project location::

    app/ml/inference/history/preparation.py

This module receives already-loaded and already-validated history state. It
constructs temporary dataframes/files only; it does not decide validity, read
persistent history itself, run Parts A/B/C, or persist permanent history
changes.

Normal path
-----------
required raw Z-cache context + accepted new raw -> temporary Part A input
Part B output -> keep only the newly generated tail
old finalized processed cache + new Part B tail -> temporary Part C input

Reconstruction path
-------------------
If genuine late data is detected inside the maintained cache, the runner first
asks ``history/loader.py`` for one extra ``cache_days`` slice immediately
before the current raw Z-cache. Preparation then builds the temporary raw
working dataset from:

    older reconstruction raw slice
    + current raw Z-cache
    + accepted late raw replacements
    + accepted requested-window forward raw rows

The resulting Part B reconstruction output is handed directly to Part C for
the affected KPI without joining stale processed-cache rows back into that
reconstructed range.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from app.core.exceptions import FileProcessingError, InputValidationError
from app.ml.preprocessing.runner import PreprocessingRunnerResult
from app.schemas.common.dataset import DatasetColumnMapping
from app.schemas.inference.request import SavedHistoryContract
from app.services.file_service import (
    ensure_directory,
    read_csv_file,
    sanitise_filename,
    write_csv_file,
)

from .loader import (
    HistoryLoadResult,
    LoadedReconstructionRawHistory,
)
from .validation import HistoryMode, HistoryRowKey, HistoryValidationResult


RAW_WORKING_FILE_NAME = "history_working_raw.csv"
PART_C_INPUT_DIRECTORY_NAME = "part_c_inputs"
NORMAL_PART_C_SUFFIX = "_history_plus_new_processed.csv"


@dataclass(frozen=True, slots=True)
class RawHistoryPreparationResult:
    """Temporary raw input prepared for the existing Part A -> Part B chain."""

    mode: HistoryMode

    working_raw_file: Path
    working_raw_dataframe: pd.DataFrame

    expected_timeline_start: pd.Timestamp
    expected_timeline_end: pd.Timestamp

    reconstruction_trigger_timestamp: pd.Timestamp | None
    reconstruction_replace_start: pd.Timestamp | None

    accepted_forward_raw_rows: pd.DataFrame
    accepted_late_raw_rows: pd.DataFrame

    retained_raw_history_rows: int
    accepted_forward_rows: int
    accepted_late_rows: int
    ignored_incoming_rows: int


@dataclass(frozen=True, slots=True)
class PreparedKPIForPartC:
    """Part-C handoff plus the Part-B rows later committed to history."""

    kpi_name: str
    mode: HistoryMode

    part_c_preprocessing_result: PreprocessingRunnerResult
    part_c_input_file: Path

    processed_update_dataframe: pd.DataFrame
    processed_update_start: pd.Timestamp
    processed_update_end: pd.Timestamp

    old_processed_rows_retained_for_part_c: int
    regenerated_rows_used_for_part_c: int


@dataclass(frozen=True, slots=True)
class PartCPreparationResult:
    """Prepared Part-C handoffs for every trained KPI."""

    mode: HistoryMode
    kpis: tuple[PreparedKPIForPartC, ...]

    def get_kpi(self, kpi_name: str) -> PreparedKPIForPartC:
        for item in self.kpis:
            if item.kpi_name == kpi_name:
                return item
        raise KeyError(f"No prepared Part C input exists for KPI {kpi_name!r}.")


def _normalize_timestamps(
    values: pd.Series,
    *,
    role: str,
    column_name: str,
) -> pd.Series:
    """
    Parse inference-history timestamps into one UTC-aware representation.

    Persisted caches can contain timezone-qualified values such as
    ``2026-03-31 23:45:00+00:00`` while newly supplied raw inference files can
    legitimately contain timezone-naive values such as
    ``2026-04-01 00:00:00``.

    Pandas' strict single-format inference can coerce one side of such a mixed
    Series to NaT even though every individual timestamp is valid. ``mixed``
    asks pandas to infer each element's valid representation independently,
    after which ``utc=True`` normalises them onto the same UTC timeline.
    """

    parsed = pd.to_datetime(
        values,
        errors="coerce",
        utc=True,
        format="mixed",
    )
    invalid = parsed.isna()

    if bool(invalid.any()):
        raise InputValidationError(
            f"{role} contains invalid timestamps.",
            error_code="INVALID_HISTORY_PREPARATION_TIMESTAMP",
            details={
                "role": role,
                "timestamp_column": column_name,
                "invalid_rows": int(invalid.sum()),
            },
        )

    return parsed


def _canonicalize_timestamp_column(
    dataframe: pd.DataFrame,
    *,
    columns: DatasetColumnMapping,
    role: str,
) -> pd.DataFrame:
    """
    Return a copy whose physical timestamp column is canonical UTC-naive.

    Part E may load a mixture of timezone-qualified persisted timestamps and
    timezone-naive incoming timestamps. ``_normalize_timestamps(...)`` first
    parses both safely into one UTC-aware internal representation.

    Before a prepared dataframe is handed to the existing Parts A/B/C,
    timestamps are converted back to UTC-naive datetimes. Parts A/B/C,
    including Prophet Phase 4, were trained and validated against timezone-
    naive timestamps, so this keeps inference-time preprocessing consistent
    with the training-time contract.
    """

    result = dataframe.copy()

    parsed = _normalize_timestamps(
        result[columns.timestamp_column],
        role=role,
        column_name=columns.timestamp_column,
    )

    result[columns.timestamp_column] = parsed.dt.tz_convert(None)

    return result


def _key_frame(
    dataframe: pd.DataFrame,
    *,
    columns: DatasetColumnMapping,
    single_kpi_name: str | None,
    role: str,
) -> pd.DataFrame:
    required = [
        columns.circle_column,
        columns.cluster_column,
        columns.site_name_column,
        columns.timestamp_column,
    ]
    if single_kpi_name is None:
        required.append(columns.kpi_name_column)

    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise InputValidationError(
            f"{role} is missing columns required for history preparation.",
            error_code="MISSING_HISTORY_PREPARATION_COLUMNS",
            details={"role": role, "missing_columns": missing},
        )

    result = pd.DataFrame(index=dataframe.index)
    if single_kpi_name is None:
        result["kpi"] = (
            dataframe[columns.kpi_name_column].astype("string").str.strip()
        )
    else:
        result["kpi"] = single_kpi_name

    result["circle"] = (
        dataframe[columns.circle_column].astype("string").str.strip()
    )
    result["cluster"] = (
        dataframe[columns.cluster_column].astype("string").str.strip()
    )
    result["site"] = (
        dataframe[columns.site_name_column].astype("string").str.strip()
    )
    result["timestamp"] = _normalize_timestamps(
        dataframe[columns.timestamp_column],
        role=role,
        column_name=columns.timestamp_column,
    )

    identity_columns = ["kpi", "circle", "cluster", "site"]
    invalid_identity = (
        result[identity_columns].isna().any(axis=1)
        | result[identity_columns].eq("").any(axis=1)
    )
    if bool(invalid_identity.any()):
        raise InputValidationError(
            f"{role} contains blank KPI/site identity values.",
            error_code="BLANK_HISTORY_PREPARATION_IDENTITY",
            details={
                "role": role,
                "invalid_rows": int(invalid_identity.sum()),
            },
        )

    return result


def _history_key(row: tuple[object, ...]) -> HistoryRowKey:
    return HistoryRowKey(
        kpi_name=str(row[0]),
        circle=str(row[1]),
        cluster=str(row[2]),
        site_name=str(row[3]),
        timestamp=pd.Timestamp(row[4]),
    )


def _first_occurrences(
    dataframe: pd.DataFrame,
    *,
    columns: DatasetColumnMapping,
    single_kpi_name: str | None,
    role: str,
) -> tuple[pd.DataFrame, dict[HistoryRowKey, object]]:
    """Apply the same keep-first key rule used by validation and Part B."""

    keys = _key_frame(
        dataframe,
        columns=columns,
        single_kpi_name=single_kpi_name,
        role=role,
    )
    duplicate_mask = keys.duplicated(
        subset=["kpi", "circle", "cluster", "site", "timestamp"],
        keep="first",
    )

    retained = dataframe.loc[~duplicate_mask].copy()
    retained_keys = keys.loc[~duplicate_mask]

    mapping: dict[HistoryRowKey, object] = {}
    for source_index, row in zip(
        retained.index,
        retained_keys.itertuples(index=False, name=None),
        strict=True,
    ):
        mapping[_history_key(row)] = source_index

    return retained, mapping


def _align_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = list(left.columns)
    for column in right.columns:
        if column not in ordered:
            ordered.append(column)
    return left.reindex(columns=ordered), right.reindex(columns=ordered)


def _context_start(
    *,
    anchor: pd.Timestamp,
    history_config: SavedHistoryContract,
    bucket_minutes: int,
) -> pd.Timestamp:
    """Honor both exact-point and calendar-span history requirements."""

    bucket = pd.Timedelta(minutes=bucket_minutes)
    point_based = (
        anchor
        - history_config.final_required_history_points * bucket
        - bucket
    )
    day_based = (
        anchor
        - pd.Timedelta(days=history_config.final_required_history_days)
        - bucket
    )
    return min(point_based, day_based)


def _sort_raw(
    dataframe: pd.DataFrame,
    *,
    columns: DatasetColumnMapping,
    single_kpi_name: str | None,
) -> pd.DataFrame:
    keys = _key_frame(
        dataframe,
        columns=columns,
        single_kpi_name=single_kpi_name,
        role="prepared raw working dataset",
    )
    ordering = (
        keys.assign(__source_position=range(len(keys)))
        .sort_values(
            by=[
                "kpi",
                "circle",
                "cluster",
                "site",
                "timestamp",
                "__source_position",
            ],
            kind="mergesort",
        )
        .index
    )

    sorted_dataframe = (
        dataframe.loc[ordering]
        .reset_index(drop=True)
    )

    return _canonicalize_timestamp_column(
        sorted_dataframe,
        columns=columns,
        role="prepared raw working dataset",
    )


def prepare_raw_working_input(
    *,
    loaded: HistoryLoadResult,
    validation: HistoryValidationResult,
    columns: DatasetColumnMapping,
    single_kpi_name: str | None,
    history_config: SavedHistoryContract,
    output_directory: str | Path,
    reconstruction_raw_history: LoadedReconstructionRawHistory | None = None,
) -> RawHistoryPreparationResult:
    """
    Build the temporary raw file consumed later by the unchanged Part A runner.

    Responsibility boundary:

    - ``loader.py`` reads incoming/cache/full-history files;
    - ``validation.py`` classifies normal vs reconstruction and accepted rows;
    - this function only combines/filter/sorts those already-loaded rows into
      the exact temporary raw working dataset;
    - ``runner.py`` later calls Parts A and B using this prepared file.

    Normal mode uses only the current raw Z-cache plus accepted forward rows.

    Reconstruction mode requires the extra older raw slice already loaded by
    ``load_reconstruction_raw_history_slice(...)`` and uses:

        older raw slice
        + current raw Z-cache
        + accepted late rows
        + accepted requested-window forward rows
    """

    output_dir = ensure_directory(output_directory)

    if validation.mode == "reconstruction":
        if reconstruction_raw_history is None:
            raise InputValidationError(
                "Reconstruction requires the older raw-history slice loaded "
                "from full raw history.",
                error_code="MISSING_RECONSTRUCTION_RAW_HISTORY_SLICE",
            )

        older_raw = reconstruction_raw_history.dataframe.copy()
        current_raw_cache = loaded.raw_cache_dataframe.copy()

        older_raw, current_raw_cache = _align_columns(
            older_raw,
            current_raw_cache,
        )

        raw_history_source = pd.concat(
            [older_raw, current_raw_cache],
            ignore_index=True,
            sort=False,
        )

        working_start = pd.Timestamp(
            reconstruction_raw_history.requested_start
        )
        reconstruction_replace_start = pd.Timestamp(
            reconstruction_raw_history.cache_start
        )

    else:
        raw_history_source = loaded.raw_cache_dataframe
        reconstruction_replace_start = None

        working_start = _context_start(
            anchor=validation.previous_history_end,
            history_config=history_config,
            bucket_minutes=validation.bucket_minutes,
        )

    raw_first, raw_lookup = _first_occurrences(
        raw_history_source,
        columns=columns,
        single_kpi_name=single_kpi_name,
        role=(
            "extended reconstruction raw history"
            if validation.mode == "reconstruction"
            else "Z-day raw history cache"
        ),
    )

    incoming_first, incoming_lookup = _first_occurrences(
        loaded.incoming_dataframe,
        columns=columns,
        single_kpi_name=single_kpi_name,
        role="incoming raw inference data",
    )

    forward_keys = set(validation.forward_new_keys)
    late_keys = set(validation.late_arrival_keys)
    accepted_keys = forward_keys | late_keys

    missing_keys = accepted_keys - set(incoming_lookup)

    if missing_keys:
        raise InputValidationError(
            "History preparation cannot locate rows classified by validation.",
            error_code="MISSING_VALIDATED_INCOMING_HISTORY_ROWS",
            details={
                "missing_keys": [
                    (
                        key.kpi_name,
                        key.circle,
                        key.cluster,
                        key.site_name,
                        str(key.timestamp),
                    )
                    for key in sorted(missing_keys)[:20]
                ]
            },
        )

    reconstruction_trigger = (
        validation.earliest_late_timestamp
        if validation.mode == "reconstruction"
        else None
    )

    raw_drop_indices = [
        raw_lookup[key]
        for key in late_keys
        if key in raw_lookup
    ]

    retained_raw = raw_first.drop(
        index=raw_drop_indices,
        errors="ignore",
    )

    retained_raw_timestamps = _normalize_timestamps(
        retained_raw[columns.timestamp_column],
        role="retained raw history",
        column_name=columns.timestamp_column,
    )

    retained_raw = retained_raw.loc[
        retained_raw_timestamps >= working_start
    ].copy()

    forward_rows = incoming_first.loc[
        [
            incoming_lookup[key]
            for key in validation.forward_new_keys
        ]
    ].copy()

    late_rows = incoming_first.loc[
        [
            incoming_lookup[key]
            for key in validation.late_arrival_keys
        ]
    ].copy()

    accepted_incoming = pd.concat(
        [late_rows, forward_rows],
        ignore_index=False,
        sort=False,
    )

    retained_raw, accepted_incoming = _align_columns(
        retained_raw,
        accepted_incoming,
    )

    combined = pd.concat(
        [retained_raw, accepted_incoming],
        ignore_index=True,
        sort=False,
    )

    combined = _sort_raw(
        combined,
        columns=columns,
        single_kpi_name=single_kpi_name,
    )

    if combined.empty:
        raise InputValidationError(
            "History preparation produced an empty raw working dataset.",
            error_code="EMPTY_PREPARED_HISTORY_WORKING_DATASET",
        )

    expected_end = (
        validation.new_period_end_exclusive
        - pd.Timedelta(minutes=validation.bucket_minutes)
    )

    if expected_end <= working_start:
        raise InputValidationError(
            "Prepared inference timeline end must be later than its history "
            "start.",
            error_code="INVALID_PREPARED_HISTORY_TIMELINE",
            details={
                "working_start": str(working_start),
                "expected_end": str(expected_end),
            },
        )

    working_file = Path(output_dir) / RAW_WORKING_FILE_NAME

    write_csv_file(
        combined,
        working_file,
    )

    ignored_rows = len(incoming_first) - len(accepted_keys)

    return RawHistoryPreparationResult(
        mode=validation.mode,
        working_raw_file=working_file,
        working_raw_dataframe=combined,
        expected_timeline_start=working_start,
        expected_timeline_end=expected_end,
        reconstruction_trigger_timestamp=reconstruction_trigger,
        reconstruction_replace_start=reconstruction_replace_start,
        accepted_forward_raw_rows=forward_rows.reset_index(drop=True),
        accepted_late_raw_rows=late_rows.reset_index(drop=True),
        retained_raw_history_rows=len(retained_raw),
        accepted_forward_rows=len(forward_rows),
        accepted_late_rows=len(late_rows),
        ignored_incoming_rows=ignored_rows,
    )


def _processed_keys(
    dataframe: pd.DataFrame,
    *,
    kpi_name: str,
    columns: DatasetColumnMapping,
    role: str,
) -> pd.DataFrame:
    required = [
        columns.circle_column,
        columns.cluster_column,
        columns.site_name_column,
        columns.timestamp_column,
    ]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise InputValidationError(
            f"{role} is missing columns required for Part C preparation.",
            error_code="MISSING_PART_C_PREPARATION_COLUMNS",
            details={"kpi_name": kpi_name, "missing_columns": missing},
        )

    keys = pd.DataFrame(index=dataframe.index)
    keys["circle"] = (
        dataframe[columns.circle_column].astype("string").str.strip()
    )
    keys["cluster"] = (
        dataframe[columns.cluster_column].astype("string").str.strip()
    )
    keys["site"] = (
        dataframe[columns.site_name_column].astype("string").str.strip()
    )
    keys["timestamp"] = _normalize_timestamps(
        dataframe[columns.timestamp_column],
        role=role,
        column_name=columns.timestamp_column,
    )
    return keys


def _sort_processed(
    dataframe: pd.DataFrame,
    *,
    kpi_name: str,
    columns: DatasetColumnMapping,
    role: str,
) -> pd.DataFrame:
    keys = _processed_keys(
        dataframe,
        kpi_name=kpi_name,
        columns=columns,
        role=role,
    )
    duplicate_mask = keys.duplicated(
        subset=["circle", "cluster", "site", "timestamp"],
        keep=False,
    )
    if bool(duplicate_mask.any()):
        raise InputValidationError(
            f"{role} contains duplicate finalized site-timestamp rows.",
            error_code="DUPLICATE_PART_C_PREPARATION_ROWS",
            details={
                "kpi_name": kpi_name,
                "duplicate_rows": int(duplicate_mask.sum()),
            },
        )

    ordering = (
        keys.assign(__source_position=range(len(keys)))
        .sort_values(
            by=["circle", "cluster", "site", "timestamp", "__source_position"],
            kind="mergesort",
        )
        .index
    )

    sorted_dataframe = (
        dataframe.loc[ordering]
        .reset_index(drop=True)
    )

    return _canonicalize_timestamp_column(
        sorted_dataframe,
        columns=columns,
        role=role,
    )


def _filter_processed(
    dataframe: pd.DataFrame,
    *,
    kpi_name: str,
    columns: DatasetColumnMapping,
    timestamp: pd.Timestamp,
    inclusive: bool,
    role: str,
) -> pd.DataFrame:
    keys = _processed_keys(
        dataframe,
        kpi_name=kpi_name,
        columns=columns,
        role=role,
    )
    mask = (
        keys["timestamp"].ge(timestamp)
        if inclusive
        else keys["timestamp"].gt(timestamp)
    )
    return dataframe.loc[mask].copy()


def _results_by_kpi(
    preprocessing_results: Iterable[PreprocessingRunnerResult],
) -> dict[str, PreprocessingRunnerResult]:
    result: dict[str, PreprocessingRunnerResult] = {}
    for item in preprocessing_results:
        kpi_name = str(item.kpi_name).strip()
        if not kpi_name:
            raise InputValidationError(
                "A Part B inference result has no usable KPI name.",
                error_code="MISSING_PART_B_INFERENCE_KPI_NAME",
            )
        if kpi_name in result:
            raise InputValidationError(
                "Part B returned more than one result for the same KPI.",
                error_code="DUPLICATE_PART_B_INFERENCE_KPI_RESULT",
                details={"kpi_name": kpi_name},
            )
        result[kpi_name] = item
    return result


def prepare_part_c_inputs(
    *,
    loaded: HistoryLoadResult,
    validation: HistoryValidationResult,
    preprocessing_results: Iterable[PreprocessingRunnerResult],
    columns: DatasetColumnMapping,
    expected_kpis: tuple[str, ...],
    output_directory: str | Path,
) -> PartCPreparationResult:
    """Prepare the exact handoff consumed by the unchanged Part C runner."""

    by_kpi = _results_by_kpi(preprocessing_results)
    if set(by_kpi) != set(expected_kpis):
        raise InputValidationError(
            "Part B inference results must contain exactly the trained KPIs.",
            error_code="PART_B_INFERENCE_KPI_SET_MISMATCH",
            details={
                "expected_kpis": sorted(expected_kpis),
                "received_kpis": sorted(by_kpi),
            },
        )

    part_c_directory = ensure_directory(
        Path(output_directory) / PART_C_INPUT_DIRECTORY_NAME
    )
    prepared: list[PreparedKPIForPartC] = []

    for kpi_name in expected_kpis:
        preprocessing_result = by_kpi[kpi_name]
        source_file = Path(
            preprocessing_result.final_dataset_file
        ).expanduser().resolve()

        if not source_file.is_file():
            raise FileProcessingError(
                "The inference-time Part B final dataset does not exist.",
                error_code="MISSING_INFERENCE_PART_B_FINAL_DATASET",
                details={"kpi_name": kpi_name, "path": str(source_file)},
            )

        regenerated = read_csv_file(source_file)
        regenerated = _sort_processed(
            regenerated,
            kpi_name=kpi_name,
            columns=columns,
            role="inference-time regenerated Part B dataset",
        )

        kpi_late_timestamps = [
            key.timestamp
            for key in validation.late_arrival_keys
            if key.kpi_name == kpi_name
        ]

        kpi_reconstruction = bool(kpi_late_timestamps)

        if not kpi_reconstruction:
            old_cache = loaded.get_processed_cache(kpi_name).dataframe.copy()
            old_cache = _sort_processed(
                old_cache,
                kpi_name=kpi_name,
                columns=columns,
                role="finalized processed Z-day cache",
            )

            regenerated_tail = _filter_processed(
                regenerated,
                kpi_name=kpi_name,
                columns=columns,
                timestamp=validation.previous_history_end,
                inclusive=False,
                role="inference-time regenerated Part B dataset",
            )

            if regenerated_tail.empty:
                raise InputValidationError(
                    "Part B produced no rows after the finalized history end.",
                    error_code="EMPTY_INFERENCE_PART_B_NEW_TAIL",
                    details={
                        "kpi_name": kpi_name,
                        "previous_history_end": str(validation.previous_history_end),
                    },
                )

            combined = pd.concat(
                [old_cache, regenerated_tail],
                ignore_index=True,
                sort=False,
            )
            combined = _sort_processed(
                combined,
                kpi_name=kpi_name,
                columns=columns,
                role="normal combined Part C inference input",
            )

            safe_name = sanitise_filename(kpi_name) or "kpi"
            part_c_input_file = (
                Path(part_c_directory) / f"{safe_name}{NORMAL_PART_C_SUFFIX}"
            )
            write_csv_file(combined, part_c_input_file)

            part_c_handoff = replace(
                preprocessing_result,
                final_dataset_file=part_c_input_file,
                final_rows=len(combined),
            )

            update_frame = regenerated_tail.reset_index(drop=True)
            update_keys = _processed_keys(
                update_frame,
                kpi_name=kpi_name,
                columns=columns,
                role="normal processed history update",
            )

            prepared.append(
                PreparedKPIForPartC(
                    kpi_name=kpi_name,
                    mode="normal",
                    part_c_preprocessing_result=part_c_handoff,
                    part_c_input_file=part_c_input_file,
                    processed_update_dataframe=update_frame,
                    processed_update_start=pd.Timestamp(
                        update_keys["timestamp"].min()
                    ),
                    processed_update_end=pd.Timestamp(
                        update_keys["timestamp"].max()
                    ),
                    old_processed_rows_retained_for_part_c=len(old_cache),
                    regenerated_rows_used_for_part_c=len(regenerated_tail),
                )
            )
            continue

        old_cache = loaded.get_processed_cache(kpi_name).dataframe.copy()
        old_cache = _sort_processed(
            old_cache,
            kpi_name=kpi_name,
            columns=columns,
            role="finalized processed Z-day cache",
        )

        old_cache_keys = _processed_keys(
            old_cache,
            kpi_name=kpi_name,
            columns=columns,
            role="finalized processed Z-day cache",
        )
        reconstruction_update_start = pd.Timestamp(
            old_cache_keys["timestamp"].min()
        )

        update_frame = _filter_processed(
            regenerated,
            kpi_name=kpi_name,
            columns=columns,
            timestamp=reconstruction_update_start,
            inclusive=True,
            role="reconstructed Part B dataset",
        )

        if update_frame.empty:
            raise InputValidationError(
                "The reconstructed Part B dataset contains no rows from the "
                "current processed-cache boundary onward.",
                error_code="EMPTY_RECONSTRUCTED_PROCESSED_UPDATE",
                details={
                    "kpi_name": kpi_name,
                    "processed_cache_start": str(reconstruction_update_start),
                    "earliest_late_timestamp": str(min(kpi_late_timestamps)),
                },
            )

        update_keys = _processed_keys(
            update_frame,
            kpi_name=kpi_name,
            columns=columns,
            role="reconstructed processed history update",
        )

        prepared.append(
            PreparedKPIForPartC(
                kpi_name=kpi_name,
                mode="reconstruction",
                part_c_preprocessing_result=preprocessing_result,
                part_c_input_file=source_file,
                processed_update_dataframe=update_frame.reset_index(drop=True),
                processed_update_start=pd.Timestamp(
                    update_keys["timestamp"].min()
                ),
                processed_update_end=pd.Timestamp(
                    update_keys["timestamp"].max()
                ),
                old_processed_rows_retained_for_part_c=0,
                regenerated_rows_used_for_part_c=len(regenerated),
            )
        )

    return PartCPreparationResult(
        mode=validation.mode,
        kpis=tuple(prepared),
    )


__all__ = [
    "PartCPreparationResult",
    "PreparedKPIForPartC",
    "RawHistoryPreparationResult",
    "prepare_part_c_inputs",
    "prepare_raw_working_input",
]
