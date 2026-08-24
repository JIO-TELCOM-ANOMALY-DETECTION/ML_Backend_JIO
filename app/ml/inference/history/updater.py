"""
Commit successful inference history updates to persistent storage.

Intended project location::

    app/ml/inference/history/updater.py

This module is the only history-subsystem layer that mutates persistent raw,
processed, and Z-day cache files.

It is deliberately called only after the parent inference runner has completed
all required work successfully, including Part C and saved-model inference.
Until then, ``loader.py``, ``validation.py``, and ``preparation.py`` operate on
loaded or temporary data only.

Commit rules
------------
Raw history remains the source of truth for genuine source observations:

- normal forward observations are appended;
- a genuine late observation may replace only a previously unavailable raw
  key that validation already approved; and
- already accepted genuine historical values are never silently overwritten.

Processed history represents finalized Part-B output:

- normal mode appends the newly generated Part-B tail; and
- reconstruction mode replaces the complete regenerated Part-B working range
  that was handed directly to Part C.

After full histories are prepared in memory, both raw and processed Z-day
caches are rebuilt from the newest finalized window. All target files are
written to sibling staging files first and then replaced as one rollback-safe
commit operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Mapping
from uuid import uuid4

import pandas as pd

from app.core.exceptions import FileProcessingError, InputValidationError
from app.schemas.common.dataset import DatasetColumnMapping
from app.schemas.inference.request import SavedHistoryContract
from app.services.file_service import ensure_directory, read_csv_file, write_csv_file

from .preparation import PartCPreparationResult, RawHistoryPreparationResult
from .validation import HistoryMode


@dataclass(frozen=True, slots=True)
class KPIHistoryCommitResult:
    """Persistent processed-history result for one trained KPI."""

    kpi_name: str
    mode: HistoryMode

    processed_history_file: Path
    processed_cache_file: Path

    processed_history_rows: int
    processed_cache_rows: int

    update_start: pd.Timestamp
    update_end: pd.Timestamp


@dataclass(frozen=True, slots=True)
class HistoryCommitResult:
    """Summary of one completed persistent history commit."""

    raw_history_file: Path
    raw_cache_file: Path

    raw_history_rows: int
    raw_cache_rows: int

    finalized_end: pd.Timestamp
    cache_cutoff: pd.Timestamp

    kpis: tuple[KPIHistoryCommitResult, ...]

    def get_kpi(self, kpi_name: str) -> KPIHistoryCommitResult:
        for item in self.kpis:
            if item.kpi_name == kpi_name:
                return item
        raise KeyError(f"No history commit result exists for KPI {kpi_name!r}.")


def _required_file(path: str | Path, *, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileProcessingError(
            f"Required persistent history file does not exist for {role}.",
            error_code="MISSING_PERSISTENT_INFERENCE_HISTORY_FILE",
            details={"role": role, "path": str(resolved)},
        )
    return resolved


def _normalise_timestamps(
    values: pd.Series,
    *,
    role: str,
    column_name: str,
) -> pd.Series:
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
            error_code="INVALID_HISTORY_UPDATE_TIMESTAMP",
            details={
                "role": role,
                "timestamp_column": column_name,
                "invalid_rows": int(invalid.sum()),
            },
        )
    return parsed


def _raw_key_frame(
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
            f"{role} is missing columns required for raw-history update.",
            error_code="MISSING_RAW_HISTORY_UPDATE_COLUMNS",
            details={"role": role, "missing_columns": missing},
        )

    keys = pd.DataFrame(index=dataframe.index)
    if single_kpi_name is None:
        keys["kpi"] = (
            dataframe[columns.kpi_name_column].astype("string").str.strip()
        )
    else:
        keys["kpi"] = single_kpi_name

    keys["circle"] = (
        dataframe[columns.circle_column].astype("string").str.strip()
    )
    keys["cluster"] = (
        dataframe[columns.cluster_column].astype("string").str.strip()
    )
    keys["site"] = (
        dataframe[columns.site_name_column].astype("string").str.strip()
    )
    keys["timestamp"] = _normalise_timestamps(
        dataframe[columns.timestamp_column],
        role=role,
        column_name=columns.timestamp_column,
    )

    identity_columns = ["kpi", "circle", "cluster", "site"]
    invalid_identity = (
        keys[identity_columns].isna().any(axis=1)
        | keys[identity_columns].eq("").any(axis=1)
    )
    if bool(invalid_identity.any()):
        raise InputValidationError(
            f"{role} contains blank KPI/site identity values.",
            error_code="BLANK_RAW_HISTORY_UPDATE_IDENTITY",
            details={"role": role, "invalid_rows": int(invalid_identity.sum())},
        )

    return keys


def _processed_key_frame(
    dataframe: pd.DataFrame,
    *,
    columns: DatasetColumnMapping,
    kpi_name: str,
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
            f"{role} is missing columns required for processed-history update.",
            error_code="MISSING_PROCESSED_HISTORY_UPDATE_COLUMNS",
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
    keys["timestamp"] = _normalise_timestamps(
        dataframe[columns.timestamp_column],
        role=role,
        column_name=columns.timestamp_column,
    )
    return keys


def _align_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = list(left.columns)
    for column in right.columns:
        if column not in ordered:
            ordered.append(column)
    return left.reindex(columns=ordered), right.reindex(columns=ordered)


def _sort_raw(
    dataframe: pd.DataFrame,
    *,
    columns: DatasetColumnMapping,
    single_kpi_name: str | None,
    role: str,
) -> pd.DataFrame:
    keys = _raw_key_frame(
        dataframe,
        columns=columns,
        single_kpi_name=single_kpi_name,
        role=role,
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
    return dataframe.loc[ordering].reset_index(drop=True)


def _sort_processed(
    dataframe: pd.DataFrame,
    *,
    columns: DatasetColumnMapping,
    kpi_name: str,
    role: str,
) -> pd.DataFrame:
    keys = _processed_key_frame(
        dataframe,
        columns=columns,
        kpi_name=kpi_name,
        role=role,
    )

    duplicate_mask = keys.duplicated(
        subset=["circle", "cluster", "site", "timestamp"],
        keep=False,
    )
    if bool(duplicate_mask.any()):
        raise InputValidationError(
            f"{role} contains duplicate site-timestamp rows.",
            error_code="DUPLICATE_PROCESSED_HISTORY_UPDATE_ROWS",
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
    return dataframe.loc[ordering].reset_index(drop=True)


def _raw_key_tuples(keys: pd.DataFrame) -> pd.Series:
    return pd.Series(
        list(
            zip(
                keys["kpi"],
                keys["circle"],
                keys["cluster"],
                keys["site"],
                keys["timestamp"],
                strict=True,
            )
        ),
        index=keys.index,
        dtype="object",
    )


def _processed_key_tuples(keys: pd.DataFrame) -> pd.Series:
    return pd.Series(
        list(
            zip(
                keys["circle"],
                keys["cluster"],
                keys["site"],
                keys["timestamp"],
                strict=True,
            )
        ),
        index=keys.index,
        dtype="object",
    )


def _canonicalise_raw_history(
    dataframe: pd.DataFrame,
    *,
    columns: DatasetColumnMapping,
    single_kpi_name: str | None,
) -> pd.DataFrame:
    """Apply the established keep-first policy to persistent raw history."""

    keys = _raw_key_frame(
        dataframe,
        columns=columns,
        single_kpi_name=single_kpi_name,
        role="full raw history",
    )
    duplicate_mask = keys.duplicated(
        subset=["kpi", "circle", "cluster", "site", "timestamp"],
        keep="first",
    )
    canonical = dataframe.loc[~duplicate_mask].copy()
    return _sort_raw(
        canonical,
        columns=columns,
        single_kpi_name=single_kpi_name,
        role="canonical full raw history",
    )


def _merge_raw_history(
    *,
    existing: pd.DataFrame,
    raw_preparation: RawHistoryPreparationResult,
    columns: DatasetColumnMapping,
    single_kpi_name: str | None,
) -> pd.DataFrame:
    existing = _canonicalise_raw_history(
        existing,
        columns=columns,
        single_kpi_name=single_kpi_name,
    )

    late = raw_preparation.accepted_late_raw_rows.copy()
    forward = raw_preparation.accepted_forward_raw_rows.copy()

    existing_keys = _raw_key_frame(
        existing,
        columns=columns,
        single_kpi_name=single_kpi_name,
        role="full raw history",
    )
    existing_tuples = _raw_key_tuples(existing_keys)

    if not late.empty:
        late_keys = _raw_key_frame(
            late,
            columns=columns,
            single_kpi_name=single_kpi_name,
            role="accepted late raw observations",
        )
        late_tuples = set(_raw_key_tuples(late_keys).tolist())
        existing = existing.loc[~existing_tuples.isin(late_tuples)].copy()

    if not forward.empty:
        current_keys = _raw_key_frame(
            existing,
            columns=columns,
            single_kpi_name=single_kpi_name,
            role="raw history before forward append",
        )
        current_tuples = set(_raw_key_tuples(current_keys).tolist())

        forward_keys = _raw_key_frame(
            forward,
            columns=columns,
            single_kpi_name=single_kpi_name,
            role="accepted forward raw observations",
        )
        forward_tuples = set(_raw_key_tuples(forward_keys).tolist())

        overlap = current_tuples & forward_tuples
        if overlap:
            raise InputValidationError(
                "A supposedly forward raw observation already exists in full "
                "raw history. Persistent history and the validated Z-cache "
                "are inconsistent.",
                error_code="FORWARD_RAW_HISTORY_ALREADY_EXISTS",
                details={"overlap_count": len(overlap)},
            )

    existing, late = _align_columns(existing, late)
    existing, forward = _align_columns(existing, forward)

    raw_parts = [
        frame
        for frame in (existing, late, forward)
        if not frame.empty
    ]
    updated = pd.concat(
        raw_parts,
        ignore_index=True,
        sort=False,
    )
    return _sort_raw(
        updated,
        columns=columns,
        single_kpi_name=single_kpi_name,
        role="updated full raw history",
    )


def _merge_processed_history(
    *,
    existing: pd.DataFrame,
    update: pd.DataFrame,
    mode: HistoryMode,
    kpi_name: str,
    columns: DatasetColumnMapping,
    update_start: pd.Timestamp,
    update_end: pd.Timestamp,
) -> pd.DataFrame:
    existing = _sort_processed(
        existing,
        columns=columns,
        kpi_name=kpi_name,
        role="full processed history",
    )
    update = _sort_processed(
        update,
        columns=columns,
        kpi_name=kpi_name,
        role="prepared processed history update",
    )

    existing_keys = _processed_key_frame(
        existing,
        columns=columns,
        kpi_name=kpi_name,
        role="full processed history",
    )
    update_keys = _processed_key_frame(
        update,
        columns=columns,
        kpi_name=kpi_name,
        role="prepared processed history update",
    )

    if mode == "normal":
        existing_tuples = set(_processed_key_tuples(existing_keys).tolist())
        update_tuples = set(_processed_key_tuples(update_keys).tolist())
        overlap = existing_tuples & update_tuples
        if overlap:
            raise InputValidationError(
                "Normal processed-history append overlaps already finalized "
                "processed rows.",
                error_code="NORMAL_PROCESSED_HISTORY_OVERLAP",
                details={"kpi_name": kpi_name, "overlap_count": len(overlap)},
            )
        retained = existing

    elif mode == "reconstruction":
        update_series = set(
            zip(
                update_keys["circle"],
                update_keys["cluster"],
                update_keys["site"],
                strict=True,
            )
        )
        existing_series = pd.Series(
            list(
                zip(
                    existing_keys["circle"],
                    existing_keys["cluster"],
                    existing_keys["site"],
                    strict=True,
                )
            ),
            index=existing_keys.index,
            dtype="object",
        )

        replace_mask = (
            existing_series.isin(update_series)
            & existing_keys["timestamp"].ge(update_start)
            & existing_keys["timestamp"].le(update_end)
        )
        retained = existing.loc[~replace_mask].copy()

    else:
        raise InputValidationError(
            "Unknown processed history update mode.",
            error_code="UNKNOWN_PROCESSED_HISTORY_UPDATE_MODE",
            details={"kpi_name": kpi_name, "mode": mode},
        )

    retained, update = _align_columns(retained, update)
    merged = pd.concat([retained, update], ignore_index=True, sort=False)
    return _sort_processed(
        merged,
        columns=columns,
        kpi_name=kpi_name,
        role="updated full processed history",
    )


def _cache_from_finalized_window(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str,
    cutoff: pd.Timestamp,
    finalized_end: pd.Timestamp,
    role: str,
) -> pd.DataFrame:
    timestamps = _normalise_timestamps(
        dataframe[timestamp_column],
        role=role,
        column_name=timestamp_column,
    )
    cache = dataframe.loc[
        timestamps.ge(cutoff) & timestamps.le(finalized_end)
    ].copy()
    if cache.empty:
        raise InputValidationError(
            f"{role} cache would be empty after applying the Z-day window.",
            error_code="EMPTY_UPDATED_INFERENCE_HISTORY_CACHE",
            details={
                "role": role,
                "cutoff": str(cutoff),
                "finalized_end": str(finalized_end),
            },
        )
    return cache


def _validate_path_mappings(
    *,
    expected_kpis: tuple[str, ...],
    processed_history_by_kpi: Mapping[str, str | Path],
    processed_cache_by_kpi: Mapping[str, str | Path],
    raw_history_file: str | Path,
    raw_cache_file: str | Path,
) -> None:
    expected = set(expected_kpis)
    if not expected or len(expected) != len(expected_kpis):
        raise InputValidationError(
            "expected_kpis must be non-empty and contain no duplicates.",
            error_code="INVALID_HISTORY_UPDATE_KPI_SET",
        )

    if set(processed_history_by_kpi) != expected:
        raise InputValidationError(
            "Processed full-history paths must exactly match trained KPIs.",
            error_code="PROCESSED_HISTORY_UPDATE_KPI_MISMATCH",
        )
    if set(processed_cache_by_kpi) != expected:
        raise InputValidationError(
            "Processed cache paths must exactly match trained KPIs.",
            error_code="PROCESSED_CACHE_UPDATE_KPI_MISMATCH",
        )

    all_paths = [
        Path(raw_history_file).expanduser().resolve(),
        Path(raw_cache_file).expanduser().resolve(),
        *(
            Path(processed_history_by_kpi[kpi]).expanduser().resolve()
            for kpi in expected_kpis
        ),
        *(
            Path(processed_cache_by_kpi[kpi]).expanduser().resolve()
            for kpi in expected_kpis
        ),
    ]
    if len(all_paths) != len(set(all_paths)):
        raise InputValidationError(
            "Persistent raw/processed history and cache targets must be unique.",
            error_code="DUPLICATE_HISTORY_UPDATE_TARGET_PATH",
        )


def _stage_and_commit(
    payloads: Mapping[Path, pd.DataFrame],
) -> None:
    """
    Stage every CSV first, then replace targets with rollback protection.

    This is not a database transaction, but no target is touched until all
    candidate files have been serialized successfully. If a replacement fails,
    original files are restored from sibling backups.
    """

    transaction_id = uuid4().hex
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []

    try:
        for target, dataframe in payloads.items():
            ensure_directory(target.parent)
            stage = target.with_name(
                f".{target.stem}.pending-{transaction_id}{target.suffix}"
            )
            write_csv_file(dataframe, stage)
            staged[target] = stage

        for target in payloads:
            backup = target.with_name(
                f".{target.name}.backup-{transaction_id}"
            )
            shutil.copy2(target, backup)
            backups[target] = backup

        for target in payloads:
            os.replace(staged[target], target)
            replaced.append(target)

    except Exception:
        for target in reversed(replaced):
            backup = backups.get(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)

        for target, backup in backups.items():
            if target not in replaced and backup.exists():
                backup.unlink(missing_ok=True)

        for stage in staged.values():
            stage.unlink(missing_ok=True)
        raise

    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
        for stage in staged.values():
            stage.unlink(missing_ok=True)


def commit_history_updates(
    *,
    raw_preparation: RawHistoryPreparationResult,
    part_c_preparation: PartCPreparationResult,
    columns: DatasetColumnMapping,
    single_kpi_name: str | None,
    history_config: SavedHistoryContract,
    bucket_minutes: int,
    expected_kpis: tuple[str, ...],
    raw_history_file: str | Path,
    raw_cache_file: str | Path,
    processed_history_by_kpi: Mapping[str, str | Path],
    processed_cache_by_kpi: Mapping[str, str | Path],
) -> HistoryCommitResult:
    """
    Persist one successful inference run's raw/processed history state.

    Call this only after the parent inference runner has successfully completed
    Part C and model inference. All updated histories/caches are first built in
    memory, then staged, and only then committed to their final paths.
    """

    if bucket_minutes <= 0:
        raise InputValidationError(
            "bucket_minutes must be positive for history update.",
            error_code="INVALID_HISTORY_UPDATE_BUCKET_MINUTES",
        )

    if set(item.kpi_name for item in part_c_preparation.kpis) != set(expected_kpis):
        raise InputValidationError(
            "Prepared Part C history updates do not match the trained KPI set.",
            error_code="PART_C_HISTORY_UPDATE_KPI_MISMATCH",
        )

    _validate_path_mappings(
        expected_kpis=expected_kpis,
        processed_history_by_kpi=processed_history_by_kpi,
        processed_cache_by_kpi=processed_cache_by_kpi,
        raw_history_file=raw_history_file,
        raw_cache_file=raw_cache_file,
    )

    resolved_raw_history = _required_file(
        raw_history_file,
        role="full raw history",
    )
    resolved_raw_cache = _required_file(
        raw_cache_file,
        role="Z-day raw history cache",
    )

    resolved_processed_history: dict[str, Path] = {}
    resolved_processed_cache: dict[str, Path] = {}
    for kpi_name in expected_kpis:
        resolved_processed_history[kpi_name] = _required_file(
            processed_history_by_kpi[kpi_name],
            role=f"full processed history for KPI {kpi_name!r}",
        )
        resolved_processed_cache[kpi_name] = _required_file(
            processed_cache_by_kpi[kpi_name],
            role=f"processed Z-day cache for KPI {kpi_name!r}",
        )

    full_raw = read_csv_file(resolved_raw_history)
    updated_raw = _merge_raw_history(
        existing=full_raw,
        raw_preparation=raw_preparation,
        columns=columns,
        single_kpi_name=single_kpi_name,
    )

    updated_processed: dict[str, pd.DataFrame] = {}
    kpi_items = {
        item.kpi_name: item
        for item in part_c_preparation.kpis
    }

    finalized_ends: list[pd.Timestamp] = []
    for kpi_name in expected_kpis:
        item = kpi_items[kpi_name]
        full_processed = read_csv_file(resolved_processed_history[kpi_name])
        updated = _merge_processed_history(
            existing=full_processed,
            update=item.processed_update_dataframe,
            mode=item.mode,
            kpi_name=kpi_name,
            columns=columns,
            update_start=item.processed_update_start,
            update_end=item.processed_update_end,
        )
        updated_processed[kpi_name] = updated
        finalized_ends.append(item.processed_update_end)

    if len(set(finalized_ends)) != 1:
        raise InputValidationError(
            "All KPI processed updates must finish at the same finalized "
            "timestamp before history can be committed.",
            error_code="MISMATCHED_HISTORY_UPDATE_END_TIMESTAMPS",
            details={
                kpi_name: str(kpi_items[kpi_name].processed_update_end)
                for kpi_name in expected_kpis
            },
        )

    finalized_end = finalized_ends[0]
    cache_cutoff = (
        finalized_end
        - pd.Timedelta(days=history_config.cache_days)
        + pd.Timedelta(minutes=bucket_minutes)
    )

    raw_cache = _cache_from_finalized_window(
        updated_raw,
        timestamp_column=columns.timestamp_column,
        cutoff=cache_cutoff,
        finalized_end=finalized_end,
        role="raw history",
    )
    raw_cache = _sort_raw(
        raw_cache,
        columns=columns,
        single_kpi_name=single_kpi_name,
        role="updated raw Z-day cache",
    )

    processed_caches: dict[str, pd.DataFrame] = {}
    for kpi_name in expected_kpis:
        cache = _cache_from_finalized_window(
            updated_processed[kpi_name],
            timestamp_column=columns.timestamp_column,
            cutoff=cache_cutoff,
            finalized_end=finalized_end,
            role=f"processed history for KPI {kpi_name!r}",
        )
        processed_caches[kpi_name] = _sort_processed(
            cache,
            columns=columns,
            kpi_name=kpi_name,
            role=f"updated processed Z-day cache for KPI {kpi_name!r}",
        )

    payloads: dict[Path, pd.DataFrame] = {
        resolved_raw_history: updated_raw,
        resolved_raw_cache: raw_cache,
    }
    for kpi_name in expected_kpis:
        payloads[resolved_processed_history[kpi_name]] = updated_processed[kpi_name]
        payloads[resolved_processed_cache[kpi_name]] = processed_caches[kpi_name]

    _stage_and_commit(payloads)

    commit_items = tuple(
        KPIHistoryCommitResult(
            kpi_name=kpi_name,
            mode=kpi_items[kpi_name].mode,
            processed_history_file=resolved_processed_history[kpi_name],
            processed_cache_file=resolved_processed_cache[kpi_name],
            processed_history_rows=len(updated_processed[kpi_name]),
            processed_cache_rows=len(processed_caches[kpi_name]),
            update_start=kpi_items[kpi_name].processed_update_start,
            update_end=kpi_items[kpi_name].processed_update_end,
        )
        for kpi_name in expected_kpis
    )

    return HistoryCommitResult(
        raw_history_file=resolved_raw_history,
        raw_cache_file=resolved_raw_cache,
        raw_history_rows=len(updated_raw),
        raw_cache_rows=len(raw_cache),
        finalized_end=finalized_end,
        cache_cutoff=cache_cutoff,
        kpis=commit_items,
    )


__all__ = [
    "HistoryCommitResult",
    "KPIHistoryCommitResult",
    "commit_history_updates",
]