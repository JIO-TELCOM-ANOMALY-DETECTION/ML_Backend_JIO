"""
Validate inference-time history continuity and reconstruction requirements.

Intended project location::

    app/ml/inference/history/validation.py

This module is the decision layer between ``loader.py`` and
``preparation.py``.

It receives already-loaded working data plus already-resolved settings from
the parent inference/history runner. It does not read files, mutate history,
run Parts A/B/C, or persist updates.

Its responsibilities are to:

1. validate the structural columns needed for history analysis;
2. normalize timestamps only for comparison/validation purposes;
3. define the requested X-day forward inference window and ignore otherwise
   valid forward rows that lie after that window for the current call;
4. verify that the maintained Z-day caches are sufficient for normal
   inference readiness;
5. calculate the old-to-new bridge gap and enforce both configured limits;
6. tolerate duplicate rows using the existing Part B ``keep_first`` policy;
7. identify overlap with already-accepted historical raw values;
8. identify newly arrived genuine values for previously missing/invalid
   historical positions;
9. ignore historical incoming rows that lie before the maintained Z-day
   processed window;
10. classify eligible in-cache late data as ``reconstruction`` without
    requiring all reconstruction context to already exist in the raw Z-cache;
11. return immutable metadata for ``preparation.py``.

Important processing rule
-------------------------
Normal inference later follows::

    historical raw + new raw -> Part A -> Part B
    -> keep only newly processed X-day rows
    -> old finalized processed history + new processed rows
    -> Part C

If genuine late historical data is detected inside the supported processed
cache, the affected range is classified for reconstruction::

    extended reconstruction raw range -> Part A -> Part B
    -> regenerated Part B range directly -> Part C

Validation no longer requires the complete reconstruction lookback to already
exist inside the raw Z-cache. The reconstruction loading/preparation layers are
responsible for extending the temporary raw working range from full raw
history when needed.

Historical incoming rows older than the maintained processed-cache window are
ignored for the current call rather than rejecting forward inference.

The distinction is decided here but the actual dataframe construction belongs
to ``preparation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np
import pandas as pd

from app.core.exceptions import InputValidationError
from app.schemas.common.dataset import DatasetColumnMapping
from app.schemas.inference.request import SavedHistoryContract
from app.schemas.preprocessing import PreprocessingConfig

from .loader import HistoryLoadResult


HistoryMode = Literal[
    "normal",
    "reconstruction",
]


@dataclass(frozen=True, slots=True, order=True)
class HistorySeriesKey:
    """Stable KPI + site identity used by history validation."""

    kpi_name: str
    circle: str
    cluster: str
    site_name: str

    def as_tuple(
        self,
    ) -> tuple[str, str, str, str]:
        return (
            self.kpi_name,
            self.circle,
            self.cluster,
            self.site_name,
        )


@dataclass(frozen=True, slots=True, order=True)
class HistoryRowKey:
    """Stable KPI + site + timestamp key used to classify incoming rows."""

    kpi_name: str
    circle: str
    cluster: str
    site_name: str
    timestamp: pd.Timestamp

    @property
    def series(
        self,
    ) -> HistorySeriesKey:
        return HistorySeriesKey(
            kpi_name=self.kpi_name,
            circle=self.circle,
            cluster=self.cluster,
            site_name=self.site_name,
        )


@dataclass(frozen=True, slots=True)
class HistorySeriesGap:
    """Bridge-gap result for one finalized KPI/site series."""

    series: HistorySeriesKey
    previous_processed_end: pd.Timestamp
    new_period_start: pd.Timestamp

    gap_points: int
    gap_days: float
    gap_ratio: float


@dataclass(frozen=True, slots=True)
class HistorySeriesReadiness:
    """Per-series cache sufficiency measured before the new inference period."""

    series: HistorySeriesKey

    raw_cache_min_timestamp: pd.Timestamp
    raw_cache_max_timestamp: pd.Timestamp

    processed_cache_min_timestamp: pd.Timestamp
    processed_cache_max_timestamp: pd.Timestamp

    available_raw_history_points: int
    required_raw_history_points: int

    available_raw_history_days: float
    required_raw_history_days: int

    available_processed_history_points: int
    required_processed_history_points: int


@dataclass(frozen=True, slots=True)
class HistoryValidationResult:
    """
    Immutable decision/result returned to history preparation.

    ``late_arrival_keys`` are genuine newly arrived values for historical
    timestamps that were previously unavailable in raw history but are inside
    the finalized processed-cache range.

    ``existing_real_overlap_keys`` are historical duplicates for which the
    previously accepted raw observation remains authoritative.

    ``ignored_historical_keys`` are overlapping incoming rows that do not
    provide a new usable real value and therefore must not alter history.
    """

    mode: HistoryMode

    bucket_minutes: int
    points_per_day: int
    inference_days: int

    previous_history_end: pd.Timestamp
    new_period_start: pd.Timestamp
    new_period_end_exclusive: pd.Timestamp

    gap_points: int
    gap_days: float
    gap_ratio: float

    series_gaps: tuple[HistorySeriesGap, ...]

    incoming_duplicate_rows: int
    raw_cache_duplicate_rows: int

    forward_new_keys: tuple[HistoryRowKey, ...]
    existing_real_overlap_keys: tuple[HistoryRowKey, ...]
    late_arrival_keys: tuple[HistoryRowKey, ...]
    ignored_historical_keys: tuple[HistoryRowKey, ...]

    earliest_late_timestamp: pd.Timestamp | None

    series_readiness: tuple[HistorySeriesReadiness, ...]

    @property
    def late_data_detected(
        self,
    ) -> bool:
        return bool(
            self.late_arrival_keys
        )


_INTERNAL_KPI_COLUMN = "__history_kpi__"
_INTERNAL_TIMESTAMP_COLUMN = "__history_timestamp__"
_INTERNAL_VALUE_COLUMN = "__history_numeric_value__"
_INTERNAL_ACCEPTED_VALUE_COLUMN = "__history_accepted_value__"

_INTERNAL_KEY_COLUMNS = (
    _INTERNAL_KPI_COLUMN,
    "__history_circle__",
    "__history_cluster__",
    "__history_site__",
    _INTERNAL_TIMESTAMP_COLUMN,
)

_INTERNAL_SERIES_COLUMNS = _INTERNAL_KEY_COLUMNS[:-1]


def _require_nonempty_dataframe(
    dataframe: pd.DataFrame,
    *,
    role: str,
) -> None:
    if dataframe.empty:
        raise InputValidationError(
            f"{role} contains no rows.",
            error_code="EMPTY_INFERENCE_HISTORY_DATASET",
            details={
                "role": role,
            },
        )


def _require_columns(
    dataframe: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    role: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise InputValidationError(
            f"{role} is missing required history columns.",
            error_code="MISSING_INFERENCE_HISTORY_COLUMNS",
            details={
                "role": role,
                "missing_columns": missing,
            },
        )


def _normalise_identity_column(
    series: pd.Series,
    *,
    logical_name: str,
    role: str,
) -> pd.Series:
    if series.isna().any():
        raise InputValidationError(
            f"{role} contains missing {logical_name} values.",
            error_code="MISSING_INFERENCE_HISTORY_IDENTITY",
            details={
                "role": role,
                "identity_field": logical_name,
            },
        )

    normalized = (
        series
        .astype("string")
        .str.strip()
    )

    invalid = (
        normalized.isna()
        | normalized.eq("")
    )

    if bool(
        invalid.any()
    ):
        raise InputValidationError(
            f"{role} contains blank {logical_name} values.",
            error_code="BLANK_INFERENCE_HISTORY_IDENTITY",
            details={
                "role": role,
                "identity_field": logical_name,
                "invalid_rows": int(
                    invalid.sum()
                ),
            },
        )

    return normalized.astype(str)


def _normalise_timestamps(
    series: pd.Series,
    *,
    role: str,
    timestamp_column: str,
) -> pd.Series:
    """
    Parse timestamps into UTC solely for stable comparison.

    The original dataframes remain unchanged. Naive timestamps are interpreted
    consistently by pandas as UTC for comparison purposes; interval lengths are
    preserved, while Part A remains responsible for the actual source timeline
    semantics used during processing.
    """

    parsed = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    invalid = parsed.isna()

    if bool(
        invalid.any()
    ):
        raise InputValidationError(
            f"{role} contains invalid timestamps.",
            error_code="INVALID_INFERENCE_HISTORY_TIMESTAMP",
            details={
                "role": role,
                "timestamp_column": timestamp_column,
                "invalid_rows": int(
                    invalid.sum()
                ),
            },
        )

    return parsed


def _accepted_real_value_mask(
    values: pd.Series,
    *,
    preprocessing_config: PreprocessingConfig,
    role: str,
    value_column: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Identify source values Part B can accept as original valid observations.

    This mirrors the relevant Part B source-value rules:

    - blank/NaN values are missing, not genuine late replacements;
    - non-blank non-numeric values are invalid input and must fail;
    - values inside [lower_bound, upper_bound] are accepted;
    - slightly out-of-range values are accepted only when
      clip_values_within_tolerance is enabled and the value lies within
      boundary_tolerance; and
    - infinities and values outside the accepted range are not usable as
      genuine late replacements.
    """

    as_string = values.astype("string")

    missing_or_blank = (
        values.isna()
        | as_string.str.strip().eq("")
    )

    numeric = pd.to_numeric(
        values.mask(
            missing_or_blank,
            np.nan,
        ),
        errors="coerce",
    )

    unexpected_non_numeric = (
        ~missing_or_blank
        & numeric.isna()
    )

    if bool(
        unexpected_non_numeric.any()
    ):
        raise InputValidationError(
            f"{role} contains non-blank KPI values that are not numeric.",
            error_code="NON_NUMERIC_INFERENCE_HISTORY_VALUES",
            details={
                "role": role,
                "value_column": value_column,
                "invalid_rows": int(
                    unexpected_non_numeric.sum()
                ),
            },
        )

    finite = pd.Series(
        np.isfinite(
            numeric.to_numpy(
                dtype=float,
                na_value=np.nan,
            )
        ),
        index=values.index,
    )

    valid_values = preprocessing_config.valid_values

    if valid_values.clip_values_within_tolerance:
        lower_allowed = (
            valid_values.lower_bound
            - valid_values.boundary_tolerance
        )
        upper_allowed = (
            valid_values.upper_bound
            + valid_values.boundary_tolerance
        )
    else:
        lower_allowed = (
            valid_values.lower_bound
        )
        upper_allowed = (
            valid_values.upper_bound
        )

    accepted = (
        ~missing_or_blank
        & finite
        & numeric.ge(
            lower_allowed
        )
        & numeric.le(
            upper_allowed
        )
    )

    return numeric, accepted


def _prepare_raw_analysis_frame(
    dataframe: pd.DataFrame,
    *,
    columns: DatasetColumnMapping,
    expected_kpis: tuple[str, ...],
    single_kpi_name: str | None,
    preprocessing_config: PreprocessingConfig,
    role: str,
) -> pd.DataFrame:
    """Build a normalized comparison-only view of one raw dataset."""

    _require_nonempty_dataframe(
        dataframe,
        role=role,
    )

    required = (
        columns.circle_column,
        columns.cluster_column,
        columns.site_name_column,
        columns.timestamp_column,
        columns.value_column,
    )

    if single_kpi_name is None:
        required = (
            *required,
            columns.kpi_name_column,
        )

    _require_columns(
        dataframe,
        columns=required,
        role=role,
    )

    normalized = pd.DataFrame(
        index=dataframe.index,
    )

    normalized["__history_circle__"] = (
        _normalise_identity_column(
            dataframe[
                columns.circle_column
            ],
            logical_name="circle",
            role=role,
        )
    )

    normalized["__history_cluster__"] = (
        _normalise_identity_column(
            dataframe[
                columns.cluster_column
            ],
            logical_name="cluster",
            role=role,
        )
    )

    normalized["__history_site__"] = (
        _normalise_identity_column(
            dataframe[
                columns.site_name_column
            ],
            logical_name="site name",
            role=role,
        )
    )

    if single_kpi_name is not None:
        normalized[
            _INTERNAL_KPI_COLUMN
        ] = single_kpi_name
    else:
        normalized[
            _INTERNAL_KPI_COLUMN
        ] = _normalise_identity_column(
            dataframe[
                columns.kpi_name_column
            ],
            logical_name="KPI name",
            role=role,
        )

    observed_kpis = set(
        normalized[
            _INTERNAL_KPI_COLUMN
        ].unique()
    )
    expected_kpi_set = set(
        expected_kpis
    )

    unknown_kpis = sorted(
        observed_kpis
        - expected_kpi_set
    )

    if unknown_kpis:
        raise InputValidationError(
            f"{role} contains KPI names that were not part of the saved "
            "training run.",
            error_code="UNKNOWN_INFERENCE_HISTORY_KPI",
            details={
                "role": role,
                "unknown_kpis": unknown_kpis,
                "expected_kpis": sorted(
                    expected_kpi_set
                ),
            },
        )

    normalized[
        _INTERNAL_TIMESTAMP_COLUMN
    ] = _normalise_timestamps(
        dataframe[
            columns.timestamp_column
        ],
        role=role,
        timestamp_column=columns.timestamp_column,
    )

    (
        normalized[
            _INTERNAL_VALUE_COLUMN
        ],
        normalized[
            _INTERNAL_ACCEPTED_VALUE_COLUMN
        ],
    ) = _accepted_real_value_mask(
        dataframe[
            columns.value_column
        ],
        preprocessing_config=preprocessing_config,
        role=role,
        value_column=columns.value_column,
    )

    return normalized


def _prepare_processed_analysis_frame(
    dataframe: pd.DataFrame,
    *,
    kpi_name: str,
    columns: DatasetColumnMapping,
) -> pd.DataFrame:
    """Build the comparison view for one finalized processed KPI cache."""

    role = (
        "processed Z-day cache "
        f"for KPI {kpi_name!r}"
    )

    _require_nonempty_dataframe(
        dataframe,
        role=role,
    )

    required = (
        columns.circle_column,
        columns.cluster_column,
        columns.site_name_column,
        columns.timestamp_column,
    )

    _require_columns(
        dataframe,
        columns=required,
        role=role,
    )

    normalized = pd.DataFrame(
        index=dataframe.index,
    )

    normalized[
        _INTERNAL_KPI_COLUMN
    ] = kpi_name

    normalized["__history_circle__"] = (
        _normalise_identity_column(
            dataframe[
                columns.circle_column
            ],
            logical_name="circle",
            role=role,
        )
    )

    normalized["__history_cluster__"] = (
        _normalise_identity_column(
            dataframe[
                columns.cluster_column
            ],
            logical_name="cluster",
            role=role,
        )
    )

    normalized["__history_site__"] = (
        _normalise_identity_column(
            dataframe[
                columns.site_name_column
            ],
            logical_name="site name",
            role=role,
        )
    )

    normalized[
        _INTERNAL_TIMESTAMP_COLUMN
    ] = _normalise_timestamps(
        dataframe[
            columns.timestamp_column
        ],
        role=role,
        timestamp_column=columns.timestamp_column,
    )

    duplicate_mask = normalized.duplicated(
        subset=list(
            _INTERNAL_KEY_COLUMNS
        ),
        keep=False,
    )

    if bool(
        duplicate_mask.any()
    ):
        raise InputValidationError(
            "A finalized processed history cache contains duplicate "
            "site-timestamp rows.",
            error_code="DUPLICATE_PROCESSED_HISTORY_CACHE_ROWS",
            details={
                "kpi_name": kpi_name,
                "duplicate_rows": int(
                    duplicate_mask.sum()
                ),
            },
        )

    return normalized


def _row_key_from_values(
    values,
) -> HistoryRowKey:
    return HistoryRowKey(
        kpi_name=str(
            values[0]
        ),
        circle=str(
            values[1]
        ),
        cluster=str(
            values[2]
        ),
        site_name=str(
            values[3]
        ),
        timestamp=pd.Timestamp(
            values[4]
        ),
    )


def _series_key_from_values(
    values,
) -> HistorySeriesKey:
    return HistorySeriesKey(
        kpi_name=str(
            values[0]
        ),
        circle=str(
            values[1]
        ),
        cluster=str(
            values[2]
        ),
        site_name=str(
            values[3]
        ),
    )


def _build_row_lookup(
    dataframe: pd.DataFrame,
) -> dict[HistoryRowKey, pd.Series]:
    lookup: dict[
        HistoryRowKey,
        pd.Series,
    ] = {}

    for row in dataframe.itertuples(
        index=False,
        name=None,
    ):
        key = _row_key_from_values(
            row[:5]
        )
        lookup[key] = pd.Series(
            {
                _INTERNAL_VALUE_COLUMN: row[5],
                _INTERNAL_ACCEPTED_VALUE_COLUMN: row[6],
            }
        )

    return lookup


def _build_processed_key_sets(
    processed_by_kpi: dict[str, pd.DataFrame],
) -> tuple[
    dict[str, set[HistoryRowKey]],
    dict[HistorySeriesKey, tuple[pd.Timestamp, pd.Timestamp]],
]:
    keys_by_kpi: dict[
        str,
        set[HistoryRowKey],
    ] = {}

    spans: dict[
        HistorySeriesKey,
        tuple[pd.Timestamp, pd.Timestamp],
    ] = {}

    for kpi_name, dataframe in processed_by_kpi.items():
        keys: set[
            HistoryRowKey
        ] = set()

        grouped = dataframe.groupby(
            list(
                _INTERNAL_SERIES_COLUMNS
            ),
            sort=False,
            dropna=False,
        )

        for raw_series, group in grouped:
            series = _series_key_from_values(
                raw_series
            )

            timestamps = group[
                _INTERNAL_TIMESTAMP_COLUMN
            ]

            spans[
                series
            ] = (
                pd.Timestamp(
                    timestamps.min()
                ),
                pd.Timestamp(
                    timestamps.max()
                ),
            )

            for row in group.itertuples(
                index=False,
                name=None,
            ):
                keys.add(
                    _row_key_from_values(
                        row[:5]
                    )
                )

        keys_by_kpi[
            kpi_name
        ] = keys

    return (
        keys_by_kpi,
        spans,
    )


def _series_spans(
    dataframe: pd.DataFrame,
) -> dict[
    HistorySeriesKey,
    tuple[pd.Timestamp, pd.Timestamp],
]:
    spans: dict[
        HistorySeriesKey,
        tuple[pd.Timestamp, pd.Timestamp],
    ] = {}

    grouped = dataframe.groupby(
        list(
            _INTERNAL_SERIES_COLUMNS
        ),
        sort=False,
        dropna=False,
    )

    for raw_series, group in grouped:
        series = _series_key_from_values(
            raw_series
        )

        timestamps = group[
            _INTERNAL_TIMESTAMP_COLUMN
        ]

        spans[
            series
        ] = (
            pd.Timestamp(
                timestamps.min()
            ),
            pd.Timestamp(
                timestamps.max()
            ),
        )

    return spans



def _validate_processed_cache_timelines(
    *,
    processed_by_kpi: dict[str, pd.DataFrame],
    bucket_minutes: int,
) -> pd.Timestamp:
    """
    Require finalized processed caches to be continuous and bucket-aligned.

    The normal inference path appends newly regenerated Part B rows to the
    already-finalized processed cache before Part C. That operation is only
    unambiguous when every saved KPI/site series has a continuous bucket
    timeline ending at the same finalized timestamp.
    """

    bucket = pd.Timedelta(minutes=bucket_minutes)
    series_end_times: dict[HistorySeriesKey, pd.Timestamp] = {}

    for dataframe in processed_by_kpi.values():
        grouped = dataframe.groupby(
            list(_INTERNAL_SERIES_COLUMNS),
            sort=False,
            dropna=False,
        )

        for raw_series, group in grouped:
            series = _series_key_from_values(raw_series)
            timestamps = (
                group[_INTERNAL_TIMESTAMP_COLUMN]
                .sort_values(kind="mergesort")
                .reset_index(drop=True)
            )

            if timestamps.empty:
                raise InputValidationError(
                    "A processed history series contains no timestamps.",
                    error_code="EMPTY_PROCESSED_HISTORY_SERIES",
                    details={"series": series.as_tuple()},
                )

            if len(timestamps) > 1:
                deltas = timestamps.diff().iloc[1:]
                bad_delta_mask = deltas.ne(bucket)

                if bool(bad_delta_mask.any()):
                    first_bad_index = int(bad_delta_mask[bad_delta_mask].index[0])
                    raise InputValidationError(
                        "A finalized processed history series is not continuous "
                        "at the saved bucket frequency.",
                        error_code="NON_CONTIGUOUS_PROCESSED_HISTORY_CACHE",
                        details={
                            "series": series.as_tuple(),
                            "bucket_minutes": bucket_minutes,
                            "previous_timestamp": str(
                                timestamps.iloc[first_bad_index - 1]
                            ),
                            "current_timestamp": str(
                                timestamps.iloc[first_bad_index]
                            ),
                        },
                    )

            series_end_times[series] = pd.Timestamp(timestamps.iloc[-1])

    if not series_end_times:
        raise InputValidationError(
            "No finalized processed history series are available.",
            error_code="EMPTY_PROCESSED_HISTORY_SERIES_SET",
        )

    unique_ends = set(series_end_times.values())

    if len(unique_ends) != 1:
        raise InputValidationError(
            "All finalized processed KPI/site series must end at the same "
            "timestamp before a new inference run.",
            error_code="MISMATCHED_PROCESSED_HISTORY_END_TIMESTAMPS",
            details={
                "series_end_times": {
                    "|".join(series.as_tuple()): str(end)
                    for series, end in series_end_times.items()
                },
            },
        )

    return next(iter(unique_ends))


# Reconstruction context is intentionally not validated against the raw
# Z-cache boundary here. If eligible late data exists inside the maintained
# processed-cache window, validation classifies it as reconstruction. The
# reconstruction loader/preparation path is responsible for extending the
# temporary raw working range from full raw history when required.


def _expected_intervals_before(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bucket_minutes: int,
) -> int:
    if end <= start:
        return 0

    bucket = pd.Timedelta(
        minutes=bucket_minutes
    )
    delta = end - start

    return int(
        delta // bucket
    )


def _validate_series_readiness(
    *,
    raw_spans: dict[
        HistorySeriesKey,
        tuple[pd.Timestamp, pd.Timestamp],
    ],
    processed_spans: dict[
        HistorySeriesKey,
        tuple[pd.Timestamp, pd.Timestamp],
    ],
    new_period_start: pd.Timestamp,
    bucket_minutes: int,
    points_per_day: int,
    history_config: SavedHistoryContract,
) -> tuple[HistorySeriesReadiness, ...]:
    """
    Require every finalized processed series to have enough Z-cache context.

    Raw cache sufficiency is needed to rerun Parts A/B.
    Processed cache sufficiency is needed by the normal Part C path.
    """

    raw_series = set(
        raw_spans
    )
    processed_series = set(
        processed_spans
    )

    missing_raw_series = sorted(
        processed_series
        - raw_series
    )

    if missing_raw_series:
        raise InputValidationError(
            "One or more finalized processed series have no raw Z-cache "
            "history.",
            error_code="MISSING_RAW_CACHE_SERIES",
            details={
                "missing_series": [
                    item.as_tuple()
                    for item in missing_raw_series
                ],
            },
        )

    required_processed_points = max(
        history_config.part_c_required_history_points,
        history_config.part_c_maximum_lookback_points + 1,
    )

    readiness: list[
        HistorySeriesReadiness
    ] = []

    for series in sorted(
        processed_series
    ):
        raw_min, raw_max = raw_spans[
            series
        ]
        processed_min, processed_max = (
            processed_spans[
                series
            ]
        )

        available_raw_points = (
            _expected_intervals_before(
                start=raw_min,
                end=new_period_start,
                bucket_minutes=bucket_minutes,
            )
        )

        available_raw_days = float(
            (
                new_period_start
                - raw_min
            )
            / pd.Timedelta(
                days=1
            )
        )

        if (
            available_raw_points
            < history_config.final_required_history_points
        ):
            raise InputValidationError(
                "The Z-day raw cache does not contain enough historical "
                "timeline for one trained series.",
                error_code="INSUFFICIENT_RAW_CACHE_HISTORY_POINTS",
                details={
                    "series": series.as_tuple(),
                    "available_points": available_raw_points,
                    "required_points": (
                        history_config
                        .final_required_history_points
                    ),
                    "raw_cache_start": str(
                        raw_min
                    ),
                    "new_period_start": str(
                        new_period_start
                    ),
                },
            )

        if (
            available_raw_days
            < history_config.final_required_history_days
        ):
            raise InputValidationError(
                "The Z-day raw cache does not contain the required historical "
                "time span for one trained series.",
                error_code="INSUFFICIENT_RAW_CACHE_HISTORY_DAYS",
                details={
                    "series": series.as_tuple(),
                    "available_days": (
                        available_raw_days
                    ),
                    "required_days": (
                        history_config
                        .final_required_history_days
                    ),
                },
            )

        available_processed_points = (
            _expected_intervals_before(
                start=processed_min,
                end=new_period_start,
                bucket_minutes=bucket_minutes,
            )
        )

        if (
            available_processed_points
            < required_processed_points
        ):
            raise InputValidationError(
                "The Z-day processed cache does not contain enough Part C "
                "historical context for one trained series.",
                error_code="INSUFFICIENT_PROCESSED_CACHE_HISTORY",
                details={
                    "series": series.as_tuple(),
                    "available_points": (
                        available_processed_points
                    ),
                    "required_points": (
                        required_processed_points
                    ),
                    "processed_cache_start": str(
                        processed_min
                    ),
                    "new_period_start": str(
                        new_period_start
                    ),
                },
            )

        readiness.append(
            HistorySeriesReadiness(
                series=series,
                raw_cache_min_timestamp=raw_min,
                raw_cache_max_timestamp=raw_max,
                processed_cache_min_timestamp=processed_min,
                processed_cache_max_timestamp=processed_max,
                available_raw_history_points=available_raw_points,
                required_raw_history_points=(
                    history_config
                    .final_required_history_points
                ),
                available_raw_history_days=available_raw_days,
                required_raw_history_days=(
                    history_config
                    .final_required_history_days
                ),
                available_processed_history_points=(
                    available_processed_points
                ),
                required_processed_history_points=(
                    required_processed_points
                ),
            )
        )

    return tuple(
        readiness
    )


def validate_history_inputs(
    *,
    loaded: HistoryLoadResult,
    columns: DatasetColumnMapping,
    expected_kpis: tuple[str, ...],
    single_kpi_name: str | None,
    preprocessing_config: PreprocessingConfig,
    history_config: SavedHistoryContract,
    bucket_minutes: int,
    points_per_day: int,
    inference_days: int,
) -> HistoryValidationResult:
    """
    Validate one loaded history/incoming-data combination.

    The function returns only decisions and immutable metadata. It never edits
    the loaded dataframes.

    The caller is expected to pass values already validated by the saved
    inference schema. Defensive checks remain here because this module may be
    exercised independently in tests.
    """

    if inference_days < 1:
        raise InputValidationError(
            "inference_days must be at least one.",
            error_code="INVALID_HISTORY_INFERENCE_DAYS",
            details={
                "inference_days": inference_days,
            },
        )

    if bucket_minutes <= 0:
        raise InputValidationError(
            "bucket_minutes must be positive.",
            error_code="INVALID_HISTORY_BUCKET_MINUTES",
        )

    if points_per_day <= 0:
        raise InputValidationError(
            "points_per_day must be positive.",
            error_code="INVALID_HISTORY_POINTS_PER_DAY",
        )

    if (
        bucket_minutes
        * points_per_day
        != 1_440
    ):
        raise InputValidationError(
            "bucket_minutes and points_per_day are inconsistent.",
            error_code="HISTORY_BUCKET_POINTS_MISMATCH",
            details={
                "bucket_minutes": bucket_minutes,
                "points_per_day": points_per_day,
            },
        )

    incoming = _prepare_raw_analysis_frame(
        loaded.incoming_dataframe,
        columns=columns,
        expected_kpis=expected_kpis,
        single_kpi_name=single_kpi_name,
        preprocessing_config=preprocessing_config,
        role="incoming raw inference data",
    )

    raw_cache = _prepare_raw_analysis_frame(
        loaded.raw_cache_dataframe,
        columns=columns,
        expected_kpis=expected_kpis,
        single_kpi_name=single_kpi_name,
        preprocessing_config=preprocessing_config,
        role="Z-day raw history cache",
    )

    observed_raw_kpis = set(
        raw_cache[
            _INTERNAL_KPI_COLUMN
        ].unique()
    )

    if observed_raw_kpis != set(
        expected_kpis
    ):
        raise InputValidationError(
            "The raw Z-day cache must contain every trained KPI.",
            error_code="RAW_CACHE_KPI_SET_MISMATCH",
            details={
                "expected_kpis": sorted(
                    expected_kpis
                ),
                "observed_kpis": sorted(
                    observed_raw_kpis
                ),
            },
        )

    incoming_duplicate_mask = (
        incoming.duplicated(
            subset=list(
                _INTERNAL_KEY_COLUMNS
            ),
            keep="first",
        )
    )

    raw_duplicate_mask = (
        raw_cache.duplicated(
            subset=list(
                _INTERNAL_KEY_COLUMNS
            ),
            keep="first",
        )
    )

    incoming_duplicate_rows = int(
        incoming_duplicate_mask.sum()
    )
    raw_cache_duplicate_rows = int(
        raw_duplicate_mask.sum()
    )

    # All history decisions follow the existing Part B duplicate policy:
    # retain the first physical row for a site/timestamp key.
    incoming_first = incoming.loc[
        ~incoming_duplicate_mask
    ].copy()

    raw_first = raw_cache.loc[
        ~raw_duplicate_mask
    ].copy()

    processed_by_kpi: dict[
        str,
        pd.DataFrame,
    ] = {}

    for cache in loaded.processed_caches:
        processed_by_kpi[
            cache.kpi_name
        ] = _prepare_processed_analysis_frame(
            cache.dataframe,
            kpi_name=cache.kpi_name,
            columns=columns,
        )

    if set(
        processed_by_kpi
    ) != set(
        expected_kpis
    ):
        raise InputValidationError(
            "Loaded processed caches do not match the trained KPI set.",
            error_code="PROCESSED_CACHE_KPI_SET_MISMATCH",
        )

    common_processed_end = (
        _validate_processed_cache_timelines(
            processed_by_kpi=processed_by_kpi,
            bucket_minutes=bucket_minutes,
        )
    )

    (
        processed_keys_by_kpi,
        processed_spans,
    ) = _build_processed_key_sets(
        processed_by_kpi
    )

    raw_spans = _series_spans(
        raw_first
    )

    raw_after_finalized = raw_first.loc[
        raw_first[
            _INTERNAL_TIMESTAMP_COLUMN
        ]
        > common_processed_end
    ]

    if not raw_after_finalized.empty:
        raise InputValidationError(
            "The raw Z-day cache contains timestamps newer than the "
            "finalized processed history. Persistent history appears "
            "partially committed.",
            error_code="RAW_CACHE_AHEAD_OF_PROCESSED_HISTORY",
            details={
                "common_processed_end": str(
                    common_processed_end
                ),
                "rows_ahead": int(
                    len(
                        raw_after_finalized
                    )
                ),
            },
        )

    processed_series = set(
        processed_spans
    )

    incoming_series = {
        _series_key_from_values(
            values
        )
        for values in incoming_first[
            list(
                _INTERNAL_SERIES_COLUMNS
            )
        ].itertuples(
            index=False,
            name=None,
        )
    }

    unknown_incoming_series = sorted(
        incoming_series
        - processed_series
    )

    if unknown_incoming_series:
        raise InputValidationError(
            "Incoming data contains a site/KPI series that is not present in "
            "the saved processed-history cache.",
            error_code="UNSEEN_INFERENCE_HISTORY_SERIES",
            details={
                "series": [
                    item.as_tuple()
                    for item in unknown_incoming_series
                ],
            },
        )

    raw_lookup = _build_row_lookup(
        raw_first[
            [
                *_INTERNAL_KEY_COLUMNS,
                _INTERNAL_VALUE_COLUMN,
                _INTERNAL_ACCEPTED_VALUE_COLUMN,
            ]
        ]
    )

    forward_new_keys: list[
        HistoryRowKey
    ] = []
    existing_real_overlap_keys: list[
        HistoryRowKey
    ] = []
    late_arrival_keys: list[
        HistoryRowKey
    ] = []
    ignored_historical_keys: list[
        HistoryRowKey
    ] = []

    # Classify each incoming first-occurrence row relative to that series'
    # finalized processed-history boundary. The processed cache, rather than
    # the last observed raw point, defines what has already been finalized.
    for row in incoming_first[
        [
            *_INTERNAL_KEY_COLUMNS,
            _INTERNAL_VALUE_COLUMN,
            _INTERNAL_ACCEPTED_VALUE_COLUMN,
        ]
    ].itertuples(
        index=False,
        name=None,
    ):
        key = _row_key_from_values(
            row[:5]
        )

        incoming_accepted = bool(
            row[6]
        )

        processed_span = processed_spans.get(
            key.series
        )

        if processed_span is None:
            raise InputValidationError(
                "Incoming series is missing from processed-history metadata.",
                error_code="MISSING_PROCESSED_HISTORY_SERIES_METADATA",
                details={
                    "series": key.series.as_tuple(),
                },
            )

        processed_start, processed_end = (
            processed_span
        )

        if key.timestamp > processed_end:
            forward_new_keys.append(
                key
            )
            continue

        if key.timestamp < processed_start:
            # Historical data older than the maintained processed-cache window
            # is outside the supported reconstruction range for this call.
            # Ignore it and continue with any valid in-cache late data and
            # requested forward inference rows.
            ignored_historical_keys.append(
                key
            )
            continue

        if key not in processed_keys_by_kpi[
            key.kpi_name
        ]:
            raise InputValidationError(
                "An overlapping historical timestamp is not represented in "
                "the finalized processed cache.",
                error_code="UNKNOWN_HISTORICAL_OVERLAP_TIMESTAMP",
                details={
                    "series": key.series.as_tuple(),
                    "timestamp": str(
                        key.timestamp
                    ),
                },
            )

        existing_raw = raw_lookup.get(
            key
        )

        if existing_raw is not None:
            existing_accepted = bool(
                existing_raw[
                    _INTERNAL_ACCEPTED_VALUE_COLUMN
                ]
            )

            if existing_accepted:
                # Once a genuine source value has been accepted, later
                # overlapping uploads cannot silently replace it.
                existing_real_overlap_keys.append(
                    key
                )
                continue

            if incoming_accepted:
                if not history_config.late_data_reconstruction_enabled:
                    raise InputValidationError(
                        "Genuine late data was detected but reconstruction is "
                        "disabled by the saved history contract.",
                        error_code="LATE_DATA_RECONSTRUCTION_DISABLED",
                        details={
                            "series": key.series.as_tuple(),
                            "timestamp": str(
                                key.timestamp
                            ),
                        },
                    )

                late_arrival_keys.append(
                    key
                )
                continue

            ignored_historical_keys.append(
                key
            )
            continue

        # The timestamp exists in finalized processed history but had no
        # original raw row. A newly supplied accepted value is genuine late
        # data for a previously absent/missing timestamp.
        if incoming_accepted:
            if not history_config.late_data_reconstruction_enabled:
                raise InputValidationError(
                    "Genuine late data was detected but reconstruction is "
                    "disabled by the saved history contract.",
                    error_code="LATE_DATA_RECONSTRUCTION_DISABLED",
                    details={
                        "series": key.series.as_tuple(),
                        "timestamp": str(
                            key.timestamp
                        ),
                    },
                )

            late_arrival_keys.append(
                key
            )
        else:
            ignored_historical_keys.append(
                key
            )

    if not forward_new_keys:
        raise InputValidationError(
            "Incoming data contains no forward timestamps beyond the "
            "finalized processed history. A normal inference call must "
            "include new data.",
            error_code="NO_FORWARD_INFERENCE_DATA",
        )

    new_period_start = min(
        key.timestamp
        for key in forward_new_keys
    )

    new_period_end_exclusive = (
        new_period_start
        + pd.Timedelta(
            days=inference_days
        )
    )

    # The incoming file may contain more future data than this inference
    # request asks to process. Only the requested X-day forward window is
    # accepted for this call. Later valid forward rows remain untouched: they
    # are not sent to Parts A/B/C, are not scored, and are not committed to
    # persistent raw/processed history or caches.
    forward_new_keys = [
        key
        for key in forward_new_keys
        if key.timestamp
        < new_period_end_exclusive
    ]

    if not forward_new_keys:
        raise InputValidationError(
            "Incoming data contains no forward rows inside the requested "
            "inference window.",
            error_code="NO_FORWARD_DATA_IN_REQUESTED_INFERENCE_WINDOW",
            details={
                "new_period_start": str(
                    new_period_start
                ),
                "new_period_end_exclusive": str(
                    new_period_end_exclusive
                ),
                "inference_days": inference_days,
            },
        )

    bucket = pd.Timedelta(
        minutes=bucket_minutes
    )

    expected_incoming_points = (
        inference_days
        * points_per_day
    )

    # Validate the bridge from every finalized series to the common new
    # inference-period start. This catches a lagging site/KPI even when other
    # series happen to be continuous.
    series_gaps: list[
        HistorySeriesGap
    ] = []

    for series in sorted(
        processed_series
    ):
        processed_end = (
            processed_spans[
                series
            ][1]
        )

        if new_period_start <= processed_end:
            gap_points_for_series = 0
        else:
            bridge_delta = (
                new_period_start
                - processed_end
            )

            if (
                bridge_delta
                % bucket
                != pd.Timedelta(0)
            ):
                raise InputValidationError(
                    "The new inference period is not aligned to the saved "
                    "bucket relative to one finalized series.",
                    error_code="INFERENCE_HISTORY_BUCKET_MISALIGNMENT",
                    details={
                        "series": series.as_tuple(),
                        "previous_processed_end": str(
                            processed_end
                        ),
                        "new_period_start": str(
                            new_period_start
                        ),
                        "bucket_minutes": bucket_minutes,
                    },
                )

            bridge_steps = int(
                bridge_delta
                // bucket
            )

            gap_points_for_series = max(
                0,
                bridge_steps - 1,
            )

        gap_days_for_series = (
            gap_points_for_series
            * bucket_minutes
            / 1_440.0
        )

        denominator = (
            gap_points_for_series
            + expected_incoming_points
        )

        gap_ratio_for_series = (
            float(
                gap_points_for_series
                / denominator
            )
            if denominator > 0
            else 0.0
        )

        if (
            gap_days_for_series
            > history_config.max_gap_days
        ):
            raise InputValidationError(
                "The gap between one finalized series and the new inference "
                "period exceeds max_gap_days.",
                error_code="INFERENCE_HISTORY_MAX_GAP_DAYS_EXCEEDED",
                details={
                    "series": series.as_tuple(),
                    "gap_points": (
                        gap_points_for_series
                    ),
                    "gap_days": (
                        gap_days_for_series
                    ),
                    "max_gap_days": (
                        history_config.max_gap_days
                    ),
                },
            )

        if (
            gap_ratio_for_series
            > history_config.max_gap_ratio
        ):
            raise InputValidationError(
                "The gap between one finalized series and the new inference "
                "period exceeds max_gap_ratio.",
                error_code="INFERENCE_HISTORY_MAX_GAP_RATIO_EXCEEDED",
                details={
                    "series": series.as_tuple(),
                    "gap_points": (
                        gap_points_for_series
                    ),
                    "gap_ratio": (
                        gap_ratio_for_series
                    ),
                    "max_gap_ratio": (
                        history_config.max_gap_ratio
                    ),
                },
            )

        series_gaps.append(
            HistorySeriesGap(
                series=series,
                previous_processed_end=(
                    processed_end
                ),
                new_period_start=(
                    new_period_start
                ),
                gap_points=(
                    gap_points_for_series
                ),
                gap_days=(
                    gap_days_for_series
                ),
                gap_ratio=(
                    gap_ratio_for_series
                ),
            )
        )

    gap_points = max(
        (
            item.gap_points
            for item in series_gaps
        ),
        default=0,
    )
    gap_days = max(
        (
            item.gap_days
            for item in series_gaps
        ),
        default=0.0,
    )
    gap_ratio = max(
        (
            item.gap_ratio
            for item in series_gaps
        ),
        default=0.0,
    )

    previous_history_end = (
        common_processed_end
    )

    # Eligible late data inside the maintained processed-cache window is
    # allowed to trigger reconstruction even when the required earlier raw
    # context starts before the current raw Z-cache. The later reconstruction
    # loading/preparation stage will supply that older context from full raw
    # history.

    series_readiness = _validate_series_readiness(
        raw_spans=raw_spans,
        processed_spans=processed_spans,
        new_period_start=new_period_start,
        bucket_minutes=bucket_minutes,
        points_per_day=points_per_day,
        history_config=history_config,
    )

    earliest_late_timestamp = (
        min(
            key.timestamp
            for key in late_arrival_keys
        )
        if late_arrival_keys
        else None
    )

    mode: HistoryMode = (
        "reconstruction"
        if late_arrival_keys
        else "normal"
    )

    return HistoryValidationResult(
        mode=mode,
        bucket_minutes=bucket_minutes,
        points_per_day=points_per_day,
        inference_days=inference_days,
        previous_history_end=previous_history_end,
        new_period_start=new_period_start,
        new_period_end_exclusive=(
            new_period_end_exclusive
        ),
        gap_points=gap_points,
        gap_days=gap_days,
        gap_ratio=gap_ratio,
        series_gaps=tuple(
            series_gaps
        ),
        incoming_duplicate_rows=(
            incoming_duplicate_rows
        ),
        raw_cache_duplicate_rows=(
            raw_cache_duplicate_rows
        ),
        forward_new_keys=tuple(
            sorted(
                forward_new_keys
            )
        ),
        existing_real_overlap_keys=tuple(
            sorted(
                existing_real_overlap_keys
            )
        ),
        late_arrival_keys=tuple(
            sorted(
                late_arrival_keys
            )
        ),
        ignored_historical_keys=tuple(
            sorted(
                ignored_historical_keys
            )
        ),
        earliest_late_timestamp=(
            earliest_late_timestamp
        ),
        series_readiness=series_readiness,
    )


__all__ = [
    "HistoryMode",
    "HistoryRowKey",
    "HistorySeriesGap",
    "HistorySeriesKey",
    "HistorySeriesReadiness",
    "HistoryValidationResult",
    "validate_history_inputs",
]