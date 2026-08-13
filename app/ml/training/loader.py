"""
Part D chronological split planner.

Intended project location::

    app/ml/training/loader.py

Despite the historical module name ``loader.py``, this module does NOT load
the full engineered feature dataset into memory and does NOT materialize
X_train/y_train/etc.

Its responsibility is to:

1. receive the validated Part C dataset contract;
2. resolve the selected model's split percentages;
3. scan only site-identity + timestamp columns;
4. reserve the first ``minimum_history_points`` for each site series;
5. calculate chronological train/validation/calibration/test boundaries; and
6. return lightweight dataclasses describing those ranges.

Later model-stage modules use these ranges to load only the rows they need.

No physical split CSV files are created.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import TypeAlias

import pandas as pd

from app.schemas.training.common import TrainingModelType
from app.schemas.training.isolation_forest.isolation_forest import (
    IsolationForestTrainingConfig,
)
from app.schemas.training.xgboost.xgboost import (
    XGBoostTrainingConfig,
)

from .validation import TrainingValidationResult


# Only identity/timestamp columns are scanned, so a fairly large chunk is safe.
# This is an execution detail, not a user-facing training setting.
SPLIT_SCAN_CHUNK_SIZE = 250_000


TrainingConfig: TypeAlias = (
    XGBoostTrainingConfig
    | IsolationForestTrainingConfig
)


@dataclass(
    frozen=True,
    slots=True,
)
class SeriesIdentity:
    """Identify one chronological KPI series inside a Part C dataset."""

    circle: str
    cluster: str
    site_name: str

    def as_tuple(
        self,
    ) -> tuple[str, str, str]:
        """Return identity in the same order as validation.site_identity_columns."""

        return (
            self.circle,
            self.cluster,
            self.site_name,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class SplitRange:
    """
    Describe one inclusive chronological range for one site series.

    ``start_position`` and ``end_position`` are zero-based positions within
    that site's chronological series, not global CSV row numbers.

    Timestamps are also stored so later model modules can filter the source
    dataset without depending on global row positions.
    """

    start_position: int
    end_position: int

    start_timestamp: datetime
    end_timestamp: datetime

    row_count: int


@dataclass(
    frozen=True,
    slots=True,
)
class SeriesSplitPlan:
    """Store all split ranges for one site series."""

    identity: SeriesIdentity
    total_points: int

    history: SplitRange
    train: SplitRange

    validation: SplitRange | None
    calibration: SplitRange | None
    test: SplitRange


@dataclass(
    frozen=True,
    slots=True,
)
class TrainingSplitPlan:
    """
    Lightweight split plan returned to the Part D runner.

    The plan contains boundaries only. It contains no feature matrices,
    targets, or copied training data.
    """

    input_file: Path
    model_type: TrainingModelType
    hyperparameter_tuning_enabled: bool

    site_identity_columns: tuple[str, str, str]
    timestamp_column: str

    minimum_history_points: int

    # Immutable ordered pairs, for example:
    # (("train", 60.0), ("validation", 10.0), ...)
    post_history_percentages: tuple[tuple[str, float], ...]

    series: tuple[SeriesSplitPlan, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class _PositionRange:
    """Internal row-position range before timestamps are attached."""

    start_position: int
    end_position: int
    row_count: int


@dataclass(
    frozen=True,
    slots=True,
)
class _UnresolvedSeriesPlan:
    """Internal split plan containing positions but not timestamps."""

    identity: SeriesIdentity
    total_points: int

    history: _PositionRange
    train: _PositionRange

    validation: _PositionRange | None
    calibration: _PositionRange | None
    test: _PositionRange


def _normalise_identity(
    raw_identity: object,
) -> SeriesIdentity:
    """Convert pandas group keys into the stable SeriesIdentity dataclass."""

    if not isinstance(
        raw_identity,
        tuple,
    ):
        raise ValueError(
            "Training loader expected exactly three site identity columns."
        )

    if len(
        raw_identity
    ) != 3:
        raise ValueError(
            "Training loader expected exactly three site identity values."
        )

    return SeriesIdentity(
        circle=str(
            raw_identity[0]
        ),
        cluster=str(
            raw_identity[1]
        ),
        site_name=str(
            raw_identity[2]
        ),
    )


def _resolve_model_percentages(
    *,
    model_type: TrainingModelType,
    training_config: TrainingConfig,
) -> tuple[
    bool,
    tuple[tuple[str, float], ...],
]:
    """
    Resolve the active chronological split sequence for the selected model.

    XGBoost
    --------
    tuning ON:
        train -> validation -> calibration -> test

    tuning OFF:
        train -> calibration -> test

    Isolation Forest
    ----------------
    tuning ON:
        train -> validation -> test

    tuning OFF:
        train -> test
    """

    if model_type == "xgboost":
        if not isinstance(
            training_config,
            XGBoostTrainingConfig,
        ):
            raise TypeError(
                "model_type='xgboost' requires XGBoostTrainingConfig."
            )

        resolved = training_config.resolve_split()

        if resolved.uses_validation_split:
            percentages = (
                (
                    "train",
                    float(
                        resolved.train_percentage
                    ),
                ),
                (
                    "validation",
                    float(
                        resolved.validation_percentage
                    ),
                ),
                (
                    "calibration",
                    float(
                        resolved.calibration_percentage
                    ),
                ),
                (
                    "test",
                    float(
                        resolved.test_percentage
                    ),
                ),
            )

        else:
            percentages = (
                (
                    "train",
                    float(
                        resolved.train_percentage
                    ),
                ),
                (
                    "calibration",
                    float(
                        resolved.calibration_percentage
                    ),
                ),
                (
                    "test",
                    float(
                        resolved.test_percentage
                    ),
                ),
            )

        return (
            bool(
                training_config.hyperparameter_tuning_enabled
            ),
            percentages,
        )

    if model_type == "isolation_forest":
        if not isinstance(
            training_config,
            IsolationForestTrainingConfig,
        ):
            raise TypeError(
                "model_type='isolation_forest' requires "
                "IsolationForestTrainingConfig."
            )

        resolved = training_config.resolve_split()

        if resolved.uses_validation_split:
            percentages = (
                (
                    "train",
                    float(
                        resolved.train_percentage
                    ),
                ),
                (
                    "validation",
                    float(
                        resolved.validation_percentage
                    ),
                ),
                (
                    "test",
                    float(
                        resolved.test_percentage
                    ),
                ),
            )

        else:
            percentages = (
                (
                    "train",
                    float(
                        resolved.train_percentage
                    ),
                ),
                (
                    "test",
                    float(
                        resolved.test_percentage
                    ),
                ),
            )

        return (
            bool(
                training_config.hyperparameter_tuning_enabled
            ),
            percentages,
        )

    raise ValueError(
        f"Unsupported training model_type: {model_type}"
    )


def _allocate_stage_counts(
    *,
    available_points: int,
    percentages: tuple[tuple[str, float], ...],
) -> dict[str, int]:
    """
    Convert percentages into exact integer row counts.

    Cumulative boundaries are used rather than independently rounding every
    stage. Therefore:
    - every available row is assigned exactly once;
    - no rows are duplicated;
    - no rows are lost because of percentage rounding.
    """

    if available_points <= 0:
        raise ValueError(
            "Post-history available_points must be positive."
        )

    if not percentages:
        raise ValueError(
            "At least one post-history split percentage is required."
        )

    stage_counts: dict[str, int] = {}

    previous_boundary = 0
    cumulative_percentage = Decimal("0")

    for index, (
        stage_name,
        percentage,
    ) in enumerate(
        percentages
    ):
        is_last_stage = (
            index
            == len(
                percentages
            )
            - 1
        )

        if is_last_stage:
            boundary = available_points

        else:
            cumulative_percentage += Decimal(
                str(
                    percentage
                )
            )

            boundary = int(
                (
                    Decimal(
                        available_points
                    )
                    * cumulative_percentage
                    / Decimal("100")
                ).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )

        count = (
            boundary
            - previous_boundary
        )

        stage_counts[
            stage_name
        ] = count

        previous_boundary = boundary

    if sum(
        stage_counts.values()
    ) != available_points:
        raise RuntimeError(
            "Internal split allocation error: allocated rows do not equal "
            "available post-history rows."
        )

    return stage_counts


def _minimum_stage_points(
    validation_result: TrainingValidationResult,
) -> dict[str, int]:
    """Map active split names to raw minimum-point requirements."""

    return {
        "train": validation_result.minimum_train_points,
        "validation": validation_result.minimum_validation_points,
        "calibration": validation_result.minimum_calibration_points,
        "test": validation_result.minimum_test_points,
    }


def _validate_raw_stage_counts(
    *,
    identity: SeriesIdentity,
    stage_counts: dict[str, int],
    validation_result: TrainingValidationResult,
) -> None:
    """
    Perform an early raw-row minimum check.

    Later model modules must repeat minimum checks AFTER applying trusted-status,
    complete-feature-row, and target-null filtering. This loader check only
    proves the raw chronological range is large enough in principle.
    """

    minimums = _minimum_stage_points(
        validation_result
    )

    failures: list[str] = []

    for stage_name, row_count in stage_counts.items():
        minimum = minimums[
            stage_name
        ]

        if row_count < minimum:
            failures.append(
                f"{stage_name}: {row_count} < minimum {minimum}"
            )

    if failures:
        raise ValueError(
            "Series does not contain enough raw points for the requested "
            "training split after reserving Part C history. "
            f"Series={identity.as_tuple()}. "
            + "; ".join(
                failures
            )
        )


def _count_series_and_validate_order(
    *,
    validation_result: TrainingValidationResult,
    allowed_series: frozenset[SeriesIdentity] | None,
) -> dict[SeriesIdentity, int]:
    """
    First lightweight CSV pass.

    Count rows per site series and require strictly increasing timestamps for
    every series in physical file order.

    Part C is expected to produce chronologically ordered series. Refusing an
    unsorted file is safer than silently assigning incorrect split boundaries.
    """

    identity_columns = list(
        validation_result.site_identity_columns
    )

    timestamp_column = (
        validation_result.timestamp_column
    )

    use_columns = [
        *identity_columns,
        timestamp_column,
    ]

    counts: dict[
        SeriesIdentity,
        int,
    ] = {}

    last_timestamp_by_series: dict[
        SeriesIdentity,
        pd.Timestamp,
    ] = {}

    reader = pd.read_csv(
        validation_result.input_file,
        usecols=use_columns,
        chunksize=SPLIT_SCAN_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        chunk[
            timestamp_column
        ] = pd.to_datetime(
            chunk[
                timestamp_column
            ],
            errors="raise",
        )

        grouped = chunk.groupby(
            identity_columns,
            sort=False,
            dropna=False,
        )

        for raw_identity, group in grouped:
            identity = _normalise_identity(
                raw_identity
            )

            if (
                allowed_series is not None
                and identity not in allowed_series
            ):
                continue

            timestamps = group[
                timestamp_column
            ].reset_index(
                drop=True
            )

            if timestamps.empty:
                continue

            # Strictly increasing prevents duplicate timestamps as well.
            if len(
                timestamps
            ) > 1:
                differences = timestamps.diff().iloc[
                    1:
                ]

                if bool(
                    (
                        differences
                        <= pd.Timedelta(0)
                    ).any()
                ):
                    raise ValueError(
                        "Part C series timestamps must be strictly increasing "
                        "before Part D split planning. "
                        f"Series={identity.as_tuple()}, "
                        f"chunk={chunk_number}."
                    )

            previous_last = last_timestamp_by_series.get(
                identity
            )

            current_first = timestamps.iloc[
                0
            ]

            if (
                previous_last is not None
                and current_first <= previous_last
            ):
                raise ValueError(
                    "Part C series timestamps are not strictly chronological "
                    "across CSV chunks. "
                    f"Series={identity.as_tuple()}, "
                    f"chunk={chunk_number}."
                )

            counts[
                identity
            ] = (
                counts.get(
                    identity,
                    0,
                )
                + len(
                    group
                )
            )

            last_timestamp_by_series[
                identity
            ] = timestamps.iloc[
                -1
            ]

    if not counts:
        raise ValueError(
            "No site series were found in the validated Part C dataset."
        )

    return counts


def _position_range(
    *,
    start_position: int,
    row_count: int,
) -> _PositionRange:
    """Build one inclusive row-position range."""

    if row_count <= 0:
        raise ValueError(
            "Every active training split must contain at least one row."
        )

    return _PositionRange(
        start_position=start_position,
        end_position=(
            start_position
            + row_count
            - 1
        ),
        row_count=row_count,
    )


def _build_unresolved_series_plan(
    *,
    identity: SeriesIdentity,
    total_points: int,
    validation_result: TrainingValidationResult,
    percentages: tuple[tuple[str, float], ...],
) -> _UnresolvedSeriesPlan:
    """Create exact row-position ranges for one site series."""

    history_points = (
        validation_result.minimum_history_points
    )

    if total_points <= history_points:
        raise ValueError(
            "Series does not contain any post-history rows after reserving "
            "Part C minimum_history_points. "
            f"Series={identity.as_tuple()}, "
            f"total_points={total_points}, "
            f"minimum_history_points={history_points}."
        )

    available_points = (
        total_points
        - history_points
    )

    stage_counts = _allocate_stage_counts(
        available_points=available_points,
        percentages=percentages,
    )

    _validate_raw_stage_counts(
        identity=identity,
        stage_counts=stage_counts,
        validation_result=validation_result,
    )

    history = _position_range(
        start_position=0,
        row_count=history_points,
    )

    current_position = history_points

    resolved_ranges: dict[
        str,
        _PositionRange,
    ] = {}

    for stage_name, _ in percentages:
        stage_range = _position_range(
            start_position=current_position,
            row_count=stage_counts[
                stage_name
            ],
        )

        resolved_ranges[
            stage_name
        ] = stage_range

        current_position = (
            stage_range.end_position
            + 1
        )

    if current_position != total_points:
        raise RuntimeError(
            "Internal split planning error: final row position does not equal "
            "series row count."
        )

    train = resolved_ranges.get(
        "train"
    )

    test = resolved_ranges.get(
        "test"
    )

    if (
        train is None
        or test is None
    ):
        raise RuntimeError(
            "Internal split planning error: train/test split missing."
        )

    return _UnresolvedSeriesPlan(
        identity=identity,
        total_points=total_points,
        history=history,
        train=train,
        validation=resolved_ranges.get(
            "validation"
        ),
        calibration=resolved_ranges.get(
            "calibration"
        ),
        test=test,
    )


def _build_unresolved_plans(
    *,
    counts: dict[SeriesIdentity, int],
    validation_result: TrainingValidationResult,
    percentages: tuple[tuple[str, float], ...],
) -> tuple[_UnresolvedSeriesPlan, ...]:
    """Build position-only plans for every series."""

    return tuple(
        _build_unresolved_series_plan(
            identity=identity,
            total_points=total_points,
            validation_result=validation_result,
            percentages=percentages,
        )
        for identity, total_points in counts.items()
    )


def _required_boundary_positions(
    plan: _UnresolvedSeriesPlan,
) -> dict[
    int,
    tuple[
        tuple[str, str],
        ...,
    ],
]:
    """
    Map chronological row positions to one or more split-boundary labels.

    A one-row range has the same start and end position, so one position may
    legitimately carry both boundary labels.
    """

    ranges: tuple[
        tuple[str, _PositionRange | None],
        ...,
    ] = (
        (
            "history",
            plan.history,
        ),
        (
            "train",
            plan.train,
        ),
        (
            "validation",
            plan.validation,
        ),
        (
            "calibration",
            plan.calibration,
        ),
        (
            "test",
            plan.test,
        ),
    )

    positions: dict[
        int,
        list[
            tuple[str, str]
        ],
    ] = {}

    for stage_name, stage_range in ranges:
        if stage_range is None:
            continue

        positions.setdefault(
            stage_range.start_position,
            [],
        ).append(
            (
                stage_name,
                "start",
            )
        )

        positions.setdefault(
            stage_range.end_position,
            [],
        ).append(
            (
                stage_name,
                "end",
            )
        )

    return {
        position: tuple(
            labels
        )
        for position, labels in positions.items()
    }


def _scan_boundary_timestamps(
    *,
    validation_result: TrainingValidationResult,
    unresolved_plans: tuple[_UnresolvedSeriesPlan, ...],
    allowed_series: frozenset[SeriesIdentity] | None,
) -> dict[
    SeriesIdentity,
    dict[
        tuple[str, str],
        datetime,
    ],
]:
    """
    Second lightweight CSV pass.

    Capture only timestamps that correspond to planned split boundaries.
    No feature matrices or full timestamp arrays are retained.
    """

    identity_columns = list(
        validation_result.site_identity_columns
    )

    timestamp_column = (
        validation_result.timestamp_column
    )

    use_columns = [
        *identity_columns,
        timestamp_column,
    ]

    required_positions = {
        plan.identity: _required_boundary_positions(
            plan
        )
        for plan in unresolved_plans
    }

    captured: dict[
        SeriesIdentity,
        dict[
            tuple[str, str],
            datetime,
        ],
    ] = {
        plan.identity: {}
        for plan in unresolved_plans
    }

    seen_counts: dict[
        SeriesIdentity,
        int,
    ] = {
        plan.identity: 0
        for plan in unresolved_plans
    }

    reader = pd.read_csv(
        validation_result.input_file,
        usecols=use_columns,
        chunksize=SPLIT_SCAN_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        chunk[
            timestamp_column
        ] = pd.to_datetime(
            chunk[
                timestamp_column
            ],
            errors="raise",
        )

        grouped = chunk.groupby(
            identity_columns,
            sort=False,
            dropna=False,
        )

        for raw_identity, group in grouped:
            identity = _normalise_identity(
                raw_identity
            )

            if identity not in required_positions:
                if (
                    allowed_series is not None
                    and identity not in allowed_series
                ):
                    continue

                raise RuntimeError(
                    "Series changed between split-planning passes: "
                    f"{identity.as_tuple()}"
                )

            group_timestamps = group[
                timestamp_column
            ].reset_index(
                drop=True
            )

            group_start_position = seen_counts[
                identity
            ]

            group_end_position = (
                group_start_position
                + len(
                    group_timestamps
                )
                - 1
            )

            for required_position, boundary_labels in required_positions[
                identity
            ].items():
                if (
                    required_position
                    < group_start_position
                    or required_position
                    > group_end_position
                ):
                    continue

                local_position = (
                    required_position
                    - group_start_position
                )

                timestamp = group_timestamps.iloc[
                    local_position
                ].to_pydatetime()

                for boundary_label in boundary_labels:
                    captured[
                        identity
                    ][
                        boundary_label
                    ] = timestamp

            seen_counts[
                identity
            ] = (
                group_end_position
                + 1
            )

    for plan in unresolved_plans:
        expected_row_count = (
            plan.total_points
        )

        actual_row_count = seen_counts.get(
            plan.identity,
            0,
        )

        if actual_row_count != expected_row_count:
            raise RuntimeError(
                "Series row count changed between split-planning passes. "
                f"Series={plan.identity.as_tuple()}, "
                f"expected={expected_row_count}, "
                f"actual={actual_row_count}."
            )

        expected_boundaries = sum(
            len(
                labels
            )
            for labels in required_positions[
                plan.identity
            ].values()
        )

        actual_boundaries = len(
            captured[
                plan.identity
            ]
        )

        if actual_boundaries != expected_boundaries:
            raise RuntimeError(
                "Could not resolve every split timestamp boundary. "
                f"Series={plan.identity.as_tuple()}, "
                f"expected_boundaries={expected_boundaries}, "
                f"captured_boundaries={actual_boundaries}."
            )

    return captured


def _attach_timestamps(
    *,
    stage_name: str,
    position_range: _PositionRange,
    captured: dict[
        tuple[str, str],
        datetime,
    ],
) -> SplitRange:
    """Convert one internal row-position range into the public SplitRange."""

    start_timestamp = captured[
        (
            stage_name,
            "start",
        )
    ]

    end_timestamp = captured[
        (
            stage_name,
            "end",
        )
    ]

    return SplitRange(
        start_position=position_range.start_position,
        end_position=position_range.end_position,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        row_count=position_range.row_count,
    )


def _finalize_series_plans(
    *,
    unresolved_plans: tuple[_UnresolvedSeriesPlan, ...],
    captured_timestamps: dict[
        SeriesIdentity,
        dict[
            tuple[str, str],
            datetime,
        ],
    ],
) -> tuple[SeriesSplitPlan, ...]:
    """Attach captured timestamps to every position-only series plan."""

    finalized: list[
        SeriesSplitPlan
    ] = []

    for plan in unresolved_plans:
        captured = captured_timestamps[
            plan.identity
        ]

        validation_range = (
            _attach_timestamps(
                stage_name="validation",
                position_range=plan.validation,
                captured=captured,
            )
            if plan.validation is not None
            else None
        )

        calibration_range = (
            _attach_timestamps(
                stage_name="calibration",
                position_range=plan.calibration,
                captured=captured,
            )
            if plan.calibration is not None
            else None
        )

        finalized.append(
            SeriesSplitPlan(
                identity=plan.identity,
                total_points=plan.total_points,
                history=_attach_timestamps(
                    stage_name="history",
                    position_range=plan.history,
                    captured=captured,
                ),
                train=_attach_timestamps(
                    stage_name="train",
                    position_range=plan.train,
                    captured=captured,
                ),
                validation=validation_range,
                calibration=calibration_range,
                test=_attach_timestamps(
                    stage_name="test",
                    position_range=plan.test,
                    captured=captured,
                ),
            )
        )

    return tuple(
        finalized
    )


def build_training_split_plan(
    *,
    validation_result: TrainingValidationResult,
    training_config: TrainingConfig,
    allowed_series: frozenset[SeriesIdentity] | None = None,
) -> TrainingSplitPlan:
    """
    Build chronological Part D split boundaries for every validated site series.

    This is the public loader entry point.

    Returns
    -------
    TrainingSplitPlan
        Lightweight dataclasses containing only identities, row positions,
        timestamps, row counts, and resolved split percentages.

    Notes
    -----
    - History is always the first ``minimum_history_points`` rows per series.
    - Percentages apply only to rows AFTER history.
    - Tuning-disabled modes automatically use the resolved schemas where the
      validation share has already been removed/absorbed as designed.
    - When ``allowed_series`` is supplied by runner.py, only those
      runner-approved valid KPI series participate in splitting. Retained
      invalid Part C series are ignored completely.
    - The input Part C CSV must remain unchanged between the two lightweight
      scans performed by this function.
    """

    if len(
        validation_result.site_identity_columns
    ) != 3:
        raise ValueError(
            "Training loader requires exactly three site identity columns."
        )

    tuning_enabled, percentages = _resolve_model_percentages(
        model_type=validation_result.model_type,
        training_config=training_config,
    )

    if (
        allowed_series is not None
        and not allowed_series
    ):
        raise ValueError(
            "allowed_series cannot be empty when supplied."
        )

    counts = _count_series_and_validate_order(
        validation_result=validation_result,
        allowed_series=allowed_series,
    )

    unresolved_plans = _build_unresolved_plans(
        counts=counts,
        validation_result=validation_result,
        percentages=percentages,
    )

    captured_timestamps = _scan_boundary_timestamps(
        validation_result=validation_result,
        unresolved_plans=unresolved_plans,
        allowed_series=allowed_series,
    )

    series_plans = _finalize_series_plans(
        unresolved_plans=unresolved_plans,
        captured_timestamps=captured_timestamps,
    )

    return TrainingSplitPlan(
        input_file=validation_result.input_file,
        model_type=validation_result.model_type,
        hyperparameter_tuning_enabled=tuning_enabled,
        site_identity_columns=(
            validation_result.site_identity_columns[0],
            validation_result.site_identity_columns[1],
            validation_result.site_identity_columns[2],
        ),
        timestamp_column=validation_result.timestamp_column,
        minimum_history_points=validation_result.minimum_history_points,
        post_history_percentages=percentages,
        series=series_plans,
    )


__all__ = [
    "SeriesIdentity",
    "SeriesSplitPlan",
    "SplitRange",
    "TrainingSplitPlan",
    "build_training_split_plan",
]