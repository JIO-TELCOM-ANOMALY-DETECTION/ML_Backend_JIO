"""
Part D training runner.

Intended project location::

    app/ml/training/runner.py

This is the orchestration layer for Part D.

Flow
----
1. receive the Part C engineered dataset and resolved Part C contract;
2. verify the explicit minimum_history_points supplied by the Part C runner;
3. inspect kpi_usability and identify valid/invalid site series;
4. reject when no valid series exist;
5. call shared Part D validation;
6. call loader.py to calculate chronological split boundaries;
7. dispatch to the selected model implementation;
8. process every valid series independently;
9. save each completed model through model_output.py;
10. finalize aggregate JSON artifacts; and
11. return a compact Part D result to the parent pipeline.

Invalid series are skipped. Any processing failure for a series marked
``kpi_usability='valid'`` fails the complete Part D run.

The runner performs orchestration only. Model calculations remain in the model
folders, split calculations remain in loader.py, and persistence remains in
model_output.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Mapping, TypeAlias

import pandas as pd

from app.schemas.common.dataset import DatasetColumnMapping
from app.schemas.feature_engineering import ResolvedFeatureEngineeringConfig
from app.schemas.training.common import TrainingModelType
from app.schemas.training.isolation_forest.isolation_forest import (
    IsolationForestTrainingConfig,
)
from app.schemas.training.xgboost.xgboost import (
    XGBoostTrainingConfig,
)

from .isolation_forest import (
    test_isolation_forest_series,
    train_isolation_forest_series,
)
from .loader import (
    SeriesIdentity,
    SeriesSplitPlan,
    TrainingSplitPlan,
    build_training_split_plan,
)
from .model_output import (
    ModelOutputPaths,
    ModelOutputSession,
    SeriesModelArtifact,
    abort_model_output,
    finalize_model_output,
    prepare_model_output,
    save_series_model,
)
from .validation import (
    TrainingValidationResult,
    validate_training_dataset,
)
from .xgboost import (
    calibrate_xgboost_series,
    test_xgboost_series,
    train_xgboost_series,
)


TrainingConfig: TypeAlias = (
    XGBoostTrainingConfig
    | IsolationForestTrainingConfig
)


KPI_USABILITY_COLUMN = "kpi_usability"
KPI_INVALID_REASON_COLUMN = "kpi_invalid_reason"

VALID_KPI_USABILITY = "valid"
INVALID_KPI_USABILITY = "invalid"
ALLOWED_KPI_USABILITY = frozenset(
    {
        VALID_KPI_USABILITY,
        INVALID_KPI_USABILITY,
    }
)

USABILITY_READ_CHUNK_SIZE = 250_000


@dataclass(
    frozen=True,
    slots=True,
)
class SkippedSeries:
    """One series intentionally excluded before model execution."""

    identity: SeriesIdentity
    usability: str
    invalid_reason: str | None


@dataclass(
    frozen=True,
    slots=True,
)
class PartDTrainingResult:
    """Compact result returned to the parent pipeline after Part D succeeds."""

    status: str
    model_type: TrainingModelType

    input_file: Path
    output_directory: Path

    minimum_history_points: int

    total_series: int
    valid_series: int
    trained_series: int
    skipped_invalid_series: int

    skipped_series: tuple[SkippedSeries, ...]

    validation_result: TrainingValidationResult
    split_plan: TrainingSplitPlan
    output_paths: ModelOutputPaths


def _normalize_input_file(
    input_file: str | Path,
) -> Path:
    path = Path(
        input_file
    ).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            "Part D input dataset does not exist:\n"
            f"  {path}"
        )

    return path


def _validate_history_contract(
    *,
    minimum_history_points: int,
    feature_config: ResolvedFeatureEngineeringConfig,
) -> None:
    """
    Require the explicit Part C runner value to match its resolved config.

    This prevents the Part D loader from silently splitting with a different
    history requirement than the one Part C actually used to engineer features.
    """

    if isinstance(
        minimum_history_points,
        bool,
    ) or not isinstance(
        minimum_history_points,
        int,
    ):
        raise TypeError(
            "minimum_history_points supplied by Part C runner must be an int."
        )

    if minimum_history_points <= 0:
        raise ValueError(
            "minimum_history_points supplied by Part C runner must be > 0."
        )

    if (
        minimum_history_points
        != feature_config.minimum_history_points
    ):
        raise ValueError(
            "Part C minimum_history_points does not match the resolved "
            "feature-engineering configuration. "
            f"runner={minimum_history_points}, "
            f"resolved={feature_config.minimum_history_points}."
        )


def _validate_model_dispatch(
    *,
    model_type: TrainingModelType,
    training_config: TrainingConfig,
) -> None:
    """Require the explicit model switch to match the supplied schema."""

    if model_type == "xgboost":
        if not isinstance(
            training_config,
            XGBoostTrainingConfig,
        ):
            raise TypeError(
                "model_type='xgboost' requires XGBoostTrainingConfig."
            )
        return

    if model_type == "isolation_forest":
        if not isinstance(
            training_config,
            IsolationForestTrainingConfig,
        ):
            raise TypeError(
                "model_type='isolation_forest' requires "
                "IsolationForestTrainingConfig."
            )
        return

    raise ValueError(
        f"Unsupported training model_type: {model_type}"
    )


def _identity_from_group_key(
    raw_identity: object,
) -> SeriesIdentity:
    if not isinstance(
        raw_identity,
        tuple,
    ) or len(
        raw_identity
    ) != 3:
        raise ValueError(
            "Part D runner requires exactly three site identity columns."
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


def _clean_reason(
    values: pd.Series,
) -> str | None:
    """Collapse nonblank invalid-reason values into one readable string."""

    normalized = (
        values
        .dropna()
        .astype("string")
        .str.strip()
    )

    normalized = normalized.loc[
        normalized.ne("")
    ]

    if normalized.empty:
        return None

    unique_values = list(
        dict.fromkeys(
            normalized.tolist()
        )
    )

    return "; ".join(
        unique_values
    )


def _inspect_kpi_usability(
    *,
    input_file: Path,
    dataset_columns: DatasetColumnMapping,
) -> tuple[
    frozenset[SeriesIdentity],
    tuple[SkippedSeries, ...],
]:
    """
    Identify valid and invalid series before Part D model execution.

    Only identity/usability metadata is scanned, and the scan is chunked so the
    runner does not load the complete KPI dataset into memory.

    The current pipeline contract allows only ``valid`` and ``invalid`` and
    requires usability to remain constant within each series.
    """

    identity_columns = (
        dataset_columns.circle_column,
        dataset_columns.cluster_column,
        dataset_columns.site_name_column,
    )

    header = pd.read_csv(
        input_file,
        nrows=0,
    )

    required_columns = [
        *identity_columns,
        KPI_USABILITY_COLUMN,
        KPI_INVALID_REASON_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in header.columns
    ]

    if missing_columns:
        raise ValueError(
            "Part D runner is missing required KPI usability columns: "
            f"{missing_columns}"
        )

    usability_by_series: dict[
        SeriesIdentity,
        str,
    ] = {}

    reasons_by_series: dict[
        SeriesIdentity,
        list[str],
    ] = {}

    row_count = 0

    reader = pd.read_csv(
        input_file,
        usecols=required_columns,
        chunksize=USABILITY_READ_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        row_count += len(
            chunk
        )

        if bool(
            chunk[
                list(
                    identity_columns
                )
            ].isna().any().any()
        ):
            raise ValueError(
                "Part D input contains missing site identity values "
                f"in usability chunk {chunk_number}."
            )

        chunk[
            KPI_USABILITY_COLUMN
        ] = (
            chunk[
                KPI_USABILITY_COLUMN
            ]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        invalid_usability_mask = (
            chunk[
                KPI_USABILITY_COLUMN
            ].isna()
            | ~chunk[
                KPI_USABILITY_COLUMN
            ].isin(
                ALLOWED_KPI_USABILITY
            )
        )

        if bool(
            invalid_usability_mask.any()
        ):
            examples = (
                chunk.loc[
                    invalid_usability_mask,
                    KPI_USABILITY_COLUMN,
                ]
                .head(20)
                .astype("string")
                .tolist()
            )

            raise ValueError(
                "kpi_usability must contain only 'valid' or 'invalid'. "
                f"Examples of invalid values: {examples}"
            )

        grouped = chunk.groupby(
            list(
                identity_columns
            ),
            sort=False,
            dropna=False,
        )

        for raw_identity, group in grouped:
            identity = _identity_from_group_key(
                raw_identity
            )

            usability_values = (
                group[
                    KPI_USABILITY_COLUMN
                ]
                .drop_duplicates()
                .tolist()
            )

            if len(
                usability_values
            ) != 1:
                raise ValueError(
                    "kpi_usability is inconsistent within series "
                    f"{identity.as_tuple()}: {usability_values}"
                )

            series_usability = str(
                usability_values[
                    0
                ]
            )

            previous = usability_by_series.get(
                identity
            )

            if (
                previous is not None
                and previous != series_usability
            ):
                raise ValueError(
                    "kpi_usability changes across CSV chunks for series "
                    f"{identity.as_tuple()}: "
                    f"{previous!r} -> {series_usability!r}"
                )

            usability_by_series[
                identity
            ] = series_usability

            reason = _clean_reason(
                group[
                    KPI_INVALID_REASON_COLUMN
                ]
            )

            if reason:
                series_reasons = reasons_by_series.setdefault(
                    identity,
                    [],
                )

                for item in reason.split(
                    "; "
                ):
                    if item not in series_reasons:
                        series_reasons.append(
                            item
                        )

    if row_count <= 0:
        raise ValueError(
            "Part D input dataset contains no rows."
        )

    valid_series: set[
        SeriesIdentity
    ] = set()

    skipped_series: list[
        SkippedSeries
    ] = []

    for identity, series_usability in usability_by_series.items():
        if series_usability == VALID_KPI_USABILITY:
            valid_series.add(
                identity
            )

        else:
            reasons = reasons_by_series.get(
                identity,
                [],
            )

            skipped_series.append(
                SkippedSeries(
                    identity=identity,
                    usability=series_usability,
                    invalid_reason=(
                        "; ".join(
                            reasons
                        )
                        if reasons
                        else None
                    ),
                )
            )

    if not valid_series:
        raise ValueError(
            "Part D cannot train because no series are marked "
            "kpi_usability='valid'."
        )

    return (
        frozenset(
            valid_series
        ),
        tuple(
            skipped_series
        ),
    )


def _valid_series_plans(
    *,
    split_plan: TrainingSplitPlan,
    valid_series: frozenset[SeriesIdentity],
) -> tuple[SeriesSplitPlan, ...]:
    """Select loader plans belonging to runner-approved valid series only."""

    plans = tuple(
        series_plan
        for series_plan in split_plan.series
        if series_plan.identity in valid_series
    )

    planned_identities = {
        series_plan.identity
        for series_plan in plans
    }

    missing = (
        set(
            valid_series
        )
        - planned_identities
    )

    if missing:
        missing_values = sorted(
            (
                identity.as_tuple()
                for identity in missing
            )
        )

        raise ValueError(
            "One or more valid KPI series were not represented in loader "
            f"split plans: {missing_values[:20]}"
        )

    return plans


def _merge_row_counts(
    *parts: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge stage row-count mappings without losing duplicate stage names."""

    merged: dict[
        str,
        Any,
    ] = {}

    for part in parts:
        for key, value in part.items():
            if key in merged:
                raise ValueError(
                    "Duplicate usable-row-count key while combining model "
                    f"stage results: {key}"
                )

            merged[
                key
            ] = value

    return merged


def _run_xgboost_series(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: XGBoostTrainingConfig,
    output_session: ModelOutputSession,
) -> SeriesModelArtifact:
    """Execute train -> residual calibration -> final test for one series."""

    training_result = train_xgboost_series(
        validation_result=validation_result,
        split_plan=split_plan,
        series_plan=series_plan,
        training_config=training_config,
    )

    calibration_result = calibrate_xgboost_series(
        trained_model=training_result.model,
        validation_result=validation_result,
        split_plan=split_plan,
        series_plan=series_plan,
        training_config=training_config,
    )

    testing_result = test_xgboost_series(
        trained_model=training_result.model,
        validation_result=validation_result,
        split_plan=split_plan,
        series_plan=series_plan,
        training_config=training_config,
        decision_settings=(
            calibration_result.decision_settings
        ),
    )

    saved_model = save_series_model(
        session=output_session,
        model_type="xgboost",
        identity=series_plan.identity,
        trained_model=training_result.model,
    )

    usable_row_counts = _merge_row_counts(
        training_result.usable_row_counts,
        {
            "calibration": (
                calibration_result.usable_row_count
            ),
        },
        testing_result.usable_row_counts,
    )

    return SeriesModelArtifact(
        identity=series_plan.identity,
        series_key=saved_model.series_key,
        relative_model_file=(
            saved_model.relative_model_file
        ),
        selected_hyperparameters=(
            training_result.selected_hyperparameters
        ),
        final_model_parameters=(
            training_result.final_model_parameters
        ),
        tuning_summary=(
            training_result.tuning_summary
        ),
        training_summary=(
            training_result.training_summary
        ),
        calibration_summary=(
            calibration_result.calibration_summary
        ),
        testing_summary=(
            testing_result.test_summary
        ),
        decision_settings=(
            calibration_result.decision_settings
        ),
        usable_row_counts=(
            usable_row_counts
        ),
    )


def _run_isolation_forest_series(
    *,
    validation_result: TrainingValidationResult,
    split_plan: TrainingSplitPlan,
    series_plan: SeriesSplitPlan,
    training_config: IsolationForestTrainingConfig,
    output_session: ModelOutputSession,
) -> SeriesModelArtifact:
    """Execute train -> final test for one Isolation Forest series."""

    training_result = (
        train_isolation_forest_series(
            validation_result=validation_result,
            split_plan=split_plan,
            series_plan=series_plan,
            training_config=training_config,
        )
    )

    testing_result = (
        test_isolation_forest_series(
            trained_model=training_result.model,
            validation_result=validation_result,
            split_plan=split_plan,
            series_plan=series_plan,
            training_config=training_config,
            decision_settings=(
                training_result.decision_settings
            ),
        )
    )

    saved_model = save_series_model(
        session=output_session,
        model_type="isolation_forest",
        identity=series_plan.identity,
        trained_model=training_result.model,
    )

    usable_row_counts = _merge_row_counts(
        training_result.usable_row_counts,
        testing_result.usable_row_counts,
    )

    return SeriesModelArtifact(
        identity=series_plan.identity,
        series_key=saved_model.series_key,
        relative_model_file=(
            saved_model.relative_model_file
        ),
        selected_hyperparameters=(
            training_result.selected_hyperparameters
        ),
        final_model_parameters=(
            training_result.final_model_parameters
        ),
        tuning_summary=(
            training_result.tuning_summary
        ),
        training_summary=(
            training_result.training_summary
        ),
        calibration_summary=None,
        testing_summary=(
            testing_result.test_summary
        ),
        decision_settings=(
            training_result.decision_settings
        ),
        usable_row_counts=(
            usable_row_counts
        ),
    )


def run_training(
    *,
    model_type: TrainingModelType,
    input_file: str | Path,
    output_directory: str | Path,
    dataset_columns: DatasetColumnMapping,
    feature_config: ResolvedFeatureEngineeringConfig,
    minimum_history_points: int,
    training_config: TrainingConfig,
    metadata: Mapping[str, Any] | None = None,
) -> PartDTrainingResult:
    """
    Run complete Part D training for one Part C engineered KPI dataset.

    ``minimum_history_points`` is intentionally supplied explicitly by the Part C
    runner and must match ``feature_config.minimum_history_points``.
    """

    path = _normalize_input_file(
        input_file
    )

    _validate_model_dispatch(
        model_type=model_type,
        training_config=training_config,
    )

    _validate_history_contract(
        minimum_history_points=minimum_history_points,
        feature_config=feature_config,
    )

    (
        valid_series,
        skipped_series,
    ) = _inspect_kpi_usability(
        input_file=path,
        dataset_columns=dataset_columns,
    )

    validation_result = validate_training_dataset(
        model_type=model_type,
        input_file=path,
        dataset_columns=dataset_columns,
        feature_config=feature_config,
        training_config=training_config,
    )

    split_plan = build_training_split_plan(
        validation_result=validation_result,
        training_config=training_config,
        allowed_series=valid_series,
    )

    series_plans = _valid_series_plans(
        split_plan=split_plan,
        valid_series=valid_series,
    )

    output_session = prepare_model_output(
        output_directory
    )

    series_artifacts: list[
        SeriesModelArtifact
    ] = []

    try:
        for series_plan in series_plans:
            try:
                if model_type == "xgboost":
                    if not isinstance(
                        training_config,
                        XGBoostTrainingConfig,
                    ):
                        raise TypeError(
                            "Internal runner model/config mismatch."
                        )

                    artifact = _run_xgboost_series(
                        validation_result=validation_result,
                        split_plan=split_plan,
                        series_plan=series_plan,
                        training_config=training_config,
                        output_session=output_session,
                    )

                elif model_type == "isolation_forest":
                    if not isinstance(
                        training_config,
                        IsolationForestTrainingConfig,
                    ):
                        raise TypeError(
                            "Internal runner model/config mismatch."
                        )

                    artifact = (
                        _run_isolation_forest_series(
                            validation_result=validation_result,
                            split_plan=split_plan,
                            series_plan=series_plan,
                            training_config=training_config,
                            output_session=output_session,
                        )
                    )

                else:
                    raise ValueError(
                        "Unsupported model_type reached execution stage: "
                        f"{model_type}"
                    )

                series_artifacts.append(
                    artifact
                )

                # The estimator is already serialized. Explicit collection
                # helps long runs containing hundreds of independent models.
                gc.collect()

            except Exception as error:
                raise RuntimeError(
                    "Part D model execution failed for valid series "
                    f"{series_plan.identity.as_tuple()}."
                ) from error

        run_metadata: dict[
            str,
            Any,
        ] = {
            "minimum_history_points_from_part_c_runner": (
                minimum_history_points
            ),
            "total_series": (
                len(valid_series)
                + len(skipped_series)
            ),
            "valid_series": len(
                valid_series
            ),
            "trained_series": len(
                series_artifacts
            ),
            "skipped_invalid_series": len(
                skipped_series
            ),
            "skipped_series": [
                {
                    "identity": {
                        "circle": item.identity.circle,
                        "cluster": item.identity.cluster,
                        "site_name": item.identity.site_name,
                    },
                    "usability": item.usability,
                    "invalid_reason": item.invalid_reason,
                }
                for item in skipped_series
            ],
        }

        if metadata:
            run_metadata[
                "parent_pipeline_metadata"
            ] = dict(
                metadata
            )

        output_paths = finalize_model_output(
            session=output_session,
            validation_result=validation_result,
            split_plan=split_plan,
            training_config=training_config,
            series_artifacts=tuple(
                series_artifacts
            ),
            metadata=run_metadata,
        )

    except Exception:
        abort_model_output(
            output_session
        )
        raise

    return PartDTrainingResult(
        status="completed",
        model_type=model_type,
        input_file=path,
        output_directory=Path(
            output_directory
        ).expanduser(),
        minimum_history_points=(
            minimum_history_points
        ),
        total_series=(
            len(valid_series)
            + len(skipped_series)
        ),
        valid_series=len(
            valid_series
        ),
        trained_series=len(
            series_artifacts
        ),
        skipped_invalid_series=len(
            skipped_series
        ),
        skipped_series=(
            skipped_series
        ),
        validation_result=(
            validation_result
        ),
        split_plan=(
            split_plan
        ),
        output_paths=(
            output_paths
        ),
    )


__all__ = [
    "PartDTrainingResult",
    "SkippedSeries",
    "run_training",
]