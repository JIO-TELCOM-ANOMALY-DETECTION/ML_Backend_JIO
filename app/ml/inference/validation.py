"""
Validate the Part C -> saved-model inference handoff.

Intended project location::

    app/ml/inference/validation.py

This module validates that the actual feature-engineered outputs returned by
the inference history subsystem can be paired safely with the saved Part D
model artifacts resolved by ``loader.py``.

Responsibility boundary
-----------------------
The history subsystem has already verified that inference-time Part C resolved
the exact same feature-engineering configuration used by the saved training
run. This module does NOT recalculate or revalidate Part C's feature logic.

Instead, it validates the next boundary:

    Part C output file
        +
    saved KPI/site model mapping
        +
    saved model feature/decision contract
        ->
    safe model-inference input

The parent runner should use the module in two stages:

1. ``validate_inference_handoff(...)``
   performs a lightweight run-level validation. It scans Part C headers and
   site metadata, checks model-file existence, and builds one validation plan
   per saved site without deserializing any estimator.

2. ``load_and_validate_series_data(...)``
   is called on demand for ONE saved site immediately before that site's model
   is loaded. It reads only that site's required model-input columns and
   returns its dataframe plus row-level readiness metadata.

This preserves the intended memory rule:

    lightweight metadata for all sites
    + one site's dataframe
    + one site's estimator

rather than loading all model estimators into memory at once.

This module does not:

- resolve ``run_id`` or read ``inference_config.json``;
- deserialize models;
- execute XGBoost or Isolation Forest;
- apply anomaly thresholds;
- save model-inference outputs;
- commit history; or
- rerun Parts A/B/C.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np
import pandas as pd

from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
)
from app.ml.feature_engineering.runner import FeatureEngineeringRunnerResult
from app.schemas.common.dataset import DatasetColumnMapping

from .loader import (
    InferenceKPIArtifactPlan,
    InferenceLoadResult,
    InferenceModelType,
    InferenceSeriesArtifact,
)


KPI_USABILITY_COLUMN = "kpi_usability"
VALID_KPI_USABILITY = "valid"
INVALID_KPI_USABILITY = "invalid"
ALLOWED_KPI_USABILITY = frozenset(
    {
        VALID_KPI_USABILITY,
        INVALID_KPI_USABILITY,
    }
)

SITE_SCAN_CHUNK_SIZE = 250_000
SERIES_LOAD_CHUNK_SIZE = 100_000


SeriesIdentityTuple = tuple[
    str,
    str,
    str,
]


@dataclass(frozen=True, slots=True)
class InferenceSeriesValidationPlan:
    """
    Lightweight validation result for one saved KPI/site model.

    No estimator and no full site dataframe is stored here.
    """

    artifact: InferenceSeriesArtifact
    feature_file: Path

    site_row_count: int
    site_usability: Literal[
        "valid",
        "invalid",
    ]

    minimum_timestamp: pd.Timestamp
    maximum_timestamp: pd.Timestamp

    ready_for_model_inference: bool
    skip_reason: str | None


@dataclass(frozen=True, slots=True)
class UnmodelledPartCSeries:
    """
    One site present in Part C output but without a saved Part D model.

    This is not automatically an error because Part D intentionally skipped
    sites that were invalid during training.
    """

    kpi_name: str
    identity: SeriesIdentityTuple
    usability: Literal[
        "valid",
        "invalid",
    ]
    row_count: int


@dataclass(frozen=True, slots=True)
class InferenceKPIValidationPlan:
    """Validated Part C/model mapping for one KPI."""

    kpi_name: str
    feature_result: FeatureEngineeringRunnerResult
    feature_file: Path

    model_feature_columns: tuple[str, ...]
    target_column: str | None

    series: tuple[
        InferenceSeriesValidationPlan,
        ...,
    ]

    unmodelled_part_c_series: tuple[
        UnmodelledPartCSeries,
        ...,
    ]

    @property
    def ready_series_count(
        self,
    ) -> int:
        return sum(
            1
            for item in self.series
            if item.ready_for_model_inference
        )

    @property
    def skipped_modelled_series_count(
        self,
    ) -> int:
        return (
            len(self.series)
            - self.ready_series_count
        )

    def get_series(
        self,
        series_key: str,
    ) -> InferenceSeriesValidationPlan:
        for item in self.series:
            if item.artifact.series_key == series_key:
                return item

        raise KeyError(
            f"No validated series {series_key!r} exists for "
            f"KPI {self.kpi_name!r}."
        )


@dataclass(frozen=True, slots=True)
class InferenceValidationResult:
    """
    Complete lightweight validation result for one inference run.
    """

    model_type: InferenceModelType

    kpis: tuple[
        InferenceKPIValidationPlan,
        ...,
    ]

    total_saved_series: int
    ready_saved_series: int
    skipped_saved_series: int
    unmodelled_part_c_series: int

    def get_kpi(
        self,
        kpi_name: str,
    ) -> InferenceKPIValidationPlan:
        for item in self.kpis:
            if item.kpi_name == kpi_name:
                return item

        raise KeyError(
            f"No validated inference KPI {kpi_name!r} exists."
        )

    def iter_ready_series(
        self,
    ) -> Iterator[InferenceSeriesValidationPlan]:
        """
        Yield only saved site models whose current Part C series is valid.

        The parent runner should iterate this method and load one estimator at
        a time through ``loader.load_series_model(...)``.
        """

        for kpi in self.kpis:
            for series in kpi.series:
                if series.ready_for_model_inference:
                    yield series


@dataclass(frozen=True, slots=True)
class ValidatedSeriesData:
    """
    One site's actual Part C rows prepared for model inference.

    ``dataframe`` contains only the required identity/timestamp/usability/model
    input columns plus XGBoost target or Isolation-Forest direction feature as
    applicable.

    ``scorable_mask`` marks rows whose complete saved model feature vector is
    finite. Model-specific inference may further restrict rows to the parent
    runner's requested output window.
    """

    series_plan: InferenceSeriesValidationPlan
    dataframe: pd.DataFrame

    model_feature_columns: tuple[str, ...]
    target_column: str | None
    direction_feature: str | None

    scorable_mask: pd.Series

    total_rows: int
    scorable_rows: int
    unscorable_rows: int

    minimum_timestamp: pd.Timestamp
    maximum_timestamp: pd.Timestamp


def _normalise_identity_value(
    value: object,
    *,
    field_name: str,
    role: str,
) -> str:
    if pd.isna(
        value
    ):
        raise InputValidationError(
            f"{role} contains a missing site identity value.",
            error_code="MISSING_INFERENCE_SITE_IDENTITY",
            details={
                "role": role,
                "field": field_name,
            },
        )

    normalized = str(
        value
    ).strip()

    if not normalized:
        raise InputValidationError(
            f"{role} contains a blank site identity value.",
            error_code="BLANK_INFERENCE_SITE_IDENTITY",
            details={
                "role": role,
                "field": field_name,
            },
        )

    return normalized


def _identity_from_values(
    values: tuple[
        object,
        object,
        object,
    ],
    *,
    role: str,
) -> SeriesIdentityTuple:
    return (
        _normalise_identity_value(
            values[0],
            field_name="circle",
            role=role,
        ),
        _normalise_identity_value(
            values[1],
            field_name="cluster",
            role=role,
        ),
        _normalise_identity_value(
            values[2],
            field_name="site_name",
            role=role,
        ),
    )


def _require_file(
    path: str | Path,
    *,
    role: str,
    require_nonempty: bool = True,
) -> Path:
    resolved = Path(
        path
    ).expanduser().resolve()

    if not resolved.is_file():
        raise FileProcessingError(
            f"Required inference file does not exist for {role}.",
            error_code="MISSING_INFERENCE_VALIDATION_FILE",
            details={
                "role": role,
                "path": str(
                    resolved
                ),
            },
        )

    if (
        require_nonempty
        and resolved.stat().st_size <= 0
    ):
        raise FileProcessingError(
            f"Required inference file is empty for {role}.",
            error_code="EMPTY_INFERENCE_VALIDATION_FILE",
            details={
                "role": role,
                "path": str(
                    resolved
                ),
            },
        )

    return resolved


def _feature_results_by_kpi(
    feature_results: Iterable[
        FeatureEngineeringRunnerResult
    ],
) -> dict[
    str,
    FeatureEngineeringRunnerResult,
]:
    mapping: dict[
        str,
        FeatureEngineeringRunnerResult,
    ] = {}

    for result in feature_results:
        kpi_name = str(
            result.kpi_name
        ).strip()

        if not kpi_name:
            raise InputValidationError(
                "A Part C result has no usable KPI name.",
                error_code="MISSING_INFERENCE_PART_C_KPI_NAME",
            )

        if kpi_name in mapping:
            raise InputValidationError(
                "Part C returned more than one result for the same KPI.",
                error_code="DUPLICATE_INFERENCE_PART_C_KPI_RESULT",
                details={
                    "kpi_name": kpi_name,
                },
            )

        mapping[
            kpi_name
        ] = result

    return mapping


def _required_header_columns(
    *,
    artifact_plan: InferenceKPIArtifactPlan,
    columns: DatasetColumnMapping,
) -> tuple[str, ...]:
    required: list[
        str
    ] = [
        columns.circle_column,
        columns.cluster_column,
        columns.site_name_column,
        columns.timestamp_column,
        KPI_USABILITY_COLUMN,
        *artifact_plan.model_feature_columns,
    ]

    if artifact_plan.target_column is not None:
        required.append(
            artifact_plan.target_column
        )

    if artifact_plan.model_type == "isolation_forest":
        for series in artifact_plan.series:
            decision_settings = (
                series
                .decision_settings
            )

            direction_feature = (
                decision_settings
                .direction_feature
            )

            # One-sided Isolation Forest inference requires the saved
            # direction feature. For "both", model anomalies are kept
            # regardless of direction, so the direction feature is optional
            # unless it is already one of the saved model features.
            if (
                decision_settings.threshold_direction
                in {"lower_only", "upper_only"}
                and direction_feature not in required
            ):
                required.append(
                    direction_feature
                )

    # Preserve order while removing repeated names.
    return tuple(
        dict.fromkeys(
            required
        )
    )


def _validate_feature_header(
    *,
    feature_file: Path,
    artifact_plan: InferenceKPIArtifactPlan,
    columns: DatasetColumnMapping,
) -> tuple[str, ...]:
    try:
        header = pd.read_csv(
            feature_file,
            nrows=0,
        )
    except Exception as error:
        raise FileProcessingError(
            "Part C output header could not be read for inference validation.",
            error_code="INFERENCE_PART_C_HEADER_READ_FAILED",
            details={
                "kpi_name": artifact_plan.kpi_name,
                "feature_file": str(
                    feature_file
                ),
                "reason": str(
                    error
                ),
            },
        ) from error

    actual_columns = tuple(
        str(
            column
        )
        for column in header.columns
    )

    required_columns = _required_header_columns(
        artifact_plan=artifact_plan,
        columns=columns,
    )

    missing = [
        column
        for column in required_columns
        if column not in actual_columns
    ]

    if missing:
        raise InputValidationError(
            "Part C output is missing columns required by the saved model "
            "contract.",
            error_code="INFERENCE_MODEL_INPUT_COLUMNS_MISSING",
            details={
                "kpi_name": artifact_plan.kpi_name,
                "missing_columns": missing,
                "feature_file": str(
                    feature_file
                ),
            },
        )

    # The estimator input matrix must preserve the exact saved training order.
    positions = [
        actual_columns.index(
            column
        )
        for column in artifact_plan.model_feature_columns
    ]

    if positions != sorted(
        positions
    ):
        raise InputValidationError(
            "Saved model feature columns appear in a different order in the "
            "Part C output file.",
            error_code="INFERENCE_MODEL_FEATURE_ORDER_MISMATCH",
            details={
                "kpi_name": artifact_plan.kpi_name,
                "saved_feature_order": list(
                    artifact_plan
                    .model_feature_columns
                ),
            },
        )

    return actual_columns


def _validate_feature_result_contract(
    *,
    feature_result: FeatureEngineeringRunnerResult,
    load_result: InferenceLoadResult,
) -> Path:
    """
    Reconfirm the exact Part C result contract before inspecting its CSV.

    The history runner already performs these checks. They remain here as a
    defensive boundary because validation may be used independently in tests.
    """

    saved_feature_contract = (
        load_result
        .saved
        .feature_engineering
    )

    if (
        feature_result.resolved_config
        != saved_feature_contract.resolved
    ):
        raise InputValidationError(
            "Part C result resolved configuration does not match the saved "
            "training-time feature contract.",
            error_code="INFERENCE_PART_C_RESOLUTION_MISMATCH",
            details={
                "kpi_name": feature_result.kpi_name,
            },
        )

    if (
        tuple(
            feature_result.feature_columns
        )
        != tuple(
            saved_feature_contract
            .resolved
            .feature_columns
        )
    ):
        raise InputValidationError(
            "Part C result feature_columns do not match the saved resolved "
            "feature contract.",
            error_code="INFERENCE_PART_C_FEATURE_COLUMNS_MISMATCH",
            details={
                "kpi_name": feature_result.kpi_name,
                "part_c_feature_columns": list(
                    feature_result.feature_columns
                ),
                "saved_feature_columns": list(
                    saved_feature_contract
                    .resolved
                    .feature_columns
                ),
            },
        )

    return _require_file(
        feature_result.final_dataset_file,
        role=(
            "Part C final feature dataset "
            f"for KPI {feature_result.kpi_name!r}"
        ),
    )


def _validate_all_model_files(
    artifact_plan: InferenceKPIArtifactPlan,
) -> None:
    """
    Validate saved model-file existence without deserializing estimators.
    """

    for artifact in artifact_plan.series:
        path = _require_file(
            artifact.model_file,
            role=(
                f"saved {artifact.model_type} model for "
                f"KPI {artifact.kpi_name!r}, "
                f"series {artifact.series_key!r}"
            ),
        )

        expected_suffix = (
            ".json"
            if artifact.model_type == "xgboost"
            else ".joblib"
        )

        if path.suffix.lower() != expected_suffix:
            raise InputValidationError(
                "Saved model file extension does not match the model type.",
                error_code="INFERENCE_MODEL_EXTENSION_MISMATCH",
                details={
                    "kpi_name": artifact.kpi_name,
                    "series_key": artifact.series_key,
                    "model_type": artifact.model_type,
                    "model_file": str(
                        path
                    ),
                    "expected_suffix": expected_suffix,
                },
            )


@dataclass
class _MutableSiteScanState:
    """Internal aggregation state used while scanning one Part C file."""

    row_count: int = 0
    usability: str | None = None
    minimum_timestamp: pd.Timestamp | None = None
    maximum_timestamp: pd.Timestamp | None = None


def _scan_part_c_sites(
    *,
    feature_file: Path,
    artifact_plan: InferenceKPIArtifactPlan,
    columns: DatasetColumnMapping,
    expected_row_count: int,
) -> dict[
    SeriesIdentityTuple,
    _MutableSiteScanState,
]:
    """
    Scan only site identity/usability/timestamp columns in chunks.

    This validates site-level mapping without loading the full feature matrix.
    """

    usecols = [
        columns.circle_column,
        columns.cluster_column,
        columns.site_name_column,
        columns.timestamp_column,
        KPI_USABILITY_COLUMN,
    ]

    states: dict[
        SeriesIdentityTuple,
        _MutableSiteScanState,
    ] = {}

    total_rows = 0

    try:
        reader = pd.read_csv(
            feature_file,
            usecols=usecols,
            chunksize=SITE_SCAN_CHUNK_SIZE,
            low_memory=False,
        )

        for chunk_number, chunk in enumerate(
            reader,
            start=1,
        ):
            total_rows += len(
                chunk
            )

            timestamps = pd.to_datetime(
                chunk[
                    columns.timestamp_column
                ],
                errors="coerce",
                utc=True,
            )

            invalid_timestamps = timestamps.isna()

            if bool(
                invalid_timestamps.any()
            ):
                raise InputValidationError(
                    "Part C output contains invalid timestamps.",
                    error_code="INVALID_INFERENCE_PART_C_TIMESTAMP",
                    details={
                        "kpi_name": artifact_plan.kpi_name,
                        "chunk_number": chunk_number,
                        "invalid_rows": int(
                            invalid_timestamps.sum()
                        ),
                    },
                )

            usability = (
                chunk[
                    KPI_USABILITY_COLUMN
                ]
                .astype("string")
                .str.strip()
                .str.casefold()
            )

            invalid_usability = (
                usability.isna()
                | ~usability.isin(
                    ALLOWED_KPI_USABILITY
                )
            )

            if bool(
                invalid_usability.any()
            ):
                raise InputValidationError(
                    "Part C output kpi_usability must contain only 'valid' "
                    "or 'invalid'.",
                    error_code="INVALID_INFERENCE_PART_C_USABILITY",
                    details={
                        "kpi_name": artifact_plan.kpi_name,
                        "chunk_number": chunk_number,
                        "invalid_rows": int(
                            invalid_usability.sum()
                        ),
                    },
                )

            working = pd.DataFrame(
                {
                    "circle": (
                        chunk[
                            columns.circle_column
                        ]
                    ),
                    "cluster": (
                        chunk[
                            columns.cluster_column
                        ]
                    ),
                    "site": (
                        chunk[
                            columns.site_name_column
                        ]
                    ),
                    "timestamp": timestamps,
                    "usability": usability,
                }
            )

            grouped = working.groupby(
                [
                    "circle",
                    "cluster",
                    "site",
                ],
                sort=False,
                dropna=False,
            )

            for raw_identity, group in grouped:
                identity = _identity_from_values(
                    raw_identity,
                    role=(
                        "Part C output "
                        f"for KPI {artifact_plan.kpi_name!r}"
                    ),
                )

                usability_values = (
                    group[
                        "usability"
                    ]
                    .drop_duplicates()
                    .tolist()
                )

                if len(
                    usability_values
                ) != 1:
                    raise InputValidationError(
                        "Part C kpi_usability changes within one site series.",
                        error_code="INCONSISTENT_INFERENCE_SITE_USABILITY",
                        details={
                            "kpi_name": artifact_plan.kpi_name,
                            "identity": identity,
                            "values": usability_values,
                        },
                    )

                current_usability = str(
                    usability_values[0]
                )

                state = states.setdefault(
                    identity,
                    _MutableSiteScanState(),
                )

                if (
                    state.usability is not None
                    and state.usability
                    != current_usability
                ):
                    raise InputValidationError(
                        "Part C kpi_usability changes across CSV chunks for "
                        "one site series.",
                        error_code="INCONSISTENT_INFERENCE_SITE_USABILITY",
                        details={
                            "kpi_name": artifact_plan.kpi_name,
                            "identity": identity,
                            "previous": state.usability,
                            "current": current_usability,
                        },
                    )

                group_min = pd.Timestamp(
                    group[
                        "timestamp"
                    ].min()
                )
                group_max = pd.Timestamp(
                    group[
                        "timestamp"
                    ].max()
                )

                state.row_count += len(
                    group
                )
                state.usability = (
                    current_usability
                )
                state.minimum_timestamp = (
                    group_min
                    if state.minimum_timestamp is None
                    else min(
                        state.minimum_timestamp,
                        group_min,
                    )
                )
                state.maximum_timestamp = (
                    group_max
                    if state.maximum_timestamp is None
                    else max(
                        state.maximum_timestamp,
                        group_max,
                    )
                )

    except InputValidationError:
        raise
    except Exception as error:
        raise FileProcessingError(
            "Part C output could not be scanned for saved model/site "
            "compatibility.",
            error_code="INFERENCE_PART_C_SITE_SCAN_FAILED",
            details={
                "kpi_name": artifact_plan.kpi_name,
                "feature_file": str(
                    feature_file
                ),
                "reason": str(
                    error
                ),
            },
        ) from error

    if total_rows <= 0:
        raise InputValidationError(
            "Part C output contains no rows.",
            error_code="EMPTY_INFERENCE_PART_C_DATASET",
            details={
                "kpi_name": artifact_plan.kpi_name,
            },
        )

    if (
        expected_row_count >= 0
        and total_rows != expected_row_count
    ):
        raise InputValidationError(
            "Part C result row count does not match its saved final dataset.",
            error_code="INFERENCE_PART_C_ROW_COUNT_MISMATCH",
            details={
                "kpi_name": artifact_plan.kpi_name,
                "reported_rows": expected_row_count,
                "actual_rows": total_rows,
            },
        )

    return states


def _build_kpi_validation_plan(
    *,
    load_result: InferenceLoadResult,
    artifact_plan: InferenceKPIArtifactPlan,
    feature_result: FeatureEngineeringRunnerResult,
) -> InferenceKPIValidationPlan:
    columns = (
        load_result
        .saved
        .dataset
        .columns
    )

    feature_file = _validate_feature_result_contract(
        feature_result=feature_result,
        load_result=load_result,
    )

    _validate_feature_header(
        feature_file=feature_file,
        artifact_plan=artifact_plan,
        columns=columns,
    )

    _validate_all_model_files(
        artifact_plan
    )

    site_states = _scan_part_c_sites(
        feature_file=feature_file,
        artifact_plan=artifact_plan,
        columns=columns,
        expected_row_count=int(
            feature_result.total_rows
        ),
    )

    saved_identity_to_artifact = {
        artifact.identity: artifact
        for artifact in artifact_plan.series
    }

    series_plans: list[
        InferenceSeriesValidationPlan
    ] = []

    for artifact in artifact_plan.series:
        state = site_states.get(
            artifact.identity
        )

        if state is None:
            raise InputValidationError(
                "A saved site model has no matching site series in the "
                "current Part C output.",
                error_code="MISSING_INFERENCE_MODEL_SITE_DATA",
                details={
                    "kpi_name": artifact.kpi_name,
                    "series_key": artifact.series_key,
                    "identity": artifact.identity,
                    "feature_file": str(
                        feature_file
                    ),
                },
            )

        if (
            state.usability is None
            or state.minimum_timestamp is None
            or state.maximum_timestamp is None
        ):
            raise InputValidationError(
                "Part C site scan produced incomplete metadata.",
                error_code="INCOMPLETE_INFERENCE_SITE_SCAN_METADATA",
                details={
                    "kpi_name": artifact.kpi_name,
                    "identity": artifact.identity,
                },
            )

        ready = (
            state.usability
            == VALID_KPI_USABILITY
        )

        series_plans.append(
            InferenceSeriesValidationPlan(
                artifact=artifact,
                feature_file=feature_file,
                site_row_count=(
                    state.row_count
                ),
                site_usability=(
                    state.usability  # type: ignore[arg-type]
                ),
                minimum_timestamp=(
                    state.minimum_timestamp
                ),
                maximum_timestamp=(
                    state.maximum_timestamp
                ),
                ready_for_model_inference=ready,
                skip_reason=(
                    None
                    if ready
                    else (
                        "Current Part C series is marked "
                        "kpi_usability='invalid'."
                    )
                ),
            )
        )

    unmodelled: list[
        UnmodelledPartCSeries
    ] = []

    for identity, state in sorted(
        site_states.items()
    ):
        if identity in saved_identity_to_artifact:
            continue

        if (
            state.usability is None
        ):
            raise InputValidationError(
                "Part C unmodelled site scan produced no usability value.",
                error_code="INCOMPLETE_UNMODELLED_SITE_METADATA",
                details={
                    "kpi_name": artifact_plan.kpi_name,
                    "identity": identity,
                },
            )

        unmodelled.append(
            UnmodelledPartCSeries(
                kpi_name=artifact_plan.kpi_name,
                identity=identity,
                usability=(
                    state.usability  # type: ignore[arg-type]
                ),
                row_count=(
                    state.row_count
                ),
            )
        )

    return InferenceKPIValidationPlan(
        kpi_name=artifact_plan.kpi_name,
        feature_result=feature_result,
        feature_file=feature_file,
        model_feature_columns=(
            artifact_plan
            .model_feature_columns
        ),
        target_column=(
            artifact_plan.target_column
        ),
        series=tuple(
            series_plans
        ),
        unmodelled_part_c_series=tuple(
            unmodelled
        ),
    )


def validate_inference_handoff(
    *,
    load_result: InferenceLoadResult,
    feature_results: Iterable[
        FeatureEngineeringRunnerResult
    ],
) -> InferenceValidationResult:
    """
    Validate the complete Part C -> saved model handoff.

    No estimator is deserialized and no complete feature dataframe is retained
    in memory.
    """

    feature_by_kpi = _feature_results_by_kpi(
        feature_results
    )

    expected_kpis = tuple(
        load_result
        .saved
        .dataset
        .kpi_names
    )

    if set(
        feature_by_kpi
    ) != set(
        expected_kpis
    ):
        raise InputValidationError(
            "Part C inference results do not match the trained KPI set.",
            error_code="INFERENCE_PART_C_KPI_SET_MISMATCH",
            details={
                "expected_kpis": sorted(
                    expected_kpis
                ),
                "received_kpis": sorted(
                    feature_by_kpi
                ),
            },
        )

    if (
        load_result.model_plan.model_type
        != load_result.saved.model_type
    ):
        raise InputValidationError(
            "Resolved model load plan type does not match the saved inference "
            "contract.",
            error_code="INFERENCE_VALIDATION_MODEL_TYPE_MISMATCH",
            details={
                "saved_model_type": (
                    load_result.saved.model_type
                ),
                "plan_model_type": (
                    load_result
                    .model_plan
                    .model_type
                ),
            },
        )

    kpi_plans: list[
        InferenceKPIValidationPlan
    ] = []

    for kpi_name in expected_kpis:
        artifact_plan = (
            load_result
            .model_plan
            .get_kpi(
                kpi_name
            )
        )

        feature_result = (
            feature_by_kpi[
                kpi_name
            ]
        )

        kpi_plans.append(
            _build_kpi_validation_plan(
                load_result=load_result,
                artifact_plan=artifact_plan,
                feature_result=feature_result,
            )
        )

    total_saved = sum(
        len(
            kpi.series
        )
        for kpi in kpi_plans
    )

    ready_saved = sum(
        kpi.ready_series_count
        for kpi in kpi_plans
    )

    skipped_saved = (
        total_saved
        - ready_saved
    )

    unmodelled_count = sum(
        len(
            kpi.unmodelled_part_c_series
        )
        for kpi in kpi_plans
    )

    return InferenceValidationResult(
        model_type=(
            load_result
            .model_plan
            .model_type
        ),
        kpis=tuple(
            kpi_plans
        ),
        total_saved_series=total_saved,
        ready_saved_series=ready_saved,
        skipped_saved_series=skipped_saved,
        unmodelled_part_c_series=(
            unmodelled_count
        ),
    )


def _required_series_data_columns(
    *,
    series_plan: InferenceSeriesValidationPlan,
    columns: DatasetColumnMapping,
) -> tuple[
    tuple[str, ...],
    str | None,
]:
    artifact = series_plan.artifact

    required: list[
        str
    ] = [
        columns.circle_column,
        columns.cluster_column,
        columns.site_name_column,
        columns.timestamp_column,
        KPI_USABILITY_COLUMN,
        *artifact.model_feature_columns,
    ]

    if artifact.target_column is not None:
        required.append(
            artifact.target_column
        )

    direction_feature: str | None = None

    if artifact.model_type == "isolation_forest":
        saved_direction_feature = (
            artifact
            .decision_settings
            .direction_feature
        )

        threshold_direction = (
            artifact
            .decision_settings
            .threshold_direction
        )

        # lower_only / upper_only require the direction feature.
        # For both, retain it only when it is already one of the saved
        # model features, so subtype counts remain optional.
        if (
            threshold_direction
            in {"lower_only", "upper_only"}
            or saved_direction_feature
            in artifact.model_feature_columns
        ):
            direction_feature = (
                saved_direction_feature
            )

            if direction_feature not in required:
                required.append(
                    direction_feature
                )

    return (
        tuple(
            dict.fromkeys(
                required
            )
        ),
        direction_feature,
    )


def _numeric_series(
    values: pd.Series,
    *,
    column_name: str,
    kpi_name: str,
    identity: SeriesIdentityTuple,
) -> pd.Series:
    """
    Convert one model-input column to float without hiding bad text values.

    Missing values are retained as NaN because Part C warm-up rows can be
    legitimately unscorable. Nonblank nonnumeric values are rejected.
    """

    as_string = values.astype(
        "string"
    )

    blank = (
        values.isna()
        | as_string.str.strip().eq("")
    )

    numeric = pd.to_numeric(
        values.mask(
            blank,
            np.nan,
        ),
        errors="coerce",
    )

    invalid_text = (
        ~blank
        & numeric.isna()
    )

    if bool(
        invalid_text.any()
    ):
        raise InputValidationError(
            "A saved-model input column contains nonnumeric text.",
            error_code="NON_NUMERIC_INFERENCE_MODEL_INPUT",
            details={
                "kpi_name": kpi_name,
                "identity": identity,
                "column": column_name,
                "invalid_rows": int(
                    invalid_text.sum()
                ),
            },
        )

    return numeric.astype(
        "float64"
    )


def load_and_validate_series_data(
    *,
    series_plan: InferenceSeriesValidationPlan,
    columns: DatasetColumnMapping,
) -> ValidatedSeriesData:
    """
    Load and validate exactly ONE site's Part C rows.

    The parent runner should call this immediately before loading the matching
    saved estimator.

    The function does not choose the final inference timestamp window; that
    remains a parent-runner orchestration decision.

    Row readiness is model-aware:
    - Isolation Forest requires a complete finite saved model feature vector.
    - XGBoost additionally requires a finite observed target because inference
      calculates ``residual = actual_target - predicted_target``.
    """

    if not series_plan.ready_for_model_inference:
        raise InputValidationError(
            "Cannot load model-inference rows for a site currently marked "
            "invalid by Part C.",
            error_code="INFERENCE_SITE_NOT_READY",
            details={
                "kpi_name": (
                    series_plan
                    .artifact
                    .kpi_name
                ),
                "series_key": (
                    series_plan
                    .artifact
                    .series_key
                ),
                "identity": (
                    series_plan
                    .artifact
                    .identity
                ),
                "reason": (
                    series_plan.skip_reason
                ),
            },
        )

    artifact = series_plan.artifact

    (
        required_columns,
        direction_feature,
    ) = _required_series_data_columns(
        series_plan=series_plan,
        columns=columns,
    )

    selected_chunks: list[
        pd.DataFrame
    ] = []

    try:
        reader = pd.read_csv(
            series_plan.feature_file,
            usecols=list(
                required_columns
            ),
            chunksize=SERIES_LOAD_CHUNK_SIZE,
            low_memory=False,
        )

        for chunk in reader:
            identity_frame = pd.DataFrame(
                {
                    "circle": (
                        chunk[
                            columns.circle_column
                        ]
                        .astype("string")
                        .str.strip()
                    ),
                    "cluster": (
                        chunk[
                            columns.cluster_column
                        ]
                        .astype("string")
                        .str.strip()
                    ),
                    "site": (
                        chunk[
                            columns.site_name_column
                        ]
                        .astype("string")
                        .str.strip()
                    ),
                }
            )

            mask = (
                identity_frame[
                    "circle"
                ].eq(
                    artifact.identity[0]
                )
                & identity_frame[
                    "cluster"
                ].eq(
                    artifact.identity[1]
                )
                & identity_frame[
                    "site"
                ].eq(
                    artifact.identity[2]
                )
            )

            if bool(
                mask.any()
            ):
                selected_chunks.append(
                    chunk.loc[
                        mask
                    ].copy()
                )

    except Exception as error:
        raise FileProcessingError(
            "Part C output could not be loaded for one-site model inference.",
            error_code="INFERENCE_SERIES_DATA_LOAD_FAILED",
            details={
                "kpi_name": artifact.kpi_name,
                "series_key": artifact.series_key,
                "identity": artifact.identity,
                "feature_file": str(
                    series_plan.feature_file
                ),
                "reason": str(
                    error
                ),
            },
        ) from error

    if not selected_chunks:
        raise InputValidationError(
            "Validated site series disappeared from the Part C output before "
            "model execution.",
            error_code="INFERENCE_SERIES_DATA_MISSING_AT_EXECUTION",
            details={
                "kpi_name": artifact.kpi_name,
                "series_key": artifact.series_key,
                "identity": artifact.identity,
            },
        )

    dataframe = pd.concat(
        selected_chunks,
        axis=0,
        ignore_index=True,
        sort=False,
    )

    if len(
        dataframe
    ) != series_plan.site_row_count:
        raise InputValidationError(
            "Site row count changed between inference validation and "
            "on-demand model loading.",
            error_code="INFERENCE_SERIES_ROW_COUNT_CHANGED",
            details={
                "kpi_name": artifact.kpi_name,
                "identity": artifact.identity,
                "validated_rows": (
                    series_plan.site_row_count
                ),
                "loaded_rows": len(
                    dataframe
                ),
            },
        )

    timestamps = pd.to_datetime(
        dataframe[
            columns.timestamp_column
        ],
        errors="coerce",
        utc=True,
    )

    if bool(
        timestamps.isna().any()
    ):
        raise InputValidationError(
            "Site data contains invalid timestamps during model execution.",
            error_code="INVALID_INFERENCE_SERIES_TIMESTAMP",
            details={
                "kpi_name": artifact.kpi_name,
                "identity": artifact.identity,
                "invalid_rows": int(
                    timestamps.isna().sum()
                ),
            },
        )

    duplicate_timestamps = (
        timestamps.duplicated(
            keep=False
        )
    )

    if bool(
        duplicate_timestamps.any()
    ):
        raise InputValidationError(
            "One site contains duplicate Part C timestamps.",
            error_code="DUPLICATE_INFERENCE_SERIES_TIMESTAMPS",
            details={
                "kpi_name": artifact.kpi_name,
                "identity": artifact.identity,
                "duplicate_rows": int(
                    duplicate_timestamps.sum()
                ),
            },
        )

    dataframe = dataframe.assign(
        **{
            columns.timestamp_column: timestamps,
        }
    )

    dataframe = dataframe.sort_values(
        by=columns.timestamp_column,
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    for feature_name in artifact.model_feature_columns:
        dataframe[
            feature_name
        ] = _numeric_series(
            dataframe[
                feature_name
            ],
            column_name=feature_name,
            kpi_name=artifact.kpi_name,
            identity=artifact.identity,
        )

    if artifact.target_column is not None:
        dataframe[
            artifact.target_column
        ] = _numeric_series(
            dataframe[
                artifact.target_column
            ],
            column_name=(
                artifact.target_column
            ),
            kpi_name=artifact.kpi_name,
            identity=artifact.identity,
        )

    if (
        direction_feature is not None
        and direction_feature
        not in artifact.model_feature_columns
    ):
        dataframe[
            direction_feature
        ] = _numeric_series(
            dataframe[
                direction_feature
            ],
            column_name=direction_feature,
            kpi_name=artifact.kpi_name,
            identity=artifact.identity,
        )

    feature_matrix = dataframe.loc[
        :,
        list(
            artifact.model_feature_columns
        ),
    ]

    finite_feature_mask = pd.Series(
        np.isfinite(
            feature_matrix.to_numpy(
                dtype=float
            )
        ).all(
            axis=1
        ),
        index=dataframe.index,
        dtype=bool,
    )

    scorable_mask = finite_feature_mask.copy()

    if artifact.target_column is not None:
        finite_target_mask = pd.Series(
            np.isfinite(
                dataframe[
                    artifact.target_column
                ].to_numpy(
                    dtype=float
                )
            ),
            index=dataframe.index,
            dtype=bool,
        )

        scorable_mask &= finite_target_mask

    minimum_timestamp = pd.Timestamp(
        dataframe[
            columns.timestamp_column
        ].min()
    )
    maximum_timestamp = pd.Timestamp(
        dataframe[
            columns.timestamp_column
        ].max()
    )

    return ValidatedSeriesData(
        series_plan=series_plan,
        dataframe=dataframe,
        model_feature_columns=(
            artifact.model_feature_columns
        ),
        target_column=(
            artifact.target_column
        ),
        direction_feature=(
            direction_feature
        ),
        scorable_mask=(
            scorable_mask
        ),
        total_rows=len(
            dataframe
        ),
        scorable_rows=int(
            scorable_mask.sum()
        ),
        unscorable_rows=int(
            (
                ~scorable_mask
            ).sum()
        ),
        minimum_timestamp=(
            minimum_timestamp
        ),
        maximum_timestamp=(
            maximum_timestamp
        ),
    )


__all__ = [
    "InferenceKPIValidationPlan",
    "InferenceSeriesValidationPlan",
    "InferenceValidationResult",
    "UnmodelledPartCSeries",
    "ValidatedSeriesData",
    "load_and_validate_series_data",
    "validate_inference_handoff",
]