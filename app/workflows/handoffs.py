"""
Persisted A->B->C->D handoff contracts.

Historical integration tests passed rich Python result objects directly in one
process:

A -> B:
    KPIEDARunResult
    + DatasetInputConfig
    + Part A resolved BucketConfig

B -> C:
    PreprocessingRunnerResult
    + DatasetInputConfig
    + same resolved BucketConfig

C -> D:
    final_dataset_file
    + resolved Part C feature contract
    + minimum_history_points
    + DatasetColumnMapping
    + training request/config supplied by the Part D workflow

Production requests may be separated by minutes, days, or process restarts.
Therefore successful stages persist JSON-safe snapshots and this module
reconstructs the exact A/B/C runner result objects needed by the next stage.

Part E is intentionally excluded. D -> inference_bootstrap -> Part E is a
persistent saved-run lifecycle, not a normal result-object handoff.
"""

from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel

from app.core.exceptions import (
    ArtifactResolutionError,
    RunManagementError,
)
from app.ml.eda.runner import (
    EDARunnerResult,
    KPIEDARunResult,
)
from app.ml.feature_engineering import (
    FeatureEngineeringRunnerResult,
)
from app.ml.preprocessing.runner import (
    PreprocessingRunnerResult,
)
from app.schemas.common.dataset import (
    BucketConfig,
    DatasetColumnMapping,
    DatasetInputConfig,
)
from app.schemas.feature_engineering import (
    ResolvedFeatureEngineeringConfig,
)

from .common import (
    WORKFLOW_HANDOFF_SCHEMA_VERSION,
    load_json_object,
    normalize_kpi_name,
    to_json_safe,
)


@dataclass(frozen=True, slots=True)
class PartAHandoff:
    """Reconstructed state required by the Part B workflow."""

    dataset_config: DatasetInputConfig
    eda_result: EDARunnerResult

    @property
    def bucket(
        self,
    ) -> BucketConfig:
        return self.eda_result.preparation.bucket

    @property
    def kpi_results(
        self,
    ) -> tuple[KPIEDARunResult, ...]:
        return self.eda_result.kpi_results


@dataclass(frozen=True, slots=True)
class PartBHandoff:
    """Reconstructed state required by the Part C workflow."""

    dataset_config: DatasetInputConfig
    bucket: BucketConfig
    preprocessing_results: tuple[
        PreprocessingRunnerResult,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class PartCHandoff:
    """Reconstructed state required by the Part D workflow."""

    dataset_config: DatasetInputConfig
    feature_engineering_results: tuple[
        FeatureEngineeringRunnerResult,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class PartDRunnerInput:
    """
    C-derived arguments consumed by one Part D run_training() call.

    model_type and training_config deliberately do not live here because they
    belong to the Part D stage request.
    """

    kpi_name: str
    input_file: Path
    dataset_columns: DatasetColumnMapping
    feature_config: ResolvedFeatureEngineeringConfig
    minimum_history_points: int
    metadata: Mapping[str, Any]


def _snapshot_envelope(
    *,
    stage: Literal[
        "part_a",
        "part_b",
        "part_c",
    ],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "handoff_schema_version": (
            WORKFLOW_HANDOFF_SCHEMA_VERSION
        ),
        "stage": stage,
        "payload": to_json_safe(
            dict(
                payload
            )
        ),
    }


def build_part_a_result_snapshot(
    *,
    dataset_config: DatasetInputConfig,
    eda_result: EDARunnerResult,
) -> dict[str, Any]:
    """
    Build the result.json payload persisted after successful Part A.

    dataset_config is the runtime config whose input_file points to the
    attempt-owned immutable source copy.
    """
    return _snapshot_envelope(
        stage="part_a",
        payload={
            "dataset_config": (
                dataset_config
            ),
            "eda_result": (
                eda_result
            ),
        },
    )


def build_part_b_result_snapshot(
    *,
    dataset_config: DatasetInputConfig,
    bucket: BucketConfig,
    preprocessing_results: Sequence[
        PreprocessingRunnerResult,
    ],
) -> dict[str, Any]:
    """
    Build the result.json payload persisted after all KPI Part B runs succeed.
    """
    results = tuple(
        preprocessing_results
    )

    _validate_unique_kpi_results(
        results,
        role="Part B",
    )

    return _snapshot_envelope(
        stage="part_b",
        payload={
            "dataset_config": (
                dataset_config
            ),
            "bucket": bucket,
            "preprocessing_results": (
                results
            ),
        },
    )


def build_part_c_result_snapshot(
    *,
    dataset_config: DatasetInputConfig,
    feature_engineering_results: Sequence[
        FeatureEngineeringRunnerResult,
    ],
) -> dict[str, Any]:
    """
    Build the result.json payload persisted after all KPI Part C runs succeed.
    """
    results = tuple(
        feature_engineering_results
    )

    _validate_unique_kpi_results(
        results,
        role="Part C",
    )

    return _snapshot_envelope(
        stage="part_c",
        payload={
            "dataset_config": (
                dataset_config
            ),
            "feature_engineering_results": (
                results
            ),
        },
    )


def _unwrap_snapshot(
    *,
    result_file: str | Path,
    expected_stage: Literal[
        "part_a",
        "part_b",
        "part_c",
    ],
) -> dict[str, Any]:
    payload = load_json_object(
        result_file
    )

    received_version = payload.get(
        "handoff_schema_version"
    )

    if (
        received_version
        != WORKFLOW_HANDOFF_SCHEMA_VERSION
    ):
        raise ArtifactResolutionError(
            "The persisted workflow handoff uses an unsupported schema version.",
            error_code="UNSUPPORTED_WORKFLOW_HANDOFF_VERSION",
            details={
                "result_file": str(
                    Path(
                        result_file
                    )
                    .expanduser()
                    .resolve()
                ),
                "expected_version": (
                    WORKFLOW_HANDOFF_SCHEMA_VERSION
                ),
                "received_version": (
                    received_version
                ),
            },
        )

    received_stage = payload.get(
        "stage"
    )

    if received_stage != expected_stage:
        raise ArtifactResolutionError(
            "The persisted workflow handoff belongs to a different stage.",
            error_code="WORKFLOW_HANDOFF_STAGE_MISMATCH",
            details={
                "expected_stage": (
                    expected_stage
                ),
                "received_stage": (
                    received_stage
                ),
            },
        )

    stage_payload = payload.get(
        "payload"
    )

    if not isinstance(
        stage_payload,
        dict,
    ):
        raise ArtifactResolutionError(
            "The persisted workflow handoff payload is invalid.",
            error_code="INVALID_WORKFLOW_HANDOFF_PAYLOAD",
            details={
                "expected_stage": (
                    expected_stage
                ),
            },
        )

    return stage_payload


def _strip_annotated(
    expected_type: Any,
) -> Any:
    origin = get_origin(
        expected_type
    )

    if origin is Annotated:
        args = get_args(
            expected_type
        )

        if args:
            return args[0]

    return expected_type


def _deserialize_value(
    value: Any,
    expected_type: Any,
) -> Any:
    """
    Reconstruct a JSON value using the target dataclass/Pydantic annotation.

    The algorithm is intentionally annotation-driven. It does not persist or
    import arbitrary class names from JSON, avoiding a type-injection surface.
    """
    expected_type = _strip_annotated(
        expected_type
    )

    if expected_type is Any:
        return value

    origin = get_origin(
        expected_type
    )
    args = get_args(
        expected_type
    )

    if value is None:
        if (
            origin
            in (
                Union,
                types.UnionType,
            )
            and type(
                None
            )
            in args
        ):
            return None

        raise TypeError(
            "A required persisted workflow value is null."
        )

    if origin in (
        Union,
        types.UnionType,
    ):
        non_none = [
            item
            for item in args
            if item is not type(
                None
            )
        ]

        if len(
            non_none
        ) == 1:
            return _deserialize_value(
                value,
                non_none[0],
            )

        last_error: Exception | None = None

        for candidate in non_none:
            try:
                return _deserialize_value(
                    value,
                    candidate,
                )
            except (
                TypeError,
                ValueError,
                KeyError,
            ) as error:
                last_error = error

        raise TypeError(
            "Persisted workflow value does not match any allowed union type."
        ) from last_error

    if origin is Literal:
        if value not in args:
            raise ValueError(
                "Persisted workflow value does not match the expected literal."
            )

        return value

    if origin in (
        list,
        Sequence,
    ):
        element_type = (
            args[0]
            if args
            else Any
        )

        if not isinstance(
            value,
            list,
        ):
            raise TypeError(
                "Expected a persisted JSON array."
            )

        return [
            _deserialize_value(
                item,
                element_type,
            )
            for item in value
        ]

    if origin is tuple:
        if not isinstance(
            value,
            list,
        ):
            raise TypeError(
                "Expected a persisted JSON array for a tuple."
            )

        if (
            len(
                args
            ) == 2
            and args[1] is Ellipsis
        ):
            return tuple(
                _deserialize_value(
                    item,
                    args[0],
                )
                for item in value
            )

        if args:
            if len(
                value
            ) != len(
                args
            ):
                raise ValueError(
                    "Persisted tuple length does not match its annotation."
                )

            return tuple(
                _deserialize_value(
                    item,
                    item_type,
                )
                for item, item_type
                in zip(
                    value,
                    args,
                    strict=True,
                )
            )

        return tuple(
            value
        )

    if origin in (
        set,
        frozenset,
    ):
        if not isinstance(
            value,
            list,
        ):
            raise TypeError(
                "Expected a persisted JSON array for a set."
            )

        element_type = (
            args[0]
            if args
            else Any
        )

        items = [
            _deserialize_value(
                item,
                element_type,
            )
            for item in value
        ]

        return (
            frozenset(
                items
            )
            if origin is frozenset
            else set(
                items
            )
        )

    if origin in (
        dict,
        Mapping,
    ):
        if not isinstance(
            value,
            dict,
        ):
            raise TypeError(
                "Expected a persisted JSON object."
            )

        key_type = (
            args[0]
            if args
            else Any
        )
        item_type = (
            args[1]
            if len(
                args
            ) > 1
            else Any
        )

        return {
            _deserialize_value(
                key,
                key_type,
            ): _deserialize_value(
                item,
                item_type,
            )
            for key, item in value.items()
        }

    if expected_type is Path:
        return Path(
            value
        ).expanduser().resolve()

    if expected_type is datetime:
        return datetime.fromisoformat(
            str(
                value
            )
        )

    if expected_type is date:
        return date.fromisoformat(
            str(
                value
            )
        )

    if (
        isinstance(
            expected_type,
            type,
        )
        and issubclass(
            expected_type,
            Enum,
        )
    ):
        return expected_type(
            value
        )

    if (
        isinstance(
            expected_type,
            type,
        )
        and issubclass(
            expected_type,
            BaseModel,
        )
    ):
        return expected_type.model_validate(
            value
        )

    if (
        isinstance(
            expected_type,
            type,
        )
        and is_dataclass(
            expected_type
        )
    ):
        if not isinstance(
            value,
            dict,
        ):
            raise TypeError(
                "Expected a JSON object for a persisted dataclass."
            )

        type_hints = get_type_hints(
            expected_type
        )

        constructor: dict[
            str,
            Any,
        ] = {}

        for field in fields(
            expected_type
        ):
            if field.name not in value:
                raise KeyError(
                    f"Persisted dataclass field is missing: {field.name}"
                )

            constructor[
                field.name
            ] = _deserialize_value(
                value[
                    field.name
                ],
                type_hints.get(
                    field.name,
                    field.type,
                ),
            )

        return expected_type(
            **constructor
        )

    if isinstance(
        expected_type,
        type,
    ):
        module_name = (
            expected_type.__module__
        )
        type_name = (
            expected_type.__name__
        )

        if (
            module_name.startswith(
                "pandas"
            )
            and type_name
            in {
                "Timestamp",
                "Timedelta",
            }
        ):
            return expected_type(
                value
            )

        if expected_type in (
            str,
            int,
            float,
            bool,
        ):
            return expected_type(
                value
            )

    # Type aliases such as collections.abc.Mapping can arrive here depending
    # on Python's resolved annotations.
    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            dict,
            list,
        ),
    ):
        return value

    raise TypeError(
        "Unsupported workflow handoff target type: "
        f"{expected_type!r}"
    )


def _deserialize_or_raise(
    value: Any,
    expected_type: Any,
    *,
    role: str,
) -> Any:
    try:
        return _deserialize_value(
            value,
            expected_type,
        )
    except Exception as error:
        raise ArtifactResolutionError(
            "A persisted workflow handoff could not be reconstructed.",
            error_code="WORKFLOW_HANDOFF_RECONSTRUCTION_FAILED",
            details={
                "role": role,
                "target_type": (
                    getattr(
                        expected_type,
                        "__name__",
                        repr(
                            expected_type
                        ),
                    )
                ),
                "original_error": str(
                    error
                ),
            },
        ) from error


def _validate_path_containment(
    value: Any,
    *,
    run_directory: str | Path,
    role: str,
) -> None:
    """
    Reject reconstructed Path values escaping the owning run root.

    This protects path fields nested inside runner result.json, complementing
    Run Management's containment check on the result.json artifact itself.
    """
    run_root = Path(
        run_directory
    ).expanduser().resolve()

    def visit(
        item: Any,
    ) -> None:
        if item is None:
            return

        if isinstance(
            item,
            Path,
        ):
            resolved = item.expanduser().resolve()

            try:
                resolved.relative_to(
                    run_root
                )
            except ValueError as error:
                raise ArtifactResolutionError(
                    "A reconstructed workflow handoff path escapes the run directory.",
                    error_code="WORKFLOW_HANDOFF_PATH_OUTSIDE_RUN",
                    details={
                        "role": role,
                        "run_directory": str(
                            run_root
                        ),
                        "path": str(
                            resolved
                        ),
                    },
                ) from error

            return

        if isinstance(
            item,
            BaseModel,
        ):
            for field_name in item.model_fields:
                visit(
                    getattr(
                        item,
                        field_name,
                    )
                )

            return

        if is_dataclass(
            item
        ) and not isinstance(
            item,
            type,
        ):
            for field in fields(
                item
            ):
                visit(
                    getattr(
                        item,
                        field.name,
                    )
                )

            return

        if isinstance(
            item,
            Mapping,
        ):
            for child in item.values():
                visit(
                    child
                )

            return

        if isinstance(
            item,
            (
                tuple,
                list,
                set,
                frozenset,
            ),
        ):
            for child in item:
                visit(
                    child
                )

    visit(
        value
    )


def _validate_unique_kpi_results(
    results: Sequence[Any],
    *,
    role: str,
) -> None:
    seen: set[str] = set()

    for result in results:
        if not hasattr(
            result,
            "kpi_name",
        ):
            raise RunManagementError(
                "A workflow result does not expose kpi_name.",
                error_code="WORKFLOW_RESULT_KPI_NAME_MISSING",
                details={
                    "role": role,
                    "received_type": (
                        type(
                            result
                        ).__name__
                    ),
                },
            )

        kpi_name = normalize_kpi_name(
            getattr(
                result,
                "kpi_name",
            )
        )

        if kpi_name in seen:
            raise RunManagementError(
                "A workflow stage produced more than one result for the same KPI.",
                error_code="DUPLICATE_WORKFLOW_KPI_RESULT",
                details={
                    "role": role,
                    "kpi_name": (
                        kpi_name
                    ),
                },
            )

        seen.add(
            kpi_name
        )


def _require_nonempty_results(
    results: Sequence[Any],
    *,
    role: str,
) -> None:
    if not results:
        raise ArtifactResolutionError(
            "A persisted workflow handoff contains no KPI results.",
            error_code="EMPTY_WORKFLOW_KPI_RESULTS",
            details={
                "role": role,
            },
        )


def load_part_a_handoff(
    *,
    result_file: str | Path,
    run_directory: str | Path,
) -> PartAHandoff:
    """Reconstruct the exact Part A state consumed by Part B."""
    payload = _unwrap_snapshot(
        result_file=result_file,
        expected_stage="part_a",
    )

    dataset_config = _deserialize_or_raise(
        payload.get(
            "dataset_config"
        ),
        DatasetInputConfig,
        role="Part A dataset config",
    )

    eda_result = _deserialize_or_raise(
        payload.get(
            "eda_result"
        ),
        EDARunnerResult,
        role="Part A EDARunnerResult",
    )

    _require_nonempty_results(
        eda_result.kpi_results,
        role="Part A",
    )

    _validate_path_containment(
        dataset_config,
        run_directory=(
            run_directory
        ),
        role="Part A dataset config",
    )
    _validate_path_containment(
        eda_result,
        run_directory=(
            run_directory
        ),
        role="Part A result",
    )

    return PartAHandoff(
        dataset_config=(
            dataset_config
        ),
        eda_result=eda_result,
    )


def load_part_b_handoff(
    *,
    result_file: str | Path,
    run_directory: str | Path,
) -> PartBHandoff:
    """Reconstruct the exact Part B results consumed by Part C."""
    payload = _unwrap_snapshot(
        result_file=result_file,
        expected_stage="part_b",
    )

    dataset_config = _deserialize_or_raise(
        payload.get(
            "dataset_config"
        ),
        DatasetInputConfig,
        role="Part B dataset config",
    )

    bucket = _deserialize_or_raise(
        payload.get(
            "bucket"
        ),
        BucketConfig,
        role="Part B resolved bucket",
    )

    raw_results = payload.get(
        "preprocessing_results"
    )

    results = _deserialize_or_raise(
        raw_results,
        tuple[
            PreprocessingRunnerResult,
            ...,
        ],
        role="Part B preprocessing results",
    )

    _require_nonempty_results(
        results,
        role="Part B",
    )
    _validate_unique_kpi_results(
        results,
        role="Part B",
    )

    _validate_path_containment(
        dataset_config,
        run_directory=(
            run_directory
        ),
        role="Part B dataset config",
    )
    _validate_path_containment(
        results,
        run_directory=(
            run_directory
        ),
        role="Part B results",
    )

    return PartBHandoff(
        dataset_config=(
            dataset_config
        ),
        bucket=bucket,
        preprocessing_results=(
            results
        ),
    )


def load_part_c_handoff(
    *,
    result_file: str | Path,
    run_directory: str | Path,
) -> PartCHandoff:
    """Reconstruct the exact Part C results consumed by Part D."""
    payload = _unwrap_snapshot(
        result_file=result_file,
        expected_stage="part_c",
    )

    dataset_config = _deserialize_or_raise(
        payload.get(
            "dataset_config"
        ),
        DatasetInputConfig,
        role="Part C dataset config",
    )

    results = _deserialize_or_raise(
        payload.get(
            "feature_engineering_results"
        ),
        tuple[
            FeatureEngineeringRunnerResult,
            ...,
        ],
        role="Part C feature-engineering results",
    )

    _require_nonempty_results(
        results,
        role="Part C",
    )
    _validate_unique_kpi_results(
        results,
        role="Part C",
    )

    _validate_path_containment(
        dataset_config,
        run_directory=(
            run_directory
        ),
        role="Part C dataset config",
    )
    _validate_path_containment(
        results,
        run_directory=(
            run_directory
        ),
        role="Part C results",
    )

    return PartCHandoff(
        dataset_config=(
            dataset_config
        ),
        feature_engineering_results=(
            results
        ),
    )


def build_part_d_runner_inputs(
    handoff: PartCHandoff,
) -> tuple[
    PartDRunnerInput,
    ...,
]:
    """
    Convert reconstructed Part C state into the C-derived Part D call contract.

    The future Part D workflow supplies:
        model_type=<Part D request>
        training_config=<Part D request>
        output_directory=<execution KPI output>

    This helper supplies exactly the remaining arguments proven by the tested
    A->D integration pipelines.
    """
    items: list[
        PartDRunnerInput
    ] = []

    for result in (
        handoff
        .feature_engineering_results
    ):
        kpi_name = normalize_kpi_name(
            result.kpi_name
        )

        minimum_history_points = int(
            result
            .resolved_config
            .minimum_history_points
        )

        if minimum_history_points < 1:
            raise ArtifactResolutionError(
                "Part C returned an invalid minimum-history contract.",
                error_code="INVALID_PART_C_MINIMUM_HISTORY",
                details={
                    "kpi_name": (
                        kpi_name
                    ),
                    "minimum_history_points": (
                        minimum_history_points
                    ),
                },
            )

        if (
            tuple(
                result.feature_columns
            )
            != tuple(
                result
                .resolved_config
                .feature_columns
            )
        ):
            raise ArtifactResolutionError(
                "Part C feature_columns do not match its resolved feature contract.",
                error_code="PART_C_FEATURE_CONTRACT_MISMATCH",
                details={
                    "kpi_name": (
                        kpi_name
                    ),
                },
            )

        items.append(
            PartDRunnerInput(
                kpi_name=kpi_name,
                input_file=Path(
                    result
                    .final_dataset_file
                )
                .expanduser()
                .resolve(),
                dataset_columns=(
                    handoff
                    .dataset_config
                    .columns
                ),
                feature_config=(
                    result
                    .resolved_config
                ),
                minimum_history_points=(
                    minimum_history_points
                ),
                metadata={
                    "kpi_name": (
                        kpi_name
                    ),
                    "part_c_output_directory": str(
                        result
                        .output_directory
                    ),
                    "part_c_final_dataset": str(
                        result
                        .final_dataset_file
                    ),
                },
            )
        )

    return tuple(
        items
    )


def preprocessing_results_by_kpi(
    handoff: PartBHandoff,
) -> dict[
    str,
    PreprocessingRunnerResult,
]:
    """Index Part B results by original KPI name without altering the result."""
    return {
        normalize_kpi_name(
            result.kpi_name
        ): result
        for result in (
            handoff
            .preprocessing_results
        )
    }


def feature_results_by_kpi(
    handoff: PartCHandoff,
) -> dict[
    str,
    FeatureEngineeringRunnerResult,
]:
    """Index Part C results by original KPI name without altering the result."""
    return {
        normalize_kpi_name(
            result.kpi_name
        ): result
        for result in (
            handoff
            .feature_engineering_results
        )
    }


__all__ = [
    "PartAHandoff",
    "PartBHandoff",
    "PartCHandoff",
    "PartDRunnerInput",
    "build_part_a_result_snapshot",
    "build_part_b_result_snapshot",
    "build_part_c_result_snapshot",
    "load_part_a_handoff",
    "load_part_b_handoff",
    "load_part_c_handoff",
    "build_part_d_runner_inputs",
    "preprocessing_results_by_kpi",
    "feature_results_by_kpi",
]
