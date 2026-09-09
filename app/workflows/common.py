"""
Shared utilities for production A-D workflows.

This module contains workflow-level mechanics that are common to several stage
workflows but are not run-management responsibilities and are not ML logic.

The historical tested A->B->C->D pipelines established two important rules
preserved here:

1. Part A must run against the immutable attempt-owned input snapshot.
2. When Part C configuration is omitted in a training workflow, its default
   profile is selected from Part D's model_type.

Actual A/B/C/D execution remains in the stage-specific workflow files.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import (
    ArtifactResolutionError,
    RunManagementError,
)
from app.schemas.common.dataset import DatasetInputConfig
from app.schemas.feature_engineering import FeatureEngineeringConfig
from app.schemas.training.common import TrainingModelType
from app.services.file_service import (
    read_json_file,
    sanitise_filename,
)


WORKFLOW_HANDOFF_SCHEMA_VERSION = settings.workflow_schema_version


def build_runtime_dataset_config(
    dataset_config: DatasetInputConfig,
    *,
    input_file: str | Path,
) -> DatasetInputConfig:
    """
    Rebuild DatasetInputConfig so Part A reads the attempt-owned input snapshot.

    The caller's original file path must not remain authoritative after Run
    Management has copied the source into the immutable attempt directory.
    """
    payload = dataset_config.model_dump(
        mode="python",
    )
    payload["input_file"] = Path(
        input_file
    ).expanduser().resolve()

    return DatasetInputConfig.model_validate(
        payload
    )


def resolve_feature_engineering_config(
    feature_config: FeatureEngineeringConfig | None,
    *,
    model_type: TrainingModelType,
) -> FeatureEngineeringConfig:
    """
    Resolve the Part C configuration exactly as the tested A->D pipelines did.

    Explicit Part C configuration is preserved unchanged. Only an omitted
    configuration selects the model-matched default profile.
    """
    if feature_config is not None:
        return feature_config

    return FeatureEngineeringConfig(
        default_profile=model_type
    )


def normalize_kpi_name(
    value: object,
) -> str:
    """
    Return one real, non-blank KPI string.

    Non-string values are rejected rather than converted into misleading names
    such as ``"None"`` or ``"123"``.
    """
    if not isinstance(
        value,
        str,
    ):
        raise RunManagementError(
            "A workflow result contains an invalid KPI name type.",
            error_code="INVALID_WORKFLOW_KPI_NAME",
            details={
                "received_type": (
                    type(
                        value
                    ).__name__
                ),
            },
        )

    normalized = value.strip()

    if not normalized:
        raise RunManagementError(
            "A workflow result contains a blank KPI name.",
            error_code="BLANK_WORKFLOW_KPI_NAME",
        )

    return normalized


def safe_kpi_directory_name(
    kpi_name: object,
    *,
    fallback_index: int,
) -> str:
    """
    Produce the same safe KPI directory convention used by the tested pipelines.
    """
    normalized = normalize_kpi_name(
        kpi_name
    )

    safe_name = sanitise_filename(
        normalized
    )

    if safe_name:
        return safe_name

    if fallback_index < 1:
        raise RunManagementError(
            "fallback_index must be at least one.",
            error_code="INVALID_KPI_FALLBACK_INDEX",
            details={
                "fallback_index": fallback_index,
            },
        )

    return f"kpi_{fallback_index}"


def to_json_safe(
    value: Any,
) -> Any:
    """
    Convert workflow/module result objects into strict JSON-safe values.

    Supported project values include:
    - Pydantic models;
    - dataclasses;
    - Path;
    - datetime/date;
    - Enum;
    - pandas Timestamp / Timedelta scalar values;
    - NumPy scalar values;
    - mappings and normal collections.

    Non-finite floats and opaque pandas/runtime objects are rejected instead of
    being silently stringified. This matters because result.json is later
    reconstructed into exact A/B/C runner contracts.
    """
    if value is None:
        return None

    if isinstance(
        value,
        BaseModel,
    ):
        return to_json_safe(
            value.model_dump(
                mode="json"
            )
        )

    if is_dataclass(
        value
    ) and not isinstance(
        value,
        type,
    ):
        return {
            field.name: to_json_safe(
                getattr(
                    value,
                    field.name,
                )
            )
            for field in fields(
                value
            )
        }

    if isinstance(
        value,
        Path,
    ):
        return str(
            value.expanduser().resolve()
        )

    if isinstance(
        value,
        datetime,
    ):
        return value.isoformat()

    if isinstance(
        value,
        date,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Enum,
    ):
        return to_json_safe(
            value.value
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): to_json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (tuple, list),
    ):
        return [
            to_json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        (set, frozenset),
    ):
        return [
            to_json_safe(
                item
            )
            for item in sorted(
                value,
                key=lambda item: str(
                    item
                ),
            )
        ]

    if isinstance(
        value,
        float,
    ):
        if not isfinite(
            value
        ):
            raise RunManagementError(
                "A workflow result contains a non-finite float.",
                error_code="NON_FINITE_WORKFLOW_FLOAT",
                details={
                    "value": repr(
                        value
                    ),
                },
            )

        return value

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    value_type = type(
        value
    )
    module_name = (
        value_type.__module__
    )
    type_name = (
        value_type.__name__
    )

    # Only the pandas scalar types used by current runner result contracts are
    # accepted. DataFrame/Series and other pandas objects must not be silently
    # stringified into a persistence snapshot.
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
        if hasattr(
            value,
            "isoformat",
        ):
            return value.isoformat()

        return str(
            value
        )

    if module_name.startswith(
        "pandas"
    ):
        raise RunManagementError(
            "A workflow result contains an unsupported pandas object.",
            error_code="UNSUPPORTED_WORKFLOW_PANDAS_OBJECT",
            details={
                "received_type": (
                    f"{module_name}."
                    f"{value_type.__qualname__}"
                ),
            },
        )

    # Convert NumPy scalar values to their Python scalar equivalent. The
    # recursive call then applies finite-float validation as well.
    if (
        module_name.startswith(
            "numpy"
        )
        and hasattr(
            value,
            "item",
        )
    ):
        return to_json_safe(
            value.item()
        )

    raise RunManagementError(
        "A workflow result contains a value that cannot be persisted safely.",
        error_code="UNSUPPORTED_WORKFLOW_RESULT_VALUE",
        details={
            "received_type": (
                f"{module_name}."
                f"{value_type.__qualname__}"
            ),
        },
    )


def load_json_object(
    file_path: str | Path,
) -> dict[str, Any]:
    """Load one persisted workflow result and require a top-level JSON object."""
    path = Path(
        file_path
    ).expanduser().resolve()

    payload = read_json_file(
        path
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ArtifactResolutionError(
            "A persisted workflow handoff must contain a JSON object.",
            error_code="INVALID_WORKFLOW_HANDOFF_JSON",
            details={
                "result_file": str(
                    path
                ),
                "received_type": (
                    type(
                        payload
                    ).__name__
                ),
            },
        )

    return payload


__all__ = [
    "WORKFLOW_HANDOFF_SCHEMA_VERSION",
    "build_runtime_dataset_config",
    "resolve_feature_engineering_config",
    "normalize_kpi_name",
    "safe_kpi_directory_name",
    "to_json_safe",
    "load_json_object",
]
