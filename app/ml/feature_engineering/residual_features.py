"""Generate residual features for Part C feature engineering.

Intended project location::

    app/ml/feature_engineering/residual_features.py

A residual measures the difference between the selected current KPI source and
its generated seasonal baseline::

    residual(t) = selected_current_source(t) - baseline(t)

The residual request owns two independent source choices:

- ``request.source`` selects the current value used in the subtraction; and
- ``request.baseline.source`` selects the historical source used by the
  baseline algorithm.

This module deliberately does not duplicate seasonal-anchor logic. It delegates
baseline calculation to ``baseline_features.py`` and returns both schema-defined
columns in deterministic order::

    baseline
    residual

General rules:

- input represents one valid site already ordered by timestamp;
- the existing weighted-seasonal baseline generator is reused unchanged;
- a missing current value or missing baseline naturally produces ``NaN``;
- no filling, clipping, or infinity conversion beyond defensive rejection is
  performed here;
- source row count and index are preserved;
- prepared sources and the original Part B dataframe are never modified; and
- ``include_baseline_column`` must remain ``True`` as required by the schema.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from app.schemas.feature_engineering import ResidualFeatureRequest

from .baseline_features import (
    BASELINE_COLUMN,
    generate_weighted_seasonal_baseline,
)
from .source_preparation import PreparedFeatureSources


RESIDUAL_COLUMN: Final = "residual"
RESIDUAL_OUTPUT_COLUMNS: Final = (
    BASELINE_COLUMN,
    RESIDUAL_COLUMN,
)


def _select_residual_current_source(
    *,
    request: ResidualFeatureRequest,
    sources: PreparedFeatureSources,
) -> pd.Series:
    """Return the copied numeric current source used by the residual formula."""

    if not isinstance(sources, PreparedFeatureSources):
        raise TypeError(
            "sources must be an instance of PreparedFeatureSources; received "
            f"{type(sources).__name__}."
        )

    selected = pd.to_numeric(
        sources.bundle.select(request.source),
        errors="coerce",
    ).astype("float64")

    expected_index = sources.bundle.raw_value.index

    if len(selected) != len(expected_index):
        raise ValueError(
            "Selected residual source changed the prepared site row count."
        )

    if not selected.index.equals(expected_index):
        raise ValueError(
            "Selected residual source is not aligned with raw_value."
        )

    finite_values = selected.dropna().to_numpy(dtype="float64")

    if finite_values.size and not np.isfinite(finite_values).all():
        raise ValueError(
            f"Selected residual source {request.source!r} contains positive "
            "or negative infinity."
        )

    return selected.copy()


def _validate_residual_output(
    output: pd.DataFrame,
    *,
    expected_index: pd.Index,
) -> None:
    """Defensively verify the residual module's output contract."""

    if len(output) != len(expected_index):
        raise RuntimeError(
            "Residual-feature generation unexpectedly changed the row count."
        )

    if not output.index.equals(expected_index):
        raise RuntimeError(
            "Residual-feature generation unexpectedly changed the source "
            "index."
        )

    if tuple(output.columns) != RESIDUAL_OUTPUT_COLUMNS:
        raise RuntimeError(
            "Residual-feature generation produced unexpected columns. "
            f"Expected {list(RESIDUAL_OUTPUT_COLUMNS)}, received "
            f"{output.columns.tolist()}."
        )

    if output.columns.has_duplicates:
        duplicates = output.columns[
            output.columns.duplicated()
        ].tolist()
        raise RuntimeError(
            "Residual-feature generation produced duplicate columns: "
            f"{duplicates}"
        )

    numeric = output.apply(
        pd.to_numeric,
        errors="coerce",
    )

    for column in RESIDUAL_OUTPUT_COLUMNS:
        invalid_numeric = (
            output[column].notna()
            & numeric[column].isna()
        )

        if invalid_numeric.any():
            raise RuntimeError(
                f"Residual-feature output '{column}' contains non-numeric "
                "values."
            )

    finite_values = numeric.to_numpy(dtype="float64")
    finite_mask = ~np.isnan(finite_values)

    if finite_mask.any() and not np.isfinite(finite_values[finite_mask]).all():
        raise RuntimeError(
            "Residual-feature generation produced positive or negative "
            "infinity."
        )


def generate_residual_features(
    request: ResidualFeatureRequest,
    *,
    sources: PreparedFeatureSources,
    site_dataframe: pd.DataFrame,
    timestamp_column: str,
    bucket_minutes: int,
) -> pd.DataFrame:
    """Generate the schema-defined baseline and residual columns for one site.

    Formula::

        residual(t) = selected_current_source(t) - baseline(t)

    ``request.source`` and ``request.baseline.source`` are independent. For
    example, the current value may use ``raw_value`` while the baseline uses a
    separately configured historical source.

    Args:
        request:
            Validated ``ResidualFeatureRequest``.
        sources:
            Prepared raw and optional corrected-history sources for one valid
            timestamp-ordered site.
        site_dataframe:
            The same site dataframe used to prepare ``sources``.
        timestamp_column:
            Resolved physical timestamp-column name.
        bucket_minutes:
            Resolved dataset frequency in whole minutes.

    Returns:
        pd.DataFrame:
            Two ``float64`` columns, ``baseline`` then ``residual``, aligned
            with the prepared site index.
    """

    if not isinstance(request, ResidualFeatureRequest):
        raise TypeError(
            "generate_residual_features requires ResidualFeatureRequest; "
            f"received {type(request).__name__}."
        )

    if request.include_baseline_column is not True:
        raise ValueError(
            "ResidualFeatureRequest.include_baseline_column must remain True "
            "because the current schema requires both baseline and residual "
            "output columns."
        )

    current_values = _select_residual_current_source(
        request=request,
        sources=sources,
    )

    baseline_result = generate_weighted_seasonal_baseline(
        request.baseline,
        sources=sources,
        site_dataframe=site_dataframe,
        timestamp_column=timestamp_column,
        bucket_minutes=bucket_minutes,
    )

    baseline_values = pd.to_numeric(
        baseline_result.values,
        errors="coerce",
    ).astype("float64")

    if not baseline_values.index.equals(current_values.index):
        raise RuntimeError(
            "Generated baseline is not aligned with the selected residual "
            "current source."
        )

    residual_values = (
        current_values - baseline_values
    ).astype("float64")
    residual_values.name = RESIDUAL_COLUMN

    # The baseline generator already guarantees finite-or-NaN output. Keep a
    # defensive replacement so an unexpected arithmetic overflow can never
    # escape as an infinite model feature.
    residual_values = residual_values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    baseline_values = baseline_values.copy()
    baseline_values.name = BASELINE_COLUMN

    output = pd.concat(
        [baseline_values, residual_values],
        axis=1,
        copy=False,
    )

    _validate_residual_output(
        output,
        expected_index=current_values.index,
    )

    return output


__all__ = [
    "RESIDUAL_COLUMN",
    "RESIDUAL_OUTPUT_COLUMNS",
    "generate_residual_features",
]