"""Generate direct value features for Part C feature engineering.

Intended project location::

    app/ml/feature_engineering/value_features.py

This module currently supports only ``current_value``.

Rules:

- ``current_value`` is always copied from the original Part B KPI value;
- corrected history and local median cleaning are never used;
- outliers and missing values remain exactly as supplied by Part B;
- the original KPI column is never modified; and
- output row count and index remain aligned with the site dataframe.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
from pandas.api.types import is_numeric_dtype

from app.schemas.feature_engineering import CurrentValueFeatureRequest

from .source_preparation import PreparedFeatureSources


CURRENT_VALUE_COLUMN: Final = "current_value"


def generate_value_features(
    request: CurrentValueFeatureRequest,
    *,
    sources: PreparedFeatureSources,
) -> pd.DataFrame:
    """Generate the unchanged current KPI value for one valid site.

    Args:
        request:
            Validated ``current_value`` feature request. The schema deliberately
            exposes no source or cleaning settings for this feature.
        sources:
            Prepared source bundle for the timestamp-ordered site.

    Returns:
        pd.DataFrame:
            One ``float64`` column named ``current_value`` aligned with the
            original site index.

    Raises:
        TypeError:
            If a different feature-request model is supplied.
        ValueError:
            If the prepared raw source is not a one-dimensional, numeric,
            index-aligned Series.
    """

    if not isinstance(request, CurrentValueFeatureRequest):
        raise TypeError(
            "value_features.py supports only CurrentValueFeatureRequest; "
            f"received {type(request).__name__}."
        )

    raw_value = sources.bundle.raw_value

    if not isinstance(raw_value, pd.Series) or raw_value.ndim != 1:
        raise ValueError(
            "Prepared raw_value must be a one-dimensional pandas Series."
        )

    if not is_numeric_dtype(raw_value.dtype):
        raise ValueError(
            "Prepared raw_value must already be numeric before value-feature "
            "generation."
        )

    # Copy only. Do not clean, fill, clip, shift, or otherwise alter Part B
    # values. Converting to float64 matches the Part C numerical feature policy.
    current_value = raw_value.astype("float64", copy=True)
    current_value.name = CURRENT_VALUE_COLUMN

    output = current_value.to_frame()

    if len(output) != len(raw_value):
        raise RuntimeError(
            "Generating current_value unexpectedly changed the row count."
        )

    if not output.index.equals(raw_value.index):
        raise RuntimeError(
            "Generating current_value unexpectedly changed the source index."
        )

    return output


__all__ = [
    "CURRENT_VALUE_COLUMN",
    "generate_value_features",
]
