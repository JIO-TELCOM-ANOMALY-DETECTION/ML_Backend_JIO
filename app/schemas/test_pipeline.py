"""
Define the temporary API request schema for running Part A EDA
and Part B preprocessing through FastAPI.

This module combines the existing dataset and preprocessing schemas.
It does not execute the pipeline or contain processing logic.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from app.schemas.common.dataset import DatasetInputConfig
from app.schemas.preprocessing import PreprocessingConfig


class PipelineRunRequest(BaseModel):
    """
    Store all validated inputs required for one temporary pipeline run.

    The request contains:

    - the source dataset path and dataset mappings;
    - all Part B preprocessing settings; and
    - the root directory where EDA and preprocessing outputs are written.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    dataset: DatasetInputConfig = Field(
        description=(
            "Dataset path, column mapping, bucket information, KPI identity, "
            "and expected timeline settings."
        ),
    )

    preprocessing: PreprocessingConfig = Field(
        description=(
            "Complete Part B preprocessing configuration."
        ),
    )

    output_directory: Path = Field(
        description=(
            "Root directory where the workflow should save EDA "
            "and preprocessing outputs."
        ),
    )

    @field_validator(
        "output_directory",
        mode="before",
    )
    @classmethod
    def normalise_output_directory(
        cls,
        value: str | Path,
    ) -> Path:
        """
        Convert the supplied output-directory value into a normalized Path.

        Args:
            value:
                Output directory supplied through JSON.

        Returns:
            Path:
                Normalized output-directory path.
        """

        return Path(
            value
        ).expanduser()


__all__ = [
    "PipelineRunRequest",
]