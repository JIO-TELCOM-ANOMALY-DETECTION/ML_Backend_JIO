"""
Connect the completed Part A EDA runner with the completed
Part B preprocessing runner for temporary server testing.

This workflow:

1. receives the validated pipeline request;
2. creates separate EDA and preprocessing output roots;
3. runs Part A EDA once;
4. loops through every KPI result returned by Part A;
5. runs Part B preprocessing once for each KPI; and
6. returns all structured runner results.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from pathlib import Path

from app.ml.eda.runner import (
    EDARunnerResult,
    run_eda_pipeline,
)
from app.ml.preprocessing.runner import (
    PreprocessingRunnerResult,
    run_preprocessing_pipeline,
)
from app.schemas.test_pipeline import PipelineRunRequest
from app.services.file_service import (
    ensure_directory,
    sanitise_filename,
)


@dataclass(frozen=True, slots=True)
class KPIWorkflowResult:
    """
    Store the preprocessing result for one KPI.

    Attributes:
        kpi_name:
            Original KPI name returned by Part A.

        preprocessing_result:
            Complete structured result returned by the Part B runner.
    """

    kpi_name: str
    preprocessing_result: PreprocessingRunnerResult


@dataclass(frozen=True, slots=True)
class TestPipelineWorkflowResult:
    """
    Store the complete result of the temporary pipeline workflow.

    Attributes:
        output_directory:
            Root output directory supplied through the request.

        eda_result:
            Complete structured result returned by Part A.

        kpi_results:
            Part B preprocessing results for every KPI returned by Part A.
    """

    output_directory: Path
    eda_result: EDARunnerResult
    kpi_results: tuple[KPIWorkflowResult, ...]


def run_test_pipeline_workflow(
    *,
    request: PipelineRunRequest,
    logger: Logger,
) -> TestPipelineWorkflowResult:
    """
    Run Part A EDA followed by Part B preprocessing for every KPI.

    Args:
        request:
            Validated dataset, preprocessing, and output-directory settings.

        logger:
            Logger shared across the complete test pipeline.

    Returns:
        TestPipelineWorkflowResult:
            Structured EDA and preprocessing results.

    Raises:
        InputValidationError:
            If dataset or preprocessing validation fails.

        FileProcessingError:
            If an input or output file cannot be processed.

        Exception:
            Any unexpected error raised by the existing runners.
    """

    output_root = ensure_directory(
        Path(
            request.output_directory
        ).expanduser().resolve()
    )

    eda_root = ensure_directory(
        output_root / "eda"
    )

    preprocessing_root = ensure_directory(
        output_root / "preprocessing"
    )

    logger.info(
        "Starting temporary EDA and preprocessing pipeline test. "
        "Output root: %s.",
        output_root,
    )

    eda_result = run_eda_pipeline(
        config=request.dataset,
        logger=logger,
        output_directory=eda_root,
    )

    preprocessing_results: list[
        KPIWorkflowResult
    ] = []

    for index, kpi_result in enumerate(
        eda_result.kpi_results,
        start=1,
    ):
        raw_kpi_name = str(
            kpi_result.prepared_dataset.kpi_name
        ).strip()

        safe_kpi_name = sanitise_filename(
            raw_kpi_name
        )

        if not safe_kpi_name:
            safe_kpi_name = f"kpi_{index}"

        kpi_output_directory = ensure_directory(
            preprocessing_root
            / safe_kpi_name
        )

        logger.info(
            "Starting preprocessing for KPI '%s'. "
            "Output directory: %s.",
            raw_kpi_name,
            kpi_output_directory,
        )

        preprocessing_result = (
            run_preprocessing_pipeline(
                kpi_result=kpi_result,
                output_directory=(
                    kpi_output_directory
                ),
                dataset_config=request.dataset,
                preprocessing_config=(
                    request.preprocessing
                ),
                bucket=(
                    eda_result.preparation.bucket
                ),
                logger=logger,
            )
        )

        preprocessing_results.append(
            KPIWorkflowResult(
                kpi_name=raw_kpi_name,
                preprocessing_result=(
                    preprocessing_result
                ),
            )
        )

    result = TestPipelineWorkflowResult(
        output_directory=output_root,
        eda_result=eda_result,
        kpi_results=tuple(
            preprocessing_results
        ),
    )

    logger.info(
        "Temporary pipeline test completed for %d KPI dataset(s). "
        "Output root: %s.",
        len(result.kpi_results),
        result.output_directory,
    )

    return result


__all__ = [
    "KPIWorkflowResult",
    "TestPipelineWorkflowResult",
    "run_test_pipeline_workflow",
]