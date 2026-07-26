"""
Run read-only sanity checks for the Part A EDA module connections.

This script verifies that:

- the shared dataset schema exposes all three timeline modes;
- the runner resolves each mode correctly;
- data-integrity analysis receives and constructs each timeline mode;
- runner imports still point to dataset preparation, site analysis,
  server validation, and data integrity;
- key function signatures still match the runner's keyword calls.

The script does not read, create, modify, or delete any pipeline dataset or
report file.

Recommended location:

    ml_backend/tests/sanity_check_eda_connections.py

Run from any working directory with:

    python <path-to-ml_backend>/tests/sanity_check_eda_connections.py

The script locates the nearest parent directory containing ``app`` and adds
that backend project root to Python's module search path.

Update the module constants below only if these files live in a package other
than ``app.ml.eda``.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def configure_project_import_path() -> Path:
    """
    Add the backend project root to Python's module search path.

    The sanity script may be stored inside ``ml_backend/tests``. When a
    Python file is executed directly, Python normally adds the script's own
    directory to ``sys.path`` but does not automatically add its parent.
    This function walks upward until it finds the directory containing
    ``app`` and inserts that directory at the beginning of ``sys.path``.

    Args:
        None.

    Returns:
        Path:
            Located backend project root containing the ``app`` package.

    Raises:
        RuntimeError:
            If no parent directory containing ``app`` can be found.
    """

    script_path = Path(__file__).resolve()

    for candidate in script_path.parents:
        app_directory = candidate / "app"

        if app_directory.is_dir():
            project_root = candidate
            project_root_text = str(project_root)

            if project_root_text not in sys.path:
                sys.path.insert(
                    0,
                    project_root_text,
                )

            return project_root

    raise RuntimeError(
        "Could not locate the backend project root. The sanity script "
        "must be stored somewhere inside the ml_backend project that "
        "contains the app directory."
    )


PROJECT_ROOT = configure_project_import_path()


DATASET_MODULE = "app.schemas.common.dataset"
RUNNER_MODULE = "app.ml.eda.runner"
DATA_PREPARATION_MODULE = "app.ml.eda.data_preparation"
SITE_ANALYSIS_MODULE = "app.ml.eda.site_analysis"
SERVER_VALIDATION_MODULE = "app.ml.eda.server_validation"
DATA_INTEGRITY_MODULE = "app.ml.eda.data_integrity"


@dataclass(frozen=True, slots=True)
class SanityCheckResult:
    """
    Store the outcome of one sanity check.

    Attributes:
        name:
            Human-readable check name.
        passed:
            Whether the check succeeded.
        details:
            Short result description.
    """

    name: str
    passed: bool
    details: str


def require_symbol(
    module: Any,
    symbol_name: str,
) -> Any:
    """
    Return a required module symbol or raise a clear assertion.

    Args:
        module:
            Imported Python module to inspect.
        symbol_name:
            Name expected to exist in the module.

    Returns:
        Any:
            Located module attribute.

    Raises:
        AssertionError:
            If the requested symbol does not exist.
    """

    assert hasattr(module, symbol_name), (
        f"{module.__name__} is missing required symbol "
        f"'{symbol_name}'."
    )

    return getattr(
        module,
        symbol_name,
    )


def require_parameters(
    function: Any,
    required_parameters: set[str],
) -> None:
    """
    Validate that a callable still accepts required parameter names.

    Args:
        function:
            Callable whose signature must be inspected.
        required_parameters:
            Parameter names required by orchestration calls.

    Returns:
        None.

    Raises:
        AssertionError:
            If one or more required parameters are unavailable.
    """

    available_parameters = set(
        inspect.signature(function).parameters
    )

    missing_parameters = (
        required_parameters
        - available_parameters
    )

    assert not missing_parameters, (
        f"{function.__module__}.{function.__name__} is missing "
        f"parameter(s): {sorted(missing_parameters)}."
    )


def build_dataset_config(
    DatasetInputConfig: Any,
    timeline: dict[str, Any] | None = None,
) -> Any:
    """
    Create an in-memory dataset configuration for timeline tests.

    Args:
        DatasetInputConfig:
            Imported Pydantic dataset configuration class.
        timeline:
            Optional expected-timeline configuration.

    Returns:
        Any:
            Validated DatasetInputConfig instance.

    Raises:
        pydantic.ValidationError:
            If the supplied test configuration is invalid.
    """

    values: dict[str, Any] = {
        "input_file": "sanity_check.csv",
        "bucket": {
            "bucket_type": "Minutes",
            "bucket_size": 15,
        },
        "single_kpi_name": "SANITY_CHECK_KPI",
    }

    if timeline is not None:
        values["expected_timeline"] = timeline

    return DatasetInputConfig(
        **values
    )


def check_required_symbols(
    modules: dict[str, Any],
) -> SanityCheckResult:
    """
    Check that all symbols imported or called by the runner still exist.

    Args:
        modules:
            Mapping containing imported pipeline modules.

    Returns:
        SanityCheckResult:
            Successful symbol-existence result.

    Raises:
        AssertionError:
            If a required class or function is missing.
    """

    required_symbols = {
        "dataset": {
            "BucketConfig",
            "DatasetColumnMapping",
            "DatasetInputConfig",
            "ExpectedTimelineConfig",
        },
        "runner": {
            "EDARunnerResult",
            "KPIEDARunResult",
            "convert_bucket_to_frequency",
            "resolve_integrity_timeline_inputs",
            "build_kpi_summary_directory",
            "run_eda_pipeline",
        },
        "preparation": {
            "DatasetPreparationResult",
            "PreparedKPIDataset",
            "prepare_dataset",
        },
        "site": {
            "SiteAnalysisResult",
            "analyse_sites",
        },
        "server": {
            "ServerValidationResult",
            "validate_server_consistency",
        },
        "integrity": {
            "DataIntegrityResult",
            "ExpectedTimeline",
            "analyse_data_integrity",
            "resolve_expected_timeline",
        },
    }

    for module_key, symbol_names in required_symbols.items():
        module = modules[module_key]

        for symbol_name in symbol_names:
            require_symbol(
                module,
                symbol_name,
            )

    return SanityCheckResult(
        name="Required symbols",
        passed=True,
        details="All runner dependencies and timeline symbols exist.",
    )


def check_runner_connections(
    modules: dict[str, Any],
) -> SanityCheckResult:
    """
    Confirm that runner references point to the expected module functions.

    Args:
        modules:
            Mapping containing imported pipeline modules.

    Returns:
        SanityCheckResult:
            Successful runner-connection result.

    Raises:
        AssertionError:
            If a runner dependency points to an unexpected object.
    """

    runner = modules["runner"]

    expected_connections = {
        "prepare_dataset": modules["preparation"].prepare_dataset,
        "analyse_sites": modules["site"].analyse_sites,
        "validate_server_consistency": (
            modules["server"].validate_server_consistency
        ),
        "analyse_data_integrity": (
            modules["integrity"].analyse_data_integrity
        ),
    }

    for runner_name, expected_function in expected_connections.items():
        actual_function = require_symbol(
            runner,
            runner_name,
        )

        assert actual_function is expected_function, (
            f"runner.{runner_name} is not connected to "
            f"{expected_function.__module__}.{expected_function.__name__}."
        )

    return SanityCheckResult(
        name="Runner connections",
        passed=True,
        details=(
            "Preparation, site, server, and integrity calls remain "
            "connected."
        ),
    )


def check_function_signatures(
    modules: dict[str, Any],
) -> SanityCheckResult:
    """
    Check function signatures used by the runner's keyword calls.

    Args:
        modules:
            Mapping containing imported pipeline modules.

    Returns:
        SanityCheckResult:
            Successful signature-compatibility result.

    Raises:
        AssertionError:
            If a required function parameter has been removed or renamed.
    """

    require_parameters(
        modules["preparation"].prepare_dataset,
        {
            "config",
            "logger",
        },
    )

    require_parameters(
        modules["site"].analyse_sites,
        {
            "dataframe",
            "source_file",
            "output_directory",
            "columns",
            "logger",
        },
    )

    require_parameters(
        modules["server"].validate_server_consistency,
        {
            "dataframe",
            "source_file",
            "columns",
            "logger",
        },
    )

    require_parameters(
        modules["integrity"].analyse_data_integrity,
        {
            "dataframe",
            "source_file",
            "site_summary_file",
            "output_directory",
            "columns",
            "logger",
            "expected_frequency",
            "expected_start",
            "expected_end",
            "num_days",
            "use_global_min_max",
        },
    )

    require_parameters(
        modules["runner"].run_eda_pipeline,
        {
            "config",
            "logger",
        },
    )

    return SanityCheckResult(
        name="Function signatures",
        passed=True,
        details="All runner keyword calls remain signature-compatible.",
    )


def check_timeline_modes(
    modules: dict[str, Any],
) -> SanityCheckResult:
    """
    Validate schema, runner, and integrity behavior for all timeline modes.

    Args:
        modules:
            Mapping containing imported pipeline modules.

    Returns:
        SanityCheckResult:
            Successful timeline-mode handoff result.

    Raises:
        AssertionError:
            If a mode is absent, resolves incorrectly, or fails to reach
            data-integrity timeline construction.
    """

    DatasetInputConfig = modules["dataset"].DatasetInputConfig
    resolve_runner_inputs = (
        modules["runner"].resolve_integrity_timeline_inputs
    )
    resolve_expected_timeline = (
        modules["integrity"].resolve_expected_timeline
    )

    test_cases = (
        (
            "global_min_max_default",
            None,
            (
                None,
                None,
                None,
                True,
            ),
            "global_min_max",
        ),
        (
            "global_min_max_explicit",
            {
                "timeline_mode": "global_min_max",
            },
            (
                None,
                None,
                None,
                True,
            ),
            "global_min_max",
        ),
        (
            "num_days",
            {
                "timeline_mode": "num_days",
                "num_days": 2,
            },
            (
                None,
                None,
                2,
                False,
            ),
            "num_days",
        ),
        (
            "date_range",
            {
                "timeline_mode": "date_range",
                "expected_start": "2026-01-01 00:00:00",
                "expected_end": "2026-01-01 01:00:00",
            },
            (
                pd.Timestamp("2026-01-01 00:00:00"),
                pd.Timestamp("2026-01-01 01:00:00"),
                None,
                False,
            ),
            "date_range",
        ),
    )

    valid_timestamps = pd.Series(
        pd.to_datetime(
            [
                "2026-01-01 00:00:00",
                "2026-01-01 00:15:00",
                "2026-01-01 01:00:00",
            ]
        )
    )

    for (
        case_name,
        timeline_values,
        expected_runner_values,
        expected_integrity_method,
    ) in test_cases:
        config = build_dataset_config(
            DatasetInputConfig,
            timeline_values,
        )

        runner_values = resolve_runner_inputs(
            config
        )

        assert runner_values == expected_runner_values, (
            f"{case_name} resolved incorrectly in runner: "
            f"{runner_values!r}."
        )

        (
            expected_start,
            expected_end,
            num_days,
            use_global_min_max,
        ) = runner_values

        timeline = resolve_expected_timeline(
            valid_timestamps=valid_timestamps,
            expected_frequency="15min",
            expected_start=expected_start,
            expected_end=expected_end,
            num_days=num_days,
            use_global_min_max=use_global_min_max,
        )

        assert timeline.method == expected_integrity_method, (
            f"{case_name} reached integrity as '{timeline.method}', not "
            f"'{expected_integrity_method}'."
        )

        assert len(timeline.index) > 0, (
            f"{case_name} generated an empty timeline."
        )

    return SanityCheckResult(
        name="Timeline modes",
        passed=True,
        details=(
            "date_range, num_days, and global_min_max pass from schema "
            "through runner into data integrity."
        ),
    )


def check_invalid_timeline_inputs(
    modules: dict[str, Any],
) -> SanityCheckResult:
    """
    Ensure contradictory timeline settings are rejected by the schema.

    Args:
        modules:
            Mapping containing imported pipeline modules.

    Returns:
        SanityCheckResult:
            Successful invalid-input rejection result.

    Raises:
        AssertionError:
            If an invalid timeline combination is accepted.
    """

    DatasetInputConfig = modules["dataset"].DatasetInputConfig

    invalid_timeline_values = (
        {
            "timeline_mode": "date_range",
        },
        {
            "timeline_mode": "num_days",
        },
        {
            "timeline_mode": "global_min_max",
            "num_days": 5,
        },
        {
            "timeline_mode": "date_range",
            "expected_start": "2026-01-02 00:00:00",
            "expected_end": "2026-01-01 00:00:00",
        },
    )

    for timeline_values in invalid_timeline_values:
        try:
            build_dataset_config(
                DatasetInputConfig,
                timeline_values,
            )
        except Exception:
            continue

        raise AssertionError(
            "The schema accepted invalid timeline settings: "
            f"{timeline_values!r}."
        )

    return SanityCheckResult(
        name="Invalid timeline rejection",
        passed=True,
        details="Contradictory and incomplete timeline settings are rejected.",
    )


def import_pipeline_modules() -> dict[str, Any]:
    """
    Import all modules involved in the Part A EDA handoff.

    Args:
        None.

    Returns:
        dict[str, Any]:
            Imported modules keyed by their pipeline role.

    Raises:
        ImportError:
            If a module or one of its required imports is unavailable.
    """

    return {
        "dataset": importlib.import_module(
            DATASET_MODULE
        ),
        "runner": importlib.import_module(
            RUNNER_MODULE
        ),
        "preparation": importlib.import_module(
            DATA_PREPARATION_MODULE
        ),
        "site": importlib.import_module(
            SITE_ANALYSIS_MODULE
        ),
        "server": importlib.import_module(
            SERVER_VALIDATION_MODULE
        ),
        "integrity": importlib.import_module(
            DATA_INTEGRITY_MODULE
        ),
    }


def run_sanity_checks() -> tuple[SanityCheckResult, ...]:
    """
    Run all read-only EDA connection and timeline checks.

    Args:
        None.

    Returns:
        tuple[SanityCheckResult, ...]:
            Successful result for each completed check.

    Raises:
        ImportError:
            If the module package paths are incorrect or imports are broken.
        AssertionError:
            If any required connection, symbol, signature, or mode fails.
    """

    modules = import_pipeline_modules()

    checks = (
        check_required_symbols,
        check_runner_connections,
        check_function_signatures,
        check_timeline_modes,
        check_invalid_timeline_inputs,
    )

    return tuple(
        check(modules)
        for check in checks
    )


def main() -> None:
    """
    Execute sanity checks and print a compact pass/fail report.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit:
            With status 1 when a check fails.
    """

    print("=" * 80)
    print("PART A EDA CONNECTION SANITY CHECK")
    print("=" * 80)

    try:
        results = run_sanity_checks()
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}")
        raise SystemExit(1) from error

    for result in results:
        print(
            f"[PASS] {result.name}: {result.details}"
        )

    print("-" * 80)
    print("ALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()