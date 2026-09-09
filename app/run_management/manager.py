"""
High-level controller for run identity, attempts, stage executions, registry
lookup, immutable lineage, artifact resolution, and Part E request coordination.

This module deliberately does not import or execute Parts A-E. Workflows call
RunManager to obtain safe contexts, then call the already-existing ML runners.

FIX INCLUDED:
Part E invocation registration is now failure-aware:
- if the invocation record is created but completion persistence fails, the
  invocation is best-effort marked failed instead of being left "running";
- if the invocation completes but registry synchronization fails, inference is
  still treated as completed because Manifest V2 is the technical truth;
- session.complete() does not allow __exit__ to incorrectly add a second
  request-level failure after completion finalization has already been handled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from app.core.config import settings
from app.core.exceptions import (
    ArtifactResolutionError,
    RunManagementError,
    RunNotFoundError,
    RunRegistryError,
)
from app.services.file_service import (
    copy_input_file,
    ensure_directory,
    remove_tree_with_retry,
    replace_path_with_retry,
    write_json_file,
)
from app.services.run_directory_service import (
    create_run_directories,
    get_run_directory,
    validate_run_directory,
    validate_run_id,
)
from app.services.run_manifest_service import (
    append_manifest_event,
    complete_inference_invocation,
    complete_stage_execution,
    convert_exception_to_record,
    create_attempt_record,
    create_run_manifest,
    fail_inference_invocation,
    fail_stage_execution,
    get_current_utc_timestamp,
    load_run_manifest,
    record_output_file,
    save_run_manifest,
    set_active_inference_attempt,
    set_attempt_status,
    set_run_status,
    start_inference_invocation,
    start_stage_execution,
    update_run_metadata,
    validate_record_id,
    validate_stage_name,
)

from app.run_management.artifact_resolver import (
    get_inference_state_directory,
    resolve_active_inference_state,
    resolve_attempt_input,
    resolve_prerequisite_results,
)
from app.run_management.identifiers import (
    generate_attempt_id,
    generate_execution_id,
    generate_inference_request_id,
    validate_part_e_compatible_run_id,
)
from app.run_management.lineage import (
    derive_attempt_status,
    ensure_inference_activation_allowed,
    ensure_stage_can_start,
    get_attempt_record,
)
from app.run_management.locking import RunLock
from app.run_management.models import (
    AttemptContext,
    InferenceRequestContext,
    OutputRegistration,
    RunContext,
    RunRegistryRecord,
    StageExecutionContext,
)
from app.run_management.registry import (
    LocalJsonRunRegistry,
    RunRegistry,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _remap_path_prefix(
    value: Any,
    *,
    old_root: Path,
    new_root: Path,
) -> Any:
    """Recursively remap absolute paths rooted below old_root to new_root."""
    if isinstance(value, dict):
        return {
            key: _remap_path_prefix(
                item,
                old_root=old_root,
                new_root=new_root,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _remap_path_prefix(
                item,
                old_root=old_root,
                new_root=new_root,
            )
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _remap_path_prefix(
                item,
                old_root=old_root,
                new_root=new_root,
            )
            for item in value
        ]

    if isinstance(value, (str, Path)):
        candidate = Path(value).expanduser()

        if not candidate.is_absolute():
            return value

        resolved = candidate.resolve()

        try:
            relative = resolved.relative_to(old_root)
        except ValueError:
            return str(value) if isinstance(value, Path) else value

        return str((new_root / relative).resolve())

    return value


def _payload_to_dict(payload: Any) -> dict[str, Any]:
    """Convert a Pydantic model or mapping into a JSON-oriented dictionary."""
    if hasattr(payload, "model_dump"):
        value = payload.model_dump(mode="json")
    elif isinstance(payload, Mapping):
        value = dict(payload)
    else:
        raise RunManagementError(
            "Run Management requires a mapping-like request payload.",
            error_code="INVALID_RUN_REQUEST_PAYLOAD",
            details={
                "received_type": type(payload).__name__,
            },
        )

    if not isinstance(value, dict):
        raise RunManagementError(
            "The request payload must resolve to a JSON object.",
            error_code="INVALID_RUN_REQUEST_PAYLOAD",
            details={
                "received_type": type(value).__name__,
            },
        )

    return value


class _InferenceSession(AbstractContextManager["_InferenceSession"]):
    """
    Context manager keeping the run lock for one complete Part E operation.

    Part E owns invocation_id. Run Management owns the pre-call request snapshot.
    """

    def __init__(
        self,
        manager: "RunManager",
        run_id: str,
        request_payload: Any,
        input_file: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._manager = manager
        self._run_id = validate_part_e_compatible_run_id(
            validate_run_id(run_id)
        )
        self._request_payload = request_payload
        self._input_file = input_file
        self._metadata = dict(metadata or {})

        self._lock: RunLock | None = None
        self.context: InferenceRequestContext | None = None
        self._finalized = False

    def __enter__(self) -> "_InferenceSession":
        run_context = self._manager.resolve_run(self._run_id)

        self._lock = RunLock(run_context.run_directory)
        self._lock.acquire()

        try:
            self.context = self._manager._prepare_inference_request_locked(
                run_context=run_context,
                request_payload=self._request_payload,
                input_file=self._input_file,
                metadata=self._metadata,
            )
        except Exception:
            self._lock.release()
            self._lock = None
            raise

        return self

    def complete(
        self,
        *,
        invocation_id: str,
        result_payload: Any,
        output_directory: str | Path,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Register and finalize the exact Part E invocation.

        FIX:
        _complete_inference_request_locked() now owns all registration-failure
        recording. Therefore this session is finalized even if that method
        raises, preventing __exit__ from incorrectly recording a second
        request-level failure for the same completion attempt.
        """
        if self.context is None:
            raise RunManagementError(
                "The inference session has not been entered.",
                error_code="INFERENCE_SESSION_NOT_STARTED",
            )

        if self._finalized:
            raise RunManagementError(
                "The inference session is already finalized.",
                error_code="INFERENCE_SESSION_ALREADY_FINALIZED",
            )

        try:
            self._manager._complete_inference_request_locked(
                context=self.context,
                invocation_id=invocation_id,
                result_payload=result_payload,
                output_directory=output_directory,
                metrics=metrics,
            )
        except Exception:
            # The manager has already persisted the correct failure state
            # (request-level or invocation-level) on every completion-path error.
            self._finalized = True
            raise

        self._finalized = True

    def fail(self, error: Exception) -> None:
        """Persist a Part E request failure when no successful invocation exists."""
        if self.context is None:
            raise RunManagementError(
                "The inference session has not been entered.",
                error_code="INFERENCE_SESSION_NOT_STARTED",
            )

        if self._finalized:
            return

        self._manager._fail_inference_request_locked(
            context=self.context,
            error=error,
        )
        self._finalized = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        pending_error: Exception | None = None

        try:
            if (
                exc_value is not None
                and not self._finalized
                and self.context is not None
            ):
                try:
                    self._manager._fail_inference_request_locked(
                        context=self.context,
                        error=(
                            exc_value
                            if isinstance(exc_value, Exception)
                            else RunManagementError(
                                str(exc_value),
                                error_code="INFERENCE_BASE_EXCEPTION",
                            )
                        ),
                    )
                    self._finalized = True
                except Exception:
                    # Never mask the original Part E exception.
                    pass

            elif exc_value is None and not self._finalized:
                pending_error = RunManagementError(
                    "The inference session exited without being completed or failed.",
                    error_code="INFERENCE_SESSION_NOT_FINALIZED",
                )

                if self.context is not None:
                    try:
                        self._manager._fail_inference_request_locked(
                            context=self.context,
                            error=pending_error,
                        )
                        self._finalized = True
                    except Exception:
                        pass

        finally:
            if self._lock is not None:
                self._lock.release()
                self._lock = None

        if pending_error is not None:
            raise pending_error

        return False


class RunManager:
    """High-level façade for production run-management operations."""

    def __init__(self, registry: RunRegistry | None = None) -> None:
        if registry is not None:
            self.registry = registry
            return

        if settings.run_registry_backend != "local_json":
            raise RunRegistryError(
                "The configured run-registry backend is unsupported by V1.",
                error_code="UNSUPPORTED_RUN_REGISTRY_BACKEND",
                details={
                    "backend": settings.run_registry_backend,
                },
            )

        self.registry = LocalJsonRunRegistry()

    # ------------------------------------------------------------------
    # RUN LOOKUP / CREATION
    # ------------------------------------------------------------------

    def _sync_registry_from_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        create: bool = False,
    ) -> RunRegistryRecord:
        record = RunRegistryRecord.from_manifest(manifest)

        if create:
            return self.registry.create(record)

        return self.registry.update(record)

    def _sync_registry_best_effort(
        self,
        manifest: Mapping[str, Any],
    ) -> RunRegistryRecord:
        """
        Refresh the secondary run registry without invalidating a committed
        Manifest V2 transition.

        Manifest V2 is the technical lifecycle truth. The local/DB registry is
        a lookup index and can be rebuilt/refreshed from the manifest later.
        Therefore an index synchronization failure must not turn an already
        committed stage/attempt/run transition into an API failure.
        """
        record = RunRegistryRecord.from_manifest(manifest)

        try:
            return self.registry.update(record)
        except (RunRegistryError, RunNotFoundError):
            return record

    def create_run(
        self,
        *,
        run_metadata: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> RunContext:
        supplied_run_id = (
            validate_part_e_compatible_run_id(validate_run_id(run_id))
            if run_id is not None
            else None
        )

        if supplied_run_id is not None and self.registry.exists(supplied_run_id):
            raise RunRegistryError(
                "The requested run ID is already registered.",
                error_code="RUN_ALREADY_EXISTS",
                details={"run_id": supplied_run_id},
            )

        created_paths = None

        try:
            created_paths = create_run_directories(
                supplied_run_id,
                exist_ok=False,
            )

            final_run_id = validate_part_e_compatible_run_id(
                str(created_paths["run_id"])
            )

            run_directory = Path(created_paths["run"]).resolve()

            manifest = create_run_manifest(
                run_directory,
                final_run_id,
                run_metadata=run_metadata,
            )

            record = self._sync_registry_from_manifest(
                manifest,
                create=True,
            )

            return RunContext(
                run_id=final_run_id,
                run_directory=run_directory,
                manifest_path=Path(created_paths["manifest"]).resolve(),
                registry_record=record,
            )

        except Exception:
            if created_paths is not None:
                for key in ("run", "logs"):
                    path = Path(created_paths[key])
                    try:
                        if path.exists():
                            remove_tree_with_retry(path)
                    except OSError:
                        pass
            raise

    def resolve_run(self, run_id: str) -> RunContext:
        cleaned_run_id = validate_run_id(run_id)
        record = self.registry.get(cleaned_run_id)

        expected_directory = get_run_directory(cleaned_run_id).resolve()

        if record.run_directory != expected_directory:
            raise RunRegistryError(
                "The registry run directory does not match the configured run root.",
                error_code="RUN_REGISTRY_DIRECTORY_MISMATCH",
                details={
                    "run_id": cleaned_run_id,
                    "registry_directory": str(record.run_directory),
                    "expected_directory": str(expected_directory),
                },
            )

        try:
            run_directory = validate_run_directory(cleaned_run_id).resolve()
        except Exception as error:
            raise RunNotFoundError(
                "The registered run directory is unavailable.",
                error_code="REGISTERED_RUN_DIRECTORY_NOT_FOUND",
                details={
                    "run_id": cleaned_run_id,
                    "run_directory": str(expected_directory),
                    "original_error": str(error),
                },
            ) from error

        manifest = load_run_manifest(run_directory)

        if manifest.get("run_id") != cleaned_run_id:
            raise RunRegistryError(
                "The run manifest identity does not match the requested run.",
                error_code="RUN_MANIFEST_ID_MISMATCH",
                details={
                    "requested_run_id": cleaned_run_id,
                    "manifest_run_id": manifest.get("run_id"),
                },
            )

        refreshed = RunRegistryRecord.from_manifest(manifest)

        if refreshed.to_dict() != record.to_dict():
            try:
                record = self.registry.update(refreshed)
            except (RunRegistryError, RunNotFoundError):
                # Manifest V2 is authoritative. Return the refreshed technical
                # state even when the secondary lookup index cannot be updated
                # during this request.
                record = refreshed

        return RunContext(
            run_id=cleaned_run_id,
            run_directory=run_directory,
            manifest_path=run_directory / "manifest.json",
            registry_record=record,
        )

    def list_runs(self) -> list[RunRegistryRecord]:
        return self.registry.list_runs()

    def get_manifest(self, run_id: str) -> dict[str, Any]:
        context = self.resolve_run(run_id)
        return load_run_manifest(context.run_directory)

    def update_metadata(
        self,
        run_id: str,
        metadata: Mapping[str, Any],
    ) -> RunContext:
        context = self.resolve_run(run_id)

        with RunLock(context.run_directory):
            manifest = update_run_metadata(
                context.run_directory,
                metadata,
            )
            record = self._sync_registry_best_effort(manifest)

        return RunContext(
            run_id=context.run_id,
            run_directory=context.run_directory,
            manifest_path=context.manifest_path,
            registry_record=record,
        )

    # ------------------------------------------------------------------
    # ATTEMPTS
    # ------------------------------------------------------------------

    def _attempt_paths(
        self,
        run_directory: Path,
        attempt_id: str,
    ) -> dict[str, Path]:
        cleaned_attempt_id = validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )

        attempt_directory = (
            run_directory / "attempts" / cleaned_attempt_id
        ).resolve()

        return {
            "attempt": attempt_directory,
            "request": attempt_directory / "request.json",
            "input": attempt_directory / "input",
            "stages": attempt_directory / "stages",
        }

    def create_attempt(
        self,
        run_id: str,
        *,
        request_payload: Any,
        input_file: str | Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> AttemptContext:
        run_context = self.resolve_run(run_id)

        with RunLock(run_context.run_directory):
            manifest = load_run_manifest(run_context.run_directory)
            attempt_id = generate_attempt_id(manifest["attempts"].keys())
            paths = self._attempt_paths(
                run_context.run_directory,
                attempt_id,
            )
            manifest_created = False

            try:
                ensure_directory(paths["attempt"])
                ensure_directory(paths["input"])
                ensure_directory(paths["stages"])

                request_file = write_json_file(
                    data=_payload_to_dict(request_payload),
                    file_path=paths["request"],
                    overwrite=False,
                )

                source_path = Path(input_file).expanduser().resolve()

                copied_input = copy_input_file(
                    source_path=source_path,
                    destination_directory=paths["input"],
                    destination_filename=source_path.name,
                    overwrite=False,
                )

                manifest = create_attempt_record(
                    run_context.run_directory,
                    attempt_id,
                    request_file=request_file,
                    input_file=copied_input,
                    metadata=metadata,
                )
                manifest_created = True

                manifest = set_run_status(
                    run_context.run_directory,
                    "created",
                )

                self._sync_registry_best_effort(manifest)

            except Exception:
                if not manifest_created:
                    try:
                        if paths["attempt"].exists():
                            remove_tree_with_retry(paths["attempt"])
                    except OSError:
                        pass
                raise

            return AttemptContext(
                run_id=run_context.run_id,
                attempt_id=attempt_id,
                run_directory=run_context.run_directory,
                attempt_directory=paths["attempt"],
                input_directory=paths["input"],
                stages_directory=paths["stages"],
                request_file=Path(request_file).resolve(),
                input_file=Path(copied_input).resolve(),
            )

    def resolve_attempt(
        self,
        run_id: str,
        attempt_id: str,
    ) -> AttemptContext:
        run_context = self.resolve_run(run_id)
        cleaned_attempt_id = validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )

        manifest = load_run_manifest(run_context.run_directory)
        attempt = get_attempt_record(manifest, cleaned_attempt_id)
        paths = self._attempt_paths(
            run_context.run_directory,
            cleaned_attempt_id,
        )

        request_file_value = attempt.get("request_file")

        if not request_file_value:
            raise ArtifactResolutionError(
                "The attempt does not register its request snapshot.",
                error_code="ATTEMPT_REQUEST_FILE_MISSING",
                details={"attempt_id": cleaned_attempt_id},
            )

        request_file = Path(request_file_value).expanduser().resolve()

        try:
            request_file.relative_to(run_context.run_directory)
        except ValueError as error:
            raise ArtifactResolutionError(
                "The attempt request snapshot escapes the owning run directory.",
                error_code="ATTEMPT_REQUEST_OUTSIDE_RUN",
                details={
                    "attempt_id": cleaned_attempt_id,
                    "request_file": str(request_file),
                },
            ) from error

        if not request_file.is_file():
            raise ArtifactResolutionError(
                "The attempt request snapshot does not exist.",
                error_code="ATTEMPT_REQUEST_FILE_NOT_FOUND",
                details={
                    "attempt_id": cleaned_attempt_id,
                    "request_file": str(request_file),
                },
            )

        input_path = resolve_attempt_input(
            manifest,
            run_context.run_directory,
            cleaned_attempt_id,
        )

        return AttemptContext(
            run_id=run_context.run_id,
            attempt_id=cleaned_attempt_id,
            run_directory=run_context.run_directory,
            attempt_directory=paths["attempt"],
            input_directory=paths["input"],
            stages_directory=paths["stages"],
            request_file=request_file,
            input_file=input_path,
        )

    # ------------------------------------------------------------------
    # TRAINING STAGE EXECUTIONS
    # ------------------------------------------------------------------

    def _stage_execution_paths(
        self,
        run_directory: Path,
        attempt_id: str,
        stage_name: str,
        execution_id: str,
    ) -> dict[str, Path]:
        cleaned_attempt_id = validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )
        cleaned_stage_name = validate_stage_name(stage_name)
        cleaned_execution_id = validate_record_id(
            execution_id,
            field_name="execution_id",
        )

        attempt_directory = (
            run_directory / "attempts" / cleaned_attempt_id
        ).resolve()

        stage_directory = (
            attempt_directory / "stages" / cleaned_stage_name
        ).resolve()

        execution_directory = (
            stage_directory / cleaned_execution_id
        ).resolve()

        output_directory = execution_directory / "outputs"

        return {
            "attempt": attempt_directory,
            "stage": stage_directory,
            "execution": execution_directory,
            "request": execution_directory / "request.json",
            "outputs": output_directory,
            "result": execution_directory / "result.json",
            "error": execution_directory / "error.json",
            "bootstrap_state": output_directory / "inference_state",
        }

    def _recover_started_stage_preparation_failure_locked(
        self,
        *,
        run_directory: Path,
        attempt_id: str,
        stage_name: str,
        execution_id: str,
        error_file: Path,
        error: Exception,
    ) -> None:
        """
        Best-effort recovery when stage registration succeeded but preparation
        failed before a StageExecutionContext could be returned.

        Without this recovery, the manifest can be left permanently with
        active_execution_id -> a "running" execution even though the HTTP
        request has already failed.
        """
        error_record = convert_exception_to_record(error)
        error_record["details"] = {
            **dict(error_record.get("details", {})),
            "registration_phase": "stage_preparation",
        }

        persisted_error_file: Path | None = None

        try:
            persisted_error_file = write_json_file(
                data=error_record,
                file_path=error_file,
                overwrite=True,
            )
        except Exception:
            # Preserve the original preparation exception.
            persisted_error_file = None

        try:
            fail_stage_execution(
                run_directory,
                attempt_id,
                stage_name,
                execution_id,
                error,
                error_file=persisted_error_file,
            )
        except Exception:
            # If the manifest itself cannot be repaired, preserve the original
            # error. The existing running record remains forensic evidence.
            return

        try:
            manifest = load_run_manifest(run_directory)
            attempt = get_attempt_record(manifest, attempt_id)
            attempt_status = derive_attempt_status(attempt)

            manifest = set_attempt_status(
                run_directory,
                attempt_id,
                attempt_status,
            )
            manifest = set_run_status(
                run_directory,
                attempt_status,
            )

            self._sync_registry_best_effort(manifest)
        except Exception:
            # Stage-level recovery above is the critical invariant: clear the
            # active running execution. Higher-level status repair is best effort.
            pass

    def prepare_stage_execution(
        self,
        run_id: str,
        attempt_id: str,
        stage_name: str,
        *,
        request_payload: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> StageExecutionContext:
        run_context = self.resolve_run(run_id)
        cleaned_attempt_id = validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )
        cleaned_stage_name = validate_stage_name(stage_name)

        with RunLock(run_context.run_directory):
            manifest = load_run_manifest(run_context.run_directory)

            ensure_stage_can_start(
                manifest,
                cleaned_attempt_id,
                cleaned_stage_name,
            )

            stage = manifest["attempts"][cleaned_attempt_id][
                "stages"
            ][cleaned_stage_name]

            execution_id = generate_execution_id(
                stage["executions"].keys()
            )

            paths = self._stage_execution_paths(
                run_context.run_directory,
                cleaned_attempt_id,
                cleaned_stage_name,
                execution_id,
            )

            execution_registered = False

            try:
                ensure_directory(paths["execution"])
                ensure_directory(paths["outputs"])

                if cleaned_stage_name == "inference_bootstrap":
                    ensure_directory(paths["bootstrap_state"])

                request_file = write_json_file(
                    data=_payload_to_dict(request_payload),
                    file_path=paths["request"],
                    overwrite=False,
                )

                prerequisites = resolve_prerequisite_results(
                    manifest,
                    run_context.run_directory,
                    cleaned_attempt_id,
                    cleaned_stage_name,
                )

                manifest = start_stage_execution(
                    run_context.run_directory,
                    cleaned_attempt_id,
                    cleaned_stage_name,
                    execution_id,
                    request_file=request_file,
                    output_directory=paths["outputs"],
                    metadata=metadata,
                )
                execution_registered = True

                manifest = set_attempt_status(
                    run_context.run_directory,
                    cleaned_attempt_id,
                    "running",
                )

                manifest = set_run_status(
                    run_context.run_directory,
                    "running",
                )

                # Registry is a secondary index. At this point Manifest V2
                # already contains the complete "running" technical state, so
                # registry refresh must not invalidate stage preparation.
                self._sync_registry_best_effort(manifest)

            except Exception as error:
                if execution_registered:
                    self._recover_started_stage_preparation_failure_locked(
                        run_directory=run_context.run_directory,
                        attempt_id=cleaned_attempt_id,
                        stage_name=cleaned_stage_name,
                        execution_id=execution_id,
                        error_file=paths["error"],
                        error=error,
                    )
                else:
                    try:
                        if paths["execution"].exists():
                            remove_tree_with_retry(paths["execution"])
                    except OSError:
                        pass
                raise

            return StageExecutionContext(
                run_id=run_context.run_id,
                attempt_id=cleaned_attempt_id,
                stage_name=cleaned_stage_name,
                execution_id=execution_id,
                run_directory=run_context.run_directory,
                attempt_directory=paths["attempt"],
                stage_directory=paths["stage"],
                execution_directory=paths["execution"],
                request_file=Path(request_file).resolve(),
                output_directory=paths["outputs"],
                result_file=paths["result"],
                error_file=paths["error"],
                prerequisite_results=prerequisites,
                inference_state_directory=(
                    paths["bootstrap_state"]
                    if cleaned_stage_name == "inference_bootstrap"
                    else None
                ),
            )

    def _derive_and_persist_statuses(
        self,
        run_directory: Path,
        attempt_id: str,
    ) -> dict[str, Any]:
        manifest = load_run_manifest(run_directory)
        attempt = get_attempt_record(manifest, attempt_id)
        attempt_status = derive_attempt_status(attempt)

        manifest = set_attempt_status(
            run_directory,
            attempt_id,
            attempt_status,
        )

        manifest = set_run_status(
            run_directory,
            attempt_status,
        )

        self._sync_registry_best_effort(manifest)
        return manifest

    def _activate_inference_attempt_locked(
        self,
        run_directory: Path,
        attempt_id: str,
    ) -> dict[str, Any]:
        manifest = load_run_manifest(run_directory)

        ensure_inference_activation_allowed(
            manifest,
            attempt_id,
        )

        get_inference_state_directory(
            run_directory,
            attempt_id,
            require_config=True,
        )

        manifest = set_active_inference_attempt(
            run_directory,
            attempt_id,
        )

        self._sync_registry_best_effort(manifest)
        return manifest

    def activate_inference_attempt(
        self,
        run_id: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        run_context = self.resolve_run(run_id)
        cleaned_attempt_id = validate_record_id(
            attempt_id,
            field_name="attempt_id",
        )

        with RunLock(run_context.run_directory):
            return self._activate_inference_attempt_locked(
                run_context.run_directory,
                cleaned_attempt_id,
            )

    def _publish_bootstrap_state(
        self,
        context: StageExecutionContext,
    ) -> Path:
        staging = context.inference_state_directory

        if staging is None:
            raise RunManagementError(
                "Bootstrap execution does not have an inference-state staging directory.",
                error_code="BOOTSTRAP_STATE_DIRECTORY_MISSING",
            )

        config_file = staging / "inference_config.json"

        if not config_file.is_file():
            raise ArtifactResolutionError(
                "Inference bootstrap did not produce inference_config.json.",
                error_code="INFERENCE_CONFIG_NOT_CREATED",
                details={
                    "expected_file": str(config_file),
                    "attempt_id": context.attempt_id,
                    "execution_id": context.execution_id,
                },
            )

        stable = get_inference_state_directory(
            context.run_directory,
            context.attempt_id,
            require_config=False,
        )

        if stable.exists():
            raise ArtifactResolutionError(
                "A stable inference state already exists for this attempt.",
                error_code="INFERENCE_STATE_ALREADY_PUBLISHED",
                details={
                    "attempt_id": context.attempt_id,
                    "state_directory": str(stable),
                },
            )

        ensure_directory(stable.parent)

        try:
            replace_path_with_retry(
                staging,
                stable,
            )
        except OSError as error:
            raise ArtifactResolutionError(
                "Unable to publish the completed inference state.",
                error_code="INFERENCE_STATE_PUBLISH_FAILED",
                details={
                    "staging_directory": str(staging),
                    "target_directory": str(stable),
                    "original_error": str(error),
                },
            ) from error

        return stable

    def complete_stage(
        self,
        context: StageExecutionContext,
        *,
        result_payload: Any,
        metrics: Mapping[str, Any] | None = None,
        outputs: Sequence[OutputRegistration] = (),
    ) -> dict[str, Any]:
        with RunLock(context.run_directory):
            manifest = load_run_manifest(context.run_directory)

            stage = manifest["attempts"][context.attempt_id][
                "stages"
            ][context.stage_name]

            if stage.get("active_execution_id") != context.execution_id:
                raise RunManagementError(
                    "The execution is no longer the active stage execution.",
                    error_code="EXECUTION_IS_NOT_ACTIVE",
                    details={
                        "attempt_id": context.attempt_id,
                        "stage_name": context.stage_name,
                        "execution_id": context.execution_id,
                        "active_execution_id": stage.get(
                            "active_execution_id"
                        ),
                    },
                )

            result_dictionary = _payload_to_dict(result_payload)
            published_state: Path | None = None
            staging_state = context.inference_state_directory
            remapped_outputs: list[OutputRegistration] = []

            try:
                if context.stage_name == "inference_bootstrap":
                    if staging_state is None:
                        raise RunManagementError(
                            "Bootstrap execution does not have an inference-state staging directory.",
                            error_code="BOOTSTRAP_STATE_DIRECTORY_MISSING",
                        )

                    staging_root = staging_state.resolve()

                    for output in outputs:
                        source = Path(
                            output.file_path
                        ).expanduser().resolve()

                        try:
                            source.relative_to(staging_root)
                        except ValueError as error:
                            raise ArtifactResolutionError(
                                "Inference-bootstrap output is outside the staged saved-run root.",
                                error_code="BOOTSTRAP_OUTPUT_OUTSIDE_STATE_ROOT",
                                details={
                                    "output_file": str(source),
                                    "staging_root": str(staging_root),
                                },
                            ) from error

                        if not source.is_file():
                            raise ArtifactResolutionError(
                                "An inference-bootstrap output file does not exist.",
                                error_code="BOOTSTRAP_OUTPUT_NOT_FOUND",
                                details={
                                    "output_file": str(source),
                                },
                            )

                    published_state = self._publish_bootstrap_state(
                        context
                    )

                    result_dictionary = _remap_path_prefix(
                        result_dictionary,
                        old_root=staging_root,
                        new_root=published_state,
                    )

                    for output in outputs:
                        source = Path(
                            output.file_path
                        ).expanduser().resolve()
                        relative = source.relative_to(staging_root)

                        remapped_outputs.append(
                            OutputRegistration(
                                file_path=published_state / relative,
                                output_type=output.output_type,
                                include_checksum=output.include_checksum,
                            )
                        )

                else:
                    remapped_outputs = list(outputs)

                write_json_file(
                    data=result_dictionary,
                    file_path=context.result_file,
                    overwrite=True,
                )

                for output in remapped_outputs:
                    record_output_file(
                        context.run_directory,
                        output.file_path,
                        attempt_id=context.attempt_id,
                        stage_name=context.stage_name,
                        execution_id=context.execution_id,
                        output_type=output.output_type,
                        include_checksum=output.include_checksum,
                    )

                manifest = complete_stage_execution(
                    context.run_directory,
                    context.attempt_id,
                    context.stage_name,
                    context.execution_id,
                    result_file=context.result_file,
                    metrics=metrics,
                )

            except Exception:
                if (
                    published_state is not None
                    and staging_state is not None
                ):
                    try:
                        if (
                            published_state.exists()
                            and not staging_state.exists()
                        ):
                            replace_path_with_retry(
                                published_state,
                                staging_state,
                            )
                    except OSError:
                        pass
                raise

            if context.stage_name == "inference_bootstrap":
                manifest = self._activate_inference_attempt_locked(
                    context.run_directory,
                    context.attempt_id,
                )

            return self._derive_and_persist_statuses(
                context.run_directory,
                context.attempt_id,
            )

    def fail_stage(
        self,
        context: StageExecutionContext,
        error: Exception,
    ) -> dict[str, Any]:
        with RunLock(context.run_directory):
            manifest = load_run_manifest(
                context.run_directory
            )
            stage = manifest["attempts"][context.attempt_id][
                "stages"
            ][context.stage_name]
            execution = stage.get(
                "executions",
                {},
            ).get(
                context.execution_id
            )

            # Completion may already be durably committed even if a later
            # attempt/run-status or secondary-registry operation raised.
            # Never corrupt that successful execution by rewriting it failed.
            if (
                isinstance(execution, Mapping)
                and execution.get("status") == "completed"
                and stage.get("successful_execution_id")
                == context.execution_id
            ):
                return self._derive_and_persist_statuses(
                    context.run_directory,
                    context.attempt_id,
                )

            error_record = convert_exception_to_record(error)

            write_json_file(
                data=error_record,
                file_path=context.error_file,
                overwrite=True,
            )

            fail_stage_execution(
                context.run_directory,
                context.attempt_id,
                context.stage_name,
                context.execution_id,
                error,
                error_file=context.error_file,
            )

            return self._derive_and_persist_statuses(
                context.run_directory,
                context.attempt_id,
            )

    # ------------------------------------------------------------------
    # PART E
    # ------------------------------------------------------------------

    def _inference_request_paths(
        self,
        state_directory: Path,
        request_id: str,
    ) -> dict[str, Path]:
        cleaned_request_id = validate_record_id(
            request_id,
            field_name="request_id",
        )

        request_directory = (
            state_directory
            / "requests"
            / cleaned_request_id
        ).resolve()

        return {
            "request": request_directory,
            "request_file": request_directory / "request.json",
            "input": request_directory / "input",
            "result": request_directory / "result.json",
            "error": request_directory / "error.json",
        }

    def _prepare_inference_request_locked(
        self,
        *,
        run_context: RunContext,
        request_payload: Any,
        input_file: str | Path,
        metadata: Mapping[str, Any],
    ) -> InferenceRequestContext:
        manifest = load_run_manifest(run_context.run_directory)

        training_attempt_id, state_directory = (
            resolve_active_inference_state(
                manifest,
                run_context.run_directory,
            )
        )

        requests_root = state_directory / "requests"
        ensure_directory(requests_root)

        existing_ids = [
            path.name
            for path in requests_root.iterdir()
            if path.is_dir()
        ]

        request_id = generate_inference_request_id(existing_ids)

        paths = self._inference_request_paths(
            state_directory,
            request_id,
        )

        ensure_directory(paths["request"])
        ensure_directory(paths["input"])

        prepared_at_utc = _utc_now()

        try:
            request_file = write_json_file(
                data={
                    **_payload_to_dict(request_payload),
                    "_run_management": {
                        "request_id": request_id,
                        "training_attempt_id": training_attempt_id,
                        "prepared_at_utc": prepared_at_utc,
                        "metadata": dict(metadata),
                    },
                },
                file_path=paths["request_file"],
                overwrite=False,
            )

            source_path = Path(input_file).expanduser().resolve()

            copied_input = copy_input_file(
                source_path=source_path,
                destination_directory=paths["input"],
                destination_filename=source_path.name,
                overwrite=False,
            )

        except Exception:
            try:
                if paths["request"].exists():
                    remove_tree_with_retry(paths["request"])
            except OSError:
                pass
            raise

        return InferenceRequestContext(
            run_id=run_context.run_id,
            training_attempt_id=training_attempt_id,
            request_id=request_id,
            run_directory=run_context.run_directory,
            inference_state_directory=state_directory,
            request_directory=paths["request"],
            request_file=Path(request_file).resolve(),
            input_file=Path(copied_input).resolve(),
            result_file=paths["result"],
            error_file=paths["error"],
            prepared_at_utc=prepared_at_utc,
        )

    def inference_session(
        self,
        run_id: str,
        *,
        request_payload: Any,
        input_file: str | Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> _InferenceSession:
        return _InferenceSession(
            self,
            run_id,
            request_payload,
            input_file,
            metadata=metadata,
        )

    def _record_started_invocation_failure_locked(
        self,
        *,
        context: InferenceRequestContext,
        invocation_id: str,
        error: Exception,
    ) -> None:
        """
        Best-effort recovery for a Part E invocation that was already registered
        as running but failed while being finalized.

        This prevents an orphaned Manifest V2 status of "running".
        """
        error_record = convert_exception_to_record(error)

        error_record["details"] = {
            **dict(error_record.get("details", {})),
            "request_id": context.request_id,
            "request_file": str(context.request_file),
            "registration_phase": "invocation_finalization",
        }

        try:
            write_json_file(
                data=error_record,
                file_path=context.error_file,
                overwrite=True,
            )
        except Exception:
            # Do not replace the original persistence error.
            pass

        try:
            fail_inference_invocation(
                context.run_directory,
                invocation_id,
                error,
                error_file=context.error_file,
            )
        except Exception:
            # Preserve the original completion error. Atomic Manifest writes mean
            # the prior successful start record remains recovery evidence even if
            # this best-effort failed-state update also cannot be persisted.
            pass

    def _complete_inference_request_locked(
        self,
        *,
        context: InferenceRequestContext,
        invocation_id: str,
        result_payload: Any,
        output_directory: str | Path,
        metrics: Mapping[str, Any] | None,
    ) -> None:
        """
        Finalize a successful Part E result safely.

        FIX:
        - errors before invocation creation become request-level failures;
        - errors after invocation creation best-effort transition that invocation
          from running to failed;
        - registry sync failure after a completed manifest invocation does not
          turn a technically successful inference into a false failure because
          Manifest V2 is the technical truth and resolve_run() can refresh the
          rebuildable registry later.
        """
        cleaned_invocation_id = validate_record_id(
            invocation_id,
            field_name="invocation_id",
        )

        invocation_started = False

        try:
            output_path = Path(
                output_directory
            ).expanduser().resolve()

            try:
                output_path.relative_to(
                    context.inference_state_directory
                )
            except ValueError as error:
                raise ArtifactResolutionError(
                    "Part E returned an output directory outside the active saved-run root.",
                    error_code="INFERENCE_OUTPUT_OUTSIDE_STATE_DIRECTORY",
                    details={
                        "invocation_id": cleaned_invocation_id,
                        "output_directory": str(output_path),
                        "state_directory": str(
                            context.inference_state_directory
                        ),
                    },
                ) from error

            if not output_path.is_dir():
                raise ArtifactResolutionError(
                    "Part E returned an inference output directory that does not exist.",
                    error_code="INFERENCE_OUTPUT_DIRECTORY_NOT_FOUND",
                    details={
                        "invocation_id": cleaned_invocation_id,
                        "output_directory": str(output_path),
                    },
                )

            write_json_file(
                data=_payload_to_dict(result_payload),
                file_path=context.result_file,
                overwrite=True,
            )

            manifest = start_inference_invocation(
                context.run_directory,
                cleaned_invocation_id,
                training_attempt_id=context.training_attempt_id,
                request_file=context.request_file,
                input_file=context.input_file,
                output_directory=output_path,
                metadata={
                    "request_id": context.request_id,
                    "request_prepared_at_utc": context.prepared_at_utc,
                },
            )

            invocation_started = True

            try:
                manifest = complete_inference_invocation(
                    context.run_directory,
                    cleaned_invocation_id,
                    result_file=context.result_file,
                    metrics=metrics,
                )
            except Exception as error:
                self._record_started_invocation_failure_locked(
                    context=context,
                    invocation_id=cleaned_invocation_id,
                    error=error,
                )
                raise

        except Exception as error:
            if not invocation_started:
                try:
                    self._fail_inference_request_locked(
                        context=context,
                        error=error,
                    )
                except Exception:
                    # Preserve the original completion/validation error.
                    pass
            raise

        # The invocation is already completed in Manifest V2 at this point.
        # Registry is only a lookup/index layer. A registry write failure must
        # not misclassify the completed inference request as failed.
        self._sync_registry_best_effort(manifest)

    def _fail_inference_request_locked(
        self,
        *,
        context: InferenceRequestContext,
        error: Exception,
    ) -> None:
        error_record = convert_exception_to_record(error)

        error_record["details"] = {
            **dict(error_record.get("details", {})),
            "request_id": context.request_id,
            "request_file": str(context.request_file),
        }

        write_json_file(
            data=error_record,
            file_path=context.error_file,
            overwrite=True,
        )

        manifest = load_run_manifest(context.run_directory)

        manifest["errors"].append(
            {
                "timestamp_utc": get_current_utc_timestamp(),
                "scope": "inference_invocation",
                "training_attempt_id": context.training_attempt_id,
                **error_record,
            }
        )

        append_manifest_event(
            manifest,
            event_type="inference_request_failed",
            message=(
                "A Part E request failed before a successful invocation "
                "could be registered."
            ),
            details={
                "training_attempt_id": context.training_attempt_id,
                "request_id": context.request_id,
                "error_type": error_record["error_type"],
                "error_code": error_record["error_code"],
            },
        )

        save_run_manifest(
            context.run_directory,
            manifest,
        )

        # Registry is secondary. Do not replace the actual request failure with
        # a registry-index synchronization failure.
        self._sync_registry_best_effort(manifest)


__all__ = [
    "RunManager",
]