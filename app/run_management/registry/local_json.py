"""
V1 filesystem-backed run registry.

The registry stores only lightweight run lookup/index data. Detailed attempts,
executions, artifacts, errors, and event history remain in each run's
manifest.json.

JSON writes are atomic through file_service.write_json_file(), while an
exclusive registry lock prevents read-modify-write lost updates.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import (
    FileProcessingError,
    InputValidationError,
    RunLockError,
    RunNotFoundError,
    RunRegistryError,
)
from app.services.file_service import (
    ensure_directory,
    read_json_file,
    write_json_file,
)
from app.services.run_directory_service import (
    validate_run_id,
)

from app.run_management.locking import (
    FileLock,
    get_registry_lock_file,
)
from app.run_management.models import (
    RunRegistryRecord,
)

from .base import RunRegistry


REGISTRY_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


class LocalJsonRunRegistry(RunRegistry):
    """Atomic JSON implementation of the RunRegistry interface."""

    def __init__(
        self,
        registry_file: str | Path | None = None,
        *,
        lock_timeout_seconds: float | None = None,
    ) -> None:
        self.registry_file = Path(
            (
                settings.run_registry_file
                if registry_file is None
                else registry_file
            )
        ).expanduser().resolve()

        self.lock_timeout_seconds = (
            settings.run_lock_timeout_seconds
            if lock_timeout_seconds is None
            else float(
                lock_timeout_seconds
            )
        )

        ensure_directory(
            self.registry_file.parent
        )

        self._ensure_document()

    @property
    def lock_file(self) -> Path:
        return get_registry_lock_file(
            self.registry_file
        )

    @contextmanager
    def _locked_registry(self):
        """
        Acquire the registry lock while normalizing lock-layer failures into
        the RunRegistryError contract consumed by RunManager.
        """
        try:
            with FileLock(
                self.lock_file,
                timeout_seconds=(
                    self.lock_timeout_seconds
                ),
            ):
                yield
        except RunLockError as error:
            raise RunRegistryError(
                "Unable to lock the local run registry.",
                error_code="RUN_REGISTRY_LOCK_FAILED",
                details={
                    "registry_file": str(
                        self.registry_file
                    ),
                    "lock_file": str(
                        self.lock_file
                    ),
                    "original_error": str(
                        error
                    ),
                },
            ) from error
        except OSError as error:
            raise RunRegistryError(
                "Unable to access the local run-registry lock.",
                error_code="RUN_REGISTRY_LOCK_FAILED",
                details={
                    "registry_file": str(
                        self.registry_file
                    ),
                    "lock_file": str(
                        self.lock_file
                    ),
                    "original_error": str(
                        error
                    ),
                },
            ) from error

    def _empty_document(
        self,
    ) -> dict[str, Any]:
        timestamp = _utc_now()

        return {
            "registry_version": (
                REGISTRY_VERSION
            ),
            "created_at_utc": (
                timestamp
            ),
            "updated_at_utc": (
                timestamp
            ),
            "runs": {},
        }

    def _validate_document(
        self,
        document: Any,
    ) -> dict[str, Any]:
        if not isinstance(
            document,
            dict,
        ):
            raise RunRegistryError(
                "The run registry must contain a JSON object.",
                error_code="INVALID_RUN_REGISTRY_TYPE",
                details={
                    "registry_file": str(
                        self.registry_file
                    ),
                    "received_type": (
                        type(
                            document
                        ).__name__
                    ),
                },
            )

        if (
            document.get(
                "registry_version"
            )
            != REGISTRY_VERSION
        ):
            raise RunRegistryError(
                "The run registry version is unsupported.",
                error_code="UNSUPPORTED_RUN_REGISTRY_VERSION",
                details={
                    "registry_file": str(
                        self.registry_file
                    ),
                    "received_version": (
                        document.get(
                            "registry_version"
                        )
                    ),
                    "supported_version": (
                        REGISTRY_VERSION
                    ),
                },
            )

        if not isinstance(
            document.get(
                "runs"
            ),
            dict,
        ):
            raise RunRegistryError(
                "The run registry runs field must be a dictionary.",
                error_code="INVALID_RUN_REGISTRY_STRUCTURE",
                details={
                    "registry_file": str(
                        self.registry_file
                    ),
                },
            )

        return document

    def _read_document(
        self,
    ) -> dict[str, Any]:
        try:
            document = read_json_file(
                self.registry_file
            )
        except (
            FileProcessingError,
            InputValidationError,
        ) as error:
            raise RunRegistryError(
                "Unable to read the local run registry.",
                error_code="RUN_REGISTRY_READ_FAILED",
                details={
                    "registry_file": str(
                        self.registry_file
                    ),
                    "original_error": str(
                        error
                    ),
                },
            ) from error

        return self._validate_document(
            document
        )

    def _write_document(
        self,
        document: dict[str, Any],
    ) -> None:
        document["updated_at_utc"] = (
            _utc_now()
        )

        try:
            write_json_file(
                data=document,
                file_path=(
                    self.registry_file
                ),
                overwrite=True,
            )
        except (
            FileProcessingError,
            InputValidationError,
        ) as error:
            raise RunRegistryError(
                "Unable to write the local run registry.",
                error_code="RUN_REGISTRY_WRITE_FAILED",
                details={
                    "registry_file": str(
                        self.registry_file
                    ),
                    "original_error": str(
                        error
                    ),
                },
            ) from error

    def _ensure_document(
        self,
    ) -> None:
        with self._locked_registry():
            if self.registry_file.exists():
                self._read_document()
                return

            self._write_document(
                self._empty_document()
            )

    def create(
        self,
        record: RunRegistryRecord,
    ) -> RunRegistryRecord:
        cleaned_run_id = validate_run_id(
            record.run_id
        )

        with self._locked_registry():
            document = self._read_document()
            runs = document["runs"]

            if cleaned_run_id in runs:
                raise RunRegistryError(
                    "The run already exists in the registry.",
                    error_code="RUN_REGISTRY_ENTRY_EXISTS",
                    details={
                        "run_id": (
                            cleaned_run_id
                        ),
                    },
                )

            runs[
                cleaned_run_id
            ] = record.to_dict()

            self._write_document(
                document
            )

        return record

    def get(
        self,
        run_id: str,
    ) -> RunRegistryRecord:
        cleaned_run_id = validate_run_id(
            run_id
        )

        document = self._read_document()
        raw_record = document[
            "runs"
        ].get(
            cleaned_run_id
        )

        if raw_record is None:
            raise RunNotFoundError(
                "The requested run does not exist in the run registry.",
                error_code="RUN_NOT_FOUND",
                details={
                    "run_id": (
                        cleaned_run_id
                    ),
                },
            )

        if not isinstance(
            raw_record,
            dict,
        ):
            raise RunRegistryError(
                "The stored run-registry entry is invalid.",
                error_code="INVALID_RUN_REGISTRY_ENTRY",
                details={
                    "run_id": (
                        cleaned_run_id
                    ),
                },
            )

        try:
            record = (
                RunRegistryRecord
                .from_dict(
                    raw_record
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise RunRegistryError(
                "The stored run-registry entry cannot be decoded.",
                error_code="INVALID_RUN_REGISTRY_ENTRY",
                details={
                    "run_id": (
                        cleaned_run_id
                    ),
                    "original_error": str(
                        error
                    ),
                },
            ) from error

        if record.run_id != cleaned_run_id:
            raise RunRegistryError(
                "The registry key does not match the record run ID.",
                error_code="RUN_REGISTRY_ID_MISMATCH",
                details={
                    "registry_key": (
                        cleaned_run_id
                    ),
                    "record_run_id": (
                        record.run_id
                    ),
                },
            )

        return record

    def update(
        self,
        record: RunRegistryRecord,
    ) -> RunRegistryRecord:
        cleaned_run_id = validate_run_id(
            record.run_id
        )

        with self._locked_registry():
            document = self._read_document()
            runs = document["runs"]

            if cleaned_run_id not in runs:
                raise RunNotFoundError(
                    "The run cannot be updated because it is not registered.",
                    error_code="RUN_NOT_FOUND",
                    details={
                        "run_id": (
                            cleaned_run_id
                        ),
                    },
                )

            runs[
                cleaned_run_id
            ] = record.to_dict()

            self._write_document(
                document
            )

        return record

    def exists(
        self,
        run_id: str,
    ) -> bool:
        cleaned_run_id = validate_run_id(
            run_id
        )

        document = self._read_document()

        return (
            cleaned_run_id
            in document["runs"]
        )

    def list_runs(
        self,
    ) -> list[RunRegistryRecord]:
        document = self._read_document()

        records: list[
            RunRegistryRecord
        ] = []

        for run_id in sorted(
            document["runs"]
        ):
            raw_record = document[
                "runs"
            ][
                run_id
            ]

            if not isinstance(
                raw_record,
                dict,
            ):
                raise RunRegistryError(
                    "The run registry contains an invalid entry.",
                    error_code="INVALID_RUN_REGISTRY_ENTRY",
                    details={
                        "run_id": run_id,
                    },
                )

            try:
                records.append(
                    RunRegistryRecord.from_dict(
                        raw_record
                    )
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise RunRegistryError(
                    "The run registry contains an undecodable entry.",
                    error_code="INVALID_RUN_REGISTRY_ENTRY",
                    details={
                        "run_id": run_id,
                        "original_error": str(
                            error
                        ),
                    },
                ) from error

        return records


__all__ = [
    "REGISTRY_VERSION",
    "LocalJsonRunRegistry",
]
