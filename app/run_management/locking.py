"""
Portable filesystem locks used by Run Management.

A run-level lock serializes operations that mutate one run. The same primitive
is also used by the local JSON registry to prevent lost-update races.

Part E must hold the run lock for its entire inference call because Part E
mutates persistent history/cache state internally.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from types import TracebackType

from app.core.config import settings
from app.core.exceptions import (
    RunLockError,
)
from app.services.file_service import (
    ensure_directory,
)


logger = logging.getLogger(__name__)


class FileLock:
    """
    Small cross-platform exclusive file lock using only the standard library.

    The lock file itself is persistent; the operating-system lock, not file
    existence, represents ownership. This avoids stale lock-file deadlocks after
    process crashes.
    """

    def __init__(
        self,
        lock_file: str | Path,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.lock_file = Path(
            lock_file
        ).expanduser().resolve()

        self.timeout_seconds = float(
            (
                settings.run_lock_timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            )
        )

        self.poll_interval_seconds = float(
            poll_interval_seconds
        )

        if self.timeout_seconds <= 0:
            raise RunLockError(
                "Lock timeout must be positive.",
                error_code="INVALID_LOCK_TIMEOUT",
                details={
                    "timeout_seconds": (
                        self.timeout_seconds
                    ),
                },
            )

        if self.poll_interval_seconds <= 0:
            raise RunLockError(
                "Lock poll interval must be positive.",
                error_code="INVALID_LOCK_POLL_INTERVAL",
                details={
                    "poll_interval_seconds": (
                        self.poll_interval_seconds
                    ),
                },
            )

        self._handle = None

    @property
    def acquired(self) -> bool:
        """Return whether this object currently owns the OS-level lock."""
        return self._handle is not None

    def _try_lock(
        self,
        handle,
    ) -> bool:
        if os.name == "nt":
            import msvcrt

            try:
                handle.seek(0)
                msvcrt.locking(
                    handle.fileno(),
                    msvcrt.LK_NBLCK,
                    1,
                )
                return True
            except OSError:
                return False

        import fcntl

        try:
            fcntl.flock(
                handle.fileno(),
                (
                    fcntl.LOCK_EX
                    | fcntl.LOCK_NB
                ),
            )
            return True
        except BlockingIOError:
            return False

    def _unlock(
        self,
        handle,
    ) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(
                handle.fileno(),
                msvcrt.LK_UNLCK,
                1,
            )
            return

        import fcntl

        fcntl.flock(
            handle.fileno(),
            fcntl.LOCK_UN,
        )

    def acquire(self) -> "FileLock":
        """
        Acquire the exclusive lock or raise after the configured timeout.
        """
        if self.acquired:
            raise RunLockError(
                "This lock object is already acquired.",
                error_code="LOCK_ALREADY_ACQUIRED",
                details={
                    "lock_file": str(
                        self.lock_file
                    ),
                },
            )

        ensure_directory(
            self.lock_file.parent
        )

        # Keep one byte present because Windows region locking requires a
        # lockable byte range.
        handle = open(
            self.lock_file,
            "a+b",
        )

        try:
            handle.seek(
                0,
                os.SEEK_END,
            )

            if handle.tell() == 0:
                handle.write(
                    b"0"
                )
                handle.flush()

            deadline = (
                time.monotonic()
                + self.timeout_seconds
            )

            while True:
                if self._try_lock(
                    handle
                ):
                    self._handle = (
                        handle
                    )
                    return self

                if (
                    time.monotonic()
                    >= deadline
                ):
                    raise RunLockError(
                        "Timed out while waiting for an exclusive backend lock.",
                        error_code="RUN_LOCK_TIMEOUT",
                        details={
                            "lock_file": str(
                                self.lock_file
                            ),
                            "timeout_seconds": (
                                self.timeout_seconds
                            ),
                        },
                    )

                time.sleep(
                    self.poll_interval_seconds
                )

        except Exception:
            handle.close()
            raise

    def release(self) -> None:
        """
        Release the operating-system lock.

        Closing the file descriptor/handle is itself an OS-level lock-release
        mechanism. If the explicit unlock call fails but handle.close()
        succeeds, treat cleanup as recovered and log the anomaly rather than
        converting an already-committed backend transition into HTTP 500.
        """
        handle = self._handle

        if handle is None:
            return

        self._handle = None
        unlock_error: OSError | None = None

        try:
            self._unlock(
                handle
            )
        except OSError as error:
            unlock_error = error

        try:
            handle.close()
        except OSError as close_error:
            raise RunLockError(
                "Unable to release the backend lock cleanly.",
                error_code="RUN_LOCK_RELEASE_FAILED",
                details={
                    "lock_file": str(
                        self.lock_file
                    ),
                    "unlock_error": (
                        str(unlock_error)
                        if unlock_error is not None
                        else None
                    ),
                    "close_error": str(
                        close_error
                    ),
                },
            ) from close_error

        if unlock_error is not None:
            logger.warning(
                "Explicit unlock failed for %s, but closing the handle "
                "completed lock cleanup: %s",
                self.lock_file,
                unlock_error,
            )

    def __enter__(
        self,
    ) -> "FileLock":
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.release()
        return False


class RunLock(FileLock):
    """Exclusive lock for one logical run."""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        run_root = Path(
            run_directory
        ).expanduser().resolve()

        super().__init__(
            run_root / ".run.lock",
            timeout_seconds=(
                timeout_seconds
            ),
        )


def get_registry_lock_file(
    registry_file: str | Path,
) -> Path:
    """Return the lock-file location for one registry JSON file."""
    path = Path(
        registry_file
    ).expanduser().resolve()

    return path.with_name(
        path.name + ".lock"
    )


__all__ = [
    "FileLock",
    "RunLock",
    "get_registry_lock_file",
]
