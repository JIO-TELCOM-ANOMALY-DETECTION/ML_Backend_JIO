"""Small bounded in-process runner for accepted background workflows."""

from __future__ import annotations

import atexit
import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
from typing import Any


class BackgroundRunner:
    """Bounded ThreadPoolExecutor facade with safe admission and logging."""

    def __init__(self, *, max_workers: int = 2, max_pending: int = 8,
                 logger: logging.Logger | None = None) -> None:
        if max_workers < 1 or max_pending < 1:
            raise ValueError("Background runner limits must be positive.")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="master-pipeline"
        )
        self._admission = BoundedSemaphore(max_pending)
        self._logger = logger or logging.getLogger(
            "telecom_anomaly_backend.background_runner"
        )
        self._submit_lock = Lock()
        self._shutdown = False

    def submit(self, function: Callable[..., Any], /, *args: Any,
               **kwargs: Any) -> Future[Any]:
        acquired = self._admission.acquire(blocking=False)
        if not acquired:
            raise RuntimeError("Background runner capacity is full.")
        try:
            with self._submit_lock:
                if self._shutdown:
                    raise RuntimeError("Background runner is shut down.")
                future = self._executor.submit(function, *args, **kwargs)
        except Exception:
            self._admission.release()
            raise
        future.add_done_callback(self._on_done)
        return future

    def _on_done(self, future: Future[Any]) -> None:
        self._admission.release()
        try:
            future.result()
        except Exception:
            self._logger.exception(
                "Unexpected exception escaped an accepted background task."
            )

    def shutdown(self, *, wait: bool = True) -> None:
        with self._submit_lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=wait)


_runner = BackgroundRunner()
atexit.register(_runner.shutdown)


def get_background_runner() -> BackgroundRunner:
    return _runner


__all__ = ["BackgroundRunner", "get_background_runner"]
