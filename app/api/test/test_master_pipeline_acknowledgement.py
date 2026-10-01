from __future__ import annotations

import threading
from types import SimpleNamespace

from app.api import pipeline
from app.services.background_runner import BackgroundRunner


def test_background_runner_admits_bounded_work_and_returns_before_completion():
    started = threading.Event()
    release = threading.Event()
    runner = BackgroundRunner(max_workers=1, max_pending=1)
    try:
        future = runner.submit(lambda: (started.set(), release.wait(2)))
        assert started.wait(1)
        assert not future.done()
        try:
            runner.submit(lambda: None)
        except RuntimeError:
            pass
        else:
            raise AssertionError("runner admitted work beyond configured capacity")
        release.set()
        future.result(timeout=2)
    finally:
        runner.shutdown()


def test_worker_calls_existing_master_pipeline_once(monkeypatch):
    calls = []
    request = SimpleNamespace(run_id="run_test")
    manager = object()
    logger = object()
    monkeypatch.setattr(pipeline, "get_workflow_logger", lambda *args, **kwargs: logger)
    monkeypatch.setattr(
        pipeline,
        "run_training_pipeline",
        lambda **kwargs: calls.append(kwargs),
    )

    pipeline._run_accepted_master_pipeline(request=request, manager=manager)

    assert calls == [{"request": request, "manager": manager, "logger": logger}]