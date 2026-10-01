# Master Pipeline Request Acknowledgement Return

The additive endpoint `POST /api/v1/pipeline/master-pipeline-request-acknowledgement-return` validates the existing `TrainingPipelineRequest` and input CSV, rejects a supplied `run_id`, creates one run using `RunManager.create_run(run_metadata=...)`, copies the request with the generated run ID, and submits it to a bounded standard-library `ThreadPoolExecutor`.

The worker calls the existing `run_training_pipeline(request=..., manager=..., logger=...)` exactly once. Only the run is pre-created because the existing master pipeline and Part A own attempt creation. The acknowledgement returns `attempt_id: null`; Part A creates exactly `attempt_0001`, and the wrapper never calls `create_attempt`.

The client polls `GET /api/v1/runs/{run_id}` until `latest_attempt_id` is available, then reads `GET /api/v1/runs/{run_id}/attempts/{attempt_id}` for progress. The runner uses two workers and admits at most eight pending tasks. Futures are internal bookkeeping only. Submission failure is logged and returned as an error, while the created run is retained for inspection; no new lifecycle state is invented. Unexpected worker exceptions are logged and existing workflow persistence remains authoritative.

Accepted work survives browser disconnect while the backend process remains alive, but is **not guaranteed to survive Python/server/computer/container restart**. Cancellation is not supported.

Changed files are `app/services/background_runner.py`, `app/api/pipeline.py`, the workflow response exports, and `app/main.py`. Focused tests cover bounded admission and the exact single worker call without invoking real ML. Compilation and generated OpenAPI checks cover importability and route registration; the full pytest suite could not run because pytest is not installed in the available environment. `training_pipeline.py`, Parts A-D, `inference_bootstrap.py`, `app/ml/**`, and `reporting_backend/**` are unchanged.