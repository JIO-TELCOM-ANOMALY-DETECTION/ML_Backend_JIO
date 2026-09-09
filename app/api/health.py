"""
Production FastAPI health-check route.

This endpoint is intentionally limited to application liveness. It confirms
that the FastAPI process is running and able to respond to HTTP requests.

It deliberately does NOT inspect:
- Run Management or the run registry;
- manifests or run directories;
- ML models or saved inference state;
- history/cache state;
- Parts A-E dependencies;
- filesystem readiness;
- external services.

Deeper dependency/readiness checks, if added in the future, should use a
separate endpoint rather than changing this liveness contract.
"""

from __future__ import annotations

from fastapi import APIRouter


router = APIRouter(
    tags=["health"],
)


@router.get(
    "/health",
    summary="Check application health",
)
def health_check() -> dict[str, str]:
    """Return the fixed application-liveness response."""
    return {
        "status": "healthy",
    }


__all__ = [
    "router",
]
