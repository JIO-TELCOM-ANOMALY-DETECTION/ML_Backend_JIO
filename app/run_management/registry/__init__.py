"""
Run-registry implementations.

V1 uses LocalJsonRunRegistry. A future database-backed implementation can
implement the same RunRegistry interface without changing workflows.
"""

from .base import (
    RunRegistry,
)
from .local_json import (
    REGISTRY_VERSION,
    LocalJsonRunRegistry,
)


__all__ = [
    "RunRegistry",
    "REGISTRY_VERSION",
    "LocalJsonRunRegistry",
]
