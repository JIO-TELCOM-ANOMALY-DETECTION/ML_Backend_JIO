"""
Abstract run-registry interface.

The registry is a searchable/indexing layer. Manifest V2 and the run folder
remain the detailed technical truth of a run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.run_management.models import (
    RunRegistryRecord,
)


class RunRegistry(ABC):
    """Storage-independent interface consumed by RunManager."""

    @abstractmethod
    def create(
        self,
        record: RunRegistryRecord,
    ) -> RunRegistryRecord:
        """Create one new registry entry."""

    @abstractmethod
    def get(
        self,
        run_id: str,
    ) -> RunRegistryRecord:
        """Load one registry entry."""

    @abstractmethod
    def update(
        self,
        record: RunRegistryRecord,
    ) -> RunRegistryRecord:
        """Replace one existing registry entry."""

    @abstractmethod
    def exists(
        self,
        run_id: str,
    ) -> bool:
        """Return whether the registry contains run_id."""

    @abstractmethod
    def list_runs(
        self,
    ) -> list[RunRegistryRecord]:
        """Return all known runs."""


__all__ = [
    "RunRegistry",
]
