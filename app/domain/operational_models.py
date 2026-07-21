"""Structured liveness and readiness state."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import model_validator

from app.domain.models import DomainModel


class HealthStatus(StrEnum):
    OK = "ok"
    NOT_READY = "not_ready"


class ReadinessReport(DomainModel):
    status: HealthStatus
    checked_at: datetime
    checks: dict[str, bool]
    catalog_version: str | None = None
    provider_schema_hash: str | None = None
    provider_circuit_state: str

    @model_validator(mode="after")
    def validate_report(self) -> ReadinessReport:
        if self.checked_at.tzinfo is None:
            raise ValueError("readiness timestamp must be timezone-aware")
        expected = HealthStatus.OK if all(self.checks.values()) else HealthStatus.NOT_READY
        if self.status is not expected:
            raise ValueError("readiness status must match checks")
        return self
