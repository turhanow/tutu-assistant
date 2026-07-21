"""Cheap structured readiness calculation after startup contract checks."""

from app.domain.operational_models import HealthStatus, ReadinessReport
from app.ports.clock import Clock


def build_readiness_report(
    *,
    clock: Clock,
    catalog_version: str | None,
    destination_count: int,
    provider_schema_hash: str | None,
    provider_circuit_state: str,
    kill_switch: bool,
) -> ReadinessReport:
    checks = {
        "catalog_loaded": bool(catalog_version and destination_count > 0),
        "provider_contract_loaded": bool(provider_schema_hash),
        "provider_circuit_available": provider_circuit_state != "open",
        "kill_switch_off": not kill_switch,
    }
    return ReadinessReport(
        status=HealthStatus.OK if all(checks.values()) else HealthStatus.NOT_READY,
        checked_at=clock.now(),
        checks=checks,
        catalog_version=catalog_version,
        provider_schema_hash=provider_schema_hash,
        provider_circuit_state=provider_circuit_state,
    )
