import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.adapters.resilient_tutu import ProtectedTutuGateway
from app.bot.conversation import TripConversation
from app.bot.discovery_conversation import DiscoveryConversation
from app.domain.errors import ProviderTransientError
from app.domain.operational_models import HealthStatus
from app.services.readiness import build_readiness_report
from app.services.resilience import CircuitBreaker, CircuitState
from app.services.url_policy import require_allowed_https_url


class FakeClock:
    def now(self):
        return datetime(2026, 7, 22, 12, tzinfo=UTC)


def test_circuit_breaker_opens_then_allows_one_half_open_probe() -> None:
    now = 0.0
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_seconds=10,
        monotonic=lambda: now,
    )
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    with pytest.raises(ProviderTransientError, match="open"):
        breaker.before_call()

    now = 11
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.before_call()
    with pytest.raises(ProviderTransientError, match="probe"):
        breaker.before_call()
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_protected_gateway_enforces_call_budget_and_circuit() -> None:
    class Gateway:
        def __init__(self):
            self.fail = True

        async def search_transport(self, query):
            if self.fail:
                raise ProviderTransientError("offline")
            return []

    delegate = Gateway()
    protected = ProtectedTutuGateway(
        delegate,  # type: ignore[arg-type]
        calls_per_minute=3,
        failure_threshold=2,
        recovery_seconds=60,
    )
    with pytest.raises(ProviderTransientError):
        await protected.search_transport(object())
    with pytest.raises(ProviderTransientError):
        await protected.search_transport(object())
    assert protected.circuit_state == "open"
    with pytest.raises(ProviderTransientError, match="circuit breaker"):
        await protected.search_transport(object())


@pytest.mark.asyncio
async def test_protected_gateway_rejects_calls_above_minute_budget() -> None:
    delegate = SimpleNamespace(search_transport=AsyncMock(return_value=[]))
    protected = ProtectedTutuGateway(
        delegate,  # type: ignore[arg-type]
        calls_per_minute=1,
    )

    assert await protected.search_transport(object()) == []
    with pytest.raises(ProviderTransientError, match="budget"):
        await protected.search_transport(object())

    assert delegate.search_transport.await_count == 1
    assert protected.metrics["accepted_calls"] == 1
    assert protected.metrics["rejected_calls"] == 1


@pytest.mark.asyncio
async def test_protected_gateway_bounds_provider_concurrency() -> None:
    active = 0
    maximum = 0

    async def search_transport(query):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return [query]

    delegate = SimpleNamespace(search_transport=search_transport)
    protected = ProtectedTutuGateway(
        delegate,  # type: ignore[arg-type]
        max_inflight=2,
        calls_per_minute=10,
    )

    results = await asyncio.gather(*(protected.search_transport(index) for index in range(6)))

    assert results == [[0], [1], [2], [3], [4], [5]]
    assert maximum == 2


def test_url_policy_blocks_credentials_ip_literals_and_non_allowlisted_hosts() -> None:
    allowed = frozenset({"tutu.ru"})
    require_allowed_https_url(
        "https://www.tutu.ru/path",
        allowed_hosts=allowed,
        allow_subdomains=True,
    )
    for value in (
        "http://tutu.ru/path",
        "https://user:pass@tutu.ru/path",
        "https://127.0.0.1/path",
        "https://tutu.ru.evil.example/path",
    ):
        with pytest.raises(ValueError):
            require_allowed_https_url(
                value,
                allowed_hosts=allowed,
                allow_subdomains=True,
            )


def test_readiness_is_structured_and_fails_closed() -> None:
    ready = build_readiness_report(
        clock=FakeClock(),
        catalog_version="v1",
        destination_count=10,
        provider_schema_hash="abc",
        provider_circuit_state="closed",
        kill_switch=False,
    )
    assert ready.status is HealthStatus.OK

    stopped = build_readiness_report(
        clock=FakeClock(),
        catalog_version="v1",
        destination_count=10,
        provider_schema_hash="abc",
        provider_circuit_state="open",
        kill_switch=True,
    )
    assert stopped.status is HealthStatus.NOT_READY
    assert not stopped.checks["kill_switch_off"]


@pytest.mark.asyncio
async def test_kill_switch_prevents_known_flow_before_llm_or_provider_calls() -> None:
    conversation = TripConversation(
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
        enabled=False,
    )
    message = SimpleNamespace(reply_text=AsyncMock())
    context = SimpleNamespace(user_data={"old": "value"})

    state = await conversation.start(SimpleNamespace(effective_message=message), context)

    assert state == -1
    assert "приостановлен" in message.reply_text.call_args.args[0]
    assert context.user_data == {"old": "value"}


@pytest.mark.asyncio
async def test_discovery_feature_flag_stops_before_llm_catalog_or_provider_calls() -> None:
    conversation = DiscoveryConversation(
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
        enabled=False,
    )
    message = SimpleNamespace(reply_text=AsyncMock())

    state = await conversation.start(
        SimpleNamespace(effective_message=message),
        SimpleNamespace(user_data={}),
    )

    assert state == -1
    assert "временно отключён" in message.reply_text.call_args.args[0]
