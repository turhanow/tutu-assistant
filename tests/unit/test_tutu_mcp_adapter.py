import asyncio
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.adapters.tutu_mcp import (
    TutuMcpAdapter,
    _merge_transport_searches,
    validate_required_capabilities,
)
from app.domain.errors import ProviderContractError, ProviderToolError, ProviderTransientError
from app.domain.models import (
    HotelMode,
    HotelPreferences,
    HotelSearchQuery,
    OfferRef,
    ToolCapability,
    TransportMode,
    TransportSearchQuery,
)

FIXTURES = Path("tests/fixtures/tutu_mcp/fixtures")


class FakeSession:
    def __init__(
        self,
        payload: dict,
        *,
        is_error: bool = False,
        failures: list[Exception] | None = None,
    ) -> None:
        self.payload = payload
        self.is_error = is_error
        self.failures = list(failures or [])
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(
            isError=self.is_error,
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
        )


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def adapter_with(session: FakeSession) -> TutuMcpAdapter:
    adapter = TutuMcpAdapter("https://mcp.tutu.ru/mcp")

    async def submit(
        action: str,
        *,
        name: str | None = None,
        arguments: dict | None = None,
        retry_transient: bool = True,
    ):
        assert action == "call_tool"
        return await adapter._call_session(
            session,  # type: ignore[arg-type]
            name or "",
            arguments or {},
            retry_transient=retry_transient,
        )

    adapter._submit = submit  # type: ignore[method-assign]
    return adapter


def test_transport_comparison_keeps_successful_objective_when_other_one_fails() -> None:
    offers = _merge_transport_searches(
        (fixture("transport-search.json"), ProviderTransientError("time sort failed"))
    )

    assert offers
    assert offers[0].service_number == "128М"


@pytest.mark.asyncio
async def test_transport_query_uses_captured_schema_names() -> None:
    session = FakeSession(fixture("transport-search.json"))
    adapter = adapter_with(session)

    offers = await adapter.search_transport(
        TransportSearchQuery(
            origin="Москва",
            destination="Казань",
            departure_date=date(2026, 8, 21),
            allowed_modes={TransportMode.RAIL, TransportMode.BUS},
        )
    )

    assert len(offers) == 2
    assert len(session.calls) == 2
    assert {arguments["optimize_for"] for _, arguments in session.calls} == {"price", "time"}
    for name, arguments in session.calls:
        assert name == "search_multitransport"
        assert arguments["modes"] == ["bus", "railway"]
        assert arguments["departure_date"] == "2026-08-21"
        assert None not in arguments.values()


@pytest.mark.asyncio
async def test_hotel_query_maps_hard_filters() -> None:
    session = FakeSession(fixture("hotel-search.json"))
    adapter = adapter_with(session)

    await adapter.search_hotels(
        HotelSearchQuery(
            city="Казань",
            check_in=date(2026, 8, 21),
            check_out=date(2026, 8, 23),
            preferences=HotelPreferences(
                mode=HotelMode.REQUIRED,
                stars_min=3,
                free_cancellation_required=True,
            ),
        )
    )

    _, arguments = session.calls[0]
    assert arguments["stars"] == [3, 4, 5]
    assert arguments["free_cancellation"] is True


@pytest.mark.asyncio
async def test_single_rail_mode_routes_to_targeted_tool() -> None:
    session = FakeSession(fixture("rail-search.json"))
    adapter = adapter_with(session)

    offers = await adapter.search_transport(
        TransportSearchQuery(
            origin="Москва",
            destination="Казань",
            departure_date=date(2026, 8, 21),
            allowed_modes={TransportMode.RAIL},
        )
    )

    assert offers
    assert len(session.calls) == 2
    assert {arguments["sort"] for _, arguments in session.calls} == {
        "price_asc",
        "duration_asc",
    }
    for name, arguments in session.calls:
        assert name == "search_rail"
        assert arguments["passengers"] == 1


@pytest.mark.asyncio
async def test_tool_error_is_typed() -> None:
    adapter = adapter_with(FakeSession({}, is_error=True))

    with pytest.raises(ProviderToolError):
        await adapter.search_transport(
            TransportSearchQuery(
                origin="Москва",
                destination="Казань",
                departure_date=date(2026, 8, 21),
            )
        )


@pytest.mark.asyncio
async def test_read_only_search_retries_one_transient_failure(monkeypatch) -> None:
    session = FakeSession(
        fixture("transport-search.json"),
        failures=[httpx.ConnectError("temporary")],
    )
    adapter = adapter_with(session)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("app.adapters.tutu_mcp.asyncio.sleep", no_sleep)
    offers = await adapter.search_transport(
        TransportSearchQuery(
            origin="Москва",
            destination="Казань",
            departure_date=date(2026, 8, 21),
        )
    )

    assert offers
    assert len(session.calls) == 3


@pytest.mark.asyncio
async def test_checkout_does_not_retry_transient_failure() -> None:
    session = FakeSession({}, failures=[httpx.ConnectError("temporary")])
    adapter = adapter_with(session)
    ref = OfferRef(product_type="hotels", checkout_data={"transport": "hotels"})

    with pytest.raises(ProviderTransientError):
        await adapter.create_checkout_link(ref)

    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_transport_details_ref_remains_nested() -> None:
    session = FakeSession({"title": "Поезд 128М"})
    adapter = adapter_with(session)
    ref = OfferRef(
        product_type="railway",
        details_data={"train_number": "128М", "source": "fixture"},
    )

    details = await adapter.get_offer_details(ref)

    assert details.title == "Поезд 128М"
    _, arguments = session.calls[0]
    assert arguments == {
        "product_type": "railway",
        "details_ref": {"train_number": "128М", "source": "fixture"},
    }


@pytest.mark.asyncio
async def test_submit_has_hard_timeout_and_aborts_stuck_session() -> None:
    adapter = TutuMcpAdapter("https://mcp.tutu.ru/mcp")
    adapter._timeout = timedelta(milliseconds=10)
    adapter._commands = asyncio.Queue()
    adapter._runner = asyncio.create_task(asyncio.sleep(60))

    with pytest.raises(ProviderTransientError, match="timed out"):
        await adapter._submit("call_tool", name="get_offer_details")

    assert adapter._runner is None


@pytest.mark.asyncio
async def test_next_operation_reconnects_after_session_timeout() -> None:
    adapter = TutuMcpAdapter("https://mcp.tutu.ru/mcp")
    adapter._timeout = timedelta(milliseconds=10)
    adapter._commands = asyncio.Queue()
    adapter._runner = asyncio.create_task(asyncio.sleep(60))

    with pytest.raises(ProviderTransientError):
        await adapter._submit("call_tool", name="get_offer_details")

    starts = 0

    async def reconnect() -> None:
        nonlocal starts
        starts += 1
        commands: asyncio.Queue = asyncio.Queue()
        adapter._commands = commands

        async def serve_one() -> None:
            command = await commands.get()
            command.future.set_result({"reconnected": True})

        adapter._runner = asyncio.create_task(serve_one())

    adapter.start = reconnect  # type: ignore[method-assign]

    result = await adapter._submit("call_tool", name="get_offer_details")

    assert result == {"reconnected": True}
    assert starts == 1


def test_capability_validation_allows_additions_but_blocks_breaking_drift() -> None:
    tools = tuple(
        ToolCapability(
            name=name,
            input_schema={"properties": {field: {} for field in fields | {"future_addition"}}},
        )
        for name, fields in {
            "search_multitransport": {"origin", "destination", "departure_date"},
            "search_hotels": {"city_name", "check_in", "check_out", "adults"},
            "get_offer_details": {"product_type"},
            "create_checkout_link": {"transport"},
        }.items()
    )

    validate_required_capabilities(tools)

    broken = tools[0].model_copy(update={"input_schema": {"properties": {}}})
    with pytest.raises(ProviderContractError, match="missing required fields"):
        validate_required_capabilities((broken, *tools[1:]))
