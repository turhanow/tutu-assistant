"""Official MCP SDK adapter implementing the Tutu provider port."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Self

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.adapters.mcp_mappers import (
    map_checkout_link,
    map_hotel_search,
    map_offer_details,
    map_transport_search,
)
from app.domain.errors import (
    ProviderContractError,
    ProviderResponseError,
    ProviderToolError,
    ProviderTransientError,
)
from app.domain.models import (
    CheckoutLink,
    HotelOffer,
    HotelSearchQuery,
    OfferDetails,
    OfferRef,
    ToolCapability,
    TransportMode,
    TransportOffer,
    TransportSearchQuery,
    TutuCapabilities,
)

EXPECTED_SCHEMA_HASH = "1a00e172686c927cf2d540c27049abe79764ec0635c232c5b236eaec6ee751bd"
REQUIRED_TOOLS = frozenset(
    {"search_multitransport", "search_hotels", "get_offer_details", "create_checkout_link"}
)
REQUIRED_INPUT_FIELDS = {
    "search_multitransport": frozenset({"origin", "destination", "departure_date"}),
    "search_hotels": frozenset({"city_name", "check_in", "check_out", "adults"}),
    "get_offer_details": frozenset({"product_type"}),
    "create_checkout_link": frozenset({"transport"}),
}
MODE_ARGUMENTS = {
    TransportMode.AVIA: "avia",
    TransportMode.RAIL: "railway",
    TransportMode.BUS: "bus",
    TransportMode.ETRAIN: "etrain",
}

logger = logging.getLogger(__name__)


@dataclass
class _McpCommand:
    action: str
    future: asyncio.Future[Any]
    name: str | None = None
    arguments: dict[str, Any] | None = None
    retry_transient: bool = True


class TutuMcpAdapter:
    """A reusable async MCP session with strict domain boundary mapping."""

    def __init__(self, endpoint: str, *, timeout_seconds: int = 12) -> None:
        self._endpoint = endpoint
        self._timeout = timedelta(seconds=timeout_seconds)
        self._runner: asyncio.Task[None] | None = None
        self._ready: asyncio.Event | None = None
        self._commands: asyncio.Queue[_McpCommand] | None = None
        self._startup_error: Exception | None = None
        self._start_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._closing = False

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def start(self) -> None:
        async with self._start_lock:
            if self._closing:
                raise ProviderTransientError("Tutu MCP adapter is closing")
            if self._runner is not None and not self._runner.done():
                return
            self._runner = None
            self._ready = asyncio.Event()
            self._commands = asyncio.Queue(maxsize=100)
            self._startup_error = None
            self._runner = asyncio.create_task(self._run_session(), name="tutu-mcp-session")
            await self._ready.wait()
            if self._startup_error is not None:
                await self._runner
                self._runner = None
                raise ProviderTransientError("failed to initialize Tutu MCP session") from (
                    self._startup_error
                )

    async def _run_session(self) -> None:
        stack = AsyncExitStack()
        active_command: _McpCommand | None = None
        try:
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamable_http_client(self._endpoint)
            )
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=self._timeout,
                )
            )
            await session.initialize()
        except Exception as error:
            self._startup_error = error
            await self._close_stack(stack)
            self._ready.set()
            return
        try:
            self._ready.set()
            while True:
                active_command = await self._commands.get()
                if active_command.action == "close":
                    active_command.future.set_result(None)
                    active_command = None
                    break
                try:
                    if active_command.action == "list_tools":
                        result = await self._list_tool_pages(session)
                    else:
                        result = await self._call_session(
                            session,
                            active_command.name or "",
                            active_command.arguments or {},
                            retry_transient=active_command.retry_transient,
                        )
                except Exception as error:
                    if not active_command.future.done():
                        active_command.future.set_exception(error)
                else:
                    if not active_command.future.done():
                        active_command.future.set_result(result)
                active_command = None
        finally:
            session_error = ProviderTransientError("Tutu MCP session closed")
            if active_command is not None and not active_command.future.done():
                active_command.future.set_exception(session_error)
            if self._commands is not None:
                while not self._commands.empty():
                    pending = self._commands.get_nowait()
                    if not pending.future.done():
                        pending.future.set_exception(session_error)
            await self._close_stack(stack)

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        self._closing = True
        async with self._operation_lock:
            runner = self._runner
            if runner is None:
                return
            if self._commands is not None and not runner.done():
                future = asyncio.get_running_loop().create_future()
                await self._commands.put(_McpCommand(action="close", future=future))
                await future
            await runner
            self._runner = None
            self._ready = None
            self._commands = None

    async def discover_capabilities(self) -> TutuCapabilities:
        pages = await self._submit("list_tools")
        digest = hashlib.sha256(
            json.dumps(
                pages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        tools = tuple(
            ToolCapability(
                name=tool["name"],
                input_schema=tool["inputSchema"],
                output_schema=tool.get("outputSchema"),
                read_only=(tool.get("annotations") or {}).get("readOnlyHint"),
            )
            for page in pages
            for tool in page["tools"]
        )
        validate_required_capabilities(tools)
        if digest != EXPECTED_SCHEMA_HASH:
            logger.warning(
                "Tutu MCP schema changed but required tools remain compatible",
                extra={"schema_hash": digest},
            )
        return TutuCapabilities(
            tools=tools,
            schema_hash=digest,
            captured_at=datetime.now(UTC),
        )

    async def search_transport(self, query: TransportSearchQuery) -> list[TransportOffer]:
        if len(query.allowed_modes) == 1:
            mode = next(iter(query.allowed_modes))
            targeted_tool = {
                TransportMode.AVIA: "search_avia",
                TransportMode.RAIL: "search_rail",
                TransportMode.BUS: "search_bus",
                TransportMode.ETRAIN: "search_etrain",
            }.get(mode)
            if targeted_tool is not None:
                return await self._search_targeted(query, mode, targeted_tool)
        modes = [
            MODE_ARGUMENTS[mode] for mode in sorted(query.allowed_modes) if mode in MODE_ARGUMENTS
        ]
        payload = await self._call(
            "search_multitransport",
            {
                "origin": query.origin,
                "destination": query.destination,
                "departure_date": query.departure_date.isoformat(),
                "adults": query.adults,
                "modes": modes or None,
                "price_max": float(query.max_price) if query.max_price is not None else None,
                "page": 1,
                "page_size": 5,
                "view": "compact",
            },
        )
        return map_transport_search(payload)

    async def _search_targeted(
        self,
        query: TransportSearchQuery,
        mode: TransportMode,
        tool: str,
    ) -> list[TransportOffer]:
        arguments: dict[str, Any] = {
            "origin": query.origin,
            "destination": query.destination,
            "departure_date": query.departure_date.isoformat(),
            "price_max": float(query.max_price) if query.max_price is not None else None,
            "page": 1,
            "page_size": 5,
            "view": "compact",
        }
        if mode is TransportMode.RAIL:
            arguments["passengers"] = query.adults
        elif mode in {TransportMode.AVIA, TransportMode.BUS}:
            arguments["adults"] = query.adults
        payload = await self._call(tool, arguments)
        return map_transport_search(payload)

    async def search_hotels(self, query: HotelSearchQuery) -> list[HotelOffer]:
        preferences = query.preferences
        stars = list(range(preferences.stars_min, 6)) if preferences.stars_min else None
        payload = await self._call(
            "search_hotels",
            {
                "city_name": query.city,
                "check_in": query.check_in.isoformat(),
                "check_out": query.check_out.isoformat(),
                "adults": query.adults,
                "stars": stars,
                "min_rating": (
                    float(preferences.rating_min) if preferences.rating_min is not None else None
                ),
                "meals": sorted(preferences.meal_preferences) or None,
                "free_cancellation": preferences.free_cancellation_required or None,
                "hotel_amenities": sorted(preferences.required_amenities) or None,
                "page": 1,
                "page_size": 5,
                "view": "compact",
            },
        )
        return [
            offer.model_copy(
                update={
                    "matched_amenities": preferences.required_amenities,
                    "matched_meals": preferences.meal_preferences,
                }
            )
            for offer in map_hotel_search(payload)
        ]

    async def get_offer_details(self, ref: OfferRef) -> OfferDetails:
        if ref.product_type == "hotels":
            arguments = {"product_type": ref.product_type, **ref.details_data}
        else:
            arguments = {
                "product_type": ref.product_type,
                "details_ref": dict(ref.details_data),
            }
        payload = await self._call("get_offer_details", dict(arguments))
        return map_offer_details(payload, ref.product_type)

    async def create_checkout_link(self, ref: OfferRef) -> CheckoutLink:
        if not ref.checkout_data:
            raise ProviderResponseError("offer is missing checkout data")
        payload = await self._call(
            "create_checkout_link", dict(ref.checkout_data), retry_transient=False
        )
        return map_checkout_link(payload)

    async def _call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        retry_transient: bool = True,
    ) -> dict[str, Any]:
        return await self._submit(
            "call_tool",
            name=name,
            arguments=arguments,
            retry_transient=retry_transient,
        )

    async def _submit(
        self,
        action: str,
        *,
        name: str | None = None,
        arguments: dict[str, Any] | None = None,
        retry_transient: bool = True,
    ) -> Any:
        async with self._operation_lock:
            if self._runner is None or self._runner.done() or self._commands is None:
                await self.start()
            if self._commands is None:
                raise ProviderTransientError("Tutu MCP command queue is unavailable")
            future = asyncio.get_running_loop().create_future()
            await self._commands.put(
                _McpCommand(
                    action=action,
                    future=future,
                    name=name,
                    arguments=arguments,
                    retry_transient=retry_transient,
                )
            )
            try:
                return await asyncio.wait_for(future, timeout=self._timeout.total_seconds())
            except TimeoutError as error:
                await self._abort_runner()
                raise ProviderTransientError("Tutu MCP operation timed out") from error

    async def _abort_runner(self) -> None:
        runner = self._runner
        if runner is not None and not runner.done():
            runner.cancel()
            with suppress(asyncio.CancelledError):
                await runner
        self._runner = None
        self._ready = None
        self._commands = None

    @staticmethod
    async def _close_stack(stack: AsyncExitStack) -> None:
        with suppress(TimeoutError, RuntimeError):
            async with asyncio.timeout(3):
                await stack.aclose()

    @staticmethod
    async def _list_tool_pages(session: ClientSession) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            result = await session.list_tools(cursor=cursor)
            pages.append(result.model_dump(mode="json", by_alias=True, exclude_none=False))
            cursor = result.nextCursor
            if cursor is None:
                return pages

    async def _call_session(
        self,
        session: ClientSession,
        name: str,
        arguments: dict[str, Any],
        *,
        retry_transient: bool,
    ) -> dict[str, Any]:
        clean_arguments = {key: value for key, value in arguments.items() if value is not None}
        attempts = 2 if retry_transient else 1
        for attempt in range(attempts):
            try:
                result = await session.call_tool(name, arguments=clean_arguments)
                break
            except (TimeoutError, httpx.TransportError) as error:
                if attempt + 1 == attempts:
                    raise ProviderTransientError(f"Tutu MCP call failed: {name}") from error
                await asyncio.sleep(0.25)
        if result.isError:
            raise ProviderToolError(f"Tutu MCP tool returned an error: {name}")
        for item in result.content:
            if item.type == "text":
                try:
                    payload = json.loads(item.text)
                except json.JSONDecodeError as error:
                    raise ProviderResponseError("Tutu MCP returned invalid JSON") from error
                if isinstance(payload, dict):
                    return payload
        raise ProviderResponseError("Tutu MCP response has no JSON object")


class TutuMcpPool:
    """Bounded pool of independent MCP sessions for cross-user concurrency."""

    def __init__(
        self,
        endpoint: str,
        *,
        size: int = 3,
        timeout_seconds: int = 12,
    ) -> None:
        if not 1 <= size <= 8:
            raise ValueError("Tutu MCP pool size must be between 1 and 8")
        self._adapters = tuple(
            TutuMcpAdapter(endpoint, timeout_seconds=timeout_seconds) for _ in range(size)
        )
        self._next = 0
        self._selection_lock = asyncio.Lock()
        self._capabilities: TutuCapabilities | None = None
        self._capabilities_lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def start(self) -> None:
        started: list[TutuMcpAdapter] = []
        try:
            for adapter in self._adapters:
                await adapter.start()
                started.append(adapter)
        except Exception:
            await asyncio.gather(*(item.close() for item in started), return_exceptions=True)
            raise

    async def close(self) -> None:
        await asyncio.gather(
            *(adapter.close() for adapter in self._adapters), return_exceptions=True
        )

    async def discover_capabilities(self) -> TutuCapabilities:
        if self._capabilities is not None:
            return self._capabilities
        async with self._capabilities_lock:
            if self._capabilities is None:
                self._capabilities = await self._adapters[0].discover_capabilities()
            return self._capabilities

    async def search_transport(self, query: TransportSearchQuery) -> list[TransportOffer]:
        return await (await self._select()).search_transport(query)

    async def search_hotels(self, query: HotelSearchQuery) -> list[HotelOffer]:
        return await (await self._select()).search_hotels(query)

    async def get_offer_details(self, ref: OfferRef) -> OfferDetails:
        return await (await self._select()).get_offer_details(ref)

    async def create_checkout_link(self, ref: OfferRef) -> CheckoutLink:
        return await (await self._select()).create_checkout_link(ref)

    async def _select(self) -> TutuMcpAdapter:
        async with self._selection_lock:
            adapter = self._adapters[self._next]
            self._next = (self._next + 1) % len(self._adapters)
            return adapter


def validate_required_capabilities(tools: tuple[ToolCapability, ...]) -> None:
    """Accept additive drift and fail closed on fields the adapter actually uses."""
    by_name = {tool.name: tool for tool in tools}
    if not set(by_name) >= REQUIRED_TOOLS:
        raise ProviderContractError("Tutu MCP is missing required P0 tools")
    for name, required_fields in REQUIRED_INPUT_FIELDS.items():
        properties = by_name[name].input_schema.get("properties")
        available = set(properties) if isinstance(properties, dict) else set()
        missing = required_fields - available
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ProviderContractError(
                f"Tutu MCP tool {name} is missing required fields: {missing_text}"
            )
