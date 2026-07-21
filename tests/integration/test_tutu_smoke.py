import os
from datetime import date

import pytest

from app.adapters.tutu_mcp import EXPECTED_SCHEMA_HASH, TutuMcpAdapter
from app.domain.models import (
    HotelMode,
    HotelPreferences,
    HotelSearchQuery,
    TransportSearchQuery,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_TUTU_INTEGRATION") != "1",
    reason="set RUN_TUTU_INTEGRATION=1 to call live Tutu MCP",
)


@pytest.mark.asyncio
async def test_live_search_mapping_and_checkout() -> None:
    async with TutuMcpAdapter("https://mcp.tutu.ru/mcp") as adapter:
        capabilities = await adapter.discover_capabilities()
        transport = await adapter.search_transport(
            TransportSearchQuery(
                origin="Москва",
                destination="Казань",
                departure_date=date(2026, 8, 21),
            )
        )
        hotels = await adapter.search_hotels(
            HotelSearchQuery(
                city="Казань",
                check_in=date(2026, 8, 21),
                check_out=date(2026, 8, 23),
                preferences=HotelPreferences(mode=HotelMode.REQUIRED),
            )
        )
        checkout = await adapter.create_checkout_link(hotels[0].offer_ref)
        details = await adapter.get_offer_details(hotels[0].offer_ref)
        transport_checkout = await adapter.create_checkout_link(transport[0].offer_ref)

    assert capabilities.schema_hash == EXPECTED_SCHEMA_HASH
    assert transport
    assert hotels
    assert checkout.url.scheme == "https"
    assert details.rooms
    assert transport_checkout.url.scheme == "https"
