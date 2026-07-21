"""Run safe read-only Tutu MCP searches using captured argument names."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from inspect_tutu_mcp import DEFAULT_TIMEOUT_SECONDS, DEFAULT_URL, canonical_json
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("var/tutu_mcp/fixtures"))
    parser.add_argument("--origin", default="Москва")
    parser.add_argument("--destination", default="Казань")
    parser.add_argument("--departure-date", type=date.fromisoformat)
    parser.add_argument("--return-date", type=date.fromisoformat)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def result_payload(result: Any) -> dict[str, Any]:
    return result.model_dump(mode="json", by_alias=True, exclude_none=False)


def text_payload(result: dict[str, Any]) -> dict[str, Any]:
    for item in result.get("content", []):
        if item.get("type") == "text":
            value = json.loads(item["text"])
            if isinstance(value, dict):
                return value
    raise ValueError("Expected a JSON object in the MCP text response")


def sanitized_fixture(value: Any, *, key: str = "") -> Any:
    """Keep response shape while replacing volatile provider references."""
    lowered = key.lower()
    if isinstance(value, dict):
        return {
            child_key: sanitized_fixture(child_value, key=child_key)
            for child_key, child_value in value.items()
            if child_key not in {"photos", "review_summary"}
        }
    if isinstance(value, list):
        return [sanitized_fixture(item, key=key) for item in value]
    if value is None:
        return None
    if "url" in lowered:
        return "https://www.tutu.ru/fixture"
    if lowered in {"alias", "hotel_alias"}:
        return "fixture-hotel"
    if lowered.endswith(("_id", "_hash")) or lowered in {
        "card_id",
        "hotel_id",
        "offer_hash",
        "offer_id",
        "result_id",
        "search_id",
        "segment_hash",
        "tutu_offer_id",
    }:
        return 1 if isinstance(value, int) else "fixture-ref"
    return value


async def run_smoke(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    departure_date = args.departure_date or date.today() + timedelta(days=30)
    return_date = args.return_date or departure_date + timedelta(days=2)
    if return_date <= departure_date:
        raise ValueError("return-date must be after departure-date")

    async with (
        streamable_http_client(args.url) as (read_stream, write_stream, _),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        transport = await session.call_tool(
            "search_multitransport",
            arguments={
                "origin": args.origin,
                "destination": args.destination,
                "departure_date": departure_date.isoformat(),
                "adults": 1,
                "page": 1,
                "page_size": 3,
                "view": "compact",
            },
        )
        hotels = await session.call_tool(
            "search_hotels",
            arguments={
                "city_name": args.destination,
                "check_in": departure_date.isoformat(),
                "check_out": return_date.isoformat(),
                "adults": 1,
                "page": 1,
                "page_size": 3,
                "view": "compact",
            },
        )
        rail = await session.call_tool(
            "search_rail",
            arguments={
                "origin": args.origin,
                "destination": args.destination,
                "departure_date": departure_date.isoformat(),
                "passengers": 1,
                "page": 1,
                "page_size": 3,
                "view": "compact",
            },
        )
        avia = await session.call_tool(
            "search_avia",
            arguments={
                "origin": args.origin,
                "destination": args.destination,
                "departure_date": departure_date.isoformat(),
                "adults": 1,
                "page": 1,
                "page_size": 3,
                "view": "compact",
            },
        )
        transport_result = result_payload(transport)
        hotel_result = result_payload(hotels)
        hotel_rows = text_payload(hotel_result).get("hotels", [])
        if not hotel_rows:
            raise RuntimeError("Hotel smoke search returned no offers for checkout validation")
        checkout_ref = hotel_rows[0].get("checkout_ref")
        if not isinstance(checkout_ref, dict):
            raise RuntimeError("Hotel smoke offer does not contain checkout_ref")
        checkout = await session.call_tool(
            "create_checkout_link",
            arguments=checkout_ref,
        )
        details = await session.call_tool(
            "get_offer_details",
            arguments={
                "product_type": "hotels",
                "offer_id": hotel_rows[0].get("hotel_id"),
                "check_in": departure_date.isoformat(),
                "check_out": return_date.isoformat(),
                "adults": 1,
                "view": "compact",
            },
        )
        return (
            transport_result,
            hotel_result,
            result_payload(checkout),
            result_payload(details),
            result_payload(rail),
            result_payload(avia),
        )


async def async_main() -> None:
    args = parse_args()
    transport, hotels, checkout, details, rail, avia = await run_smoke(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transport-search.raw.json").write_text(
        canonical_json(transport), encoding="utf-8"
    )
    (args.output / "hotel-search.raw.json").write_text(canonical_json(hotels), encoding="utf-8")
    (args.output / "checkout-link.raw.json").write_text(canonical_json(checkout), encoding="utf-8")
    (args.output / "hotel-details.raw.json").write_text(canonical_json(details), encoding="utf-8")
    (args.output / "rail-search.raw.json").write_text(canonical_json(rail), encoding="utf-8")
    (args.output / "avia-search.raw.json").write_text(canonical_json(avia), encoding="utf-8")
    transport_body = text_payload(transport)
    hotel_body = text_payload(hotels)
    checkout_body = text_payload(checkout)
    details_body = text_payload(details)
    rail_body = text_payload(rail)
    avia_body = text_payload(avia)
    transport_body["variants"] = transport_body.get("variants", [])[:1]
    hotel_body["hotels"] = hotel_body.get("hotels", [])[:1]
    (args.output / "transport-search.json").write_text(
        canonical_json(sanitized_fixture(transport_body)), encoding="utf-8"
    )
    (args.output / "hotel-search.json").write_text(
        canonical_json(sanitized_fixture(hotel_body)), encoding="utf-8"
    )
    (args.output / "checkout-link.json").write_text(
        canonical_json(sanitized_fixture(checkout_body)), encoding="utf-8"
    )
    (args.output / "hotel-details.json").write_text(
        canonical_json(sanitized_fixture(details_body)), encoding="utf-8"
    )
    rail_body["offers"] = rail_body.get("offers", rail_body.get("variants", []))[:1]
    avia_body["offers"] = avia_body.get("offers", avia_body.get("variants", []))[:1]
    (args.output / "rail-search.json").write_text(
        canonical_json(sanitized_fixture(rail_body)), encoding="utf-8"
    )
    (args.output / "avia-search.json").write_text(
        canonical_json(sanitized_fixture(avia_body)), encoding="utf-8"
    )
    print(
        "Smoke completed: "
        f"transport_error={transport['isError']}, "
        f"hotel_error={hotels['isError']}, checkout_error={checkout['isError']}, "
        f"details_error={details['isError']}, rail_error={rail['isError']}, "
        f"avia_error={avia['isError']}"
    )


if __name__ == "__main__":
    asyncio.run(async_main())
