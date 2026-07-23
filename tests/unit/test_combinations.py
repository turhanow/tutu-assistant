import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

from app.adapters.mcp_mappers import map_hotel_search, map_transport_search
from app.domain.models import (
    HotelMode,
    HotelPreferences,
    TimeWindow,
    TransportPreferences,
    TripRequest,
)
from app.services.combinations import build_combinations, build_transport_pairs
from app.services.ranking import rank_options

FIXTURES = Path("tests/fixtures/tutu_mcp/fixtures")


def payload(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def trip_inputs():
    outbound = map_transport_search(payload("transport-search.json"))[0]
    return_offer = outbound.model_copy(
        update={
            "provider_id": "return-fixture",
            "origin": "Казань",
            "destination": "Москва",
            "departure_at": datetime.fromisoformat("2026-08-23T18:00:00+03:00"),
            "arrival_at": datetime.fromisoformat("2026-08-24T06:00:00+03:00"),
        }
    )
    hotel = map_hotel_search(payload("hotel-search.json"))[0]
    hotel = hotel.model_copy(update={"check_in": date(2026, 8, 22), "check_out": date(2026, 8, 23)})
    request = TripRequest(
        origin="Москва",
        destination="Казань",
        departure_date="2026-08-21",
        return_date="2026-08-23",
        hotel=HotelPreferences(mode=HotelMode.REQUIRED),
    )
    return request, outbound, return_offer, hotel


def test_builds_required_hotel_combination_and_deduplicates() -> None:
    request, outbound, return_offer, hotel = trip_inputs()
    combinations = build_combinations(
        request,
        [outbound, outbound],
        [return_offer, return_offer],
        [hotel, hotel],
    )

    assert len(combinations) == 1
    assert combinations[0].hotel is not None
    assert combinations[0].stay.nights == 1


def test_budget_hard_filter_only_removes_exact_over_budget() -> None:
    request, outbound, return_offer, hotel = trip_inputs()
    request = request.model_copy(update={"budget": 1})

    assert build_combinations(request, [outbound], [return_offer], [hotel]) == []


def test_ranking_merges_labels_for_single_winner() -> None:
    request, outbound, return_offer, hotel = trip_inputs()
    combinations = build_combinations(request, [outbound], [return_offer], [hotel])
    ranked = rank_options(combinations)

    assert len(ranked) == 1
    assert {label.value for label in ranked[0].labels} == {
        "cheapest",
        "fastest",
        "balanced",
    }


def test_transport_constraints_are_revalidated_after_provider_search() -> None:
    request, outbound, return_offer, _ = trip_inputs()
    request = request.model_copy(
        update={
            "transport": TransportPreferences(
                max_transfers=0,
                departure_window=TimeWindow(from_time=time(18, 0), to_time=time(21, 0)),
                return_window=TimeWindow(from_time=time(16, 0), to_time=time(20, 0)),
            )
        }
    )

    assert build_transport_pairs(request, [outbound], [return_offer])
    assert not build_transport_pairs(
        request,
        [outbound.model_copy(update={"transfers": 1})],
        [return_offer],
    )
    assert not build_transport_pairs(
        request,
        [outbound.model_copy(update={"departure_at": outbound.departure_at.replace(hour=10)})],
        [return_offer],
    )


def test_transport_pair_pool_preserves_cheap_fast_and_convenient_alternatives() -> None:
    request, outbound, return_offer, _ = trip_inputs()
    outbound_base = datetime.fromisoformat("2026-08-21T08:00:00+03:00")
    return_base = datetime.fromisoformat("2026-08-23T16:00:00+03:00")
    outbound_offers = [
        outbound.model_copy(
            update={
                "provider_id": f"out-{index}",
                "departure_at": outbound_base + timedelta(minutes=index),
                "arrival_at": outbound_base + timedelta(minutes=index, hours=10 - index),
                "duration": timedelta(hours=10 - index),
                "price": Decimal(100 + index * 100),
            }
        )
        for index in range(10)
    ]
    return_offers = [
        return_offer.model_copy(
            update={
                "provider_id": f"back-{index}",
                "departure_at": return_base + timedelta(minutes=index),
                "arrival_at": return_base + timedelta(minutes=index, hours=10 - index),
                "duration": timedelta(hours=10 - index),
                "price": Decimal(100 + index * 100),
            }
        )
        for index in range(10)
    ]

    pairs = build_transport_pairs(request, outbound_offers, return_offers)

    assert len(pairs) == 24
    assert len({pair[0].provider_id for pair in pairs}) > 1
    assert len({pair[1].provider_id for pair in pairs}) > 1
    assert (outbound_offers[0], return_offers[0]) in pairs
    assert (outbound_offers[-1], return_offers[-1]) in pairs


def test_hotel_offer_must_match_exact_pair_stay() -> None:
    request, outbound, return_offer, hotel = trip_inputs()
    wrong_stay = hotel.model_copy(
        update={"check_in": date(2026, 8, 21), "check_out": date(2026, 8, 23)}
    )

    assert build_combinations(request, [outbound], [return_offer], [wrong_stay]) == []
