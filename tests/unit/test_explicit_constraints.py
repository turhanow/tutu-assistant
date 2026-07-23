from datetime import date

from app.domain.models import HotelMode
from app.services.explicit_constraints import (
    explicit_hotel_mode,
    explicitly_forbids_night_travel,
    explicitly_mentions_river,
    explicitly_relaxed,
    extract_explicit_party,
    resolved_hotel_mode,
)


def test_exact_qa_party_phrases_are_preserved() -> None:
    group = extract_explicit_party("Три взрослых, нужен отель")
    family = extract_explicit_party("Двое взрослых и ребёнок, два номера")

    assert (group.adults, group.children, group.rooms) == (3, None, None)
    assert (family.adults, family.children, family.rooms) == (2, 1, 2)


def test_exact_qa_comfort_and_hotel_phrases_are_detected() -> None:
    text = "Хочется реки и прогулок, спокойно, без ночной дороги и без отеля"

    assert explicitly_mentions_river(text)
    assert explicitly_relaxed(text)
    assert explicitly_forbids_night_travel(text)
    assert explicit_hotel_mode(text) is HotelMode.FORBIDDEN


def test_overnight_trip_requires_hotel_by_default_but_explicit_no_wins() -> None:
    assert (
        resolved_hotel_mode(
            "Хочу уехать на два дня",
            HotelMode.FORBIDDEN,
            departure_date=date(2026, 8, 29),
            return_date=date(2026, 8, 30),
        )
        is HotelMode.REQUIRED
    )
    assert (
        resolved_hotel_mode(
            "Хочу уехать на два дня без отеля",
            HotelMode.REQUIRED,
            departure_date=date(2026, 8, 29),
            return_date=date(2026, 8, 30),
        )
        is HotelMode.FORBIDDEN
    )
