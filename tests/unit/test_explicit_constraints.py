from app.domain.models import HotelMode
from app.services.explicit_constraints import (
    explicit_hotel_mode,
    explicitly_forbids_night_travel,
    explicitly_mentions_river,
    explicitly_relaxed,
    extract_explicit_party,
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
