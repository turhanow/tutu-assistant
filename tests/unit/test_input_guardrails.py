from datetime import date

import pytest

from app.services.input_safety import contains_sensitive_data
from app.services.known_input_guardrails import (
    explicit_conflicts,
    extract_explicit_date_range,
    extract_explicit_trip_dates,
    nearest_weekend,
    normalize_city_answer,
    recover_known_draft,
)


@pytest.mark.parametrize(
    "text",
    [
        "Сохрани данные моей банковской карты 1111 1111 1111 1111",
        "1111 1111 1111 1111",
        "CVV 123",
        "Паспорт серия и номер 45 08 123456",
    ],
)
def test_sensitive_data_is_detected_before_llm(text: str) -> None:
    assert contains_sensitive_data(text)


@pytest.mark.parametrize(
    "text",
    [
        "Бюджет 25 000 рублей",
        "Москва — Казань, 15–16 августа",
        "Нас двое взрослых",
    ],
)
def test_travel_numbers_do_not_trigger_sensitive_data_guard(text: str) -> None:
    assert not contains_sensitive_data(text)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Из Москвы.", "Москвы"),
        ("из мск", "мск"),
        ("в Санкт-Петербург", "Санкт-Петербург"),
        ("Нижний Новгород", "Нижний Новгород"),
    ],
)
def test_city_answer_only_cleans_syntax_without_encoding_alias_knowledge(raw, expected) -> None:
    assert normalize_city_answer(raw) == expected


def test_recovery_preserves_route_and_labelled_reversed_dates() -> None:
    draft = recover_known_draft(
        "Москва — Казань, обратно 10 августа, туда 12 августа 2026 года",
        today=date(2026, 7, 22),
    )

    assert draft.origin == "Москва"
    assert draft.destination == "Казань"
    assert draft.departure_date == date(2026, 8, 12)
    assert draft.return_date == date(2026, 8, 10)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("29–30 августа", (date(2026, 8, 29), date(2026, 8, 30))),
        ("25-26 авг", (date(2026, 8, 25), date(2026, 8, 26))),
        ("25-26 авг 2026", (date(2026, 8, 25), date(2026, 8, 26))),
        ("с 25 по 26 августа", (date(2026, 8, 25), date(2026, 8, 26))),
        ("25 и 26 августа", (date(2026, 8, 25), date(2026, 8, 26))),
        ("25.08-26.08.2026", (date(2026, 8, 25), date(2026, 8, 26))),
        ("29-30.08.2026", (date(2026, 8, 29), date(2026, 8, 30))),
        ("31 декабря — 2 января", (date(2026, 12, 31), date(2027, 1, 2))),
        ("завтра и послезавтра", (date(2026, 7, 23), date(2026, 7, 24))),
        ("в эти выходные", (date(2026, 7, 25), date(2026, 7, 26))),
    ],
)
def test_explicit_date_range_preserves_user_order(text: str, expected) -> None:
    assert extract_explicit_date_range(text, today=date(2026, 7, 22)) == expected


def test_explicit_transport_and_hotel_contradictions_are_named() -> None:
    conflicts = explicit_conflicts(
        "Только самолёт, но самолёты не предлагать, отель нужен и не нужен одновременно"
    )

    assert len(conflicts) == 2
    assert any("отель" in item for item in conflicts)
    assert any("самолёт" in item for item in conflicts)


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 7, 22), (date(2026, 7, 25), date(2026, 7, 26))),
        (date(2026, 7, 25), (date(2026, 7, 25), date(2026, 7, 26))),
        (date(2026, 7, 26), (date(2026, 8, 1), date(2026, 8, 2))),
    ],
)
def test_nearest_weekend_uses_local_calendar(today: date, expected) -> None:
    assert nearest_weekend(today) == expected


@pytest.mark.parametrize(
    "text",
    ["в эти выходные", "на этих выходных", "в ближайшие выходные"],
)
def test_explicit_weekend_phrase_overrides_model_dates_deterministically(text: str) -> None:
    assert extract_explicit_trip_dates(text, today=date(2026, 7, 22)) == (
        date(2026, 7, 25),
        date(2026, 7, 26),
    )
