from datetime import date

import pytest

from app.services.input_safety import contains_sensitive_data
from app.services.known_input_guardrails import (
    explicit_conflicts,
    normalize_city_answer,
    recover_known_draft,
)


@pytest.mark.parametrize(
    "text",
    [
        "Сохрани данные моей банковской карты 1111 1111 1111 1111",
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


def test_city_answer_removes_preposition_and_normalizes_common_inflection() -> None:
    assert normalize_city_answer("Из Москвы.") == "Москва"
    assert normalize_city_answer("в Санкт-Петербург") == "Санкт-Петербург"


def test_recovery_preserves_route_and_labelled_reversed_dates() -> None:
    draft = recover_known_draft(
        "Москва — Казань, обратно 10 августа, туда 12 августа 2026 года",
        today=date(2026, 7, 22),
    )

    assert draft.origin == "Москва"
    assert draft.destination == "Казань"
    assert draft.departure_date == date(2026, 8, 12)
    assert draft.return_date == date(2026, 8, 10)


def test_explicit_transport_and_hotel_contradictions_are_named() -> None:
    conflicts = explicit_conflicts(
        "Только самолёт, но самолёты не предлагать, отель нужен и не нужен одновременно"
    )

    assert len(conflicts) == 2
    assert any("отель" in item for item in conflicts)
    assert any("самолёт" in item for item in conflicts)
