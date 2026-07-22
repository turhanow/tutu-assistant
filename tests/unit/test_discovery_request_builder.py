from datetime import date, timedelta

import pytest

from app.domain.discovery_models import (
    DateFlexibility,
    DiscoveryDraft,
    ExperienceProfile,
    RoadTolerance,
    TravelPace,
)
from app.domain.models import HotelMode
from app.services.clarification_policy import plan_clarifications
from app.services.discovery_request_builder import (
    DiscoveryInputError,
    build_discovery_request,
    missing_discovery_fields,
)

TODAY = date(2026, 7, 21)


def complete_draft(**overrides) -> DiscoveryDraft:
    values = {
        "origin": "Москва",
        "departure_date": date(2026, 8, 22),
        "return_date": date(2026, 8, 23),
        "date_flexibility": DateFlexibility.WEEKEND,
        "hotel_mode": HotelMode.OPTIONAL,
        "experience": ExperienceProfile(interests={"архитектура"}),
    }
    values.update(overrides)
    return DiscoveryDraft(**values)


def test_missing_fields_only_contains_critical_information() -> None:
    draft = DiscoveryDraft()

    assert missing_discovery_fields(draft) == (
        "origin",
        "dates",
        "motives",
        "hotel_mode",
    )
    with pytest.raises(DiscoveryInputError) as error:
        build_discovery_request(draft, today=TODAY)
    assert error.value.field == "origin"


def test_builder_preserves_explicit_profile_and_uses_safe_scope_defaults() -> None:
    draft = complete_draft(
        adults=None,
        budget=None,
        experience=ExperienceProfile(
            motives={"смена обстановки"},
            interests={"архитектура"},
            road_tolerance=RoadTolerance(
                max_one_way_duration=timedelta(hours=4),
                allow_night_travel=False,
            ),
        ),
    )

    request = build_discovery_request(draft, today=TODAY)

    assert request.origin == "Москва"
    assert request.dates.flexibility is DateFlexibility.WEEKEND
    assert request.travelers.adults == 1
    assert request.budget is None
    assert request.experience.road_tolerance.allow_night_travel is False
    assert request.hotel_mode is HotelMode.OPTIONAL


def test_builder_preserves_explicit_no_hotel_constraint() -> None:
    request = build_discovery_request(
        complete_draft(hotel_mode=HotelMode.FORBIDDEN),
        today=TODAY,
    )

    assert request.hotel_mode is HotelMode.FORBIDDEN


def test_builder_never_silently_assumes_hotel_when_preference_is_unknown() -> None:
    draft = complete_draft(hotel_mode=None)

    with pytest.raises(DiscoveryInputError, match="hotel_mode") as raised:
        build_discovery_request(draft, today=TODAY)

    assert raised.value.field == "hotel_mode"
    questions = plan_clarifications(draft)
    assert questions[0].field == "hotel_mode"
    assert "Нужен отель" in questions[0].text


@pytest.mark.parametrize(
    "overrides",
    [
        {"adults": 3},
        {"adults": 2, "children": 1},
        {"adults": 2, "rooms": 2},
    ],
)
def test_discovery_rejects_unsupported_party_without_coercion(overrides) -> None:
    with pytest.raises(DiscoveryInputError, match="без детей и один номер"):
        build_discovery_request(complete_draft(**overrides), today=TODAY)


def test_builder_rejects_past_reversed_and_oversized_windows() -> None:
    with pytest.raises(DiscoveryInputError, match="уже прошла"):
        build_discovery_request(
            complete_draft(
                departure_date=date(2026, 7, 20),
                return_date=date(2026, 7, 21),
            ),
            today=TODAY,
        )
    with pytest.raises(DiscoveryInputError, match="раньше"):
        build_discovery_request(
            complete_draft(
                departure_date=date(2026, 8, 23),
                return_date=date(2026, 8, 22),
            ),
            today=TODAY,
        )
    with pytest.raises(DiscoveryInputError, match="15 calendar days"):
        build_discovery_request(
            complete_draft(
                departure_date=date(2026, 8, 1),
                return_date=date(2026, 8, 20),
            ),
            today=TODAY,
        )


def test_clarification_policy_prioritizes_critical_and_never_exceeds_three() -> None:
    questions = plan_clarifications(DiscoveryDraft(), limit=50)

    assert [item.field for item in questions] == ["origin", "dates", "motives"]
    assert all(item.required for item in questions)


def test_clarification_policy_uses_optional_questions_after_critical_fields() -> None:
    questions = plan_clarifications(complete_draft(), limit=3)

    assert [item.field for item in questions] == [
        "budget",
        "road_tolerance",
        "travelers",
    ]
    assert not any(item.required for item in questions)
    assert plan_clarifications(complete_draft(), limit=0) == ()


def test_required_clarification_does_not_append_optional_interview_questions() -> None:
    draft = complete_draft(hotel_mode=None, budget=None)

    questions = plan_clarifications(draft, limit=3)

    assert [item.field for item in questions] == ["hotel_mode"]
    assert questions[0].required


def test_explicit_relaxed_pace_is_enough_to_describe_discovery_motive() -> None:
    draft = complete_draft(
        experience=ExperienceProfile(pace=TravelPace.RELAXED),
    )

    assert "motives" not in missing_discovery_fields(draft)
