from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.domain.content_models import (
    Activity,
    DestinationContent,
    DestinationProfile,
    EvidenceRef,
    PriceRange,
    sum_price_ranges,
)
from app.domain.discovery_models import (
    CostBreakdown,
    DateFlexibility,
    DateRange,
    DayPlan,
    DestinationCandidate,
    DiscoveryDraft,
    FeasibilitySnapshot,
    FeasibilityStatus,
    IntentParseResult,
    ScheduledActivity,
    TripIntent,
    WeekendProposal,
)
from app.domain.models import (
    OfferRef,
    PriceStatus,
    PriceSummary,
    RankedTripOption,
    SortPreference,
    TransportMode,
    TransportOffer,
    TripCombination,
    TripRequest,
    TripSearchResult,
    TripTimeMetrics,
)
from app.domain.product_models import FeedbackReason, FeedbackRecord, ProductEvent

NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)


def evidence(**overrides) -> EvidenceRef:
    values = {
        "evidence_id": "source_1",
        "title": "Official destination page",
        "source_url": "https://example.org/destination",
        "retrieved_at": NOW,
    }
    values.update(overrides)
    return EvidenceRef(**values)


def destination(**overrides) -> DestinationProfile:
    values = {
        "destination_id": "kolomna",
        "name": "Коломна",
        "region": "Московская область",
        "experience_tags": {"architecture", "gastronomy"},
        "evidence_ids": {"source_1"},
    }
    values.update(overrides)
    return DestinationProfile(**values)


def activity(activity_id: str = "kremlin", **overrides) -> Activity:
    values = {
        "activity_id": activity_id,
        "destination_id": "kolomna",
        "name": "Коломенский кремль",
        "categories": {"architecture"},
        "duration": timedelta(hours=2),
        "evidence_ids": {"source_1"},
    }
    values.update(overrides)
    return Activity(**values)


def trip_option() -> RankedTripOption:
    outbound_departure = datetime(2026, 8, 22, 8, tzinfo=UTC)
    outbound = TransportOffer(
        provider_id="out",
        mode=TransportMode.RAIL,
        origin="Москва",
        destination="Коломна",
        departure_at=outbound_departure,
        arrival_at=outbound_departure + timedelta(hours=2),
        price="1000",
        currency="RUB",
        offer_ref=OfferRef(product_type="rail"),
    )
    return_departure = datetime(2026, 8, 23, 18, tzinfo=UTC)
    return_offer = TransportOffer(
        provider_id="back",
        mode=TransportMode.RAIL,
        origin="Коломна",
        destination="Москва",
        departure_at=return_departure,
        arrival_at=return_departure + timedelta(hours=2),
        price="1000",
        currency="RUB",
        offer_ref=OfferRef(product_type="rail"),
    )
    combination = TripCombination(
        outbound=outbound,
        return_offer=return_offer,
        price=PriceSummary(status=PriceStatus.EXACT, known_total="2000", currency="RUB"),
        metrics=TripTimeMetrics(
            total_travel_duration=timedelta(hours=4),
            trip_duration=timedelta(hours=36),
            time_in_city=timedelta(hours=32),
        ),
        signature="kolomna-option",
    )
    return RankedTripOption(
        combination=combination,
        labels={SortPreference.BALANCED},
        balance_score="0.2",
    )


def scheduled(item: Activity, hour: int) -> ScheduledActivity:
    starts = datetime(2026, 8, 22, hour, tzinfo=UTC)
    return ScheduledActivity(
        activity=item,
        starts_at=starts,
        ends_at=starts + item.duration,
    )


def test_dynamic_evidence_requires_bounded_freshness() -> None:
    with pytest.raises(ValidationError, match="requires valid_until"):
        evidence(is_dynamic=True)

    item = evidence(is_dynamic=True, valid_until=NOW + timedelta(days=1))
    assert item.is_fresh_at(NOW + timedelta(hours=12))
    assert not item.is_fresh_at(NOW + timedelta(days=2))


def test_evidence_rejects_insecure_url_and_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="HTTPS"):
        evidence(source_url="http://example.org")
    with pytest.raises(ValidationError, match="timezone-aware"):
        evidence(retrieved_at=NOW.replace(tzinfo=None))


def test_price_range_is_ordered_and_only_compatible_ranges_are_summed() -> None:
    with pytest.raises(ValidationError, match="maximum"):
        PriceRange(minimum="500", maximum="100", currency="RUB")

    total = sum_price_ranges(
        (
            PriceRange(minimum="100", maximum="200", currency="RUB"),
            PriceRange(minimum="50", maximum="80", currency="RUB"),
        )
    )
    assert total == PriceRange(minimum="150", maximum="280", currency="RUB")
    assert (
        sum_price_ranges(
            (
                PriceRange(minimum="1", maximum="2", currency="RUB"),
                PriceRange(minimum="1", maximum="2", currency="USD"),
            )
        )
        is None
    )


def test_destination_content_rejects_orphan_evidence_and_foreign_activity() -> None:
    with pytest.raises(ValidationError, match="unknown evidence"):
        DestinationContent(
            destination=destination(evidence_ids={"missing"}),
            evidence=(evidence(),),
            catalog_version="v1",
        )
    with pytest.raises(ValidationError, match="different destination"):
        DestinationContent(
            destination=destination(),
            activities=(activity(destination_id="tver"),),
            evidence=(evidence(),),
            catalog_version="v1",
        )


def test_destination_content_accepts_grounded_activities() -> None:
    content = DestinationContent(
        destination=destination(),
        activities=(activity(),),
        evidence=(evidence(),),
        catalog_version="v1",
    )
    assert content.activities[0].categories == frozenset({"architecture"})


def test_discovery_date_range_is_bounded() -> None:
    period = DateRange(
        start=date(2026, 8, 22),
        end=date(2026, 8, 23),
        flexibility=DateFlexibility.WEEKEND,
    )
    assert period.flexibility is DateFlexibility.WEEKEND
    with pytest.raises(ValidationError, match="15 calendar days"):
        DateRange(start=date(2026, 8, 1), end=date(2026, 8, 20))


def test_unknown_intent_requires_discovery_payload() -> None:
    with pytest.raises(ValidationError, match="requires discovery draft"):
        IntentParseResult(intent=TripIntent.DESTINATION_UNKNOWN, confidence="0.9")

    parsed = IntentParseResult(
        intent=TripIntent.DESTINATION_UNKNOWN,
        confidence="0.9",
        discovery_draft=DiscoveryDraft(origin="Москва"),
        missing_fields=("dates", "motives"),
    )
    assert parsed.confidence == Decimal("0.9")


def test_verified_snapshot_requires_real_options_and_timezone() -> None:
    request = TripRequest(
        origin="Москва",
        destination="Коломна",
        departure_date=date(2026, 8, 22),
        return_date=date(2026, 8, 23),
    )
    empty = TripSearchResult(request=request, searched_at=NOW)
    with pytest.raises(ValidationError, match="requires at least one"):
        FeasibilitySnapshot(
            destination_id="kolomna",
            status=FeasibilityStatus.VERIFIED,
            verified_at=NOW,
            trip_result=empty,
        )

    result = empty.model_copy(update={"options": (trip_option(),)})
    snapshot = FeasibilitySnapshot(
        destination_id="kolomna",
        status=FeasibilityStatus.VERIFIED,
        verified_at=NOW,
        trip_result=result,
    )
    assert snapshot.status is FeasibilityStatus.VERIFIED


def test_cost_breakdown_never_mixes_currency_classes() -> None:
    cost = CostBreakdown(
        confirmed_total="2000",
        confirmed_currency="RUB",
        estimated=PriceRange(minimum="1000", maximum="2000", currency="RUB"),
        unknown_components=("local_transport",),
    )
    assert cost.confirmed_total == Decimal("2000")
    with pytest.raises(ValidationError, match="currencies must match"):
        cost.model_copy(
            update={"estimated": PriceRange(minimum="10", maximum="20", currency="USD")}
        ).model_validate(
            {
                **cost.model_dump(),
                "estimated": {"minimum": "10", "maximum": "20", "currency": "USD"},
            }
        )


def test_day_plan_rejects_overlapping_activities() -> None:
    first = scheduled(activity("one"), 10)
    second_activity = activity("two", duration=timedelta(hours=1))
    second = ScheduledActivity(
        activity=second_activity,
        starts_at=datetime(2026, 8, 22, 11, tzinfo=UTC),
        ends_at=datetime(2026, 8, 22, 12, tzinfo=UTC),
    )
    with pytest.raises(ValidationError, match="cannot overlap"):
        DayPlan(date=date(2026, 8, 22), activities=(first, second))


def test_complete_proposal_requires_two_anchor_activities() -> None:
    candidate = DestinationCandidate(
        destination=destination(),
        match_score="0.85",
        match_reasons=("Архитектура соответствует интересам",),
    )
    one_activity_day = DayPlan(
        date=date(2026, 8, 22),
        activities=(scheduled(activity(), 10),),
    )
    values = {
        "candidate": candidate,
        "trip_option": trip_option(),
        "days": (one_activity_day,),
        "cost": CostBreakdown(confirmed_total="2000", confirmed_currency="RUB"),
        "verified_at": NOW,
        "evidence_ids": {"source_1"},
        "trade_off": "Нет подтверждённой оценки локального транспорта",
        "content_complete": True,
    }
    with pytest.raises(ValidationError, match="two or three activities per day"):
        WeekendProposal(**values)

    proposal = WeekendProposal(**{**values, "content_complete": False})
    assert not proposal.content_complete


def test_proposal_rejects_same_place_repeated_across_trip_days() -> None:
    candidate = DestinationCandidate(
        destination=destination(),
        match_score="0.85",
        match_reasons=("Архитектура соответствует интересам",),
    )
    first = activity("place_one", name="Нижегородский кремль")
    duplicate = activity("place_alias", name="  нижегородский   кремль ")

    with pytest.raises(ValidationError, match="cannot be repeated"):
        WeekendProposal(
            candidate=candidate,
            trip_option=trip_option(),
            days=(
                DayPlan(date=date(2026, 8, 22), suggestions=(first,)),
                DayPlan(date=date(2026, 8, 23), suggestions=(duplicate,)),
            ),
            cost=CostBreakdown(confirmed_total="2000", confirmed_currency="RUB"),
            verified_at=NOW,
            evidence_ids={"source_1"},
            trade_off="Часы работы нужно проверить перед поездкой",
            content_complete=False,
        )


def test_product_events_and_feedback_are_privacy_bounded() -> None:
    event = ProductEvent(
        name="intent_classified",
        occurred_at=NOW,
        flow_id="flow_12345678",
        intent=TripIntent.DESTINATION_UNKNOWN,
        dimensions={"candidate_count": 5},
    )
    assert event.dimensions == {"candidate_count": 5}

    feedback = FeedbackRecord(
        flow_id="flow_12345678",
        reason=FeedbackReason.IRRELEVANT,
        comment="  Не   подходит   темп  ",
        created_at=NOW,
    )
    assert feedback.comment == "Не подходит темп"
