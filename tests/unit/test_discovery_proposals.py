from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from app.domain.content_models import (
    Activity,
    DestinationContent,
    DestinationProfile,
    EvidenceRef,
    PriceRange,
)
from app.domain.discovery_models import (
    CandidateShortlist,
    CostBreakdown,
    DateRange,
    DayPlan,
    DestinationCandidate,
    DiscoveryFeasibilityResult,
    DiscoveryRequest,
    FeasibilitySnapshot,
    FeasibilityStatus,
    ProposalCopy,
    ScheduledActivity,
    WeekendProposal,
)
from app.domain.errors import LlmProviderError
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
from app.ports.discovery_llm import ConversationContext
from app.services.discovery_ranking import RankingWeights, rank_proposals
from app.services.itinerary_builder import ItineraryBuilder
from app.services.proposal_builder import (
    GroundedProposalNarration,
    ProposalBuilder,
    project_grounded_facts,
)
from app.services.trip_budget import TripBudgetBuilder

NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)


class FakeClock:
    def now(self) -> datetime:
        return NOW


def evidence(destination_id: str, *, expired: bool = False) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=f"source_{destination_id}",
        title="Официальный туристический портал",
        source_url=f"https://example.org/{destination_id}",
        retrieved_at=NOW - timedelta(days=2),
        valid_until=NOW - timedelta(days=1) if expired else NOW + timedelta(days=30),
        is_dynamic=True,
    )


def destination(destination_id: str = "kolomna", **overrides) -> DestinationProfile:
    values = {
        "destination_id": destination_id,
        "name": destination_id.title(),
        "region": "Тестовый регион",
        "experience_tags": {"architecture"},
        "evidence_ids": {f"source_{destination_id}"},
    }
    values.update(overrides)
    return DestinationProfile(**values)


def activity(destination_id: str, index: int, **overrides) -> Activity:
    values = {
        "activity_id": f"activity_{destination_id}_{index}",
        "destination_id": destination_id,
        "name": f"Активность {index}",
        "categories": {"architecture"},
        "duration": timedelta(hours=1),
        "evidence_ids": {f"source_{destination_id}"},
    }
    values.update(overrides)
    return Activity(**values)


def content(destination_id: str = "kolomna", **destination_overrides) -> DestinationContent:
    return DestinationContent(
        destination=destination(destination_id, **destination_overrides),
        activities=(activity(destination_id, 1), activity(destination_id, 2)),
        evidence=(evidence(destination_id),),
        catalog_version="v1",
    )


def option(
    destination_id: str = "kolomna",
    *,
    arrival: datetime = datetime(2026, 8, 22, 17, tzinfo=UTC),
    return_departure: datetime = datetime(2026, 8, 23, 18, tzinfo=UTC),
    cost: Decimal | None = Decimal("2000"),
    time_in_city: timedelta | None = None,
    timezone_known: bool = True,
) -> RankedTripOption:
    outbound = TransportOffer(
        provider_id=f"secret-provider-out-{destination_id}",
        mode=TransportMode.RAIL,
        origin="Москва",
        destination=destination_id,
        departure_at=arrival - timedelta(hours=2),
        arrival_at=arrival,
        price="1000" if cost is not None else None,
        currency="RUB" if cost is not None else None,
        transfers=0,
        offer_ref=OfferRef(product_type="rail", checkout_data={"secret": "out"}),
        timezone_known=timezone_known,
    )
    return_offer = TransportOffer(
        provider_id=f"secret-provider-back-{destination_id}",
        mode=TransportMode.BUS,
        origin=destination_id,
        destination="Москва",
        departure_at=return_departure,
        arrival_at=return_departure + timedelta(hours=3),
        price="1000" if cost is not None else None,
        currency="RUB" if cost is not None else None,
        transfers=0,
        offer_ref=OfferRef(product_type="bus", checkout_data={"secret": "back"}),
        timezone_known=timezone_known,
    )
    known = None if cost is None else cost
    combination = TripCombination(
        outbound=outbound,
        return_offer=return_offer,
        price=PriceSummary(
            status=PriceStatus.UNKNOWN if known is None else PriceStatus.EXACT,
            known_total=known,
            currency="RUB" if known is not None else None,
            missing_components=("outbound", "return") if known is None else (),
        ),
        metrics=TripTimeMetrics(
            total_travel_duration=timedelta(hours=5),
            trip_duration=return_offer.arrival_at - outbound.departure_at,
            time_in_city=time_in_city or return_departure - arrival,
        ),
        signature=f"signature-{destination_id}",
    )
    return RankedTripOption(
        combination=combination,
        labels={SortPreference.BALANCED},
        balance_score="0.2",
    )


def candidate(destination_id: str = "kolomna", score: str = "0.8") -> DestinationCandidate:
    return DestinationCandidate(
        destination=destination(destination_id),
        match_score=score,
        match_reasons=("Совпадает интерес к архитектуре",),
    )


def proposal(
    destination_id: str,
    *,
    score: str = "0.8",
    cost: str = "2000",
    time_in_city: timedelta = timedelta(hours=25),
) -> WeekendProposal:
    trip = option(
        destination_id,
        cost=Decimal(cost),
        time_in_city=time_in_city,
    )
    item = activity(destination_id, 1)
    start = datetime(2026, 8, 22, 10, tzinfo=UTC)
    day = DayPlan(
        date=start.date(),
        activities=(
            ScheduledActivity(activity=item, starts_at=start, ends_at=start + item.duration),
            ScheduledActivity(
                activity=activity(destination_id, 2),
                starts_at=start + timedelta(hours=2),
                ends_at=start + timedelta(hours=3),
            ),
        ),
    )
    return WeekendProposal(
        candidate=candidate(destination_id, score),
        trip_option=trip,
        days=(day,),
        cost=CostBreakdown(confirmed_total=cost, confirmed_currency="RUB"),
        verified_at=NOW,
        evidence_ids={f"source_{destination_id}"},
        trade_off="Часы работы нужно проверить",
        content_complete=True,
    )


def test_itinerary_uses_real_city_window_and_splits_anchors_across_days() -> None:
    trip = option(arrival=datetime(2026, 8, 22, 16, 30, tzinfo=UTC))

    result = ItineraryBuilder().build(content(), trip, verified_at=NOW)

    scheduled = [item for day in result.days for item in day.activities]
    assert len(scheduled) == 2
    assert scheduled[0].starts_at >= trip.combination.outbound.arrival_at + timedelta(hours=1)
    assert scheduled[-1].ends_at <= (
        trip.combination.return_offer.departure_at - timedelta(minutes=90)
    )
    assert {item.starts_at.date() for item in scheduled} == {
        date(2026, 8, 22),
        date(2026, 8, 23),
    }
    assert result.content_complete


def test_itinerary_excludes_expired_dynamic_content_and_unknown_timezone() -> None:
    invalid_content = DestinationContent(
        destination=destination(),
        activities=(activity("kolomna", 1),),
        evidence=(evidence("kolomna", expired=True),),
        catalog_version="v1",
    )
    expired = ItineraryBuilder().build(invalid_content, option(), verified_at=NOW)
    assert not expired.content_complete
    assert expired.unscheduled_activity_ids == ("activity_kolomna_1",)
    assert "просроченными" in expired.warnings[-1]

    unknown_timezone = ItineraryBuilder().build(
        content(),
        option(timezone_known=False),
        verified_at=NOW,
    )
    assert not unknown_timezone.content_complete
    assert "Часовой пояс" in unknown_timezone.warnings[0]


def test_budget_keeps_confirmed_and_estimated_ranges_separate() -> None:
    local_content = content(
        estimated_daily_cost=PriceRange(minimum="500", maximum="900", currency="RUB")
    )
    itinerary = ItineraryBuilder().build(local_content, option(), verified_at=NOW)

    budget = TripBudgetBuilder().build(option(), local_content, itinerary.days)

    assert budget.confirmed_total == Decimal("2000")
    assert budget.estimated == PriceRange(minimum="1000", maximum="1800", currency="RUB")
    assert budget.unknown_components == ()


def test_budget_does_not_mix_estimate_in_another_currency() -> None:
    foreign = content(estimated_daily_cost=PriceRange(minimum="10", maximum="20", currency="USD"))
    itinerary = ItineraryBuilder().build(foreign, option(), verified_at=NOW)

    budget = TripBudgetBuilder().build(option(), foreign, itinerary.days)

    assert budget.confirmed_total == Decimal("2000")
    assert budget.estimated is None
    assert "local_expenses_other_currency" in budget.unknown_components


def test_cross_destination_ranking_is_deterministic_on_exact_ties() -> None:
    ranked = rank_proposals((proposal("city_b"), proposal("city_a")))

    assert [item.proposal.candidate.destination.destination_id for item in ranked] == [
        "city_a",
        "city_b",
    ]
    assert ranked[0].labels == {
        "best_match",
        "lowest_confirmed_cost",
        "most_time_in_city",
    }
    assert ranked[0].score_version == "discovery_v1"


def test_ranking_penalizes_transfers_and_night_travel() -> None:
    difficult = proposal("city_a")
    difficult_combination = difficult.trip_option.combination.model_copy(
        update={
            "outbound": difficult.trip_option.combination.outbound.model_copy(
                update={"transfers": 2}
            ),
            "return_offer": difficult.trip_option.combination.return_offer.model_copy(
                update={"transfers": 2}
            ),
            "metrics": difficult.trip_option.combination.metrics.model_copy(
                update={"night_travel_hours": Decimal(8)}
            ),
        }
    )
    difficult = difficult.model_copy(
        update={
            "trip_option": difficult.trip_option.model_copy(
                update={"combination": difficult_combination}
            )
        }
    )

    ranked = rank_proposals((difficult, proposal("city_z")))

    assert ranked[0].proposal.candidate.destination.destination_id == "city_z"
    with pytest.raises(ValueError, match="sum to one"):
        RankingWeights(candidate_match=Decimal(1))


class ContentGateway:
    def __init__(self, contents: dict[str, DestinationContent]) -> None:
        self.contents = contents

    async def get_content(self, destination_id, dates):
        value = self.contents[destination_id]
        if isinstance(value, Exception):
            raise value
        return value


def feasibility(*destination_ids: str) -> DiscoveryFeasibilityResult:
    request = DiscoveryRequest(
        origin="Москва",
        dates=DateRange(start=date(2026, 8, 22), end=date(2026, 8, 23)),
    )
    candidates = tuple(candidate(item) for item in destination_ids)
    shortlist = CandidateShortlist(
        request=request,
        candidates=candidates,
        catalog_version="v1",
        score_version="candidate_v1",
    )
    snapshots = []
    for item in destination_ids:
        trip = option(item)
        snapshots.append(
            FeasibilitySnapshot(
                destination_id=item,
                status=FeasibilityStatus.VERIFIED,
                verified_at=NOW,
                trip_result=TripSearchResult(
                    request=TripRequest(
                        origin="Москва",
                        destination=item,
                        departure_date=date(2026, 8, 22),
                        return_date=date(2026, 8, 23),
                    ),
                    options=(trip,),
                    searched_at=NOW,
                ),
            )
        )
    return DiscoveryFeasibilityResult(
        shortlist=shortlist,
        snapshots=tuple(snapshots),
        completed_at=NOW,
    )


@pytest.mark.asyncio
async def test_proposal_builder_isolates_content_failure_and_projects_only_safe_facts() -> None:
    gateway = ContentGateway(
        {
            "city_a": content("city_a"),
            "city_b": RuntimeError("catalog unavailable"),
        }
    )

    result = await ProposalBuilder(gateway, FakeClock()).build(feasibility("city_a", "city_b"))

    assert len(result.recommendations) == 1
    assert result.failures[0].destination_id == "city_b"
    facts = project_grounded_facts(result.recommendations)
    serialized = facts[0].model_dump_json()
    assert "secret-provider" not in serialized
    assert "checkout_data" not in serialized
    assert facts[0].transport_facts[0].startswith("Туда: rail")
    assert facts[0].transport_facts[1].startswith("Обратно: bus")


@pytest.mark.asyncio
async def test_grounded_narration_falls_back_when_llm_is_unavailable() -> None:
    class FailingNarrator:
        async def narrate(self, proposals, *, context):
            raise LlmProviderError("offline")

    ranked = rank_proposals((proposal("city_a"),))
    service = GroundedProposalNarration(FailingNarrator())  # type: ignore[arg-type]

    copies = await service.narrate(
        ranked,
        context=ConversationContext(timezone="Europe/Moscow", current_date="2026-07-21"),
    )

    assert copies[0].title == "City_A"
    assert copies[0].evidence_ids == {"source_city_a"}


@pytest.mark.asyncio
async def test_grounded_narration_rejects_copy_with_foreign_evidence() -> None:
    class ModelUnsafeNarrator:
        async def narrate(self, proposals, *, context):
            return [
                ProposalCopy(
                    proposal_id=proposals[0].proposal_id,
                    title="Непроверенный заголовок",
                    reason="Непроверенная причина",
                    trade_off="Неизвестно",
                    evidence_ids={"foreign_source"},
                )
            ]

    ranked = rank_proposals((proposal("city_a"),))
    service = GroundedProposalNarration(ModelUnsafeNarrator())  # type: ignore[arg-type]

    copies = await service.narrate(
        ranked,
        context=ConversationContext(timezone="Europe/Moscow", current_date="2026-07-21"),
    )

    assert copies[0].title == "City_A"
    assert copies[0].evidence_ids == {"source_city_a"}
