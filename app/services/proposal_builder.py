"""Compose verified logistics and trusted content into grounded proposals."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from decimal import Decimal

from app.domain.discovery_models import (
    DiscoveryFeasibilityResult,
    DiscoveryProposalResult,
    FeasibilitySnapshot,
    FeasibilityStatus,
    GroundedProposalFacts,
    ProposalBuildFailure,
    ProposalCopy,
    Recommendation,
    WeekendProposal,
)
from app.domain.errors import LlmParseError, LlmProviderError
from app.domain.models import RankedTripOption, SortPreference
from app.ports.clock import Clock
from app.ports.content import TravelContentGateway
from app.ports.discovery_llm import ConversationContext, ProposalNarrator
from app.services.discovery_ranking import rank_proposals
from app.services.itinerary_builder import ItineraryBuilder
from app.services.trip_budget import TripBudgetBuilder


class ProposalBuilder:
    def __init__(
        self,
        content_gateway: TravelContentGateway,
        clock: Clock,
        *,
        itinerary_builder: ItineraryBuilder | None = None,
        budget_builder: TripBudgetBuilder | None = None,
    ) -> None:
        self._content_gateway = content_gateway
        self._clock = clock
        self._itinerary = itinerary_builder or ItineraryBuilder()
        self._budget = budget_builder or TripBudgetBuilder()

    async def build(self, feasibility: DiscoveryFeasibilityResult) -> DiscoveryProposalResult:
        candidates = {
            item.destination.destination_id: item for item in feasibility.shortlist.candidates
        }
        eligible = tuple(
            snapshot
            for snapshot in feasibility.snapshots
            if snapshot.status in {FeasibilityStatus.VERIFIED, FeasibilityStatus.PARTIAL}
            and snapshot.trip_result is not None
            and snapshot.trip_result.options
        )
        outcomes = await asyncio.gather(
            *(
                self._build_one(
                    snapshot,
                    candidates[snapshot.destination_id],
                    feasibility,
                )
                for snapshot in eligible
            ),
            return_exceptions=True,
        )
        proposals: list[WeekendProposal] = []
        failures: list[ProposalBuildFailure] = []
        for snapshot, outcome in zip(eligible, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                failures.append(
                    ProposalBuildFailure(
                        destination_id=snapshot.destination_id,
                        category="content_unavailable",
                        retryable=False,
                        user_message="Не удалось собрать проверенную программу для направления",
                    )
                )
            else:
                proposals.append(outcome)
        return DiscoveryProposalResult(
            recommendations=rank_proposals(
                proposals,
                budget=feasibility.shortlist.request.budget,
                budget_currency=feasibility.shortlist.request.currency,
            ),
            failures=tuple(failures),
            completed_at=self._clock.now(),
        )

    async def _build_one(self, snapshot, candidate, feasibility) -> WeekendProposal:
        content = await self._content_gateway.get_content(
            snapshot.destination_id,
            feasibility.shortlist.request.dates,
        )
        if content.destination.destination_id != snapshot.destination_id:
            raise ValueError("content destination does not match feasibility snapshot")
        option = select_trip_option(snapshot)
        itinerary = self._itinerary.build(
            content,
            option,
            verified_at=snapshot.verified_at,
            preferred_categories=feasibility.shortlist.request.experience.interests,
        )
        cost = self._budget.build(option, content, itinerary.days)
        fresh_evidence_ids = {
            item.evidence_id for item in content.evidence if item.is_fresh_at(snapshot.verified_at)
        }
        evidence_ids = set(content.destination.evidence_ids).intersection(fresh_evidence_ids)
        for day in itinerary.days:
            for scheduled in day.activities:
                evidence_ids.update(scheduled.activity.evidence_ids)
            for suggestion in day.suggestions:
                evidence_ids.update(suggestion.evidence_ids)
        if not evidence_ids and not content.is_ai_generated:
            raise ValueError("proposal has no fresh destination evidence")
        content_grounded = bool(evidence_ids)
        combination_warning = (
            option.combination.warnings[0] if option.combination.warnings else None
        )
        if not content_grounded:
            trade_off = "Идеи программы подобраны AI; проверьте часы работы и условия"
            if itinerary.contains_unverified_activities:
                trade_off = "Часть программы отмечена как непроверенные идеи. " + trade_off
            if combination_warning:
                trade_off = f"{combination_warning} {trade_off}"
        elif not itinerary.content_complete:
            trade_off = "Программа неполная: не удалось разместить две проверенные активности"
        elif itinerary.contains_unverified_activities:
            trade_off = (
                "Часть программы отмечена как непроверенные идеи. "
                "Проверьте часы работы и условия перед поездкой"
            )
        elif combination_warning:
            trade_off = combination_warning
        elif cost.unknown_components:
            trade_off = "Часть расходов пока неизвестна; проверьте её перед бронированием"
        else:
            trade_off = itinerary.warnings[0]
        return WeekendProposal(
            candidate=candidate,
            trip_option=option,
            trip_alternatives=tuple(
                item
                for item in snapshot.trip_result.options
                if item.combination.signature != option.combination.signature
            ),
            days=itinerary.days,
            cost=cost,
            verified_at=snapshot.verified_at,
            evidence_ids=frozenset(evidence_ids),
            content_grounded=content_grounded,
            trade_off=trade_off,
            content_complete=itinerary.content_complete,
            suggested_activities=content.activities[:4],
        )


def select_trip_option(snapshot: FeasibilitySnapshot) -> RankedTripOption:
    if snapshot.trip_result is None or not snapshot.trip_result.options:
        raise ValueError("proposal requires at least one verified trip option")
    return next(
        (item for item in snapshot.trip_result.options if SortPreference.BALANCED in item.labels),
        snapshot.trip_result.options[0],
    )


def project_grounded_facts(
    recommendations: Sequence[Recommendation],
) -> tuple[GroundedProposalFacts, ...]:
    """Allowlist facts before any optional LLM narration call."""
    return tuple(
        _project_one(item, proposal_id=f"proposal_{index}")
        for index, item in enumerate(recommendations, start=1)
    )


class GroundedProposalNarration:
    """Narrate allowlisted facts and degrade to deterministic copy on LLM failure."""

    def __init__(self, narrator: ProposalNarrator) -> None:
        self._narrator = narrator

    async def narrate(
        self,
        recommendations: Sequence[Recommendation],
        *,
        context: ConversationContext,
    ) -> tuple[ProposalCopy, ...]:
        facts = project_grounded_facts(recommendations)
        if not facts:
            return ()
        if any(not item.evidence_ids for item in facts):
            return fallback_proposal_copies(facts)
        try:
            copies = tuple(await self._narrator.narrate(facts, context=context))
        except (LlmParseError, LlmProviderError, TimeoutError):
            return fallback_proposal_copies(facts)
        if not _copies_match_facts(copies, facts):
            return fallback_proposal_copies(facts)
        return copies


def fallback_proposal_copies(
    facts: Sequence[GroundedProposalFacts],
) -> tuple[ProposalCopy, ...]:
    return tuple(
        ProposalCopy(
            proposal_id=item.proposal_id,
            title=item.destination_name,
            reason="; ".join(item.match_reasons),
            trade_off=item.trade_off,
            evidence_ids=item.evidence_ids,
        )
        for item in facts
    )


def _copies_match_facts(
    copies: Sequence[ProposalCopy],
    facts: Sequence[GroundedProposalFacts],
) -> bool:
    expected = {item.proposal_id: item.evidence_ids for item in facts}
    if not all(isinstance(item, ProposalCopy) for item in copies):
        return False
    if len(copies) != len(expected) or {item.proposal_id for item in copies} != set(expected):
        return False
    return all(
        bool(item.evidence_ids) and item.evidence_ids.issubset(expected[item.proposal_id])
        for item in copies
    )


def _project_one(
    recommendation: Recommendation,
    *,
    proposal_id: str,
) -> GroundedProposalFacts:
    proposal = recommendation.proposal
    combination = proposal.trip_option.combination
    estimated = proposal.cost.estimated
    return GroundedProposalFacts(
        proposal_id=proposal_id,
        destination_name=proposal.candidate.destination.name,
        match_reasons=proposal.candidate.match_reasons,
        day_activity_names=tuple(
            tuple(
                [item.activity.name for item in day.activities]
                + [item.name for item in day.suggestions]
            )
            for day in proposal.days
        ),
        transport_facts=(
            _transport_fact("Туда", combination.outbound),
            _transport_fact("Обратно", combination.return_offer),
        ),
        confirmed_cost=(
            _money(proposal.cost.confirmed_total, proposal.cost.confirmed_currency)
            if proposal.cost.confirmed_total is not None
            else None
        ),
        estimated_cost=(
            f"{_decimal(estimated.minimum)}–{_decimal(estimated.maximum)} {estimated.currency}"
            if estimated is not None
            else None
        ),
        unknown_components=proposal.cost.unknown_components,
        trade_off=proposal.trade_off,
        evidence_ids=proposal.evidence_ids,
    )


def _transport_fact(label, offer) -> str:
    return (
        f"{label}: {offer.mode.value}, "
        f"{offer.departure_at.isoformat()} — {offer.arrival_at.isoformat()}"
    )


def _money(amount: Decimal, currency: str | None) -> str:
    return f"{_decimal(amount)} {currency}"


def _decimal(value: Decimal) -> str:
    return format(value, "f")
