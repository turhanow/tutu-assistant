"""Domain models for destination discovery and grounded proposals."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise

from pydantic import Field, field_validator, model_validator

from app.domain.content_models import Activity, ContentId, DestinationProfile, PriceRange
from app.domain.models import (
    Currency,
    DomainModel,
    HotelMode,
    NonNegativeMoney,
    ParsedTripDraft,
    RankedTripOption,
    TransportMode,
    TransportPreferences,
    TravelerComposition,
    TripSearchFailure,
    TripSearchResult,
)


class TripIntent(StrEnum):
    DESTINATION_KNOWN = "destination_known"
    DESTINATION_UNKNOWN = "destination_unknown"
    EVENT_LED = "event_led"


class DateFlexibility(StrEnum):
    FIXED = "fixed"
    RANGE = "range"
    WEEKEND = "weekend"


class TravelPace(StrEnum):
    RELAXED = "relaxed"
    BALANCED = "balanced"
    INTENSIVE = "intensive"


class DateRange(DomainModel):
    start: date
    end: date
    flexibility: DateFlexibility = DateFlexibility.FIXED

    @model_validator(mode="after")
    def validate_range(self) -> DateRange:
        if self.end < self.start:
            raise ValueError("date range end cannot precede start")
        if (self.end - self.start).days > 14:
            raise ValueError("discovery date window cannot exceed 15 calendar days")
        return self


class RoadTolerance(DomainModel):
    max_one_way_duration: timedelta | None = Field(
        default=None,
        ge=timedelta(minutes=30),
        le=timedelta(hours=24),
    )
    max_transfers: int | None = Field(default=None, ge=0, le=3)
    allow_night_travel: bool | None = None


class ExperienceProfile(DomainModel):
    motives: frozenset[str] = frozenset()
    interests: frozenset[str] = frozenset()
    pace: TravelPace | None = None
    road_tolerance: RoadTolerance = RoadTolerance()

    @field_validator("motives", "interests")
    @classmethod
    def normalize_values(cls, value: frozenset[str]) -> frozenset[str]:
        return frozenset(item.strip().casefold() for item in value if item.strip())


class DiscoveryRequest(DomainModel):
    origin: str = Field(min_length=1, max_length=200)
    dates: DateRange
    travelers: TravelerComposition = TravelerComposition()
    budget: NonNegativeMoney | None = None
    currency: Currency = "RUB"
    hotel_mode: HotelMode = HotelMode.OPTIONAL
    experience: ExperienceProfile = ExperienceProfile()
    transport: TransportPreferences = TransportPreferences()

    @model_validator(mode="before")
    @classmethod
    def enforce_hotel_for_overnight_trip(cls, value):
        if not isinstance(value, Mapping):
            return value
        dates = value.get("dates")
        date_range = dates if isinstance(dates, DateRange) else DateRange.model_validate(dates)
        mode = HotelMode(value.get("hotel_mode", HotelMode.OPTIONAL))
        overnight = date_range.end > date_range.start
        if overnight and mode is HotelMode.FORBIDDEN:
            date_range = date_range.model_copy(update={"end": date_range.start})
            overnight = False
        if not overnight and mode is HotelMode.REQUIRED:
            raise ValueError("a same-day discovery trip cannot require a hotel")
        normalized = dict(value)
        normalized["dates"] = date_range
        normalized["hotel_mode"] = HotelMode.REQUIRED if overnight else mode
        return normalized


class DiscoveryDraft(DomainModel):
    origin: str | None = Field(default=None, max_length=200)
    departure_date: date | None = None
    return_date: date | None = None
    date_flexibility: DateFlexibility | None = None
    adults: int | None = Field(default=None, ge=1, le=20)
    children: int = Field(default=0, ge=0, le=20)
    rooms: int = Field(default=1, ge=1, le=10)
    budget: NonNegativeMoney | None = None
    currency: Currency = "RUB"
    hotel_mode: HotelMode | None = None
    allowed_modes: frozenset[TransportMode] = frozenset()
    experience: ExperienceProfile = ExperienceProfile()


class IntentParseResult(DomainModel):
    intent: TripIntent
    confidence: Decimal = Field(ge=0, le=1)
    known_draft: ParsedTripDraft | None = None
    discovery_draft: DiscoveryDraft | None = None
    missing_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_intent_payload(self) -> IntentParseResult:
        if self.intent is TripIntent.DESTINATION_UNKNOWN and self.discovery_draft is None:
            raise ValueError("unknown-destination intent requires discovery draft")
        if self.intent is TripIntent.DESTINATION_KNOWN and self.known_draft is None:
            raise ValueError("known-destination intent requires known trip draft")
        if (
            self.intent is TripIntent.EVENT_LED
            and self.known_draft is None
            and self.discovery_draft is None
        ):
            raise ValueError("event-led intent requires at least one draft")
        return self


class DestinationCandidate(DomainModel):
    destination: DestinationProfile
    match_score: Decimal = Field(ge=0, le=1)
    match_reasons: tuple[str, ...] = Field(min_length=1, max_length=3)


class CandidateShortlist(DomainModel):
    request: DiscoveryRequest
    candidates: tuple[DestinationCandidate, ...] = Field(max_length=8)
    catalog_version: str = Field(min_length=1, max_length=50)
    score_version: str = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_unique_destinations(self) -> CandidateShortlist:
        destination_ids = [item.destination.destination_id for item in self.candidates]
        if len(destination_ids) != len(set(destination_ids)):
            raise ValueError("shortlist destinations must be unique")
        return self


class FeasibilityStatus(StrEnum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    TRANSPORT_ONLY = "transport_only"
    UNAVAILABLE = "unavailable"


class FeasibilitySnapshot(DomainModel):
    destination_id: ContentId
    status: FeasibilityStatus
    verified_at: datetime
    trip_result: TripSearchResult | None = None
    failures: tuple[TripSearchFailure, ...] = ()

    @model_validator(mode="after")
    def validate_snapshot(self) -> FeasibilitySnapshot:
        if self.verified_at.tzinfo is None:
            raise ValueError("verified_at must be timezone-aware")
        has_options = self.trip_result is not None and bool(self.trip_result.options)
        if (
            self.status
            in {
                FeasibilityStatus.VERIFIED,
                FeasibilityStatus.TRANSPORT_ONLY,
            }
            and not has_options
        ):
            raise ValueError(
                "verified or transport-only snapshot requires at least one trip option"
            )
        if self.status is FeasibilityStatus.UNAVAILABLE and has_options:
            raise ValueError("unavailable snapshot cannot contain trip options")
        return self


class DiscoveryFeasibilityResult(DomainModel):
    shortlist: CandidateShortlist
    snapshots: tuple[FeasibilitySnapshot, ...] = Field(max_length=8)
    completed_at: datetime

    @model_validator(mode="after")
    def validate_result(self) -> DiscoveryFeasibilityResult:
        if self.completed_at.tzinfo is None:
            raise ValueError("discovery completion time must be timezone-aware")
        snapshot_ids = [item.destination_id for item in self.snapshots]
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise ValueError("discovery snapshots must have unique destinations")
        candidate_ids = {item.destination.destination_id for item in self.shortlist.candidates}
        if not set(snapshot_ids).issubset(candidate_ids):
            raise ValueError("snapshot destination must belong to shortlist")
        return self


class CostBreakdown(DomainModel):
    confirmed_total: NonNegativeMoney | None = None
    confirmed_currency: Currency | None = None
    estimated: PriceRange | None = None
    unknown_components: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_money(self) -> CostBreakdown:
        if (self.confirmed_total is None) != (self.confirmed_currency is None):
            raise ValueError("confirmed total and currency must be provided together")
        if (
            self.estimated is not None
            and self.confirmed_currency is not None
            and self.estimated.currency != self.confirmed_currency
        ):
            raise ValueError("confirmed and estimated currencies must match")
        return self


class ScheduledActivity(DomainModel):
    activity: Activity
    starts_at: datetime
    ends_at: datetime

    @model_validator(mode="after")
    def validate_interval(self) -> ScheduledActivity:
        if self.starts_at.tzinfo is None or self.ends_at.tzinfo is None:
            raise ValueError("activity schedule must be timezone-aware")
        if self.ends_at <= self.starts_at:
            raise ValueError("scheduled activity must end after it starts")
        if self.ends_at - self.starts_at < self.activity.duration:
            raise ValueError("scheduled interval is shorter than activity duration")
        return self


class DayPlan(DomainModel):
    date: date
    activities: tuple[ScheduledActivity, ...] = ()
    suggestions: tuple[Activity, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def validate_schedule(self) -> DayPlan:
        ordered = sorted(self.activities, key=lambda item: item.starts_at)
        for previous, current in pairwise(ordered):
            if current.starts_at < previous.ends_at:
                raise ValueError("day plan activities cannot overlap")
        if any(item.starts_at.date() != self.date for item in self.activities):
            raise ValueError("activity must start on the day plan date")
        activity_ids = [item.activity.activity_id for item in self.activities]
        suggestion_ids = [item.activity_id for item in self.suggestions]
        if len(suggestion_ids) != len(set(suggestion_ids)):
            raise ValueError("day suggestions must be unique")
        if set(activity_ids).intersection(suggestion_ids):
            raise ValueError("scheduled activities cannot be repeated as suggestions")
        return self


class WeekendProposal(DomainModel):
    candidate: DestinationCandidate
    trip_option: RankedTripOption
    trip_alternatives: tuple[RankedTripOption, ...] = Field(default=(), max_length=3)
    days: tuple[DayPlan, ...] = Field(min_length=1, max_length=4)
    cost: CostBreakdown
    verified_at: datetime
    evidence_ids: frozenset[ContentId] = frozenset()
    content_grounded: bool = True
    trade_off: str = Field(min_length=1, max_length=300)
    content_complete: bool
    suggested_activities: tuple[Activity, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def validate_proposal(self) -> WeekendProposal:
        if self.verified_at.tzinfo is None:
            raise ValueError("proposal verified_at must be timezone-aware")
        activity_count = sum(len(day.activities) + len(day.suggestions) for day in self.days)
        if self.content_complete and any(
            not 2 <= len(day.activities) + len(day.suggestions) <= 3 for day in self.days
        ):
            raise ValueError("complete proposal requires two or three activities per day")
        if self.content_grounded and not self.evidence_ids:
            raise ValueError("grounded proposal requires evidence")
        if not self.content_grounded and self.evidence_ids:
            raise ValueError("AI-generated proposal cannot claim evidence")
        if not self.suggested_activities and activity_count == 0:
            raise ValueError("proposal requires at least one activity suggestion")
        planned = [
            activity
            for day in self.days
            for activity in (
                *(item.activity for item in day.activities),
                *day.suggestions,
            )
        ]
        planned_ids = [item.activity_id for item in planned]
        planned_names = [
            " ".join(item.name.casefold().replace("ё", "е").split()) for item in planned
        ]
        if len(planned_ids) != len(set(planned_ids)) or len(planned_names) != len(
            set(planned_names)
        ):
            raise ValueError("a place cannot be repeated within one trip plan")
        signatures = [item.combination.signature for item in self.trip_alternatives]
        if len(signatures) != len(set(signatures)):
            raise ValueError("trip alternatives must be unique")
        if self.trip_option.combination.signature in signatures:
            raise ValueError("selected trip option cannot be repeated as an alternative")
        return self


class Recommendation(DomainModel):
    proposal: WeekendProposal
    labels: frozenset[str]
    score: Decimal = Field(ge=0, le=1)
    score_version: str = Field(min_length=1, max_length=50)


class ProposalBuildFailure(DomainModel):
    """Destination-scoped failure that must not cancel other proposals."""

    destination_id: ContentId
    category: str = Field(min_length=1, max_length=100)
    retryable: bool
    user_message: str = Field(min_length=1, max_length=300)


class DiscoveryProposalResult(DomainModel):
    recommendations: tuple[Recommendation, ...] = Field(max_length=3)
    failures: tuple[ProposalBuildFailure, ...] = ()
    completed_at: datetime

    @model_validator(mode="after")
    def validate_result(self) -> DiscoveryProposalResult:
        if self.completed_at.tzinfo is None:
            raise ValueError("proposal completion time must be timezone-aware")
        destination_ids = [
            item.proposal.candidate.destination.destination_id for item in self.recommendations
        ]
        if len(destination_ids) != len(set(destination_ids)):
            raise ValueError("proposal recommendations must have unique destinations")
        return self


class ProposalCopy(DomainModel):
    proposal_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{4,64}$")
    title: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=500)
    trade_off: str = Field(min_length=1, max_length=300)
    evidence_ids: frozenset[ContentId] = frozenset()


class GroundedProposalFacts(DomainModel):
    """Allowlisted proposal projection safe to send to an LLM."""

    proposal_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{4,64}$")
    destination_name: str = Field(min_length=1, max_length=200)
    match_reasons: tuple[str, ...] = Field(min_length=1, max_length=3)
    day_activity_names: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=4)
    transport_facts: tuple[str, ...] = Field(min_length=2, max_length=4)
    confirmed_cost: str | None = Field(default=None, max_length=100)
    estimated_cost: str | None = Field(default=None, max_length=100)
    unknown_components: tuple[str, ...] = ()
    trade_off: str = Field(min_length=1, max_length=300)
    evidence_ids: frozenset[ContentId] = frozenset()
