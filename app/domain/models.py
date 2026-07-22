"""Validated domain models shared by application services and ports."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Currency = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
NonNegativeMoney = Annotated[Decimal, Field(ge=0, max_digits=14, decimal_places=2)]


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class HotelMode(StrEnum):
    FORBIDDEN = "forbidden"
    OPTIONAL = "optional"
    REQUIRED = "required"


class SortPreference(StrEnum):
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    BALANCED = "balanced"


class TransportMode(StrEnum):
    AVIA = "avia"
    RAIL = "rail"
    BUS = "bus"
    ETRAIN = "etrain"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class PriceStatus(StrEnum):
    EXACT = "exact"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class DialogStep(StrEnum):
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    SEARCHING = "searching"
    RESULTS = "results"
    DETAILS = "details"
    CHECKOUT = "checkout"


class TripComponent(StrEnum):
    OUTBOUND = "outbound"
    RETURN = "return"
    HOTEL = "hotel"


class EventConstraint(DomainModel):
    date: date
    start_time: time | None = None
    end_time: time | None = None
    end_date: date | None = None
    arrival_buffer_minutes: int = Field(default=90, ge=0)
    departure_buffer_minutes: int = Field(default=60, ge=0)

    @model_validator(mode="after")
    def validate_event_interval(self) -> EventConstraint:
        effective_end_date = self.end_date or self.date
        if effective_end_date < self.date:
            raise ValueError("event end_date cannot precede event date")
        if (
            self.start_time is not None
            and self.end_time is not None
            and effective_end_date == self.date
            and self.end_time <= self.start_time
        ):
            raise ValueError("same-day event end_time must be after start_time")
        return self


class TravelerComposition(DomainModel):
    adults: int = Field(default=1, ge=1, le=2)
    children: int = Field(default=0, ge=0, le=0)
    rooms: int = Field(default=1, ge=1, le=1)


class TimeWindow(DomainModel):
    from_time: time | None = None
    to_time: time | None = None

    @model_validator(mode="after")
    def validate_window(self) -> TimeWindow:
        if (
            self.from_time is not None
            and self.to_time is not None
            and self.to_time <= self.from_time
        ):
            raise ValueError("time window end must be after start")
        return self


class TransportPreferences(DomainModel):
    allowed_modes: frozenset[TransportMode] = frozenset()
    max_transfers: int | None = Field(default=None, ge=0)
    departure_window: TimeWindow | None = None
    return_window: TimeWindow | None = None


class HotelPreferences(DomainModel):
    mode: HotelMode = HotelMode.OPTIONAL
    stars_min: int | None = Field(default=None, ge=0, le=5)
    rating_min: Decimal | None = Field(default=None, ge=0, le=10)
    meal_preferences: frozenset[str] = frozenset()
    free_cancellation_required: bool = False
    required_amenities: frozenset[str] = frozenset()


class TripRequest(DomainModel):
    origin: str = Field(min_length=1, max_length=200)
    destination: str = Field(min_length=1, max_length=200)
    departure_date: date
    return_date: date
    travelers: TravelerComposition = TravelerComposition()
    budget: NonNegativeMoney | None = None
    currency: Currency = "RUB"
    transport: TransportPreferences = TransportPreferences()
    hotel: HotelPreferences = HotelPreferences()
    event: EventConstraint | None = None
    sort: SortPreference = SortPreference.BALANCED

    @model_validator(mode="after")
    def validate_route_and_dates(self) -> TripRequest:
        if self.origin.casefold() == self.destination.casefold():
            raise ValueError("origin and destination must differ")
        duration = (self.return_date - self.departure_date).days
        if duration < 0:
            raise ValueError("return_date cannot precede departure_date")
        if duration > 3:
            raise ValueError("P0 supports at most four calendar days")
        if self.event is not None and not (
            self.departure_date <= self.event.date <= self.return_date
        ):
            raise ValueError("event date must be inside the trip date range")
        return self


class ParsedTripDraft(DomainModel):
    origin: str | None = None
    destination: str | None = None
    departure_date: date | None = None
    return_date: date | None = None
    # Drafts preserve unsupported values so the product can explain its scope instead of
    # silently coercing a user's party into a valid provider request.
    adults: int | None = Field(default=None, ge=1, le=20)
    children: int = Field(default=0, ge=0, le=20)
    rooms: int = Field(default=1, ge=1, le=10)
    budget: NonNegativeMoney | None = None
    currency: Currency = "RUB"
    allowed_modes: frozenset[TransportMode] = frozenset()
    max_transfers: int | None = Field(default=None, ge=0)
    departure_window: TimeWindow | None = None
    return_window: TimeWindow | None = None
    hotel_mode: HotelMode | None = None
    hotel_stars_min: int | None = Field(default=None, ge=0, le=5)
    hotel_rating_min: Decimal | None = Field(default=None, ge=0, le=10)
    hotel_meals: frozenset[str] = frozenset()
    hotel_free_cancellation: bool = False
    hotel_amenities: frozenset[str] = frozenset()
    event_date: date | None = None
    event_start_time: time | None = None
    event_end_time: time | None = None
    event_end_date: date | None = None
    sort: SortPreference | None = None


class ParseResult(DomainModel):
    draft: ParsedTripDraft
    missing_fields: tuple[str, ...] = ()


class OfferRef(DomainModel):
    product_type: str
    checkout_data: Mapping[str, Any] = Field(default_factory=dict, repr=False)
    details_data: Mapping[str, Any] = Field(default_factory=dict, repr=False)


class ToolCapability(DomainModel):
    name: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None = None
    read_only: bool | None = None


class TutuCapabilities(DomainModel):
    tools: tuple[ToolCapability, ...]
    schema_hash: str
    captured_at: datetime

    @model_validator(mode="after")
    def validate_unique_names(self) -> TutuCapabilities:
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("capability tool names must be unique")
        return self


class TransportOffer(DomainModel):
    provider_id: str | None = Field(default=None, repr=False)
    mode: TransportMode
    origin: str
    destination: str
    departure_at: datetime
    arrival_at: datetime
    duration: timedelta | None = None
    price: NonNegativeMoney | None = None
    currency: Currency | None = None
    transfers: int | None = Field(default=None, ge=0)
    provider_url: AnyHttpUrl | None = None
    offer_ref: OfferRef
    timezone_known: bool = True

    @model_validator(mode="after")
    def validate_offer(self) -> TransportOffer:
        if self.arrival_at <= self.departure_at:
            raise ValueError("arrival_at must be after departure_at")
        if self.price is not None and self.currency is None:
            raise ValueError("currency is required when price is known")
        if self.provider_id is None and self.provider_url is None:
            raise ValueError("provider_id or provider_url is required")
        return self

    @field_validator("provider_url")
    @classmethod
    def require_https_url(cls, value: AnyHttpUrl | None) -> AnyHttpUrl | None:
        if value is not None and value.scheme != "https":
            raise ValueError("provider URL must use HTTPS")
        return value


class HotelOffer(DomainModel):
    hotel_id: str | None = Field(default=None, repr=False)
    offer_id: str | None = Field(default=None, repr=False)
    name: str = Field(min_length=1)
    check_in: date
    check_out: date
    price: NonNegativeMoney | None = None
    currency: Currency | None = None
    stars: int | None = Field(default=None, ge=0, le=5)
    rating: Decimal | None = Field(default=None, ge=0, le=10)
    review_count: int | None = Field(default=None, ge=0)
    room_name: str | None = None
    breakfast_included: bool | None = None
    free_cancellation: bool | None = None
    matched_amenities: frozenset[str] = frozenset()
    matched_meals: frozenset[str] = frozenset()
    provider_url: AnyHttpUrl | None = None
    offer_ref: OfferRef

    @model_validator(mode="after")
    def validate_offer(self) -> HotelOffer:
        if self.check_out <= self.check_in:
            raise ValueError("hotel check_out must be after check_in")
        if self.price is not None and self.currency is None:
            raise ValueError("currency is required when price is known")
        if self.offer_id is None and self.provider_url is None:
            raise ValueError("offer_id or provider_url is required")
        return self

    @field_validator("provider_url")
    @classmethod
    def require_https_url(cls, value: AnyHttpUrl | None) -> AnyHttpUrl | None:
        if value is not None and value.scheme != "https":
            raise ValueError("provider URL must use HTTPS")
        return value


class HotelStay(DomainModel):
    check_in: date
    check_out: date
    nights: int = Field(ge=1, le=3)
    needs_early_checkin_warning: bool = False
    needs_late_checkout_warning: bool = False

    @model_validator(mode="after")
    def validate_nights(self) -> HotelStay:
        if (self.check_out - self.check_in).days != self.nights:
            raise ValueError("nights must match the stay date range")
        return self


class TransportSearchQuery(DomainModel):
    origin: str
    destination: str
    departure_date: date
    adults: int = Field(default=1, ge=1, le=2)
    allowed_modes: frozenset[TransportMode] = frozenset()
    max_price: NonNegativeMoney | None = None


class HotelSearchQuery(DomainModel):
    city: str
    check_in: date
    check_out: date
    adults: int = Field(default=1, ge=1, le=2)
    rooms: int = Field(default=1, ge=1, le=1)
    preferences: HotelPreferences


class CheckoutLink(DomainModel):
    url: AnyHttpUrl
    fallback_url: AnyHttpUrl | None = None
    expires_at: datetime | None = None
    kind: str | None = Field(default=None, max_length=50)

    @field_validator("url", "fallback_url")
    @classmethod
    def require_https_url(cls, value: AnyHttpUrl | None) -> AnyHttpUrl | None:
        if value is not None and value.scheme != "https":
            raise ValueError("checkout URLs must use HTTPS")
        return value


class TripCheckoutItem(DomainModel):
    component: TripComponent
    link: CheckoutLink


class HotelRateDetails(DomainModel):
    price: NonNegativeMoney | None = None
    currency: Currency | None = None
    breakfast_included: bool | None = None
    free_cancellation: bool | None = None
    refundable: bool | None = None
    pay_online: bool | None = None
    checkout_url: AnyHttpUrl | None = None
    offer_pack_ref: str | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def validate_money(self) -> HotelRateDetails:
        if self.price is not None and self.currency is None:
            raise ValueError("currency is required when rate price is known")
        return self


class HotelRoomDetails(DomainModel):
    name: str
    bed_summary: str | None = None
    max_occupancy: int | None = Field(default=None, ge=1)
    rates: tuple[HotelRateDetails, ...] = ()


class OfferDetails(DomainModel):
    product_type: str
    title: str
    check_in_time: time | None = None
    check_out_time: time | None = None
    amenities: tuple[str, ...] = ()
    rooms: tuple[HotelRoomDetails, ...] = ()
    facts: tuple[str, ...] = ()


class TripSearchFailure(DomainModel):
    category: str
    component: str
    retryable: bool
    user_message: str
    cause_code: str | None = None


class PriceSummary(DomainModel):
    status: PriceStatus
    known_total: NonNegativeMoney | None = None
    currency: Currency | None = None
    missing_components: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_summary(self) -> PriceSummary:
        if self.known_total is not None and self.currency is None:
            raise ValueError("currency is required for a known total")
        if self.status is PriceStatus.EXACT and self.known_total is None:
            raise ValueError("exact price requires known_total")
        return self


class TripTimeMetrics(DomainModel):
    total_travel_duration: timedelta
    trip_duration: timedelta
    time_in_city: timedelta
    night_travel_hours: Decimal = Field(default=Decimal(0), ge=0)


class TripCombination(DomainModel):
    outbound: TransportOffer
    return_offer: TransportOffer
    hotel: HotelOffer | None = None
    stay: HotelStay | None = None
    price: PriceSummary
    metrics: TripTimeMetrics
    warnings: tuple[str, ...] = ()
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_components(self) -> TripCombination:
        if (self.hotel is None) != (self.stay is None):
            raise ValueError("hotel and stay must either both exist or both be absent")
        if self.return_offer.departure_at <= self.outbound.arrival_at:
            raise ValueError("return departure must follow outbound arrival")
        if (
            self.hotel is not None
            and self.stay is not None
            and (
                self.hotel.check_in != self.stay.check_in
                or self.hotel.check_out != self.stay.check_out
            )
        ):
            raise ValueError("hotel offer dates must exactly match the combination stay")
        return self


class RankedTripOption(DomainModel):
    combination: TripCombination
    labels: frozenset[SortPreference]
    balance_score: Decimal | None = Field(default=None, ge=0)
    reasons: tuple[str, ...] = ()


class TripSearchResult(DomainModel):
    request: TripRequest
    options: tuple[RankedTripOption, ...] = Field(default=(), max_length=3)
    failures: tuple[TripSearchFailure, ...] = ()
    searched_at: datetime


class TripTransportCandidates(DomainModel):
    """Reusable transport-stage result, before any hotel calls."""

    request: TripRequest
    outbound: tuple[TransportOffer, ...] = ()
    return_offers: tuple[TransportOffer, ...] = ()
    failures: tuple[TripSearchFailure, ...] = ()
    searched_at: datetime

    @property
    def is_feasible(self) -> bool:
        return bool(self.outbound and self.return_offers)
