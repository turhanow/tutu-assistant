"""Trusted-boundary models for destination and travel content."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Annotated
from urllib.parse import parse_qs

from pydantic import AnyHttpUrl, Field, field_validator, model_validator

from app.domain.models import Currency, DomainModel, NonNegativeMoney

ContentId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")]


class EvidenceRef(DomainModel):
    evidence_id: ContentId
    title: str = Field(min_length=1, max_length=300)
    source_url: AnyHttpUrl
    retrieved_at: datetime
    valid_until: datetime | None = None
    license_name: str | None = Field(default=None, max_length=100)
    attribution: str | None = Field(default=None, max_length=300)
    is_dynamic: bool = False

    @field_validator("source_url")
    @classmethod
    def require_https(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.scheme != "https":
            raise ValueError("evidence URL must use HTTPS")
        return value

    @model_validator(mode="after")
    def validate_freshness(self) -> EvidenceRef:
        if self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if self.valid_until is not None:
            if self.valid_until.tzinfo is None:
                raise ValueError("valid_until must be timezone-aware")
            if self.valid_until <= self.retrieved_at:
                raise ValueError("valid_until must follow retrieved_at")
        if self.is_dynamic and self.valid_until is None:
            raise ValueError("dynamic evidence requires valid_until")
        return self

    def is_fresh_at(self, moment: datetime) -> bool:
        if moment.tzinfo is None:
            raise ValueError("freshness moment must be timezone-aware")
        return self.valid_until is None or moment <= self.valid_until


class PriceRange(DomainModel):
    minimum: NonNegativeMoney
    maximum: NonNegativeMoney
    currency: Currency = "RUB"

    @model_validator(mode="after")
    def validate_order(self) -> PriceRange:
        if self.maximum < self.minimum:
            raise ValueError("price range maximum cannot be lower than minimum")
        return self


class Activity(DomainModel):
    activity_id: ContentId
    destination_id: ContentId
    name: str = Field(min_length=1, max_length=200)
    categories: frozenset[str] = Field(min_length=1)
    duration: timedelta = Field(gt=timedelta(0), le=timedelta(hours=12))
    estimated_cost: PriceRange | None = None
    indoor: bool | None = None
    evidence_ids: frozenset[ContentId] = frozenset()
    address: str | None = Field(default=None, min_length=3, max_length=300)
    map_url: AnyHttpUrl | None = None

    @field_validator("categories")
    @classmethod
    def normalize_categories(cls, value: frozenset[str]) -> frozenset[str]:
        normalized = frozenset(item.strip().casefold() for item in value if item.strip())
        if not normalized:
            raise ValueError("activity requires at least one category")
        return normalized

    @field_validator("map_url")
    @classmethod
    def require_yandex_maps_url(cls, value: AnyHttpUrl | None) -> AnyHttpUrl | None:
        if value is None:
            return None
        if value.scheme != "https" or value.host != "yandex.ru" or value.path != "/maps/":
            raise ValueError("activity map URL must use the Yandex Maps search endpoint")
        if value.username or value.password or value.fragment:
            raise ValueError("activity map URL must not contain credentials or fragments")
        query = parse_qs(value.query or "", keep_blank_values=True)
        if set(query) != {"text"} or len(query["text"]) != 1 or not query["text"][0].strip():
            raise ValueError("activity map URL must contain only one non-empty text query")
        return value


class DestinationProfile(DomainModel):
    destination_id: ContentId
    name: str = Field(min_length=1, max_length=200)
    region: str = Field(min_length=1, max_length=200)
    aliases: frozenset[str] = frozenset()
    experience_tags: frozenset[str] = Field(min_length=1)
    season_months: frozenset[int] = Field(default_factory=lambda: frozenset(range(1, 13)))
    typical_visit_duration: timedelta = Field(
        default=timedelta(days=2),
        ge=timedelta(hours=4),
        le=timedelta(days=4),
    )
    estimated_daily_cost: PriceRange | None = None
    activity_highlights: tuple[str, ...] = Field(default=(), max_length=3)
    evidence_ids: frozenset[ContentId] = frozenset()

    @field_validator("activity_highlights")
    @classmethod
    def normalize_activity_highlights(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value if item.strip())
        if len({item.casefold() for item in normalized}) != len(normalized):
            raise ValueError("activity highlights must be unique")
        return normalized

    @field_validator("season_months")
    @classmethod
    def validate_months(cls, value: frozenset[int]) -> frozenset[int]:
        if not value or any(month < 1 or month > 12 for month in value):
            raise ValueError("season_months must contain values from 1 to 12")
        return value

    @field_validator("experience_tags")
    @classmethod
    def normalize_tags(cls, value: frozenset[str]) -> frozenset[str]:
        normalized = frozenset(item.strip().casefold() for item in value if item.strip())
        if not normalized:
            raise ValueError("destination requires experience tags")
        return normalized


class DestinationContent(DomainModel):
    destination: DestinationProfile
    activities: tuple[Activity, ...] = ()
    evidence: tuple[EvidenceRef, ...]
    catalog_version: str = Field(min_length=1, max_length=50)
    is_ai_generated: bool = False

    @model_validator(mode="after")
    def validate_references(self) -> DestinationContent:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        known = set(evidence_ids)
        referenced = set(self.destination.evidence_ids)
        for activity in self.activities:
            if activity.destination_id != self.destination.destination_id:
                raise ValueError("activity belongs to a different destination")
            referenced.update(activity.evidence_ids)
        missing = referenced - known
        if missing:
            raise ValueError(f"unknown evidence references: {sorted(missing)}")
        activity_ids = [item.activity_id for item in self.activities]
        if len(activity_ids) != len(set(activity_ids)):
            raise ValueError("activity IDs must be unique")
        if not self.is_ai_generated and not referenced:
            raise ValueError("curated destination content requires evidence")
        return self


class DestinationCatalogBundle(DomainModel):
    version: str = Field(pattern=r"^v[1-9][0-9]*(?:\.[0-9]+){0,2}$")
    generated_at: datetime
    pilot_origins: frozenset[str] = Field(min_length=1)
    destinations: tuple[DestinationContent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bundle(self) -> DestinationCatalogBundle:
        if self.generated_at.tzinfo is None:
            raise ValueError("catalog generated_at must be timezone-aware")
        destination_ids = [item.destination.destination_id for item in self.destinations]
        if len(destination_ids) != len(set(destination_ids)):
            raise ValueError("destination IDs must be unique across catalog")
        if any(item.catalog_version != self.version for item in self.destinations):
            raise ValueError("destination catalog_version must match bundle version")
        return self


def sum_price_ranges(ranges: tuple[PriceRange, ...]) -> PriceRange | None:
    """Sum compatible estimates without converting currencies."""
    if not ranges:
        return None
    currency = ranges[0].currency
    if any(item.currency != currency for item in ranges):
        return None
    return PriceRange(
        minimum=sum((item.minimum for item in ranges), Decimal(0)),
        maximum=sum((item.maximum for item in ranges), Decimal(0)),
        currency=currency,
    )
