"""Privacy-bounded product analytics and feedback models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from app.domain.discovery_models import TripIntent
from app.domain.models import DomainModel

ALLOWED_PRODUCT_EVENTS = frozenset(
    {
        "conversation_started",
        "intent_classified",
        "clarification_shown",
        "clarification_answered",
        "shortlist_generated",
        "shortlist_empty",
        "verification_started",
        "destination_verified",
        "verification_completed",
        "inspiration_shown",
        "proposals_shown",
        "proposal_details_opened",
        "proposal_rejected",
        "price_rechecked",
        "handoff_requested",
        "handoff_created",
        "feedback_submitted",
    }
)

ALLOWED_EVENT_DIMENSIONS = frozenset(
    {
        "flow_type",
        "voice_version",
        "prompt_version",
        "catalog_version",
        "candidate_count",
        "verified_count",
        "proposal_count",
        "price_completeness",
        "latency_bucket",
        "failure_category",
        "reason_code",
        "revision",
    }
)


class FeedbackReason(StrEnum):
    IRRELEVANT = "irrelevant"
    TOO_EXPENSIVE = "too_expensive"
    BAD_LOGISTICS = "bad_logistics"
    UNINTERESTING = "uninteresting"
    WRONG_DATA = "wrong_data"
    TECHNICAL_PROBLEM = "technical_problem"
    OTHER = "other"


class ProductEvent(DomainModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    occurred_at: datetime
    flow_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{8,64}$")
    intent: TripIntent | None = None
    dimensions: dict[str, str | int | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_event(self) -> ProductEvent:
        if self.occurred_at.tzinfo is None:
            raise ValueError("event timestamp must be timezone-aware")
        if len(self.dimensions) > 20:
            raise ValueError("too many event dimensions")
        if self.name not in ALLOWED_PRODUCT_EVENTS:
            raise ValueError("product event is not allowlisted")
        forbidden = set(self.dimensions) - ALLOWED_EVENT_DIMENSIONS
        if forbidden:
            raise ValueError(f"event dimensions are not allowlisted: {sorted(forbidden)}")
        if any(isinstance(value, str) and len(value) > 64 for value in self.dimensions.values()):
            raise ValueError("event string dimension exceeds 64 characters")
        return self


class FeedbackRecord(DomainModel):
    flow_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{8,64}$")
    reason: FeedbackReason
    comment: str | None = Field(default=None, max_length=1_000)
    created_at: datetime

    @field_validator("comment")
    @classmethod
    def normalize_comment(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    @model_validator(mode="after")
    def validate_timestamp(self) -> FeedbackRecord:
        if self.created_at.tzinfo is None:
            raise ValueError("feedback timestamp must be timezone-aware")
        return self
