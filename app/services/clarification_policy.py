"""Bounded, deterministic question planning for discovery intake."""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.discovery_models import DiscoveryDraft
from app.prompts.clarification_v1 import QUESTIONS
from app.services.discovery_request_builder import (
    missing_discovery_fields,
    useful_optional_fields,
)


@dataclass(frozen=True, slots=True)
class ClarificationQuestion:
    field: str
    text: str
    required: bool


def plan_clarifications(
    draft: DiscoveryDraft,
    *,
    limit: int = 3,
) -> tuple[ClarificationQuestion, ...]:
    if limit < 1:
        return ()
    bounded_limit = min(limit, 3)
    critical = missing_discovery_fields(draft)
    optional = useful_optional_fields(draft)
    # Do not append nice-to-have questions while a user is resolving required fields.
    # This keeps the dialogue focused and lets safe domain defaults handle party/road limits.
    selected_fields = critical or optional
    ordered = tuple((field, field in critical) for field in selected_fields)
    return tuple(
        ClarificationQuestion(field=field, text=QUESTIONS[field], required=required)
        for field, required in ordered[:bounded_limit]
    )
