"""LLM boundaries used by the discovery application layer."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.domain.discovery_models import GroundedProposalFacts, IntentParseResult, ProposalCopy


@dataclass(frozen=True, slots=True)
class ConversationContext:
    timezone: str
    current_date: str


class IntentExtractor(Protocol):
    async def extract(
        self,
        text: str,
        *,
        context: ConversationContext,
        safety_identifier: str | None = None,
    ) -> IntentParseResult: ...


class ProposalNarrator(Protocol):
    async def narrate(
        self,
        proposals: Sequence[GroundedProposalFacts],
        *,
        context: ConversationContext,
    ) -> Sequence[ProposalCopy]: ...
