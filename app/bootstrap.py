"""Small composition helpers; this is the only layer allowed to unwrap secrets."""

from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI

from app.adapters.openai_discovery import OpenAIIntentExtractor, OpenAIProposalNarrator
from app.adapters.openai_parser import OpenAIRequestParser
from app.config import Settings


@dataclass(repr=False)
class LlmResources:
    client: AsyncOpenAI
    parser: OpenAIRequestParser
    intent_extractor: OpenAIIntentExtractor
    proposal_narrator: OpenAIProposalNarrator

    async def close(self) -> None:
        await self.client.close()


def build_llm_resources(settings: Settings) -> LlmResources:
    """Create the official SDK client without retaining a plain-text key in config."""
    api_key = settings.require_openai_api_key().get_secret_value()
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=str(settings.openai_base_url),
        max_retries=0,
        timeout=settings.openai_timeout_seconds,
    )
    return LlmResources(
        client=client,
        parser=OpenAIRequestParser(
            client,
            model=settings.openai_model,
            timeout_seconds=settings.openai_timeout_seconds,
        ),
        intent_extractor=OpenAIIntentExtractor(
            client,
            model=settings.openai_model,
            timeout_seconds=settings.openai_timeout_seconds,
        ),
        proposal_narrator=OpenAIProposalNarrator(
            client,
            model=settings.openai_model,
            timeout_seconds=settings.openai_timeout_seconds,
        ),
    )
