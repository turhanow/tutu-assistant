"""OpenAI Responses API adapters for discovery intent and grounded narration."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.domain.discovery_models import (
    DateFlexibility,
    DiscoveryDraft,
    ExperienceProfile,
    GroundedProposalFacts,
    IntentParseResult,
    ProposalCopy,
    RoadTolerance,
    TravelPace,
    TripIntent,
)
from app.domain.errors import LlmParseError, LlmProviderError
from app.domain.models import HotelMode, ParsedTripDraft, TransportMode
from app.ports.discovery_llm import ConversationContext
from app.prompts import intent_v1, narration_v2
from app.services.explicit_constraints import (
    explicitly_forbids_night_travel,
    explicitly_mentions_river,
    explicitly_relaxed,
    extract_explicit_party,
    resolved_hotel_mode,
)
from app.services.known_input_guardrails import (
    extract_explicit_trip_dates,
    normalize_optional_city,
)
from app.voice import FORBIDDEN_SLANG

MAX_INPUT_LENGTH = 2_000
MAX_NARRATION_INPUT_LENGTH = 20_000


class ExtractedIntent(StrEnum):
    DESTINATION_KNOWN = "destination_known"
    DESTINATION_UNKNOWN = "destination_unknown"
    EVENT_LED = "event_led"


class ExtractedDateFlexibility(StrEnum):
    FIXED = "fixed"
    RANGE = "range"
    WEEKEND = "weekend"


class ExtractedPace(StrEnum):
    RELAXED = "relaxed"
    BALANCED = "balanced"
    INTENSIVE = "intensive"


class ExtractedTransport(StrEnum):
    AVIA = "avia"
    RAIL = "rail"
    BUS = "bus"
    ETRAIN = "etrain"


class ExtractedHotelMode(StrEnum):
    FORBIDDEN = "forbidden"
    OPTIONAL = "optional"
    REQUIRED = "required"


class IntentExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: ExtractedIntent
    confidence: float = Field(
        ge=0,
        le=1,
        description="Intent classification confidence as a number from 0 to 1.",
    )
    origin: str | None
    destination: str | None
    departure_date: date | None
    return_date: date | None
    date_flexibility: ExtractedDateFlexibility | None
    adults: int | None = Field(default=None, ge=1, le=20)
    children: int = Field(default=0, ge=0, le=20)
    rooms: int = Field(default=1, ge=1, le=10)
    max_total_budget: str | None
    currency: str | None
    transport_preferences: list[ExtractedTransport]
    max_transfers: int | None = Field(default=None, ge=0, le=3)
    hotel_mode: ExtractedHotelMode | None
    event_date: date | None
    motives: list[str]
    interests: list[str]
    pace: ExtractedPace | None
    max_one_way_minutes: int | None = Field(default=None, ge=30, le=1_440)
    allow_night_travel: bool | None


class NarrationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    title: str
    reason: str
    trade_off: str
    evidence_ids: list[str]


class NarrationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[NarrationItem]


class NarrationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[GroundedProposalFacts]


class OpenAIIntentExtractor:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str = "gpt-5.6-sol",
        timeout_seconds: float = 12,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def extract(
        self,
        text: str,
        *,
        context: ConversationContext,
        safety_identifier: str | None = None,
    ) -> IntentParseResult:
        normalized = text.strip()
        if not normalized:
            raise LlmParseError("discovery request text is empty")
        if len(normalized) > MAX_INPUT_LENGTH:
            raise LlmParseError("discovery request text exceeds 2000 characters")
        instructions = (
            f"{intent_v1.INSTRUCTIONS}\nCurrent date: {context.current_date}. "
            f"User timezone: {context.timezone}. Prompt version: {intent_v1.PROMPT_VERSION}."
        )
        if context.expected_field in {"origin", "destination"}:
            role = "departure" if context.expected_field == "origin" else "destination"
            instructions += (
                f"\nConversation context: this turn is the user's answer to the application's "
                f"question about the {role} city. Interpret a short city name or alias as "
                f"{context.expected_field}, return its official Russian name in nominative case, "
                "and leave it null when the reference is genuinely ambiguous."
            )
        try:
            async with asyncio.timeout(self._timeout_seconds):
                for attempt in range(2):
                    response = await self._request(
                        instructions=instructions,
                        input_text=normalized,
                        output_type=IntentExtraction,
                        max_output_tokens=700,
                        safety_identifier=safety_identifier,
                    )
                    if isinstance(response, IntentExtraction):
                        try:
                            return _map_intent(
                                response,
                                source_text=normalized,
                                today=date.fromisoformat(context.current_date),
                            )
                        except (ValidationError, ValueError, InvalidOperation):
                            pass
                    if attempt == 0:
                        instructions += (
                            "\nRepair: return a schema-valid object with internally consistent "
                            "intent, confidence and draft fields."
                        )
        except TimeoutError as error:
            raise LlmProviderError("OpenAI intent extraction exceeded its time budget") from error
        raise LlmParseError("OpenAI returned no valid intent extraction")

    async def _request(
        self,
        *,
        instructions: str,
        input_text: str,
        output_type,
        max_output_tokens: int,
        safety_identifier: str | None,
    ):
        options = {
            "model": self._model,
            "instructions": instructions,
            "input": input_text,
            "text_format": output_type,
            "max_output_tokens": max_output_tokens,
            "store": False,
            "timeout": self._timeout_seconds,
        }
        if safety_identifier is not None:
            options["safety_identifier"] = safety_identifier
        try:
            response = await self._client.responses.parse(**options)
        except (APITimeoutError, APIConnectionError) as error:
            raise LlmProviderError("OpenAI is temporarily unavailable") from error
        except OpenAIError as error:
            raise LlmProviderError("OpenAI request failed") from error
        return response.output_parsed


class OpenAIProposalNarrator:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str = "gpt-5.6-sol",
        timeout_seconds: float = 12,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def narrate(
        self,
        proposals: Sequence[GroundedProposalFacts],
        *,
        context: ConversationContext,
    ) -> tuple[ProposalCopy, ...]:
        if not 1 <= len(proposals) <= 3:
            raise LlmParseError("narration requires one to three proposals")
        payload = NarrationPayload(proposals=list(proposals)).model_dump_json()
        if len(payload) > MAX_NARRATION_INPUT_LENGTH:
            raise LlmParseError("grounded narration payload is too large")
        instructions = (
            f"{narration_v2.INSTRUCTIONS}\nUser timezone: {context.timezone}. "
            f"Prompt version: {narration_v2.PROMPT_VERSION}."
        )
        try:
            async with asyncio.timeout(self._timeout_seconds):
                for attempt in range(2):
                    parsed = await self._request(instructions, payload)
                    if isinstance(parsed, NarrationBatch):
                        result = _validate_narration(proposals, parsed)
                        if result is not None:
                            return result
                    if attempt == 0:
                        instructions += (
                            "\nRepair: preserve every proposal_id and use only its supplied "
                            "evidence_ids. Remove slang, emoji and exclamation marks. Keep "
                            "trade-offs neutral. Return one item per proposal."
                        )
        except TimeoutError as error:
            raise LlmProviderError("OpenAI narration exceeded its time budget") from error
        raise LlmParseError("OpenAI returned ungrounded proposal narration")

    async def _request(self, instructions: str, payload: str):
        try:
            response = await self._client.responses.parse(
                model=self._model,
                instructions=instructions,
                input=payload,
                text_format=NarrationBatch,
                max_output_tokens=1_000,
                store=False,
                timeout=self._timeout_seconds,
            )
        except (APITimeoutError, APIConnectionError) as error:
            raise LlmProviderError("OpenAI is temporarily unavailable") from error
        except OpenAIError as error:
            raise LlmProviderError("OpenAI request failed") from error
        return response.output_parsed


def _map_intent(
    value: IntentExtraction,
    *,
    source_text: str = "",
    today: date | None = None,
) -> IntentParseResult:
    intent = TripIntent(value.intent.value)
    confidence = Decimal(str(value.confidence))
    budget = _optional_money(value.max_total_budget)
    currency = (value.currency or "RUB").upper()
    allowed_modes = frozenset(TransportMode(item.value) for item in value.transport_preferences)
    party = extract_explicit_party(source_text)
    explicit_dates = (
        extract_explicit_trip_dates(source_text, today=today) if today is not None else (None, None)
    )
    departure_date = explicit_dates[0] or value.departure_date
    return_date = explicit_dates[1] or value.return_date
    hotel_mode = resolved_hotel_mode(
        source_text,
        HotelMode(value.hotel_mode.value) if value.hotel_mode else None,
        departure_date=departure_date,
        return_date=return_date,
    )
    normalized_experience = {
        item.strip().casefold().replace("ё", "е") for item in (*value.motives, *value.interests)
    }
    relaxed_is_explicit = bool(
        normalized_experience & {"без суеты", "спокойно", "спокойный отдых", "не спеша"}
    )
    relaxed_is_explicit = relaxed_is_explicit or explicitly_relaxed(source_text)
    pace = (
        TravelPace(value.pace.value)
        if value.pace
        else TravelPace.RELAXED
        if relaxed_is_explicit
        else None
    )
    known_draft = None
    discovery_draft = None
    if intent is TripIntent.DESTINATION_KNOWN or (
        intent is TripIntent.EVENT_LED and value.destination is not None
    ):
        known_draft = ParsedTripDraft(
            origin=normalize_optional_city(value.origin),
            destination=normalize_optional_city(value.destination),
            departure_date=departure_date,
            return_date=return_date,
            adults=party.adults if party.adults is not None else value.adults,
            children=party.children if party.children is not None else value.children,
            rooms=party.rooms if party.rooms is not None else value.rooms,
            budget=budget,
            currency=currency,
            allowed_modes=allowed_modes,
            max_transfers=value.max_transfers,
            hotel_mode=hotel_mode,
            event_date=value.event_date,
        )
        required = ("origin", "destination", "departure_date", "return_date", "hotel_mode")
        missing = tuple(field for field in required if getattr(known_draft, field) is None)
    else:
        discovery_draft = DiscoveryDraft(
            origin=normalize_optional_city(value.origin),
            departure_date=departure_date,
            return_date=return_date,
            date_flexibility=(
                DateFlexibility(value.date_flexibility.value) if value.date_flexibility else None
            ),
            adults=party.adults if party.adults is not None else value.adults,
            children=party.children if party.children is not None else value.children,
            rooms=party.rooms if party.rooms is not None else value.rooms,
            budget=budget,
            currency=currency,
            hotel_mode=hotel_mode,
            allowed_modes=allowed_modes,
            experience=ExperienceProfile(
                motives=frozenset(value.motives),
                interests=frozenset(value.interests)
                | ({"river"} if explicitly_mentions_river(source_text) else set()),
                pace=pace,
                road_tolerance=RoadTolerance(
                    max_one_way_duration=(
                        None
                        if value.max_one_way_minutes is None
                        else timedelta(minutes=value.max_one_way_minutes)
                    ),
                    max_transfers=value.max_transfers,
                    allow_night_travel=(
                        False
                        if explicitly_forbids_night_travel(source_text)
                        else value.allow_night_travel
                    ),
                ),
            ),
        )
        missing_values: list[str] = []
        if discovery_draft.origin is None:
            missing_values.append("origin")
        if discovery_draft.departure_date is None or discovery_draft.return_date is None:
            missing_values.append("dates")
        if not discovery_draft.experience.motives and not discovery_draft.experience.interests:
            missing_values.append("motives")
        if discovery_draft.hotel_mode is None:
            missing_values.append("hotel_mode")
        if discovery_draft.budget is None:
            missing_values.append("budget")
        missing = tuple(missing_values)
    return IntentParseResult(
        intent=intent,
        confidence=confidence,
        known_draft=known_draft,
        discovery_draft=discovery_draft,
        missing_fields=missing,
    )


def _optional_money(value: str | None) -> Decimal | None:
    if value is None:
        return None
    amount = Decimal(value.replace(" ", "").replace(",", "."))
    if amount < 0:
        raise ValueError("budget cannot be negative")
    return amount


def _validate_narration(
    proposals: Sequence[GroundedProposalFacts],
    batch: NarrationBatch,
) -> tuple[ProposalCopy, ...] | None:
    expected = {item.proposal_id: item for item in proposals}
    if len(batch.items) != len(expected):
        return None
    if {item.proposal_id for item in batch.items} != set(expected):
        return None
    result: list[ProposalCopy] = []
    try:
        for item in batch.items:
            if not _narration_style_is_safe(item):
                return None
            allowed_evidence = expected[item.proposal_id].evidence_ids
            evidence_ids = frozenset(item.evidence_ids)
            if not evidence_ids or not evidence_ids.issubset(allowed_evidence):
                return None
            result.append(
                ProposalCopy(
                    proposal_id=item.proposal_id,
                    title=item.title,
                    reason=item.reason,
                    trade_off=item.trade_off,
                    evidence_ids=evidence_ids,
                )
            )
    except ValidationError:
        return None
    return tuple(result)


def _narration_style_is_safe(item: NarrationItem) -> bool:
    combined = " ".join((item.title, item.reason, item.trade_off)).casefold().replace("ё", "е")
    forbidden = {word.replace("ё", "е") for word in FORBIDDEN_SLANG}
    forbidden.update({"идеальный", "гарантированно понравится", "точно понравится"})
    if any(word in combined for word in forbidden):
        return False
    if "!" in combined:
        return False
    return not any(symbol in combined for symbol in ("🔥", "✨", "😎", "😂", "🚀"))
