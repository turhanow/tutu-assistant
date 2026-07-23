"""Dynamic AI destination hypotheses backed by the OpenAI Responses API."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timedelta

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.domain.content_models import Activity, DestinationContent, DestinationProfile
from app.domain.discovery_models import DateRange, DiscoveryRequest
from app.domain.errors import CatalogItemNotFoundError, LlmParseError, LlmProviderError
from app.ports.clock import Clock
from app.prompts import destination_discovery_v1
from app.services.map_links import build_yandex_maps_search_url

DYNAMIC_CATALOG_VERSION = "dynamic_v1"
MAX_DYNAMIC_CACHE_ITEMS = 64


class DiscoveredActivity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=200)
    categories: list[str] = Field(min_length=1, max_length=4)
    duration_minutes: int = Field(ge=30, le=720)
    exact_address: str | None = Field(default=None, min_length=3, max_length=300)


class DiscoveredDestination(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=200)
    region: str = Field(min_length=2, max_length=200)
    experience_tags: list[str] = Field(min_length=1, max_length=10)
    typical_visit_hours: int = Field(ge=4, le=96)
    activities: list[DiscoveredActivity] = Field(min_length=4, max_length=6)


class DestinationDiscoveryBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destinations: list[DiscoveredDestination] = Field(min_length=5, max_length=8)


class OpenAIDestinationDiscovery:
    """Generate AI profiles and retain their content for proposal composition."""

    version = DYNAMIC_CATALOG_VERSION
    pilot_origins = frozenset({"Москва"})

    def __init__(
        self,
        client: AsyncOpenAI,
        clock: Clock,
        *,
        model: str = "gpt-5.6-sol",
        timeout_seconds: float = 30,
    ) -> None:
        self._client = client
        self._clock = clock
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._content: OrderedDict[str, DestinationContent] = OrderedDict()
        self._request_cache: OrderedDict[
            str, tuple[datetime, tuple[DestinationProfile, ...]]
        ] = OrderedDict()

    async def find_candidates(
        self,
        request: DiscoveryRequest,
        *,
        limit: int,
    ) -> tuple[DestinationProfile, ...]:
        if limit < 1:
            return ()
        cache_key = _request_key(request)
        cached = self._request_cache.get(cache_key)
        if cached is not None:
            cached_at, profiles = cached
            cache_is_fresh = self._clock.now() <= cached_at + timedelta(hours=24)
            content_is_present = all(item.destination_id in self._content for item in profiles)
            if cache_is_fresh and content_is_present:
                self._request_cache.move_to_end(cache_key)
                return profiles[:limit]
            del self._request_cache[cache_key]

        payload = request.model_dump_json(exclude_none=True)
        instructions = (
            f"{destination_discovery_v1.INSTRUCTIONS}\n"
            f"Current datetime: {self._clock.now().isoformat()}. "
            f"Prompt version: {destination_discovery_v1.PROMPT_VERSION}."
        )
        try:
            async with asyncio.timeout(self._timeout_seconds):
                response = await self._client.responses.parse(
                    model=self._model,
                    instructions=instructions,
                    input=payload,
                    text_format=DestinationDiscoveryBatch,
                    max_output_tokens=4_000,
                    reasoning={"effort": "low"},
                    store=False,
                    timeout=self._timeout_seconds,
                )
        except TimeoutError as error:
            raise LlmProviderError(
                "dynamic destination discovery exceeded its time budget"
            ) from error
        except (APITimeoutError, APIConnectionError) as error:
            raise LlmProviderError("OpenAI destination discovery is unavailable") from error
        except OpenAIError as error:
            raise LlmProviderError("OpenAI destination discovery failed") from error

        parsed = response.output_parsed
        if not isinstance(parsed, DestinationDiscoveryBatch):
            raise LlmParseError("OpenAI returned no structured destination discovery")
        try:
            profiles = self._store_batch(parsed, origin=request.origin)
        except (ValidationError, ValueError) as error:
            raise LlmParseError("OpenAI returned invalid grounded destination content") from error
        if not profiles:
            raise LlmParseError("OpenAI returned no unique destination candidates")
        self._request_cache[cache_key] = (self._clock.now(), profiles)
        self._request_cache.move_to_end(cache_key)
        while len(self._request_cache) > MAX_DYNAMIC_CACHE_ITEMS:
            self._request_cache.popitem(last=False)
        return profiles[:limit]

    async def get_content(
        self,
        destination_id: str,
        dates: DateRange,
    ) -> DestinationContent:
        del dates
        try:
            content = self._content[destination_id]
        except KeyError as error:
            raise CatalogItemNotFoundError(
                "destination is absent from dynamic discovery cache"
            ) from error
        self._content.move_to_end(destination_id)
        return content

    def _store_batch(
        self,
        batch: DestinationDiscoveryBatch,
        *,
        origin: str,
    ) -> tuple[DestinationProfile, ...]:
        profiles: list[DestinationProfile] = []
        seen_names: set[str] = set()
        for item in batch.destinations:
            normalized_name = item.name.strip().casefold()
            if normalized_name == origin.strip().casefold() or normalized_name in seen_names:
                continue
            seen_names.add(normalized_name)
            content = self._map_destination(item)
            profiles.append(content.destination)
            self._content[content.destination.destination_id] = content
            self._content.move_to_end(content.destination.destination_id)
        while len(self._content) > MAX_DYNAMIC_CACHE_ITEMS:
            self._content.popitem(last=False)
        return tuple(profiles)

    def _map_destination(self, item: DiscoveredDestination) -> DestinationContent:
        destination_id = _content_id("city", item.name.casefold())
        activities: list[Activity] = []
        for activity in item.activities:
            activities.append(
                Activity(
                    activity_id=_content_id(
                        "activity", f"{destination_id}:{activity.name.casefold()}"
                    ),
                    destination_id=destination_id,
                    name=activity.name,
                    categories=frozenset(activity.categories),
                    duration=timedelta(minutes=activity.duration_minutes),
                    address=activity.exact_address,
                    map_url=build_yandex_maps_search_url(
                        name=activity.name,
                        address=activity.exact_address,
                        city=item.name,
                        region=item.region,
                    ),
                )
            )
        profile = DestinationProfile(
            destination_id=destination_id,
            name=item.name,
            region=item.region,
            experience_tags=frozenset(item.experience_tags),
            typical_visit_duration=timedelta(hours=item.typical_visit_hours),
            activity_highlights=tuple(activity.name for activity in activities[:3]),
        )
        return DestinationContent(
            destination=profile,
            activities=tuple(activities),
            evidence=(),
            catalog_version=DYNAMIC_CATALOG_VERSION,
            is_ai_generated=True,
        )


def _request_key(request: DiscoveryRequest) -> str:
    canonical = json.dumps(
        request.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _content_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"
