"""Dynamic-first destination adapters with an immutable catalog fallback."""

from __future__ import annotations

import logging

from app.domain.content_models import DestinationContent, DestinationProfile
from app.domain.discovery_models import DateRange, DiscoveryRequest
from app.domain.errors import CatalogItemNotFoundError, LlmParseError, LlmProviderError
from app.ports.catalog import DestinationCatalog
from app.ports.content import TravelContentGateway

logger = logging.getLogger(__name__)


class HybridDestinationCatalog:
    version = "hybrid_v1"

    def __init__(
        self,
        dynamic: DestinationCatalog,
        fallback: DestinationCatalog,
    ) -> None:
        self._dynamic = dynamic
        self._fallback = fallback
        self.pilot_origins = dynamic.pilot_origins & fallback.pilot_origins

    async def find_candidates(
        self,
        request: DiscoveryRequest,
        *,
        limit: int,
    ) -> tuple[DestinationProfile, ...]:
        if limit < 1:
            return ()
        try:
            dynamic = tuple(await self._dynamic.find_candidates(request, limit=limit))
        except (LlmParseError, LlmProviderError, TimeoutError, ValueError):
            logger.warning("dynamic_destination_discovery_fallback", exc_info=True)
            dynamic = ()
        fallback = await self._fallback.find_candidates(request, limit=limit)
        seen_names = {item.name.casefold() for item in dynamic}
        supplement = tuple(item for item in fallback if item.name.casefold() not in seen_names)
        return (dynamic + supplement)[:limit]


class HybridTravelContentGateway:
    def __init__(
        self,
        dynamic: TravelContentGateway,
        fallback: TravelContentGateway,
    ) -> None:
        self._dynamic = dynamic
        self._fallback = fallback

    async def get_content(
        self,
        destination_id: str,
        dates: DateRange,
    ) -> DestinationContent:
        try:
            return await self._dynamic.get_content(destination_id, dates)
        except CatalogItemNotFoundError:
            return await self._fallback.get_content(destination_id, dates)
