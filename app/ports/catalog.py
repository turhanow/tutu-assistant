"""Destination catalog boundary."""

from collections.abc import Sequence
from typing import Protocol

from app.domain.content_models import DestinationProfile
from app.domain.discovery_models import DiscoveryRequest


class DestinationCatalog(Protocol):
    @property
    def version(self) -> str: ...

    @property
    def pilot_origins(self) -> frozenset[str]: ...

    async def find_candidates(
        self,
        request: DiscoveryRequest,
        *,
        limit: int,
    ) -> Sequence[DestinationProfile]: ...
