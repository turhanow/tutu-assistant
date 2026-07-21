"""Narrow provider API consumed by application use cases."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.domain.models import (
    CheckoutLink,
    HotelOffer,
    HotelSearchQuery,
    OfferDetails,
    OfferRef,
    TransportOffer,
    TransportSearchQuery,
    TutuCapabilities,
)


class TutuGateway(Protocol):
    async def discover_capabilities(self) -> TutuCapabilities: ...

    async def search_transport(self, query: TransportSearchQuery) -> Sequence[TransportOffer]: ...

    async def search_hotels(self, query: HotelSearchQuery) -> Sequence[HotelOffer]: ...

    async def get_offer_details(self, ref: OfferRef) -> OfferDetails: ...

    async def create_checkout_link(self, ref: OfferRef) -> CheckoutLink: ...

    async def close(self) -> None: ...
