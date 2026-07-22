"""Safe offer-details lookup and explicit handoff to Tutu."""

from __future__ import annotations

import asyncio

from app.domain.errors import CheckoutError, ProviderError
from app.domain.models import (
    CheckoutLink,
    HotelOffer,
    OfferDetails,
    TransportOffer,
    TripCheckoutItem,
    TripComponent,
    TripSearchResult,
)
from app.ports.tutu import TutuGateway
from app.services.url_policy import require_allowed_https_url


class TripHandoffService:
    def __init__(self, gateway: TutuGateway) -> None:
        self._gateway = gateway

    def available_components(
        self,
        result: TripSearchResult,
        index: int,
    ) -> tuple[TripComponent, ...]:
        combination = self._combination(result, index)
        components = [TripComponent.OUTBOUND, TripComponent.RETURN]
        if combination.hotel is not None:
            components.append(TripComponent.HOTEL)
        return tuple(components)

    async def get_details(
        self,
        result: TripSearchResult,
        index: int,
        component: TripComponent,
    ) -> OfferDetails:
        offer = self._offer(result, index, component)
        return await self._gateway.get_offer_details(offer.offer_ref)

    async def create_checkout_items(
        self,
        result: TripSearchResult,
        index: int,
    ) -> tuple[TripCheckoutItem, ...]:
        components = self.available_components(result, index)
        async with asyncio.timeout(15):
            resolved = await asyncio.gather(
                *(self._checkout_item(result, index, component) for component in components)
            )
        items = [item for item in resolved if item is not None]
        if not items:
            raise CheckoutError("no safe Tutu handoff links are available")
        return tuple(items)

    async def _checkout_item(
        self,
        result: TripSearchResult,
        index: int,
        component: TripComponent,
    ) -> TripCheckoutItem | None:
        offer = self._offer(result, index, component)
        if offer.provider_url is not None:
            # Search responses already carry the most specific URL for this exact offer.
            # Prefer it over rebuilding a URL that may degrade to a day-level listing.
            link = CheckoutLink(url=offer.provider_url, kind="direct_offer")
        else:
            try:
                link = await self._gateway.create_checkout_link(offer.offer_ref)
            except ProviderError:
                return None
        self._require_tutu_host(link)
        return TripCheckoutItem(component=component, link=link)

    @staticmethod
    def _combination(result: TripSearchResult, index: int):
        if index < 0 or index >= len(result.options):
            raise CheckoutError("trip option is missing or stale")
        return result.options[index].combination

    @staticmethod
    def is_offer_specific(item: TripCheckoutItem) -> bool:
        return item.link.kind in {"direct_offer", "deeplink", "checkout_deeplink"}

    def _offer(
        self,
        result: TripSearchResult,
        index: int,
        component: TripComponent,
    ) -> TransportOffer | HotelOffer:
        combination = self._combination(result, index)
        if component is TripComponent.OUTBOUND:
            return combination.outbound
        if component is TripComponent.RETURN:
            return combination.return_offer
        if component is TripComponent.HOTEL and combination.hotel is not None:
            return combination.hotel
        raise CheckoutError("trip component is missing or stale")

    @staticmethod
    def _require_tutu_host(link: CheckoutLink) -> None:
        for url in (link.url, link.fallback_url):
            if url is None:
                continue
            try:
                require_allowed_https_url(
                    str(url),
                    allowed_hosts=frozenset({"tutu.ru"}),
                    allow_subdomains=True,
                )
            except ValueError as error:
                raise CheckoutError("provider returned a non-Tutu handoff URL") from error
