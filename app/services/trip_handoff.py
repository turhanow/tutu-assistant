"""Safe offer-details lookup and explicit handoff to Tutu."""

from __future__ import annotations

import asyncio

from app.domain.errors import CheckoutError, ProviderError
from app.domain.models import (
    CheckoutLink,
    HotelOffer,
    OfferDetails,
    TransportMode,
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
        return await self._create_checkout_items(result, index, require_all=True)

    async def create_available_checkout_items(
        self,
        result: TripSearchResult,
        index: int,
    ) -> tuple[TripCheckoutItem, ...]:
        """Return every exact link available for inline rendering without all-or-none loss."""

        return await self._create_checkout_items(result, index, require_all=False)

    async def _create_checkout_items(
        self,
        result: TripSearchResult,
        index: int,
        *,
        require_all: bool,
    ) -> tuple[TripCheckoutItem, ...]:
        components = self.available_components(result, index)
        async with asyncio.timeout(15):
            resolved = await asyncio.gather(
                *(self._checkout_item(result, index, component) for component in components)
            )
        items = [item for item in resolved if item is not None]
        resolved_components = {item.component for item in items}
        required_components = set(components)
        if require_all and not required_components.issubset(resolved_components):
            raise CheckoutError("exact checkout links are unavailable for this trip option")
        return tuple(items)

    async def _checkout_item(
        self,
        result: TripSearchResult,
        index: int,
        component: TripComponent,
    ) -> TripCheckoutItem | None:
        offer = self._offer(result, index, component)
        # Tutu currently exposes only a date-level schedule URL for commuter trains.
        # It must never be presented as checkout for the exact departure shown in a card.
        if isinstance(offer, TransportOffer) and offer.mode is TransportMode.ETRAIN:
            return None
        if offer.provider_url is not None:
            # Search responses already carry the most specific URL for this exact offer.
            # Prefer it over rebuilding a URL that may degrade to a day-level listing.
            kind = "hotel_page" if isinstance(offer, HotelOffer) else "direct_offer"
            link = CheckoutLink(url=offer.provider_url, kind=kind)
        else:
            try:
                link = await self._gateway.create_checkout_link(offer.offer_ref)
            except ProviderError:
                return None
        self._require_tutu_host(link)
        item = TripCheckoutItem(
            component=component,
            link=link,
            option_signature=self._combination(result, index).signature,
        )
        return item if self.is_offer_specific(item) else None

    @staticmethod
    def _combination(result: TripSearchResult, index: int):
        if index < 0 or index >= len(result.options):
            raise CheckoutError("trip option is missing or stale")
        return result.options[index].combination

    @staticmethod
    def is_offer_specific(item: TripCheckoutItem) -> bool:
        return item.link.kind in {
            "direct_offer",
            "hotel_page",
            "deeplink",
            "checkout_deeplink",
        }

    @staticmethod
    def button_prefix(item: TripCheckoutItem) -> str:
        if item.link.kind == "schedule_url":
            return "Открыть расписание"
        if item.component is TripComponent.HOTEL and TripHandoffService.is_offer_specific(item):
            return "Открыть выбранный отель"
        if TripHandoffService.is_offer_specific(item):
            return "Открыть выбранный билет"
        return "Смотреть варианты"

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
