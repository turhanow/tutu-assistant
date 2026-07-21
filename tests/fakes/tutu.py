from collections.abc import Sequence

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


class FakeTutuGateway:
    """Fixture-driven gateway with observable calls for use-case tests."""

    def __init__(
        self,
        *,
        capabilities: TutuCapabilities,
        transport: Sequence[TransportOffer] = (),
        hotels: Sequence[HotelOffer] = (),
        checkout: CheckoutLink | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.transport = list(transport)
        self.hotels = list(hotels)
        self.checkout = checkout
        self.calls: list[tuple[str, object]] = []

    async def discover_capabilities(self) -> TutuCapabilities:
        self.calls.append(("discover_capabilities", None))
        return self.capabilities

    async def search_transport(self, query: TransportSearchQuery) -> list[TransportOffer]:
        self.calls.append(("search_transport", query))
        return list(self.transport)

    async def search_hotels(self, query: HotelSearchQuery) -> list[HotelOffer]:
        self.calls.append(("search_hotels", query))
        return list(self.hotels)

    async def get_offer_details(self, ref: OfferRef) -> OfferDetails:
        self.calls.append(("get_offer_details", ref))
        return OfferDetails(product_type=ref.product_type, title="Fixture offer")

    async def create_checkout_link(self, ref: OfferRef) -> CheckoutLink:
        self.calls.append(("create_checkout_link", ref))
        if self.checkout is None:
            raise RuntimeError("fake checkout response is not configured")
        return self.checkout

    async def close(self) -> None:
        self.calls.append(("close", None))
