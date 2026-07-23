from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.adapters.hybrid_catalog import HybridDestinationCatalog, HybridTravelContentGateway
from app.adapters.openai_destination_discovery import (
    DestinationDiscoveryBatch,
    OpenAIDestinationDiscovery,
)
from app.domain.content_models import Activity, DestinationContent, DestinationProfile
from app.domain.discovery_models import DateRange, DiscoveryRequest, ExperienceProfile
from app.domain.errors import CatalogItemNotFoundError, LlmParseError, LlmProviderError


class FakeClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 7, 22, 18, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self.current


class FakeResponses:
    def __init__(self, parsed) -> None:
        self.parsed = parsed
        self.calls: list[dict] = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.parsed, output=[])


class FakeClient:
    def __init__(self, responses) -> None:
        self.responses = responses


def batch() -> DestinationDiscoveryBatch:
    return DestinationDiscoveryBatch.model_validate(
        {
            "destinations": [
                {
                    "name": f"Город {index}",
                    "region": f"Регион {index}",
                    "experience_tags": ["history", "walking", "relaxed"],
                    "typical_visit_hours": 30,
                    "activities": [
                        {
                            "name": f"Исторический центр {index}",
                            "categories": ["history", "walking"],
                            "duration_minutes": 120,
                            "exact_address": f"Советская улица, {index + 1}",
                        },
                        {
                            "name": f"Музей {index}",
                            "categories": ["history", "museum"],
                            "duration_minutes": 90,
                            "exact_address": None,
                        },
                        {
                            "name": f"Набережная {index}",
                            "categories": ["walking", "relaxed"],
                            "duration_minutes": 90,
                            "exact_address": f"Набережная, {index + 1}",
                        },
                        {
                            "name": f"Усадьба {index}",
                            "categories": ["history", "architecture"],
                            "duration_minutes": 90,
                            "exact_address": f"Парковая улица, {index + 1}",
                        },
                    ],
                }
                for index in range(5)
            ]
        }
    )


def request() -> DiscoveryRequest:
    return DiscoveryRequest(
        origin="Москва",
        dates=DateRange(start=date(2026, 7, 25), end=date(2026, 7, 26)),
        experience=ExperienceProfile(
            interests={"история", "прогулки"},
        ),
    )


@pytest.mark.asyncio
async def test_dynamic_catalog_uses_structured_output_and_caches_ai_content() -> None:
    responses = FakeResponses(batch())
    catalog = OpenAIDestinationDiscovery(
        FakeClient(responses),  # type: ignore[arg-type]
        FakeClock(),
    )

    profiles = await catalog.find_candidates(request(), limit=12)
    repeated = await catalog.find_candidates(request(), limit=12)
    content = await catalog.get_content(profiles[0].destination_id, request().dates)

    assert len(profiles) == 5
    assert repeated == profiles
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert "tools" not in call
    assert call["text_format"] is DestinationDiscoveryBatch
    assert call["store"] is False
    assert len(content.activities) == 4
    assert profiles[0].activity_highlights == (
        "Исторический центр 0",
        "Музей 0",
        "Набережная 0",
    )
    assert profiles[0].short_description == ("В программе — Исторический центр 0 и Музей 0.")
    assert "Набережная 0" in (profiles[0].full_description or "")
    assert profiles[0].taxi_available is True
    assert content.is_ai_generated is True
    assert content.evidence == ()
    assert content.activities[0].address == "Советская улица, 1"
    assert str(content.activities[0].map_url).startswith("https://yandex.ru/maps/?text=")
    assert "%D0%93%D0%BE%D1%80%D0%BE%D0%B4%200" in str(content.activities[0].map_url)
    assert content.activities[1].map_url is not None
    assert content.activities[0].description == ("Знакомство с историей; ориентировочно 2 ч.")


def test_dynamic_catalog_deduplicates_same_place_spelling_variants() -> None:
    parsed = batch()
    parsed.destinations[0].activities[1].name = "  исторический   центр 0 "
    catalog = OpenAIDestinationDiscovery(
        FakeClient(FakeResponses(parsed)),  # type: ignore[arg-type]
        FakeClock(),
    )

    content = catalog._map_destination(parsed.destinations[0])

    normalized = [" ".join(item.name.casefold().split()) for item in content.activities]
    assert len(normalized) == len(set(normalized))
    assert content.destination.short_description.count("Исторический центр 0") == 1


def test_yandex_query_does_not_repeat_city_already_present_in_address() -> None:
    parsed = batch()
    parsed.destinations[0].activities[0].exact_address = "Советская улица, 1, Город 0"
    catalog = OpenAIDestinationDiscovery(
        FakeClient(FakeResponses(parsed)),  # type: ignore[arg-type]
        FakeClock(),
    )

    content = catalog._map_destination(parsed.destinations[0])
    url = str(content.activities[0].map_url)

    assert url.count("%D0%93%D0%BE%D1%80%D0%BE%D0%B4%200") == 1


@pytest.mark.asyncio
async def test_dynamic_catalog_rejects_missing_structured_output() -> None:
    responses = FakeResponses(batch())
    responses.parsed = None
    catalog = OpenAIDestinationDiscovery(
        FakeClient(responses),  # type: ignore[arg-type]
        FakeClock(),
    )

    with pytest.raises(LlmParseError, match="structured"):
        await catalog.find_candidates(request(), limit=5)


@pytest.mark.parametrize(
    "url",
    (
        "https://google.com/maps/?text=Музей",
        "https://yandex.ru/maps/?text=Музей&utm_source=bot",
        "https://yandex.ru/maps/org/example/123",
    ),
)
def test_activity_rejects_noncanonical_or_tracking_map_links(url: str) -> None:
    with pytest.raises(ValueError, match="map URL"):
        Activity(
            activity_id="activity_test",
            destination_id="city_test",
            name="Музей",
            categories={"museum"},
            duration=timedelta(hours=1),
            map_url=url,
        )


class FakeCatalog:
    version = "fake"
    pilot_origins = frozenset({"Москва"})

    def __init__(self, profiles=(), error: Exception | None = None) -> None:
        self.profiles = tuple(profiles)
        self.error = error

    async def find_candidates(self, request, *, limit):
        if self.error is not None:
            raise self.error
        return self.profiles[:limit]


def profile(destination_id: str, name: str) -> DestinationProfile:
    return DestinationProfile(
        destination_id=destination_id,
        name=name,
        region="Регион",
        experience_tags={"walking"},
        evidence_ids={f"source_{destination_id}"},
    )


@pytest.mark.asyncio
async def test_hybrid_catalog_supplements_dynamic_results_and_falls_back_on_failure() -> None:
    fallback_profiles = (profile("static_a", "А"), profile("static_b", "Б"))
    hybrid = HybridDestinationCatalog(
        FakeCatalog((profile("dynamic_a", "А"),)),
        FakeCatalog(fallback_profiles),
    )

    supplemented = await hybrid.find_candidates(request(), limit=3)
    assert [item.name for item in supplemented] == ["А", "Б"]

    unavailable = HybridDestinationCatalog(
        FakeCatalog(error=LlmProviderError("offline")),
        FakeCatalog(fallback_profiles),
    )
    assert await unavailable.find_candidates(request(), limit=2) == fallback_profiles


@pytest.mark.asyncio
async def test_hybrid_catalog_keeps_dynamic_pool_primary_and_supplements_capacity() -> None:
    dynamic_profiles = tuple(
        profile(f"dynamic_{index}", f"Динамический {index}") for index in range(6)
    )
    hybrid = HybridDestinationCatalog(
        FakeCatalog(dynamic_profiles),
        FakeCatalog((profile("static_a", "Статический"),)),
    )

    result = await hybrid.find_candidates(request(), limit=12)

    assert result[:6] == dynamic_profiles
    assert result[6].name == "Статический"


class FakeContentGateway:
    def __init__(self, content: DestinationContent | None = None) -> None:
        self.content = content

    async def get_content(self, destination_id, dates):
        if self.content is None:
            raise CatalogItemNotFoundError("missing")
        return self.content


@pytest.mark.asyncio
async def test_hybrid_content_gateway_uses_static_content_when_dynamic_id_is_missing() -> None:
    dynamic_catalog = OpenAIDestinationDiscovery(
        FakeClient(FakeResponses(batch())),  # type: ignore[arg-type]
        FakeClock(),
    )
    profiles = await dynamic_catalog.find_candidates(request(), limit=5)
    static_content = await dynamic_catalog.get_content(profiles[0].destination_id, request().dates)
    gateway = HybridTravelContentGateway(
        FakeContentGateway(),
        FakeContentGateway(static_content),
    )

    result = await gateway.get_content("static-id", request().dates)

    assert result is static_content
