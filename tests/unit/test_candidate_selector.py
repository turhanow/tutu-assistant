from datetime import date, timedelta
from pathlib import Path

import pytest

from app.adapters.file_catalog import FileCatalogRepository, FileDestinationCatalog
from app.domain.content_models import DestinationProfile, PriceRange
from app.domain.discovery_models import DateRange, DiscoveryRequest, ExperienceProfile
from app.domain.errors import UnsupportedOriginError
from app.services.candidate_selector import SCORE_VERSION, CandidateSelector


class FakeCatalog:
    version = "v-test"
    pilot_origins = frozenset({"Москва"})

    def __init__(self, profiles: list[DestinationProfile]) -> None:
        self.profiles = profiles
        self.calls = []

    async def find_candidates(self, request, *, limit):
        self.calls.append((request, limit))
        return tuple(self.profiles[:limit])


def profile(
    destination_id: str,
    *,
    region: str,
    tags: set[str],
    daily_cost: PriceRange | None = None,
) -> DestinationProfile:
    return DestinationProfile(
        destination_id=destination_id,
        name=destination_id.title(),
        region=region,
        experience_tags=tags,
        typical_visit_duration=timedelta(days=2),
        estimated_daily_cost=daily_cost,
        evidence_ids={f"source_{destination_id}"},
    )


def request(**overrides) -> DiscoveryRequest:
    values = {
        "origin": "Москва",
        "dates": DateRange(start=date(2026, 8, 22), end=date(2026, 8, 23)),
        "experience": ExperienceProfile(interests={"архитектура"}),
    }
    values.update(overrides)
    return DiscoveryRequest(**values)


@pytest.mark.asyncio
async def test_selector_prefers_explicit_interest_match() -> None:
    catalog = FakeCatalog(
        [
            profile("nature_city", region="A", tags={"nature"}),
            profile("old_city", region="B", tags={"architecture", "history"}),
        ]
    )
    selector = CandidateSelector(catalog)

    shortlist = await selector.select(request())

    assert shortlist.candidates[0].destination.destination_id == "old_city"
    assert "архитектура" in shortlist.candidates[0].match_reasons[0]
    assert shortlist.catalog_version == "v-test"
    assert shortlist.score_version == SCORE_VERSION
    assert catalog.calls[0][1] == 12


@pytest.mark.asyncio
async def test_selector_normalizes_russian_and_canonical_tags_equally() -> None:
    profiles = [profile("old_city", region="A", tags={"architecture"})]
    selector = CandidateSelector(FakeCatalog(profiles))

    russian = await selector.select(request())
    canonical = await selector.select(
        request(experience=ExperienceProfile(interests={"architecture"}))
    )

    assert russian.candidates[0].match_score == canonical.candidates[0].match_score


@pytest.mark.asyncio
async def test_selector_does_not_use_estimates_as_hard_budget_filter() -> None:
    selector = CandidateSelector(
        FakeCatalog(
            [
                profile(
                    "expensive",
                    region="A",
                    tags={"architecture"},
                    daily_cost=PriceRange(minimum="6000", maximum="8000", currency="RUB"),
                ),
                profile("unknown_cost", region="B", tags={"architecture"}),
            ]
        )
    )

    shortlist = await selector.select(request(budget="10000"))

    assert {item.destination.destination_id for item in shortlist.candidates} == {
        "expensive",
        "unknown_cost",
    }


@pytest.mark.asyncio
async def test_diversity_pass_prefers_different_regions_before_filling_duplicates() -> None:
    selector = CandidateSelector(
        FakeCatalog(
            [
                profile("alpha", region="Same", tags={"architecture"}),
                profile("beta", region="Same", tags={"architecture"}),
                profile("gamma", region="Other", tags={"architecture"}),
            ]
        )
    )

    shortlist = await selector.select(request(), limit=2)

    assert [item.destination.destination_id for item in shortlist.candidates] == [
        "alpha",
        "gamma",
    ]


@pytest.mark.asyncio
async def test_selector_is_bounded_unique_and_deterministic() -> None:
    profiles = [
        profile(f"city_{index}", region=f"Region {index}", tags={"architecture"})
        for index in range(8)
    ]
    selector = CandidateSelector(FakeCatalog(list(reversed(profiles))))

    first = await selector.select(request(), limit=99)
    second = await selector.select(request(), limit=99)

    assert len(first.candidates) == 5
    assert len({item.destination.destination_id for item in first.candidates}) == 5
    assert first == second


@pytest.mark.asyncio
async def test_zero_limit_skips_catalog_call() -> None:
    catalog = FakeCatalog([profile("alpha", region="A", tags={"architecture"})])
    selector = CandidateSelector(catalog)

    shortlist = await selector.select(request(), limit=0)

    assert shortlist.candidates == ()
    assert catalog.calls == []


@pytest.mark.asyncio
async def test_real_pilot_catalog_produces_grounded_bounded_shortlist() -> None:
    catalog = FileDestinationCatalog(
        FileCatalogRepository.from_path(Path("data/destinations/v1/catalog.json"))
    )
    selector = CandidateSelector(catalog)

    shortlist = await selector.select(request(), limit=5)

    assert len(shortlist.candidates) == 5
    assert shortlist.catalog_version == "v1"
    assert all(item.destination.evidence_ids for item in shortlist.candidates)
    assert all(item.match_reasons for item in shortlist.candidates)


@pytest.mark.asyncio
async def test_space_and_science_request_prefers_kaluga_in_real_catalog() -> None:
    catalog = FileDestinationCatalog(
        FileCatalogRepository.from_path(Path("data/destinations/v1/catalog.json"))
    )
    selector = CandidateSelector(catalog)

    shortlist = await selector.select(
        request(
            experience=ExperienceProfile(
                interests={"космос", "наука", "музеи"},
            )
        )
    )

    assert shortlist.candidates[0].destination.name == "Калуга"
    assert shortlist.candidates[0].match_score == 1
    assert "наука" in " ".join(shortlist.candidates[0].match_reasons)


@pytest.mark.asyncio
async def test_river_interest_survives_russian_inflection_and_is_explained() -> None:
    catalog = FileDestinationCatalog(
        FileCatalogRepository.from_path(Path("data/destinations/v1/catalog.json"))
    )
    selector = CandidateSelector(catalog)

    shortlist = await selector.select(
        request(experience=ExperienceProfile(interests={"реки", "прогулки"}))
    )

    reasons = " ".join(
        reason for candidate in shortlist.candidates for reason in candidate.match_reasons
    )
    assert "набережные и река" in reasons
    assert "прогулки" in reasons


@pytest.mark.asyncio
async def test_unsupported_origin_stops_before_catalog_or_provider_work() -> None:
    catalog = FakeCatalog([profile("alpha", region="A", tags={"walking"})])
    selector = CandidateSelector(catalog)

    with pytest.raises(UnsupportedOriginError, match="только из: Москва"):
        await selector.select(request(origin="Владивосток"))

    assert catalog.calls == []
