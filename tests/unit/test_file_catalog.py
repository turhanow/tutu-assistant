from datetime import date
from pathlib import Path

import pytest

from app.adapters.file_catalog import (
    FileCatalogRepository,
    FileDestinationCatalog,
    FileTravelContentGateway,
)
from app.domain.discovery_models import DateRange, DiscoveryRequest
from app.domain.errors import CatalogContractError, CatalogItemNotFoundError

CATALOG_PATH = Path("data/destinations/v1/catalog.json")


def request(**overrides) -> DiscoveryRequest:
    values = {
        "origin": "Москва",
        "dates": DateRange(start=date(2026, 8, 22), end=date(2026, 8, 23)),
    }
    values.update(overrides)
    return DiscoveryRequest(**values)


def test_repository_loads_one_immutable_snapshot() -> None:
    repository = FileCatalogRepository.from_path(CATALOG_PATH)

    assert repository.version == "v1"
    assert repository.destination_count == 10
    assert repository.activity_count == 20
    assert repository.pilot_origins == frozenset({"Москва"})


def test_repository_hides_paths_and_validation_details_in_public_error(tmp_path: Path) -> None:
    invalid = tmp_path / "private-invalid-catalog.json"
    invalid.write_text('{"version":"wrong"}', encoding="utf-8")

    with pytest.raises(CatalogContractError) as error:
        FileCatalogRepository.from_path(invalid)

    assert "private-invalid-catalog.json" in str(error.value)
    assert str(tmp_path) not in str(error.value)


@pytest.mark.asyncio
async def test_catalog_returns_bounded_deterministic_candidates() -> None:
    catalog = FileDestinationCatalog(FileCatalogRepository.from_path(CATALOG_PATH))

    candidates = await catalog.find_candidates(request(), limit=5)

    assert len(candidates) == 5
    assert [item.destination_id for item in candidates] == sorted(
        item.destination_id for item in candidates
    )
    assert all(item.name != "Москва" for item in candidates)


@pytest.mark.asyncio
async def test_zero_limit_does_not_leak_catalog_content() -> None:
    catalog = FileDestinationCatalog(FileCatalogRepository.from_path(CATALOG_PATH))

    assert await catalog.find_candidates(request(), limit=0) == ()


@pytest.mark.asyncio
async def test_content_gateway_returns_grounded_content_by_canonical_id() -> None:
    gateway = FileTravelContentGateway(FileCatalogRepository.from_path(CATALOG_PATH))

    content = await gateway.get_content(
        "kolomna",
        DateRange(start=date(2026, 8, 22), end=date(2026, 8, 23)),
    )

    assert content.destination.name == "Коломна"
    assert len(content.activities) == 2
    assert content.catalog_version == "v1"


@pytest.mark.asyncio
async def test_unknown_destination_is_a_typed_catalog_failure() -> None:
    gateway = FileTravelContentGateway(FileCatalogRepository.from_path(CATALOG_PATH))

    with pytest.raises(CatalogItemNotFoundError, match="absent from catalog"):
        await gateway.get_content(
            "invented_city",
            DateRange(start=date(2026, 8, 22), end=date(2026, 8, 23)),
        )
