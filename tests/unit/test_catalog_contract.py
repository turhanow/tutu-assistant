from datetime import UTC, datetime
from pathlib import Path

from app.adapters.file_catalog import FileCatalogRepository

CATALOG_PATH = Path("data/destinations/v1/catalog.json")


def test_pilot_catalog_has_grounded_complete_minimum() -> None:
    repository = FileCatalogRepository.from_path(CATALOG_PATH)
    contents = repository.all_content()

    assert len(contents) == 10
    assert all(len(content.activities) >= 2 for content in contents)
    assert all(content.evidence for content in contents)
    assert all(content.destination.evidence_ids for content in contents)
    assert all(activity.evidence_ids for content in contents for activity in content.activities)


def test_pilot_catalog_uses_https_sources_and_no_expired_dynamic_facts() -> None:
    repository = FileCatalogRepository.from_path(CATALOG_PATH)
    checked_at = datetime(2026, 7, 21, 23, tzinfo=UTC)

    evidence = [item for content in repository.all_content() for item in content.evidence]
    assert all(item.source_url.scheme == "https" for item in evidence)
    assert all(item.is_fresh_at(checked_at) for item in evidence)
    assert not any(item.is_dynamic for item in evidence)


def test_pilot_catalog_contains_no_unverified_price_or_image_claims() -> None:
    repository = FileCatalogRepository.from_path(CATALOG_PATH)

    assert all(
        content.destination.estimated_daily_cost is None for content in repository.all_content()
    )
    assert all(
        activity.estimated_cost is None
        for content in repository.all_content()
        for activity in content.activities
    )
