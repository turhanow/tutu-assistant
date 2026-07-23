"""Validated file-backed destination catalog and content adapters."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from app.domain.content_models import (
    DestinationCatalogBundle,
    DestinationContent,
    DestinationProfile,
)
from app.domain.discovery_models import DateRange, DiscoveryRequest
from app.domain.errors import CatalogContractError, CatalogItemNotFoundError
from app.services.url_policy import require_allowed_https_url

CONTENT_SOURCE_HOSTS = frozenset(
    {
        "kolomnavisit.ru",
        "ryazantourism.ru",
        "visit-kaluga.ru",
        "visit-sposad.ru",
        "visitnizhniy.com",
        "visittula.com",
        "visittver.ru",
        "vladimirtravel.ru",
        "visityar.ru",
    }
)


class FileCatalogRepository:
    """Load one immutable catalog snapshot and serve it without runtime file reads."""

    def __init__(self, bundle: DestinationCatalogBundle) -> None:
        self._bundle = bundle
        self._by_id = {item.destination.destination_id: item for item in self._bundle.destinations}

    @classmethod
    def from_path(cls, path: str | Path) -> FileCatalogRepository:
        catalog_path = Path(path)
        try:
            raw = catalog_path.read_text(encoding="utf-8")
        except OSError as error:
            raise CatalogContractError(
                f"cannot read destination catalog: {catalog_path.name}"
            ) from error
        try:
            bundle = DestinationCatalogBundle.model_validate_json(raw)
        except ValidationError as error:
            raise CatalogContractError(
                f"invalid destination catalog: {catalog_path.name}"
            ) from error
        try:
            for content in bundle.destinations:
                for evidence in content.evidence:
                    require_allowed_https_url(
                        str(evidence.source_url),
                        allowed_hosts=CONTENT_SOURCE_HOSTS,
                        allow_subdomains=True,
                    )
        except ValueError as error:
            raise CatalogContractError("catalog contains a non-allowlisted source URL") from error
        return cls(bundle)

    @property
    def version(self) -> str:
        return self._bundle.version

    @property
    def destination_count(self) -> int:
        return len(self._bundle.destinations)

    @property
    def activity_count(self) -> int:
        return sum(len(item.activities) for item in self._bundle.destinations)

    @property
    def pilot_origins(self) -> frozenset[str]:
        return self._bundle.pilot_origins

    def all_content(self) -> tuple[DestinationContent, ...]:
        return self._bundle.destinations

    def get_content(self, destination_id: str) -> DestinationContent:
        try:
            return self._by_id[destination_id]
        except KeyError as error:
            raise CatalogItemNotFoundError(
                f"destination is absent from catalog {self.version}"
            ) from error


class FileDestinationCatalog:
    def __init__(self, repository: FileCatalogRepository) -> None:
        self._repository = repository

    @property
    def version(self) -> str:
        return self._repository.version

    @property
    def pilot_origins(self) -> frozenset[str]:
        return self._repository.pilot_origins

    async def find_candidates(
        self,
        request: DiscoveryRequest,
        *,
        limit: int,
    ) -> tuple[DestinationProfile, ...]:
        if limit < 1:
            return ()
        requested_months = {
            request.dates.start.month,
            request.dates.end.month,
        }
        origin = request.origin.casefold()
        candidates = (
            item.destination.model_copy(
                update={
                    "activity_highlights": tuple(
                        activity.name for activity in item.activities[:3]
                    )
                }
            )
            for item in self._repository.all_content()
            if item.destination.name.casefold() != origin
            and requested_months.intersection(item.destination.season_months)
        )
        return tuple(sorted(candidates, key=lambda item: item.destination_id)[:limit])


class FileTravelContentGateway:
    def __init__(self, repository: FileCatalogRepository) -> None:
        self._repository = repository

    async def get_content(
        self,
        destination_id: str,
        dates: DateRange,
    ) -> DestinationContent:
        content = self._repository.get_content(destination_id)
        requested_months = {dates.start.month, dates.end.month}
        if not requested_months.intersection(content.destination.season_months):
            raise CatalogItemNotFoundError(
                f"destination has no content for requested season in {self._repository.version}"
            )
        return content
