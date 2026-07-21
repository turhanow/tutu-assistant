"""Travel content boundary."""

from typing import Protocol

from app.domain.content_models import DestinationContent
from app.domain.discovery_models import DateRange


class TravelContentGateway(Protocol):
    async def get_content(
        self,
        destination_id: str,
        dates: DateRange,
    ) -> DestinationContent: ...
