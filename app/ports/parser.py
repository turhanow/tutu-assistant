"""Natural-language request parser boundary."""

from datetime import datetime
from typing import Protocol

from app.domain.models import ParseResult


class RequestParser(Protocol):
    async def parse(
        self,
        text: str,
        *,
        now: datetime,
        timezone: str,
        safety_identifier: str | None = None,
    ) -> ParseResult: ...
