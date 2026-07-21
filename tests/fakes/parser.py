from datetime import datetime

from app.domain.models import ParseResult


class FakeRequestParser:
    def __init__(self, result: ParseResult) -> None:
        self.result = result
        self.calls: list[tuple[str, datetime, str]] = []

    async def parse(
        self,
        text: str,
        *,
        now: datetime,
        timezone: str,
        safety_identifier: str | None = None,
    ) -> ParseResult:
        self.calls.append((text, now, timezone))
        return self.result
