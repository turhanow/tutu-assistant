"""Production clock implementation."""

from datetime import datetime
from zoneinfo import ZoneInfo


class SystemClock:
    def __init__(self, timezone: str) -> None:
        self._timezone = ZoneInfo(timezone)

    def now(self) -> datetime:
        return datetime.now(self._timezone)
