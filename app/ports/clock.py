"""Clock port keeps date-sensitive rules deterministic in tests."""

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...
