from datetime import datetime

from app.adapters.clock import SystemClock


def test_system_clock_returns_timezone_aware_current_time() -> None:
    now = SystemClock("Europe/Moscow").now()

    assert isinstance(now, datetime)
    assert now.tzinfo is not None
