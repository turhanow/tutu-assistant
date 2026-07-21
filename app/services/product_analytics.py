"""Typed best-effort product analytics facade for application services."""

from __future__ import annotations

import logging

from app.domain.discovery_models import TripIntent
from app.domain.product_models import ProductEvent
from app.ports.analytics import AnalyticsSink
from app.ports.clock import Clock

logger = logging.getLogger(__name__)


class ProductAnalytics:
    def __init__(self, sink: AnalyticsSink, clock: Clock) -> None:
        self._sink = sink
        self._clock = clock

    async def track(
        self,
        name: str,
        *,
        flow_id: str,
        intent: TripIntent | None = None,
        dimensions: dict[str, str | int | bool] | None = None,
    ) -> None:
        try:
            event = ProductEvent(
                name=name,
                occurred_at=self._clock.now(),
                flow_id=flow_id,
                intent=intent,
                dimensions=dimensions or {},
            )
            await self._sink.emit(event)
        except Exception:
            logger.exception("product_analytics_rejected")

    async def close(self) -> None:
        try:
            await self._sink.close()
        except Exception:
            logger.exception("product_analytics_close_failed")
