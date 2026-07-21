"""Bounded, failure-isolated analytics delivery to structured application logs."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Protocol

from app.domain.product_models import ProductEvent

logger = logging.getLogger(__name__)


class AnalyticsEventWriter(Protocol):
    async def write(self, event: ProductEvent) -> None: ...


class LoggingAnalyticsEventWriter:
    async def write(self, event: ProductEvent) -> None:
        logger.info(
            "product_analytics",
            extra={"product_event": event.model_dump(mode="json")},
        )


class NullAnalyticsSink:
    async def emit(self, event: ProductEvent) -> None:
        return None

    async def close(self) -> None:
        return None


class BufferedAnalyticsSink:
    """put_nowait producer with one bounded background consumer."""

    def __init__(
        self,
        writer: AnalyticsEventWriter | None = None,
        *,
        queue_size: int = 256,
        flush_timeout_seconds: float = 2,
    ) -> None:
        if queue_size < 1:
            raise ValueError("analytics queue_size must be positive")
        if flush_timeout_seconds <= 0:
            raise ValueError("analytics flush timeout must be positive")
        self._writer = writer or LoggingAnalyticsEventWriter()
        self._queue: asyncio.Queue[ProductEvent] = asyncio.Queue(maxsize=queue_size)
        self._flush_timeout_seconds = flush_timeout_seconds
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self.dropped_count = 0
        self.failed_count = 0

    async def emit(self, event: ProductEvent) -> None:
        if self._closed:
            self.dropped_count += 1
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped_count += 1

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        worker = self._worker
        if worker is None:
            return
        try:
            async with asyncio.timeout(self._flush_timeout_seconds):
                await self._queue.join()
        except TimeoutError:
            logger.warning(
                "analytics_flush_timed_out",
                extra={"event": "analytics_flush_timed_out"},
            )
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(
                self._run(),
                name="product-analytics-writer",
            )

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._writer.write(event)
            except Exception:
                self.failed_count += 1
                logger.exception("analytics_event_delivery_failed")
            finally:
                self._queue.task_done()
