"""Privacy sanitization and failure-isolated feedback use cases."""

from __future__ import annotations

import asyncio
import logging
import re

from app.domain.product_models import FeedbackReason, FeedbackRecord
from app.ports.clock import Clock
from app.ports.feedback import FeedbackSink
from app.services.product_analytics import ProductAnalytics

logger = logging.getLogger(__name__)

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s()\-]{6,}\d)(?!\w)")
_HANDLE = re.compile(r"(?<!\w)@[a-zA-Z0-9_]{5,32}\b")


class FeedbackService:
    def __init__(
        self,
        sink: FeedbackSink,
        clock: Clock,
        *,
        analytics: ProductAnalytics | None = None,
        timeout_seconds: float = 2,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("feedback timeout must be positive")
        self._sink = sink
        self._clock = clock
        self._analytics = analytics
        self._timeout_seconds = timeout_seconds

    async def submit(
        self,
        *,
        flow_id: str,
        reason: FeedbackReason,
        comment: str | None,
    ) -> str | None:
        try:
            record = FeedbackRecord(
                flow_id=flow_id,
                reason=reason,
                comment=sanitize_feedback_comment(comment),
                created_at=self._clock.now(),
            )
            async with asyncio.timeout(self._timeout_seconds):
                receipt = await self._sink.submit(record)
        except Exception:
            logger.exception("feedback_submission_failed")
            return None
        if self._analytics is not None:
            await self._analytics.track(
                "feedback_submitted",
                flow_id=flow_id,
                dimensions={"reason_code": reason.value},
            )
        return receipt

    async def delete(self, receipt_id: str) -> bool | None:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                return await self._sink.delete(receipt_id)
        except Exception:
            logger.exception("feedback_deletion_failed")
            return None


def sanitize_feedback_comment(value: str | None) -> str | None:
    if value is None:
        return None
    sanitized = _URL.sub("[link]", value)
    sanitized = _EMAIL.sub("[email]", sanitized)
    sanitized = _PHONE.sub("[phone]", sanitized)
    sanitized = _HANDLE.sub("[handle]", sanitized)
    normalized = " ".join(sanitized.split())[:1_000].strip()
    return normalized or None
