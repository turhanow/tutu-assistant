"""User feedback persistence boundary."""

from typing import Protocol

from app.domain.product_models import FeedbackRecord


class FeedbackSink(Protocol):
    async def submit(self, feedback: FeedbackRecord) -> str: ...

    async def delete(self, receipt_id: str) -> bool: ...
