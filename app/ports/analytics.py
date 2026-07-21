"""Non-blocking product analytics boundary."""

from typing import Protocol

from app.domain.product_models import ProductEvent


class AnalyticsSink(Protocol):
    async def emit(self, event: ProductEvent) -> None: ...

    async def close(self) -> None: ...
