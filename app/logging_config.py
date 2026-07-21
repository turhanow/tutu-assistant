"""Application logging that never renders configured credential values."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class RedactingFormatter(logging.Formatter):
    def __init__(self, secrets: tuple[str, ...]) -> None:
        super().__init__("%(levelname)s:%(name)s:%(message)s")
        self._secrets = tuple(secret for secret in secrets if secret)

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        for secret in self._secrets:
            rendered = rendered.replace(secret, "[REDACTED]")
        return rendered


class StructuredRedactingFormatter(logging.Formatter):
    """Emit allowlisted JSON fields and redact configured credentials."""

    _EXTRA_FIELDS = (
        "event",
        "category",
        "component",
        "latency_ms",
        "result_count",
        "retryable",
        "schema_hash",
    )

    def __init__(self, secrets: tuple[str, ...]) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                field: getattr(record, field)
                for field in self._EXTRA_FIELDS
                if hasattr(record, field)
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        product_event = getattr(record, "product_event", None)
        if isinstance(product_event, dict):
            payload["product_event"] = product_event
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
        for secret in self._secrets:
            rendered = rendered.replace(secret, "[REDACTED]")
        return rendered


def configure_logging(level: str, *, secrets: tuple[str, ...]) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredRedactingFormatter(secrets))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # These clients log complete request URLs; Telegram embeds its token in the path.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("mcp.client").setLevel(logging.WARNING)
