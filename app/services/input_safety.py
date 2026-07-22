"""Deterministic pre-LLM guard for sensitive data in Telegram messages."""

from __future__ import annotations

import re

SENSITIVE_DATA_RESPONSE = (
    "Не могу принимать или сохранять платёжные, паспортные и другие чувствительные "
    "данные. Удалите их из сообщения и отправьте только параметры поездки. Подробнее: /privacy"
)

_PAYMENT_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_SECURITY_CODE = re.compile(r"\b(?:cvv|cvc|код\s+безопасности)\s*[:=-]?\s*\d{3,4}\b", re.IGNORECASE)
_PASSPORT = re.compile(
    r"\b(?:паспорт\w*|серия\s+и\s+номер)\b.{0,30}\d(?:[ -]?\d){7,11}\b",
    re.IGNORECASE,
)


def contains_sensitive_data(text: str) -> bool:
    """Return true for high-confidence payment or identity-data patterns."""

    normalized = " ".join(text.split())
    return bool(
        _SECURITY_CODE.search(normalized)
        or _PASSPORT.search(normalized)
        # A card-shaped sequence is sensitive even when the user omits words such as
        # "карта". Requiring context here let bare payment data reach the generic router.
        or _PAYMENT_CARD.search(normalized)
    )
