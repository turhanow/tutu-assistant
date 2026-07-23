"""Compact versioned callback contract shared by discovery Telegram views."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

CALLBACK_VERSION = "d1"
MAX_CALLBACK_BYTES = 64
_FLOW_PATTERN = re.compile(r"^[0-9a-f]{8}$")
_ARGUMENT_PATTERN = re.compile(r"^[a-z0-9_-]{1,16}$")


class DiscoveryAction(StrEnum):
    DETAILS = "dt"
    PLAN = "pl"
    COMPARE = "cp"
    REFINE = "rf"
    RECHECK = "rc"
    REJECT = "rj"
    REJECT_REASON = "rr"
    HANDOFF = "go"
    NEW = "nw"


@dataclass(frozen=True, slots=True)
class DiscoveryCallback:
    flow_id: str
    revision: int
    action: DiscoveryAction
    argument: str | None = None


class CallbackCodec:
    @staticmethod
    def encode(value: DiscoveryCallback) -> str:
        if not _FLOW_PATTERN.fullmatch(value.flow_id):
            raise ValueError("callback flow_id must be eight lowercase hex characters")
        if value.revision < 1:
            raise ValueError("callback revision must be positive")
        if value.argument is not None and not _ARGUMENT_PATTERN.fullmatch(value.argument):
            raise ValueError("callback argument contains unsupported characters")
        parts = (CALLBACK_VERSION, value.flow_id, str(value.revision), value.action.value)
        encoded = ":".join((*parts, value.argument) if value.argument is not None else parts)
        if len(encoded.encode("utf-8")) > MAX_CALLBACK_BYTES:
            raise ValueError("Telegram callback_data exceeds 64 bytes")
        return encoded

    @staticmethod
    def decode(raw: str) -> DiscoveryCallback:
        parts = raw.split(":")
        if len(parts) not in {4, 5} or parts[0] != CALLBACK_VERSION:
            raise ValueError("unsupported discovery callback")
        try:
            revision = int(parts[2])
            action = DiscoveryAction(parts[3])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid discovery callback") from error
        value = DiscoveryCallback(
            flow_id=parts[1],
            revision=revision,
            action=action,
            argument=parts[4] if len(parts) == 5 else None,
        )
        if CallbackCodec.encode(value) != raw:
            raise ValueError("non-canonical discovery callback")
        return value
