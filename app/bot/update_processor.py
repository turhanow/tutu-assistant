"""Bounded Telegram concurrency while preserving per-chat update order."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from telegram import Update
from telegram.ext import BaseUpdateProcessor


class ChatSerialUpdateProcessor(BaseUpdateProcessor):
    """Run independent chats concurrently and serialize each chat's updates."""

    def __init__(self, max_concurrent_updates: int = 16) -> None:
        super().__init__(max_concurrent_updates)
        self._guard = asyncio.Lock()
        self._entries: dict[int, tuple[asyncio.Lock, int]] = {}

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        self._entries.clear()

    async def do_process_update(
        self,
        update: object,
        coroutine: Awaitable[Any],
    ) -> None:
        key = self._key(update)
        async with self._guard:
            lock, users = self._entries.get(key, (asyncio.Lock(), 0))
            self._entries[key] = (lock, users + 1)
        try:
            async with lock:
                await coroutine
        finally:
            async with self._guard:
                current_lock, users = self._entries[key]
                if users == 1:
                    self._entries.pop(key, None)
                else:
                    self._entries[key] = (current_lock, users - 1)

    @staticmethod
    def _key(update: object) -> int:
        if isinstance(update, Update):
            if update.effective_chat is not None:
                return update.effective_chat.id
            if update.effective_user is not None:
                return update.effective_user.id
            return update.update_id
        return id(update)
