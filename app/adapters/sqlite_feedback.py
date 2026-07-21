"""Small local feedback store with bounded retention and capability receipts."""

from __future__ import annotations

import asyncio
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.domain.product_models import FeedbackRecord


class SqliteFeedbackSink:
    def __init__(self, path: str | Path, *, retention_days: int = 30) -> None:
        if not 1 <= retention_days <= 365:
            raise ValueError("feedback retention must be between 1 and 365 days")
        self._path = Path(path)
        self._retention = timedelta(days=retention_days)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def submit(self, feedback: FeedbackRecord) -> str:
        await self._ensure_initialized()
        receipt_id = f"fb_{secrets.token_hex(8)}"
        expires_at = feedback.created_at + self._retention
        await asyncio.to_thread(
            self._submit_sync,
            receipt_id,
            feedback,
            expires_at,
        )
        await self.cleanup_expired(feedback.created_at)
        return receipt_id

    async def delete(self, receipt_id: str) -> bool:
        await self._ensure_initialized()
        if not _valid_receipt(receipt_id):
            return False
        return await asyncio.to_thread(self._delete_sync, receipt_id)

    async def cleanup_expired(self, now: datetime) -> int:
        await self._ensure_initialized()
        if now.tzinfo is None:
            raise ValueError("feedback cleanup time must be timezone-aware")
        return await asyncio.to_thread(self._cleanup_sync, now)

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5)
        connection.execute("PRAGMA secure_delete=ON")
        return connection

    def _initialize_sync(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    receipt_id TEXT PRIMARY KEY,
                    flow_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    comment TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS feedback_expires_at ON feedback(expires_at)"
            )

    def _submit_sync(
        self,
        receipt_id: str,
        feedback: FeedbackRecord,
        expires_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback(
                    receipt_id, flow_id, reason, comment, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    feedback.flow_id,
                    feedback.reason.value,
                    feedback.comment,
                    feedback.created_at.astimezone(UTC).isoformat(),
                    expires_at.astimezone(UTC).isoformat(),
                ),
            )

    def _delete_sync(self, receipt_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM feedback WHERE receipt_id = ?",
                (receipt_id,),
            )
            return cursor.rowcount > 0

    def _cleanup_sync(self, now: datetime) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM feedback WHERE expires_at <= ?",
                (now.astimezone(UTC).isoformat(),),
            )
            return cursor.rowcount


def _valid_receipt(value: str) -> bool:
    return (
        len(value) == 19
        and value.startswith("fb_")
        and all(character in "0123456789abcdef" for character in value[3:])
    )
