import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.adapters.buffered_analytics import BufferedAnalyticsSink
from app.adapters.sqlite_feedback import SqliteFeedbackSink
from app.bot.discovery_conversation import DiscoveryState
from app.bot.feedback_conversation import FeedbackConversation, FeedbackState
from app.domain.product_models import FeedbackReason, FeedbackRecord, ProductEvent
from app.logging_config import StructuredRedactingFormatter
from app.services.feedback_service import FeedbackService, sanitize_feedback_comment

NOW = datetime(2026, 7, 22, 12, tzinfo=UTC)


class FakeClock:
    def now(self):
        return NOW


def event(**overrides) -> ProductEvent:
    values = {
        "name": "shortlist_generated",
        "occurred_at": NOW,
        "flow_id": "flow1234",
        "dimensions": {
            "flow_type": "destination_unknown",
            "voice_version": "voice_v2",
            "candidate_count": 3,
            "catalog_version": "v1",
        },
    }
    values.update(overrides)
    return ProductEvent(**values)


def test_product_event_rejects_non_allowlisted_names_dimensions_and_raw_text() -> None:
    with pytest.raises(ValidationError, match="not allowlisted"):
        event(name="arbitrary_event")
    with pytest.raises(ValidationError, match="dimensions are not allowlisted"):
        event(dimensions={"raw_text": "Москва — Казань"})
    with pytest.raises(ValidationError, match="dimensions are not allowlisted"):
        event(dimensions={"telegram_id": 123})


@pytest.mark.asyncio
async def test_buffered_sink_never_waits_for_slow_writer_and_close_is_bounded() -> None:
    class SlowWriter:
        async def write(self, item):
            await asyncio.sleep(10)

    sink = BufferedAnalyticsSink(
        SlowWriter(),  # type: ignore[arg-type]
        queue_size=1,
        flush_timeout_seconds=0.01,
    )

    await asyncio.wait_for(sink.emit(event()), timeout=0.05)
    await asyncio.wait_for(sink.emit(event()), timeout=0.05)
    await asyncio.wait_for(sink.close(), timeout=0.1)

    assert sink.dropped_count >= 0


@pytest.mark.asyncio
async def test_writer_failure_is_counted_and_does_not_escape_to_user_flow() -> None:
    class BrokenWriter:
        async def write(self, item):
            raise RuntimeError("offline")

    sink = BufferedAnalyticsSink(BrokenWriter())  # type: ignore[arg-type]

    await sink.emit(event())
    await sink.close()

    assert sink.failed_count == 1


def test_structured_log_contains_only_typed_product_event_payload() -> None:
    formatter = StructuredRedactingFormatter(("secret",))
    record = logging.LogRecord(
        "analytics",
        logging.INFO,
        __file__,
        1,
        "product_analytics",
        (),
        None,
    )
    record.product_event = event().model_dump(mode="json")

    payload = json.loads(formatter.format(record))

    assert payload["product_event"]["name"] == "shortlist_generated"
    assert "raw_text" not in payload["product_event"]["dimensions"]


@pytest.mark.asyncio
async def test_sqlite_feedback_returns_receipt_expires_and_supports_physical_delete(
    tmp_path,
) -> None:
    sink = SqliteFeedbackSink(tmp_path / "feedback.sqlite3", retention_days=30)
    record = FeedbackRecord(
        flow_id="flow1234",
        reason=FeedbackReason.WRONG_DATA,
        comment="Неверное время",
        created_at=NOW,
    )

    receipt = await sink.submit(record)

    assert receipt.startswith("fb_")
    assert await sink.delete(receipt)
    assert not await sink.delete(receipt)

    expired_receipt = await sink.submit(
        record.model_copy(update={"created_at": NOW - timedelta(days=31)})
    )
    assert await sink.cleanup_expired(NOW) == 1
    assert not await sink.delete(expired_receipt)


def test_feedback_comment_redacts_contact_channels_and_is_bounded() -> None:
    sanitized = sanitize_feedback_comment(
        "Пишите test@example.org, +7 999 123-45-67, @private_user или https://example.org/private"
    )

    assert sanitized == "Пишите [email], [phone], [handle] или [link]"
    assert len(sanitize_feedback_comment("x" * 1_500) or "") == 1_000


@pytest.mark.asyncio
async def test_feedback_service_timeout_returns_safe_failure() -> None:
    class SlowSink:
        async def submit(self, feedback):
            await asyncio.sleep(10)

        async def delete(self, receipt_id):
            await asyncio.sleep(10)

    service = FeedbackService(
        SlowSink(),  # type: ignore[arg-type]
        FakeClock(),
        timeout_seconds=0.01,
    )

    receipt = await service.submit(
        flow_id="flow1234",
        reason=FeedbackReason.OTHER,
        comment="Комментарий",
    )
    deleted = await service.delete("fb_0000000000000000")

    assert receipt is None
    assert deleted is None


class FeedbackSink:
    def __init__(self) -> None:
        self.records = []
        self.deleted = []

    async def submit(self, record):
        self.records.append(record)
        return "fb_0123456789abcdef"

    async def delete(self, receipt_id):
        self.deleted.append(receipt_id)
        return True


def message(text=""):
    return SimpleNamespace(text=text, reply_text=AsyncMock())


@pytest.mark.asyncio
async def test_feedback_flow_returns_receipt_and_delete_command_uses_capability() -> None:
    sink = FeedbackSink()
    conversation = FeedbackConversation(
        FeedbackService(sink, FakeClock())  # type: ignore[arg-type]
    )
    context = SimpleNamespace(user_data={"discovery_flow_id": "abcd1234"}, args=[])
    incoming = message()

    state = await conversation.start(SimpleNamespace(effective_message=incoming), context)
    assert state is FeedbackState.REASON
    reason_keyboard = incoming.reply_text.call_args.kwargs["reply_markup"]
    assert all(
        len(button.callback_data.encode()) <= 64
        for row in reason_keyboard.inline_keyboard
        for button in row
    )

    query_message = message()
    query = SimpleNamespace(
        data="feedback:reason:wrong_data",
        answer=AsyncMock(),
        message=query_message,
    )
    state = await conversation.choose_reason(SimpleNamespace(callback_query=query), context)
    assert state is FeedbackState.COMMENT

    comment_message = message("Цена неверная, test@example.org")
    state = await conversation.comment(
        SimpleNamespace(effective_message=comment_message),
        context,
    )
    assert state == -1
    assert sink.records[0].flow_id == "abcd1234"
    assert sink.records[0].comment == "Цена неверная, [email]"
    assert "fb_0123456789abcdef" in comment_message.reply_text.call_args.args[0]

    delete_message = message()
    context.args = ["fb_0123456789abcdef"]
    await conversation.delete(SimpleNamespace(effective_message=delete_message), context)

    assert sink.deleted == ["fb_0123456789abcdef"]
    assert "удалено" in delete_message.reply_text.call_args.args[0]


@pytest.mark.asyncio
async def test_feedback_returns_to_existing_discovery_results() -> None:
    sink = FeedbackSink()
    conversation = FeedbackConversation(
        FeedbackService(sink, FakeClock())  # type: ignore[arg-type]
    )
    context = SimpleNamespace(
        user_data={
            "discovery_flow_id": "abcd1234",
            "discovery_proposals": object(),
        }
    )
    incoming = message()
    await conversation.start(SimpleNamespace(effective_message=incoming), context)
    query_message = message()
    reason_query = SimpleNamespace(
        data="feedback:reason:other",
        answer=AsyncMock(),
        message=query_message,
    )
    await conversation.choose_reason(SimpleNamespace(callback_query=reason_query), context)
    skip_query = SimpleNamespace(
        data="feedback:skip",
        answer=AsyncMock(),
        message=query_message,
    )

    state = await conversation.skip_comment(SimpleNamespace(callback_query=skip_query), context)

    assert state is DiscoveryState.RESULTS
    assert sink.records[0].comment is None


@pytest.mark.asyncio
async def test_feedback_feature_flag_stops_before_storage_calls() -> None:
    sink = FeedbackSink()
    conversation = FeedbackConversation(
        FeedbackService(sink, FakeClock()),  # type: ignore[arg-type]
        enabled=False,
    )
    incoming = message()

    state = await conversation.start(
        SimpleNamespace(effective_message=incoming),
        SimpleNamespace(user_data={}),
    )

    assert state == -1
    assert "временно недоступна" in incoming.reply_text.call_args.args[0]
    assert sink.records == []
