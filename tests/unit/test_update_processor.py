import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from telegram import Chat, InlineQuery, Message, Update, User

from app.bot.update_processor import ChatSerialUpdateProcessor


@pytest.mark.asyncio
async def test_updates_for_same_chat_are_serialized() -> None:
    processor = ChatSerialUpdateProcessor(4)
    processor._key = lambda update: update.chat_id  # type: ignore[method-assign]
    active = 0
    peak = 0

    async def work() -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1

    await asyncio.gather(
        processor.process_update(SimpleNamespace(chat_id=1), work()),
        processor.process_update(SimpleNamespace(chat_id=1), work()),
    )

    assert peak == 1
    assert processor._entries == {}


@pytest.mark.asyncio
async def test_different_chats_are_processed_concurrently() -> None:
    processor = ChatSerialUpdateProcessor(4)
    processor._key = lambda update: update.chat_id  # type: ignore[method-assign]
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = 0

    async def work() -> None:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await release.wait()

    tasks = [
        asyncio.create_task(processor.process_update(SimpleNamespace(chat_id=chat_id), work()))
        for chat_id in (1, 2)
    ]
    await asyncio.wait_for(both_started.wait(), timeout=1)
    release.set()
    await asyncio.gather(*tasks)

    assert started == 2


@pytest.mark.asyncio
async def test_processor_lifecycle_and_update_key_resolution() -> None:
    processor = ChatSerialUpdateProcessor(2)
    user = User(id=42, first_name="Test", is_bot=False)
    chat = Chat(id=100, type="private")
    message = Message(
        message_id=1,
        date=datetime(2026, 7, 21, tzinfo=UTC),
        chat=chat,
        from_user=user,
        text="hello",
    )
    update = Update(update_id=7, message=message)

    await processor.initialize()
    assert processor._key(update) == 100
    inline_update = Update(
        update_id=8,
        inline_query=InlineQuery(id="inline", from_user=user, query="", offset=""),
    )
    assert processor._key(inline_update) == 42
    assert processor._key(object()) != 100

    processor._entries[1] = (asyncio.Lock(), 1)
    await processor.shutdown()
    assert processor._entries == {}
