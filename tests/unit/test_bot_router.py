from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot.conversation import State
from app.bot.discovery_conversation import DiscoveryState
from app.bot.router import BotRouter, RouterState
from app.domain.discovery_models import (
    DiscoveryDraft,
    ExperienceProfile,
    IntentParseResult,
    TripIntent,
)
from app.domain.errors import LlmParseError, LlmProviderError
from app.domain.models import ParsedTripDraft


class FakeClock:
    def now(self):
        return datetime(2026, 7, 22, 12, tzinfo=UTC)


class Extractor:
    def __init__(self, result=None, *, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def extract(self, text, *, context, safety_identifier=None):
        self.calls.append((text, context, safety_identifier))
        if self.error is not None:
            raise self.error
        return self.result


class KnownConversation:
    def __init__(self):
        self.drafts = []

    async def accept_draft(self, message, context, draft):
        self.drafts.append(draft)
        return State.CONFIRM


class IdeasConversation:
    def __init__(self):
        self.results = []

    async def accept_result(self, message, context, result):
        self.results.append(result)
        return DiscoveryState.RESULTS

    async def start(self, update, context):
        return DiscoveryState.INTAKE


def update(text="тест"):
    progress = SimpleNamespace(edit_text=AsyncMock())
    message = SimpleNamespace(text=text, reply_text=AsyncMock(return_value=progress))
    return (
        SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=7)),
        progress,
    )


@pytest.mark.asyncio
async def test_router_sends_known_trip_to_existing_flow_without_second_llm_call() -> None:
    parsed = IntentParseResult(
        intent=TripIntent.DESTINATION_KNOWN,
        confidence="0.98",
        known_draft=ParsedTripDraft(
            origin="Москва",
            destination="Казань",
            departure_date=date(2026, 8, 15),
            return_date=date(2026, 8, 17),
        ),
    )
    extractor = Extractor(parsed)
    known = KnownConversation()
    ideas = IdeasConversation()
    router = BotRouter(
        extractor,  # type: ignore[arg-type]
        known,  # type: ignore[arg-type]
        ideas,  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    incoming, _ = update("Москва — Казань")

    state = await router.intake(incoming, SimpleNamespace(user_data={}))

    assert state is State.CONFIRM
    assert len(extractor.calls) == 1
    assert known.drafts[0].destination == "Казань"
    assert not ideas.results


@pytest.mark.asyncio
async def test_router_sends_unknown_destination_to_ideas_flow() -> None:
    parsed = IntentParseResult(
        intent=TripIntent.DESTINATION_UNKNOWN,
        confidence="0.95",
        discovery_draft=DiscoveryDraft(
            origin="Москва",
            departure_date=date(2026, 8, 15),
            return_date=date(2026, 8, 16),
            experience=ExperienceProfile(interests={"nature"}),
        ),
    )
    ideas = IdeasConversation()
    router = BotRouter(
        Extractor(parsed),  # type: ignore[arg-type]
        KnownConversation(),  # type: ignore[arg-type]
        ideas,  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    incoming, _ = update("Куда поехать на природу?")

    state = await router.intake(incoming, SimpleNamespace(user_data={}))

    assert state is DiscoveryState.RESULTS
    assert ideas.results == [parsed]


@pytest.mark.asyncio
async def test_router_failure_offers_explicit_choice_and_start_has_both_examples() -> None:
    router = BotRouter(
        Extractor(error=LlmProviderError("offline")),  # type: ignore[arg-type]
        KnownConversation(),  # type: ignore[arg-type]
        IdeasConversation(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    context = SimpleNamespace(user_data={"old": "state"})
    start_update, _ = update("")

    state = await router.start(start_update, context)

    assert state is RouterState.INTAKE
    assert "old" not in context.user_data
    assert len(context.user_data["flow_id"]) == 8
    start_text = start_update.effective_message.reply_text.call_args.args[0]
    assert "Если город уже есть" in start_text
    assert "предложу несколько проверенных идей" in start_text
    start_keyboard = start_update.effective_message.reply_text.call_args.kwargs["reply_markup"]
    assert [row[0].callback_data for row in start_keyboard.inline_keyboard] == [
        "route:known",
        "route:ideas",
    ]

    intake_update, progress = update("неясно")
    state = await router.intake(intake_update, context)

    assert state is RouterState.INTAKE
    keyboard = progress.edit_text.call_args.kwargs["reply_markup"]
    assert [row[0].callback_data for row in keyboard.inline_keyboard] == [
        "route:known",
        "route:ideas",
    ]


@pytest.mark.asyncio
async def test_router_mapping_failure_is_not_misreported_as_unclear_user_input() -> None:
    router = BotRouter(
        Extractor(error=LlmParseError("invalid structured result")),  # type: ignore[arg-type]
        KnownConversation(),  # type: ignore[arg-type]
        IdeasConversation(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    incoming, progress = update("полный и понятный запрос")

    state = await router.intake(incoming, SimpleNamespace(user_data={}))

    assert state is RouterState.INTAKE
    text = progress.edit_text.call_args.args[0]
    assert "техническая ошибка" in text
    assert "надёжно определить" not in text


@pytest.mark.asyncio
async def test_router_rejects_sensitive_data_before_openai() -> None:
    extractor = Extractor(error=AssertionError("extractor must not be called"))
    router = BotRouter(
        extractor,  # type: ignore[arg-type]
        KnownConversation(),  # type: ignore[arg-type]
        IdeasConversation(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    incoming, _ = update("Сохрани карту 1111 1111 1111 1111")

    state = await router.intake(incoming, SimpleNamespace(user_data={}))

    assert state is RouterState.INTAKE
    assert not extractor.calls
    assert (
        "Не могу принимать или сохранять"
        in (incoming.effective_message.reply_text.call_args.args[0])
    )


@pytest.mark.asyncio
async def test_plain_text_after_cancel_gets_explicit_recovery_buttons() -> None:
    router = BotRouter(
        Extractor(),  # type: ignore[arg-type]
        KnownConversation(),  # type: ignore[arg-type]
        IdeasConversation(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    incoming, _ = update("15–16 августа")

    await router.orphan_text(incoming, SimpleNamespace(user_data={}))

    reply = incoming.effective_message.reply_text.call_args
    assert "Активного диалога нет" in reply.args[0]
    assert len(reply.kwargs["reply_markup"].inline_keyboard) == 2
