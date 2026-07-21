from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot.conversation import State, TripConversation
from app.domain.errors import LlmProviderError, TripValidationError
from app.domain.models import (
    CheckoutLink,
    HotelMode,
    OfferDetails,
    ParsedTripDraft,
    ParseResult,
    TripCheckoutItem,
    TripComponent,
    TripSearchResult,
)
from app.services.product_analytics import ProductAnalytics
from app.services.request_builder import build_trip_request
from tests.fakes.parser import FakeRequestParser
from tests.unit.test_trip_handoff import search_result


class FakeClock:
    def now(self) -> datetime:
        return datetime(2026, 7, 21, tzinfo=UTC)


class FakePlanner:
    def __init__(self) -> None:
        self.calls = []

    async def search_trip(self, request):
        self.calls.append(request)
        return TripSearchResult(request=request, searched_at=FakeClock().now())


class FailingParser:
    def __init__(self) -> None:
        self.calls = []

    async def parse(self, text, *, now, timezone, safety_identifier=None):
        self.calls.append((text, now, timezone))
        raise LlmProviderError("unavailable")


class FakeHandoff:
    def __init__(self) -> None:
        self.details_calls = 0

    def available_components(self, result, index):
        return (TripComponent.OUTBOUND, TripComponent.RETURN, TripComponent.HOTEL)

    async def get_details(self, result, index, component):
        self.details_calls += 1
        return OfferDetails(product_type="railway", title="Поезд 128М")

    async def create_checkout_items(self, result, index):
        return (
            TripCheckoutItem(
                component=TripComponent.OUTBOUND,
                link=CheckoutLink(url="https://www.tutu.ru/fixture"),
            ),
        )


def message(text: str = ""):
    return SimpleNamespace(text=text, reply_text=AsyncMock())


def update_with_message(value):
    return SimpleNamespace(effective_message=value)


@pytest.mark.asyncio
async def test_natural_language_always_calls_parser_and_mcp_waits_for_confirmation() -> None:
    draft = ParsedTripDraft(
        origin="Москва",
        destination="Казань",
        departure_date="2026-08-21",
        return_date="2026-08-23",
        hotel_mode=HotelMode.REQUIRED,
    )
    parser = FakeRequestParser(ParseResult(draft=draft))
    planner = FakePlanner()
    conversation = TripConversation(
        parser,
        planner,
        FakeClock(),
        timezone="Europe/Moscow",  # type: ignore[arg-type]
    )
    context = SimpleNamespace(user_data={})
    incoming = message("Москва — Казань на выходные")

    state = await conversation.intake(update_with_message(incoming), context)

    assert state is State.CONFIRM
    assert len(parser.calls) == 1
    assert not planner.calls
    keyboard = incoming.reply_text.call_args.kwargs["reply_markup"]
    assert all(
        len(button.callback_data) <= 64 for row in keyboard.inline_keyboard for button in row
    )

    progress = SimpleNamespace(edit_text=AsyncMock())
    query = SimpleNamespace(
        data="trip:confirm",
        answer=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
        message=SimpleNamespace(reply_text=AsyncMock(return_value=progress)),
    )
    state = await conversation.confirm(SimpleNamespace(callback_query=query), context)

    assert state is State.RESULTS
    assert len(planner.calls) == 1


@pytest.mark.asyncio
async def test_llm_failure_switches_to_form_without_recalling_llm() -> None:
    parser = FailingParser()
    planner = FakePlanner()
    conversation = TripConversation(
        parser,
        planner,
        FakeClock(),
        timezone="Europe/Moscow",  # type: ignore[arg-type]
    )
    context = SimpleNamespace(user_data={})

    state = await conversation.intake(update_with_message(message("поездка")), context)
    assert state is State.FORM

    for answer in ("Москва", "Казань", "21.08.2026", "23.08.2026", "нет"):
        state = await conversation.form_input(
            update_with_message(message(answer)),
            context,
        )

    assert state is State.CONFIRM
    assert len(parser.calls) == 1
    assert not planner.calls


@pytest.mark.asyncio
async def test_offer_details_are_cached_inside_conversation() -> None:
    handoff = FakeHandoff()
    conversation = TripConversation(
        SimpleNamespace(),  # type: ignore[arg-type]
        FakePlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
        handoff=handoff,  # type: ignore[arg-type]
    )
    context = SimpleNamespace(
        user_data={
            "trip_result": search_result(),
            "trip_details_cache": {},
            "trip_search_id": "abc12345",
        }
    )
    query = SimpleNamespace(
        data="trip:detail:abc12345:0:outbound",
        answer=AsyncMock(),
        message=SimpleNamespace(reply_text=AsyncMock()),
    )
    update = SimpleNamespace(callback_query=query)

    await conversation.result_action(update, context)
    await conversation.result_action(update, context)

    assert handoff.details_calls == 1
    assert query.message.reply_text.call_count == 2


@pytest.mark.asyncio
async def test_stale_result_callback_cannot_open_current_option() -> None:
    handoff = FakeHandoff()
    conversation = TripConversation(
        SimpleNamespace(),  # type: ignore[arg-type]
        FakePlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
        handoff=handoff,  # type: ignore[arg-type]
    )
    context = SimpleNamespace(
        user_data={
            "trip_result": search_result(),
            "trip_details_cache": {},
            "trip_search_id": "current1",
        }
    )
    query = SimpleNamespace(
        data="trip:detail:old00000:0:outbound",
        answer=AsyncMock(),
        message=SimpleNamespace(reply_text=AsyncMock()),
    )

    state = await conversation.result_action(SimpleNamespace(callback_query=query), context)

    assert state is State.RESULTS
    assert handoff.details_calls == 0
    query.answer.assert_awaited_once_with("Это кнопка от предыдущего поиска", show_alert=True)


@pytest.mark.asyncio
async def test_old_result_button_is_acknowledged_after_state_loss_or_restart() -> None:
    conversation = TripConversation(
        SimpleNamespace(),  # type: ignore[arg-type]
        FakePlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    query = SimpleNamespace(
        data="trip:checkout:old00000:0",
        answer=AsyncMock(),
    )

    state = await conversation.stale_result_callback(
        SimpleNamespace(callback_query=query),
        SimpleNamespace(user_data={}),
    )

    assert state is None
    query.answer.assert_awaited_once_with(
        "Это кнопка от предыдущего поиска. Запустите новый поиск командой /newtrip.",
        show_alert=True,
    )


@pytest.mark.asyncio
async def test_field_edit_preserves_all_other_trip_parameters() -> None:
    draft = ParsedTripDraft(
        origin="Москва",
        destination="Казань",
        departure_date="2026-08-21",
        return_date="2026-08-23",
        hotel_mode=HotelMode.REQUIRED,
    )
    conversation = TripConversation(
        SimpleNamespace(),  # type: ignore[arg-type]
        FakePlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    context = SimpleNamespace(user_data={"trip_draft": draft})
    query_message = SimpleNamespace(reply_text=AsyncMock())
    query = SimpleNamespace(
        data="trip:editfield:budget",
        answer=AsyncMock(),
        message=query_message,
    )

    state = await conversation.confirm(SimpleNamespace(callback_query=query), context)
    assert state is State.FORM

    answer = message("30 000")
    state = await conversation.form_input(update_with_message(answer), context)

    assert state is State.CONFIRM
    updated = context.user_data["trip_draft"]
    assert updated.origin == "Москва"
    assert updated.destination == "Казань"
    assert str(updated.budget) == "30000"


def test_no_results_keyboard_always_offers_recovery_actions() -> None:
    result = TripSearchResult(
        request=search_result().request,
        searched_at=FakeClock().now(),
    )

    keyboard = TripConversation._results_keyboard(result, "abc12345")

    callbacks = [row[0].callback_data for row in keyboard.inline_keyboard]
    assert callbacks == [
        "trip:retry:abc12345",
        "trip:change:abc12345",
        "trip:new:abc12345",
    ]


@pytest.mark.asyncio
async def test_start_explains_privacy_and_session_timeout() -> None:
    conversation = TripConversation(
        SimpleNamespace(),  # type: ignore[arg-type]
        FakePlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    incoming = message()
    context = SimpleNamespace(user_data={"old": "state"})

    state = await conversation.start(update_with_message(incoming), context)

    assert state is State.INTAKE
    assert context.user_data == {}
    text = incoming.reply_text.call_args.args[0]
    assert "OpenAI" in text
    assert "30 минут" in text
    assert "/privacy" in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("help", "/newtrip"),
        ("privacy", "передаётся OpenAI"),
        ("feedback", "владельцу бота"),
        ("unknown_command", "Такой команды нет"),
    ],
)
async def test_information_commands_always_answer(method: str, expected: str) -> None:
    conversation = TripConversation(
        SimpleNamespace(),  # type: ignore[arg-type]
        FakePlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    incoming = message()

    await getattr(conversation, method)(
        update_with_message(incoming), SimpleNamespace(user_data={})
    )

    assert expected in incoming.reply_text.call_args.args[0]


@pytest.mark.asyncio
async def test_natural_language_change_merges_with_existing_draft() -> None:
    current = ParsedTripDraft(
        origin="Москва",
        destination="Казань",
        departure_date="2026-08-21",
        return_date="2026-08-23",
        hotel_mode=HotelMode.REQUIRED,
    )
    patch = ParsedTripDraft(hotel_mode=HotelMode.FORBIDDEN)
    parser = FakeRequestParser(ParseResult(draft=patch))
    conversation = TripConversation(
        parser,
        FakePlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    context = SimpleNamespace(user_data={"trip_draft": current})
    incoming = message("Давай без отеля")

    state = await conversation.modify_text(update_with_message(incoming), context)

    assert state is State.CONFIRM
    updated = context.user_data["trip_draft"]
    assert updated.origin == "Москва"
    assert updated.hotel_mode is HotelMode.FORBIDDEN


@pytest.mark.asyncio
async def test_unrecognized_change_offers_field_level_editing() -> None:
    current = ParsedTripDraft(
        origin="Москва",
        destination="Казань",
        departure_date="2026-08-21",
        return_date="2026-08-23",
        hotel_mode=HotelMode.REQUIRED,
    )
    conversation = TripConversation(
        FailingParser(),
        FakePlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    context = SimpleNamespace(user_data={"trip_draft": current})
    incoming = message("непонятное изменение")

    state = await conversation.modify_text(update_with_message(incoming), context)

    assert state is State.CONFIRM
    assert incoming.reply_text.call_count == 2
    keyboard = incoming.reply_text.call_args.kwargs["reply_markup"]
    assert len(keyboard.inline_keyboard) == 9


@pytest.mark.asyncio
async def test_components_and_checkout_are_resolved_only_for_current_search() -> None:
    handoff = FakeHandoff()
    conversation = TripConversation(
        SimpleNamespace(),  # type: ignore[arg-type]
        FakePlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
        handoff=handoff,  # type: ignore[arg-type]
    )
    context = SimpleNamespace(
        user_data={"trip_result": search_result(), "trip_search_id": "abc12345"}
    )
    message_mock = SimpleNamespace(reply_text=AsyncMock())
    components_query = SimpleNamespace(
        data="trip:components:abc12345:0",
        answer=AsyncMock(),
        message=message_mock,
    )

    state = await conversation.result_action(
        SimpleNamespace(callback_query=components_query), context
    )
    assert state is State.RESULTS
    assert "Что показать" in message_mock.reply_text.call_args.args[0]

    status = SimpleNamespace(edit_text=AsyncMock())
    checkout_message = SimpleNamespace(reply_text=AsyncMock(return_value=status))
    checkout_query = SimpleNamespace(
        data="trip:checkout:abc12345:0",
        answer=AsyncMock(),
        message=checkout_message,
    )
    state = await conversation.result_action(
        SimpleNamespace(callback_query=checkout_query), context
    )

    assert state is State.RESULTS
    handoff_text = status.edit_text.call_args.args[0]
    assert "не резервирует" in handoff_text
    assert status.edit_text.call_args.kwargs["reply_markup"].inline_keyboard


@pytest.mark.asyncio
async def test_expected_search_failure_keeps_request_and_recovery_buttons() -> None:
    class ErrorPlanner:
        async def search_trip(self, request):
            raise TripValidationError("invalid")

    draft = ParsedTripDraft(
        origin="Москва",
        destination="Казань",
        departure_date="2026-08-21",
        return_date="2026-08-23",
        hotel_mode=HotelMode.FORBIDDEN,
    )
    request = build_trip_request(draft)
    conversation = TripConversation(
        SimpleNamespace(),  # type: ignore[arg-type]
        ErrorPlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    status = SimpleNamespace(edit_text=AsyncMock())
    message_mock = SimpleNamespace(reply_text=AsyncMock(return_value=status))
    context = SimpleNamespace(user_data={"trip_request": request})

    state = await conversation._run_search(message_mock, context)

    assert state is State.CONFIRM
    assert context.user_data["trip_request"] == request
    assert status.edit_text.call_args.kwargs["reply_markup"].inline_keyboard


@pytest.mark.asyncio
async def test_cancel_and_timeout_clear_personal_state() -> None:
    conversation = TripConversation(
        SimpleNamespace(),  # type: ignore[arg-type]
        FakePlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    incoming = message()
    context = SimpleNamespace(user_data={"trip_draft": "sensitive"})

    await conversation.cancel(update_with_message(incoming), context)
    assert context.user_data == {}

    context.user_data["trip_draft"] = "sensitive"
    await conversation.timeout(update_with_message(incoming), context)
    assert context.user_data == {}
    assert "30 минут" in incoming.reply_text.call_args.args[0]


@pytest.mark.asyncio
async def test_known_trip_emits_typed_funnel_without_route_dimensions() -> None:
    class EventSink:
        def __init__(self) -> None:
            self.events = []

        async def emit(self, item):
            self.events.append(item)

        async def close(self):
            return None

    class ResultsPlanner:
        async def search_trip(self, request):
            return search_result().model_copy(update={"request": request})

    draft = ParsedTripDraft(
        origin="Москва",
        destination="Казань",
        departure_date="2026-08-21",
        return_date="2026-08-23",
        hotel_mode=HotelMode.REQUIRED,
    )
    sink = EventSink()
    conversation = TripConversation(
        FakeRequestParser(ParseResult(draft=draft)),
        ResultsPlanner(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
        analytics=ProductAnalytics(sink, FakeClock()),  # type: ignore[arg-type]
    )
    context = SimpleNamespace(user_data={})
    start_message = message()
    await conversation.start(update_with_message(start_message), context)
    incoming = message("Москва — Казань")
    state = await conversation.intake(update_with_message(incoming), context)
    assert state is State.CONFIRM
    progress = SimpleNamespace(edit_text=AsyncMock())
    query = SimpleNamespace(
        data="trip:confirm",
        answer=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
        message=SimpleNamespace(reply_text=AsyncMock(return_value=progress)),
    )

    state = await conversation.confirm(SimpleNamespace(callback_query=query), context)

    assert state is State.RESULTS
    assert [item.name for item in sink.events] == [
        "conversation_started",
        "intent_classified",
        "verification_started",
        "verification_completed",
        "proposals_shown",
    ]
    assert all("origin" not in item.dimensions for item in sink.events)
