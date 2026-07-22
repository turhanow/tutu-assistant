import asyncio
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot.callbacks import CallbackCodec, DiscoveryAction, DiscoveryCallback
from app.bot.discovery_conversation import (
    DiscoveryConversation,
    DiscoveryState,
    _no_proposals_message,
)
from app.bot.discovery_formatters import (
    format_program,
    format_proposal_sections,
    pack_html_sections,
)
from app.domain.discovery_models import (
    CandidateShortlist,
    DateRange,
    DiscoveryDraft,
    DiscoveryFeasibilityResult,
    DiscoveryProposalResult,
    DiscoveryRequest,
    ExperienceProfile,
    IntentParseResult,
    ProposalCopy,
    TravelPace,
    TripIntent,
)
from app.domain.errors import LlmParseError
from app.domain.models import CheckoutLink, HotelMode, TripCheckoutItem, TripComponent
from app.services.discovery_ranking import rank_proposals
from app.services.discovery_request_builder import build_discovery_request
from app.services.product_analytics import ProductAnalytics
from tests.unit.test_discovery_proposals import candidate, feasibility, proposal

NOW = datetime(2026, 7, 22, 12, tzinfo=UTC)


class FakeClock:
    def now(self) -> datetime:
        return NOW


class FakeExtractor:
    def __init__(self, *results: IntentParseResult) -> None:
        self.results = list(results)
        self.calls = []

    async def extract(self, text, *, context, safety_identifier=None):
        self.calls.append((text, context, safety_identifier))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeSelector:
    def __init__(self, candidates=None) -> None:
        self.calls = []
        self.candidates = (candidate("city_a"),) if candidates is None else tuple(candidates)

    async def select(self, request):
        self.calls.append(request)
        return CandidateShortlist(
            request=request,
            candidates=self.candidates,
            catalog_version="v1",
            score_version="candidate_v1",
        )


class FakeDiscoveryPlanner:
    def __init__(self) -> None:
        self.calls = []

    async def verify(self, shortlist):
        self.calls.append(shortlist)
        return DiscoveryFeasibilityResult(
            shortlist=shortlist,
            snapshots=(),
            completed_at=NOW,
        )


class FakeProposalBuilder:
    def __init__(self) -> None:
        self.calls = []

    async def build(self, feasibility):
        self.calls.append(feasibility)
        return DiscoveryProposalResult(
            recommendations=rank_proposals((proposal("city_a"),)),
            completed_at=NOW,
        )


class FakeNarration:
    def __init__(self) -> None:
        self.calls = []

    async def narrate(self, recommendations, *, context):
        self.calls.append((recommendations, context))
        return (
            ProposalCopy(
                proposal_id="proposal_1",
                title="Выходные в City_A",
                reason="Подходит для архитектурной прогулки",
                trade_off="Часы работы нужно проверить",
                evidence_ids={"source_city_a"},
            ),
        )


class FakeHandoff:
    def __init__(self) -> None:
        self.calls = []

    async def create_checkout_items(self, result, index):
        self.calls.append((result, index))
        return (
            TripCheckoutItem(
                component=TripComponent.OUTBOUND,
                link=CheckoutLink(url="https://www.tutu.ru/fixture"),
            ),
        )


def parsed(draft: DiscoveryDraft) -> IntentParseResult:
    return IntentParseResult(
        intent=TripIntent.DESTINATION_UNKNOWN,
        confidence="0.95",
        discovery_draft=draft,
    )


def complete_draft(**overrides) -> DiscoveryDraft:
    values = {
        "origin": "Москва",
        "departure_date": date(2026, 8, 15),
        "return_date": date(2026, 8, 16),
        "hotel_mode": HotelMode.OPTIONAL,
        "experience": ExperienceProfile(interests={"architecture"}),
    }
    values.update(overrides)
    return DiscoveryDraft(**values)


def telegram_message(text: str = ""):
    status = SimpleNamespace(edit_text=AsyncMock())
    value = SimpleNamespace(
        text=text,
        reply_text=AsyncMock(return_value=status),
        edit_text=AsyncMock(),
    )
    return value, status


def telegram_update(message, *, user_id: int = 42):
    return SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=user_id),
    )


def conversation(*extractor_results, handoff=None, analytics=None, selector=None):
    extractor = FakeExtractor(*extractor_results)
    selector = selector or FakeSelector()
    planner = FakeDiscoveryPlanner()
    builder = FakeProposalBuilder()
    narration = FakeNarration()
    service = DiscoveryConversation(
        extractor,  # type: ignore[arg-type]
        selector,  # type: ignore[arg-type]
        planner,  # type: ignore[arg-type]
        builder,  # type: ignore[arg-type]
        narration,  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
        handoff=handoff,
        analytics=analytics,
    )
    return service, extractor, selector, planner, builder, narration


def test_versioned_callback_codec_is_compact_and_rejects_noncanonical_data() -> None:
    value = DiscoveryCallback(
        flow_id="abcd1234",
        revision=17,
        action=DiscoveryAction.DETAILS,
        argument="2",
    )

    encoded = CallbackCodec.encode(value)

    assert CallbackCodec.decode(encoded) == value
    assert len(encoded.encode()) <= 64
    with pytest.raises(ValueError):
        CallbackCodec.decode("d1:abcd1234:017:dt:2")
    with pytest.raises(ValueError):
        CallbackCodec.encode(value.__class__("not-safe", 1, DiscoveryAction.NEW))


def test_discovery_formatters_keep_money_classes_separate_and_split_on_sections() -> None:
    recommendation = rank_proposals((proposal("city_a"),))[0]
    copy = ProposalCopy(
        proposal_id="proposal_1",
        title="<City_A>",
        reason="Архитектура & прогулки",
        trade_off="Проверить часы",
        evidence_ids={"source_city_a"},
    )

    sections = format_proposal_sections((recommendation,), (copy,))
    section_limit = max(len(item) for item in sections)
    messages = pack_html_sections(sections, limit=section_limit)

    rendered = "\n".join(messages)
    assert "&lt;City_A&gt;" in rendered
    assert "Подтверждённая стоимость" in rendered
    assert "Отель: не нужен" in rendered
    assert "Программа:" in format_program(recommendation)
    assert len(messages) == 2
    assert all(len(item) <= section_limit for item in messages)


@pytest.mark.asyncio
async def test_complete_ideas_request_reaches_verified_proposals_end_to_end() -> None:
    service, extractor, selector, planner, builder, narration = conversation(
        parsed(complete_draft())
    )
    context = SimpleNamespace(user_data={})
    incoming, status = telegram_message("Из Москвы на выходные, хочу архитектуру")

    state = await service.intake(telegram_update(incoming), context)

    assert state is DiscoveryState.RESULTS
    assert len(extractor.calls) == 1
    assert len(selector.calls) == len(planner.calls) == len(builder.calls) == 1
    assert len(narration.calls) == 1
    assert context.user_data["discovery_revision"] == 1
    final_call = status.edit_text.call_args_list[-1]
    assert "поездки, которые реально складываются" in final_call.args[0]
    keyboard = final_call.kwargs["reply_markup"]
    callbacks = [
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    ]
    assert callbacks
    assert all(len(item.encode()) <= 64 for item in callbacks)


@pytest.mark.asyncio
async def test_missing_fields_are_asked_in_one_bounded_batch_then_merged() -> None:
    service, extractor, selector, *_ = conversation(
        parsed(DiscoveryDraft()),
        parsed(complete_draft()),
    )
    context = SimpleNamespace(user_data={})
    incoming, _ = telegram_message("Хочу куда-нибудь")

    state = await service.intake(telegram_update(incoming), context)

    assert state is DiscoveryState.CLARIFY
    question_text = incoming.reply_text.call_args_list[-1].args[0]
    assert question_text.count("?") <= 3
    answer, _ = telegram_message("Москва; 15–16 августа; архитектура")
    state = await service.clarification_input(telegram_update(answer), context)

    assert state is DiscoveryState.RESULTS
    assert len(extractor.calls) == 2
    assert len(selector.calls) == 1


@pytest.mark.asyncio
async def test_mapping_failure_is_reported_as_technical_not_as_bad_user_input() -> None:
    service, *_ = conversation(LlmParseError("invalid confidence"))
    context = SimpleNamespace(user_data={})
    incoming, status = telegram_message(
        "Хочу из Москвы куда-нибудь на два дня 29–30 августа, нужен отель"
    )

    state = await service.intake(telegram_update(incoming), context)

    assert state is DiscoveryState.INTAKE
    text = status.edit_text.call_args.args[0]
    assert "техническая ошибка" in text
    assert "указать город" not in text


@pytest.mark.asyncio
async def test_sensitive_data_stops_discovery_before_progress_llm_and_state_merge() -> None:
    service, extractor, *_ = conversation()
    context = SimpleNamespace(user_data={})
    incoming, _ = telegram_message("Сохрани карту 1111 1111 1111 1111")

    state = await service.intake(telegram_update(incoming), context)

    assert state is DiscoveryState.INTAKE
    assert not extractor.calls
    assert incoming.reply_text.await_count == 1
    assert "Не могу принимать или сохранять" in incoming.reply_text.call_args.args[0]


@pytest.mark.asyncio
async def test_impossible_constraints_return_recovery_actions_without_invented_options() -> None:
    empty_selector = FakeSelector(())
    service, _, _, planner, builder, narration = conversation(
        parsed(
            complete_draft(
                origin="Владивосток",
                budget="500",
            )
        ),
        selector=empty_selector,
    )
    context = SimpleNamespace(user_data={})
    incoming, status = telegram_message("Куда-нибудь на один день за 500 рублей")

    state = await service.intake(telegram_update(incoming), context)

    assert state is DiscoveryState.RESULTS
    assert "Под эти условия направлений пока нет" in status.edit_text.call_args.args[0]
    assert incoming.reply_text.call_args.kwargs["reply_markup"] is not None
    assert not planner.calls
    assert not builder.calls
    assert not narration.calls


@pytest.mark.asyncio
async def test_stale_revision_is_rejected_and_recheck_uses_existing_shortlist() -> None:
    service, _, _, planner, _, narration = conversation(parsed(complete_draft()))
    context = SimpleNamespace(user_data={})
    incoming, status = telegram_message("Идея")
    await service.intake(telegram_update(incoming), context)
    keyboard = status.edit_text.call_args_list[-1].kwargs["reply_markup"]
    details_data = keyboard.inline_keyboard[0][0].callback_data
    recheck_data = keyboard.inline_keyboard[-2][1].callback_data

    stale_query = SimpleNamespace(
        data=details_data,
        answer=AsyncMock(),
        message=incoming,
    )
    context.user_data["discovery_revision"] = 2
    state = await service.callback(SimpleNamespace(callback_query=stale_query), context)
    assert state is DiscoveryState.RESULTS
    stale_query.answer.assert_awaited_once_with(
        "Это кнопка от предыдущей версии подборки",
        show_alert=True,
    )

    context.user_data["discovery_revision"] = 1
    replies_before_recheck = incoming.reply_text.await_count
    recheck_query = SimpleNamespace(
        data=recheck_data,
        answer=AsyncMock(),
        message=incoming,
    )
    state = await service.callback(SimpleNamespace(callback_query=recheck_query), context)

    assert state is DiscoveryState.RESULTS
    assert context.user_data["discovery_revision"] == 2
    assert len(planner.calls) == 2
    assert len(narration.calls) == 1
    incoming.edit_text.assert_awaited()
    assert incoming.reply_text.await_count == replies_before_recheck


@pytest.mark.asyncio
async def test_discovery_button_after_flow_switch_is_always_acknowledged() -> None:
    service, *_ = conversation()
    query = SimpleNamespace(data="d1:abcd1234:1:dt:0", answer=AsyncMock())

    await service.stale_callback(
        SimpleNamespace(callback_query=query),
        SimpleNamespace(user_data={}),
    )

    query.answer.assert_awaited_once_with(
        "Эта кнопка относится к предыдущей подборке. Запустите новую через /ideas.",
        show_alert=True,
    )


@pytest.mark.asyncio
async def test_refinement_increments_revision_and_rebuilds_candidates() -> None:
    service, _, selector, *_ = conversation(
        parsed(complete_draft()),
        parsed(DiscoveryDraft(budget="15000")),
    )
    context = SimpleNamespace(user_data={})
    incoming, _ = telegram_message("Идея")
    await service.intake(telegram_update(incoming), context)
    refine_message, _ = telegram_message("Бюджет до 15 000")

    state = await service.refine_input(telegram_update(refine_message), context)

    assert state is DiscoveryState.RESULTS
    assert context.user_data["discovery_revision"] == 2
    assert context.user_data["discovery_draft"].budget == 15000
    assert len(selector.calls) == 2


@pytest.mark.asyncio
async def test_refinement_can_change_explicit_hotel_requirement() -> None:
    service, *_ = conversation(
        parsed(complete_draft(hotel_mode=HotelMode.REQUIRED)),
        parsed(DiscoveryDraft(hotel_mode=HotelMode.FORBIDDEN)),
    )
    context = SimpleNamespace(user_data={})
    incoming, _ = telegram_message("Идея с отелем")
    await service.intake(telegram_update(incoming), context)
    refinement, _ = telegram_message("Давай без отеля")

    state = await service.refine_input(telegram_update(refinement), context)

    assert state is DiscoveryState.RESULTS
    assert context.user_data["discovery_draft"].hotel_mode is HotelMode.FORBIDDEN


@pytest.mark.asyncio
async def test_details_reject_reason_and_handoff_remain_inside_current_revision() -> None:
    handoff = FakeHandoff()
    service, *_ = conversation(parsed(complete_draft()), handoff=handoff)
    context = SimpleNamespace(user_data={})
    incoming, status = telegram_message("Идея")
    await service.intake(telegram_update(incoming), context)
    context.user_data["discovery_feasibility"] = feasibility("city_a")
    keyboard = status.edit_text.call_args_list[-1].kwargs["reply_markup"]

    details_query = SimpleNamespace(
        data=keyboard.inline_keyboard[0][0].callback_data,
        answer=AsyncMock(),
        message=incoming,
    )
    state = await service.callback(SimpleNamespace(callback_query=details_query), context)
    assert state is DiscoveryState.RESULTS
    assert "Программа:" in incoming.reply_text.call_args.args[0]

    reject_query = SimpleNamespace(
        data=keyboard.inline_keyboard[1][0].callback_data,
        answer=AsyncMock(),
        message=incoming,
    )
    assert keyboard.inline_keyboard[1][0].text.startswith("Что не понравилось")
    await service.callback(SimpleNamespace(callback_query=reject_query), context)
    reason_keyboard = incoming.reply_text.call_args.kwargs["reply_markup"]
    reason_query = SimpleNamespace(
        data=reason_keyboard.inline_keyboard[0][0].callback_data,
        answer=AsyncMock(),
        message=incoming,
    )
    state = await service.callback(SimpleNamespace(callback_query=reason_query), context)
    assert state is DiscoveryState.RESULTS
    assert "Спасибо" in incoming.reply_text.call_args.args[0]

    handoff_query = SimpleNamespace(
        data=keyboard.inline_keyboard[0][1].callback_data,
        answer=AsyncMock(),
        message=incoming,
    )
    state = await service.callback(SimpleNamespace(callback_query=handoff_query), context)
    assert state is DiscoveryState.RESULTS
    assert len(handoff.calls) == 1
    assert handoff.calls[0][1] == 0
    handoff_result = handoff.calls[0][0]
    assert len(handoff_result.options) == 1


def test_no_proposals_names_budget_and_recovery_levers_without_false_precision() -> None:
    request = build_discovery_request(
        complete_draft(
            budget="1000",
            hotel_mode=HotelMode.REQUIRED,
            experience=ExperienceProfile(pace=TravelPace.RELAXED),
        ),
        today=date(2026, 7, 22),
    )

    rendered = _no_proposals_message(request)

    assert "1 000 ₽" in rendered
    assert "обязательным отелем" in rendered
    assert "увеличить бюджет" in rendered


def test_empty_candidate_shortlist_model_used_by_flow_is_valid() -> None:
    request = DiscoveryRequest(
        origin="Москва",
        dates=DateRange(start=date(2026, 8, 15), end=date(2026, 8, 16)),
        experience=ExperienceProfile(interests={"nature"}),
    )
    shortlist = CandidateShortlist(
        request=request,
        candidates=(),
        catalog_version="v1",
        score_version="candidate_v1",
    )
    assert not shortlist.candidates


@pytest.mark.asyncio
async def test_two_discovery_chats_keep_flow_and_draft_state_isolated() -> None:
    class PerTextExtractor:
        async def extract(self, text, *, context, safety_identifier=None):
            origin = "Москва" if "Москв" in text else "Тула"
            return parsed(complete_draft(origin=origin))

    selector = FakeSelector()
    service = DiscoveryConversation(
        PerTextExtractor(),  # type: ignore[arg-type]
        selector,  # type: ignore[arg-type]
        FakeDiscoveryPlanner(),  # type: ignore[arg-type]
        FakeProposalBuilder(),  # type: ignore[arg-type]
        FakeNarration(),  # type: ignore[arg-type]
        FakeClock(),
        timezone="Europe/Moscow",
    )
    first_context = SimpleNamespace(user_data={})
    second_context = SimpleNamespace(user_data={})
    first_message, _ = telegram_message("Из Москвы")
    second_message, _ = telegram_message("Из Тулы")

    states = await asyncio.gather(
        service.intake(telegram_update(first_message, user_id=1), first_context),
        service.intake(telegram_update(second_message, user_id=2), second_context),
    )

    assert states == [DiscoveryState.RESULTS, DiscoveryState.RESULTS]
    assert (
        first_context.user_data["discovery_flow_id"]
        != second_context.user_data["discovery_flow_id"]
    )
    assert first_context.user_data["discovery_draft"].origin == "Москва"
    assert second_context.user_data["discovery_draft"].origin == "Тула"


@pytest.mark.asyncio
async def test_discovery_happy_path_emits_complete_allowlisted_funnel() -> None:
    class EventSink:
        def __init__(self) -> None:
            self.events = []

        async def emit(self, item):
            self.events.append(item)

        async def close(self):
            return None

    sink = EventSink()
    analytics = ProductAnalytics(sink, FakeClock())  # type: ignore[arg-type]
    service, *_ = conversation(parsed(complete_draft()), analytics=analytics)
    context = SimpleNamespace(user_data={})
    incoming, _ = telegram_message("Из Москвы на выходные")

    state = await service.intake(telegram_update(incoming), context)

    assert state is DiscoveryState.RESULTS
    names = [item.name for item in sink.events]
    assert names == [
        "intent_classified",
        "shortlist_generated",
        "verification_started",
        "verification_completed",
        "proposals_shown",
    ]
    assert all("raw_text" not in item.dimensions for item in sink.events)
    assert all(item.flow_id == context.user_data["discovery_flow_id"] for item in sink.events)
