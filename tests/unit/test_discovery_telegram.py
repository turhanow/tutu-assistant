import asyncio
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
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
    format_details,
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
from app.domain.models import (
    CheckoutLink,
    HotelMode,
    SortPreference,
    TripCheckoutItem,
    TripComponent,
)
from app.services.discovery_ranking import rank_proposals
from app.services.discovery_request_builder import build_discovery_request
from app.services.product_analytics import ProductAnalytics
from tests.unit.test_discovery_proposals import candidate, feasibility, proposal
from tests.unit.test_trip_handoff import search_result

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


@pytest.mark.asyncio
async def test_onboarding_origin_survives_first_free_form_discovery_prompt() -> None:
    extracted = complete_draft(origin=None)
    service, *_ = conversation(parsed(extracted))
    context = SimpleNamespace(user_data={})
    intro_message, _ = telegram_message()

    start_state = await service.start_from_origin(
        telegram_update(intro_message),
        context,
        "Москва",
    )
    request_message, _ = telegram_message(
        "В эти выходные хочу старинный город с усадьбами, без суеты"
    )
    result_state = await service.intake(telegram_update(request_message), context)

    assert start_state is DiscoveryState.INTAKE
    assert result_state is DiscoveryState.RESULTS
    assert context.user_data["discovery_draft"].origin == "Москва"


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


def test_every_recommendation_has_its_own_itinerary_button() -> None:
    service, *_ = conversation(parsed(complete_draft()))
    result = DiscoveryProposalResult(
        recommendations=rank_proposals(
            (proposal("city_a"), proposal("city_b"), proposal("city_c"))
        ),
        completed_at=NOW,
    )
    context = SimpleNamespace(user_data={"discovery_flow_id": "abcd1234", "discovery_revision": 1})

    keyboard = service._proposals_keyboard(context, result)

    buttons = [button for row in keyboard.inline_keyboard for button in row]
    plan_buttons = [button for button in buttons if button.text.startswith("План на 2 дня")]
    assert len(plan_buttons) == 3


def test_discovery_formatters_keep_money_classes_separate_and_split_on_sections() -> None:
    recommendation = rank_proposals((proposal("city_a"),))[0]
    days = recommendation.proposal.days
    linked_activity = (
        days[0]
        .activities[0]
        .activity.model_copy(
            update={
                "address": "Советская улица, 3",
                "map_url": "https://yandex.ru/maps/?text=Test%20Place%2C%20City",
            }
        )
    )
    linked_schedule = days[0].activities[0].model_copy(update={"activity": linked_activity})
    days = (
        days[0].model_copy(update={"activities": (linked_schedule, *days[0].activities[1:])}),
        *days[1:],
    )
    recommendation = recommendation.model_copy(
        update={"proposal": recommendation.proposal.model_copy(update={"days": days})}
    )
    trip_option = recommendation.proposal.trip_option
    combination = trip_option.combination
    combination = combination.model_copy(
        update={
            "outbound": combination.outbound.model_copy(
                update={"service_number": "6104", "carrier": "ЦППК"}
            ),
            "return_offer": combination.return_offer.model_copy(
                update={"service_number": "B-42", "carrier": "Тестовый перевозчик"}
            ),
            "hotel": search_result().options[0].combination.hotel,
        }
    )
    recommendation = recommendation.model_copy(
        update={
            "proposal": recommendation.proposal.model_copy(
                update={"trip_option": trip_option.model_copy(update={"combination": combination})}
            )
        }
    )
    copy = ProposalCopy(
        proposal_id="proposal_1",
        title="<City_A>",
        reason="Архитектура & прогулки",
        trade_off="Проверить часы",
        evidence_ids={"source_city_a"},
    )

    checkout_items = {
        "city_a": (
            TripCheckoutItem(
                component=TripComponent.OUTBOUND,
                link=CheckoutLink(
                    url="https://www.tutu.ru/booking/outbound",
                    kind="direct_offer",
                ),
            ),
            TripCheckoutItem(
                component=TripComponent.RETURN,
                link=CheckoutLink(
                    url="https://www.tutu.ru/schedule/return",
                    kind="schedule_url",
                ),
            ),
            TripCheckoutItem(
                component=TripComponent.HOTEL,
                link=CheckoutLink(
                    url="https://www.tutu.ru/hotel/selected",
                    kind="hotel_page",
                ),
            ),
        )
    }
    sections = format_proposal_sections((recommendation,), (copy,), checkout_items)
    section_limit = max(len(item) for item in sections)
    messages = pack_html_sections(sections, limit=section_limit)

    rendered = "\n".join(messages)
    assert "&lt;City_A&gt;" in rendered
    assert "Предварительная стоимость" in rendered
    assert 'Отель: <a href="https://yandex.ru/maps/?text=' in rendered
    assert 'номер: <a href="https://www.tutu.ru/hotel/selected"' in rendered
    assert "на карте</a>" not in rendered
    assert 'Туда: <a href="https://www.tutu.ru/booking/outbound">' in rendered
    assert "поезд № 6104 · ЦППК</a> · 22.08, 15:00 → 22.08, 17:00" in rendered
    assert "Обратно: автобус № B-42 · Тестовый перевозчик · 23.08, 18:00" in rendered
    assert 'href="https://www.tutu.ru/booking/outbound"' in rendered
    assert "расписание на Tutu</a>" not in rendered
    assert "Выбор дороги: лучший баланс цены, времени в пути" in rendered
    assert "Коротко:" in rendered
    assert "Test%20Place%2C%20City" in rendered
    assert rendered.index("Коротко:") < rendered.index("Туда:")
    assert "Что делать:" not in rendered
    assert "План на 2 дня:" in format_program(recommendation)
    assert messages


def test_message_packer_counts_visible_text_instead_of_long_checkout_url() -> None:
    long_checkout_url = "https://www.tutu.ru/booking?payload=" + ("x" * 6_000)
    section = f'<a href="{long_checkout_url}">Конкретный билет</a> · 25.07, 10:00'

    messages = pack_html_sections((section,), limit=100)

    assert messages == (section,)
    assert long_checkout_url in messages[0]


def test_message_packer_splits_truly_long_visible_card_without_failure() -> None:
    section = "\n".join(f"Строка {index}: " + ("текст " * 8) for index in range(12))

    messages = pack_html_sections((section,), limit=100)

    assert len(messages) > 1
    assert all(len(message) <= 100 for message in messages)


def test_proposal_uses_suggestions_when_no_activity_fits_schedule() -> None:
    recommendation = rank_proposals((proposal("city_a"),))[0]
    source_activity = recommendation.proposal.days[0].activities[0].activity
    linked_activity = source_activity.model_copy(
        update={
            "map_url": "https://yandex.ru/maps/?text=Suggested%20Museum",
        }
    )
    empty_days = tuple(
        day.model_copy(update={"activities": ()}) for day in recommendation.proposal.days
    )
    recommendation = recommendation.model_copy(
        update={
            "proposal": recommendation.proposal.model_copy(
                update={
                    "days": empty_days,
                    "content_complete": False,
                    "suggested_activities": (linked_activity,),
                }
            )
        }
    )
    copy = ProposalCopy(
        proposal_id="proposal_1",
        title="City A",
        reason="Подходит по интересам",
        trade_off="Проверьте часы работы",
        evidence_ids={"source_city_a"},
    )

    rendered = "\n".join(format_proposal_sections((recommendation,), (copy,)))

    assert "Коротко:" in rendered
    assert "Активность 1" in rendered


def test_hotel_without_room_name_keeps_map_and_booking_links_separate() -> None:
    recommendation = rank_proposals((proposal("city_a"),))[0]
    trip_option = recommendation.proposal.trip_option
    combination = trip_option.combination
    hotel = search_result().options[0].combination.hotel.model_copy(update={"room_name": None})
    recommendation = recommendation.model_copy(
        update={
            "proposal": recommendation.proposal.model_copy(
                update={
                    "trip_option": trip_option.model_copy(
                        update={"combination": combination.model_copy(update={"hotel": hotel})}
                    )
                }
            )
        }
    )
    copy = ProposalCopy(
        proposal_id="proposal_1",
        title="City A",
        reason="Подходит по интересам",
        trade_off="Проверьте условия",
        evidence_ids={"source_city_a"},
    )
    checkout = {
        "city_a": (
            TripCheckoutItem(
                component=TripComponent.HOTEL,
                link=CheckoutLink(
                    url="https://www.tutu.ru/hotel/selected",
                    kind="hotel_page",
                ),
            ),
        )
    }

    rendered = "\n".join(format_proposal_sections((recommendation,), (copy,), checkout))

    assert 'Отель: <a href="https://yandex.ru/maps/?text=' in rendered
    assert '<a href="https://www.tutu.ru/hotel/selected">забронировать</a>' in rendered


def test_card_and_program_expose_cheaper_and_faster_verified_alternatives() -> None:
    recommendation = rank_proposals((proposal("city_a"),))[0]
    selected = recommendation.proposal.trip_option
    cheap_combination = selected.combination.model_copy(
        update={
            "signature": "cheap-alternative",
            "price": selected.combination.price.model_copy(update={"known_total": Decimal("1500")}),
        }
    )
    fast_combination = selected.combination.model_copy(
        update={
            "signature": "fast-alternative",
            "price": selected.combination.price.model_copy(update={"known_total": Decimal("2600")}),
            "metrics": selected.combination.metrics.model_copy(
                update={"total_travel_duration": timedelta(minutes=120)}
            ),
        }
    )
    alternatives = (
        selected.model_copy(
            update={
                "combination": cheap_combination,
                "labels": frozenset({SortPreference.CHEAPEST}),
            }
        ),
        selected.model_copy(
            update={
                "combination": fast_combination,
                "labels": frozenset({SortPreference.FASTEST}),
            }
        ),
    )
    recommendation = recommendation.model_copy(
        update={
            "proposal": recommendation.proposal.model_copy(
                update={"trip_alternatives": alternatives}
            )
        }
    )
    copy = ProposalCopy(
        proposal_id="proposal_1",
        title="City A",
        reason="Подходит по интересам",
        trade_off="Проверьте условия",
        evidence_ids={"source_city_a"},
    )

    card = "\n".join(format_proposal_sections((recommendation,), (copy,)))
    program = format_program(recommendation)

    assert "Сравнить варианты: дешевле за 1 500 ₽; быстрее за 2 600 ₽" in card
    assert "Что ещё можно выбрать" in program
    assert "Бюджетный" in program
    assert "Быстрый" in program
    assert "🚗 Транспорт:" in program
    assert "🏨 Отель:" in program
    assert program.count("Туда:") == 3


def test_story_map_details_and_plan_contain_city_places_hotel_and_taxi() -> None:
    recommendation = rank_proposals((proposal("city_a"),))[0]
    proposal_value = recommendation.proposal
    destination = proposal_value.candidate.destination.model_copy(
        update={
            "short_description": "Старинный город для прогулок без спешки.",
            "full_description": (
                "Компактный исторический город с архитектурой нескольких эпох. "
                "Основные места удобно осмотреть пешком за выходные."
            ),
            "taxi_available": True,
        }
    )
    candidate_value = proposal_value.candidate.model_copy(update={"destination": destination})
    source = proposal_value.days[0].activities[0]
    linked = source.activity.model_copy(
        update={
            "description": "Главный архитектурный ансамбль города.",
            "map_url": "https://yandex.ru/maps/?text=Main%20Place",
        }
    )
    days = (
        proposal_value.days[0].model_copy(
            update={
                "activities": (
                    source.model_copy(update={"activity": linked}),
                    *proposal_value.days[0].activities[1:],
                )
            }
        ),
    )
    hotel = (
        search_result()
        .options[0]
        .combination.hotel.model_copy(
            update={"check_in_time": time(14), "check_out_time": time(12)}
        )
    )
    combination = proposal_value.trip_option.combination.model_copy(update={"hotel": hotel})
    recommendation = recommendation.model_copy(
        update={
            "proposal": proposal_value.model_copy(
                update={
                    "candidate": candidate_value,
                    "days": days,
                    "trip_option": proposal_value.trip_option.model_copy(
                        update={"combination": combination}
                    ),
                }
            )
        }
    )

    checkout_items = (
        TripCheckoutItem(
            component=TripComponent.OUTBOUND,
            link=CheckoutLink(
                url="https://www.tutu.ru/booking/exact-outbound",
                kind="deeplink",
            ),
        ),
        TripCheckoutItem(
            component=TripComponent.RETURN,
            link=CheckoutLink(
                url="https://www.tutu.ru/booking/exact-return",
                kind="deeplink",
            ),
        ),
        TripCheckoutItem(
            component=TripComponent.HOTEL,
            link=CheckoutLink(
                url="https://www.tutu.ru/hotel/exact",
                kind="hotel_page",
            ),
        ),
    )
    details = format_details(recommendation, checkout_items)
    plan = format_program(recommendation, checkout_items)

    assert "Подробнее:" in details
    assert "Фото и отзывы" in details
    assert "Main%20Place" in details
    assert "Стоимость транспорта" in details
    assert "Цена:" in details
    assert "booking/exact-outbound" in details
    assert "booking/exact-return" in details
    assert "hotel/exact" in details
    assert "Заезд:" in plan and "14:00" in plan
    assert "Выезд:" in plan and "12:00" in plan
    assert "Такси в городе: да" in plan
    assert "booking/exact-outbound" in plan
    assert "booking/exact-return" in plan
    assert "hotel/exact" in plan
    assert "https://yandex.ru/maps/?text=" in plan


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
    assert "куда можно уехать" in final_call.args[0]
    keyboard = final_call.kwargs["reply_markup"]
    assert keyboard.inline_keyboard[0][0].text == "Решились? Подобрать варианты"
    callbacks = [
        button.callback_data
        for row in keyboard.inline_keyboard
        for button in row
        if button.callback_data
    ]
    assert callbacks
    assert all(len(item.encode()) <= 64 for item in callbacks)

    compare_query = SimpleNamespace(
        data=keyboard.inline_keyboard[0][0].callback_data,
        answer=AsyncMock(),
        message=incoming,
    )
    await service.callback(SimpleNamespace(callback_query=compare_query), context)
    assert "поездки, которые реально складываются" in incoming.edit_text.call_args.args[0]


@pytest.mark.asyncio
async def test_inline_checkout_links_are_resolved_for_verified_recommendations() -> None:
    handoff = FakeHandoff()
    service, *_ = conversation(handoff=handoff)
    proposals = DiscoveryProposalResult(
        recommendations=rank_proposals((proposal("city_a"),)),
        completed_at=NOW,
    )

    links = await service._resolve_inline_checkout_items(
        proposals,
        feasibility("city_a"),
    )

    assert set(links) == {"city_a"}
    assert links["city_a"][0].component is TripComponent.OUTBOUND
    assert len(handoff.calls) == 1


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
async def test_date_follow_up_is_applied_immediately_without_second_llm_call() -> None:
    incomplete = complete_draft(departure_date=None, return_date=None)
    service, extractor, selector, *_ = conversation(parsed(incomplete))
    context = SimpleNamespace(user_data={})
    incoming, _ = telegram_message("Хочу спокойные выходные")

    state = await service.intake(telegram_update(incoming), context)

    assert state is DiscoveryState.CLARIFY
    assert context.user_data["discovery_pending_field"] == "dates"
    answer, _ = telegram_message("29–30 августа 2026")
    state = await service.clarification_input(telegram_update(answer), context)

    assert state is DiscoveryState.RESULTS
    assert len(extractor.calls) == 1
    assert len(selector.calls) == 1
    request = selector.calls[0]
    assert request.dates.start == date(2026, 8, 29)
    assert request.dates.end == date(2026, 8, 30)
    assert "discovery_pending_field" not in context.user_data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer_text", "expected"),
    [
        ("завтра и послезавтра", (date(2026, 7, 23), date(2026, 7, 24))),
        ("в эти выходные", (date(2026, 7, 25), date(2026, 7, 26))),
        ("25-26 авг", (date(2026, 8, 25), date(2026, 8, 26))),
        ("25-26 авг 2026", (date(2026, 8, 25), date(2026, 8, 26))),
    ],
)
async def test_natural_date_follow_ups_continue_discovery(answer_text, expected) -> None:
    incomplete = complete_draft(departure_date=None, return_date=None)
    service, extractor, selector, *_ = conversation(parsed(incomplete))
    context = SimpleNamespace(user_data={})
    incoming, _ = telegram_message("Хочу молодёжно потусить")
    assert await service.intake(telegram_update(incoming), context) is DiscoveryState.CLARIFY

    answer, _ = telegram_message(answer_text)
    state = await service.clarification_input(telegram_update(answer), context)

    assert state is DiscoveryState.RESULTS
    assert len(extractor.calls) == 1
    assert len(selector.calls) == 1
    request = selector.calls[0]
    assert (request.dates.start, request.dates.end) == expected


@pytest.mark.asyncio
async def test_broad_month_is_clarified_without_leaking_validation_details() -> None:
    broad_month = complete_draft(
        departure_date=date(2026, 8, 1),
        return_date=date(2026, 8, 31),
    )
    service, extractor, selector, *_ = conversation(parsed(broad_month))
    context = SimpleNamespace(user_data={})
    incoming, _ = telegram_message("В августе хочу в небольшой город прогуляться")

    state = await service.intake(telegram_update(incoming), context)

    assert state is DiscoveryState.CLARIFY
    assert context.user_data["discovery_pending_field"] == "dates"
    reply = incoming.reply_text.call_args_list[-1].args[0]
    assert "конкретные даты" in reply
    assert "validation error" not in reply
    assert "pydantic" not in reply.casefold()
    assert len(extractor.calls) == 1
    assert not selector.calls


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
    inspiration_keyboard = status.edit_text.call_args_list[-1].kwargs["reply_markup"]
    compare_query = SimpleNamespace(
        data=inspiration_keyboard.inline_keyboard[0][0].callback_data,
        answer=AsyncMock(),
        message=incoming,
    )
    await service.callback(SimpleNamespace(callback_query=compare_query), context)
    keyboard = incoming.edit_text.call_args.kwargs["reply_markup"]
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
    assert (
        context.user_data["discovery_shortlist"].request.dates.start
        == context.user_data["discovery_shortlist"].request.dates.end
    )


@pytest.mark.asyncio
async def test_new_prompt_after_comparison_replaces_previous_preferences() -> None:
    replacement = complete_draft(
        origin=None,
        budget="12000",
        experience=ExperienceProfile(interests={"nature"}),
    )
    service, *_ = conversation(parsed(complete_draft()), parsed(replacement))
    context = SimpleNamespace(user_data={})
    incoming, status = telegram_message("Хочу архитектуру")
    await service.intake(telegram_update(incoming), context)
    inspiration_keyboard = status.edit_text.call_args.kwargs["reply_markup"]
    compare = SimpleNamespace(
        data=inspiration_keyboard.inline_keyboard[0][0].callback_data,
        answer=AsyncMock(),
        message=incoming,
    )
    await service.callback(SimpleNamespace(callback_query=compare), context)
    replacement_message, _ = telegram_message("Теперь хочу природу, до 12 тысяч")

    state = await service.refine_input(telegram_update(replacement_message), context)

    assert state is DiscoveryState.RESULTS
    draft = context.user_data["discovery_draft"]
    assert draft.origin == "Москва"
    assert draft.budget == 12000
    assert draft.experience.interests == {"nature"}
    assert "architecture" not in draft.experience.interests


@pytest.mark.asyncio
async def test_details_reject_reason_and_handoff_remain_inside_current_revision() -> None:
    handoff = FakeHandoff()
    service, *_ = conversation(parsed(complete_draft()), handoff=handoff)
    context = SimpleNamespace(user_data={})
    incoming, status = telegram_message("Идея")
    await service.intake(telegram_update(incoming), context)
    context.user_data["discovery_feasibility"] = feasibility("city_a")
    inspiration_keyboard = status.edit_text.call_args_list[-1].kwargs["reply_markup"]
    compare_query = SimpleNamespace(
        data=inspiration_keyboard.inline_keyboard[0][0].callback_data,
        answer=AsyncMock(),
        message=incoming,
    )
    await service.callback(SimpleNamespace(callback_query=compare_query), context)
    keyboard = incoming.edit_text.call_args.kwargs["reply_markup"]

    details_query = SimpleNamespace(
        data=keyboard.inline_keyboard[0][0].callback_data,
        answer=AsyncMock(),
        message=incoming,
    )
    state = await service.callback(SimpleNamespace(callback_query=details_query), context)
    assert state is DiscoveryState.RESULTS
    assert "Подробнее:" in incoming.reply_text.call_args.args[0]

    plan_query = SimpleNamespace(
        data=keyboard.inline_keyboard[0][1].callback_data,
        answer=AsyncMock(),
        message=incoming,
    )
    state = await service.callback(SimpleNamespace(callback_query=plan_query), context)
    assert state is DiscoveryState.RESULTS
    assert "План на 2 дня:" in incoming.reply_text.call_args.args[0]
    assert "Такси в городе" in incoming.reply_text.call_args.args[0]

    reject_query = SimpleNamespace(
        data=keyboard.inline_keyboard[2][0].callback_data,
        answer=AsyncMock(),
        message=incoming,
    )
    assert keyboard.inline_keyboard[2][0].text.startswith("Что не понравилось")
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
        data=keyboard.inline_keyboard[1][0].callback_data,
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
        "inspiration_shown",
    ]
    assert all("raw_text" not in item.dimensions for item in sink.events)
    assert all(item.flow_id == context.user_data["discovery_flow_id"] for item in sink.events)
