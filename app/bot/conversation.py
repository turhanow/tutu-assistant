"""Telegram conversation adapter with recoverable, versioned trip flows."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
import warnings
from datetime import timedelta
from enum import IntEnum

from pydantic import ValidationError
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from telegram.warnings import PTBUserWarning

from app.bot.formatters import (
    COMPONENT_LABELS,
    format_confirmation,
    format_details,
    format_results,
)
from app.domain.discovery_models import TripIntent
from app.domain.errors import LlmParseError, LlmProviderError, TutuAssistantError
from app.domain.models import ParsedTripDraft, TripComponent, TripRequest, TripSearchResult
from app.ports.clock import Clock
from app.ports.parser import RequestParser
from app.services.input_safety import SENSITIVE_DATA_RESPONSE, contains_sensitive_data
from app.services.known_input_guardrails import explicit_conflicts, recover_known_draft
from app.services.product_analytics import ProductAnalytics
from app.services.request_builder import (
    DraftInputError,
    apply_form_answer,
    build_trip_request,
    next_missing_field,
)
from app.services.trip_handoff import TripHandoffService
from app.services.trip_planner import TripPlanner
from app.voice import HELP, PRIVACY, Voice

logger = logging.getLogger(__name__)

_DRAFT = "trip_draft"
_PENDING_FIELD = "trip_pending_field"
_REQUEST = "trip_request"
_RESULT = "trip_result"
_DETAILS_CACHE = "trip_details_cache"
_SEARCH_ID = "trip_search_id"
_RATE_EVENTS = "trip_rate_events"
_ANALYTICS_FLOW_ID = "flow_id"
KNOWN_RESULT_CALLBACK_PATTERN = (
    r"^trip:(components|checkout|detail):[0-9a-f]{8}:\d+"
    r"(?::(outbound|return|hotel))?$"
)
KNOWN_CONTROL_CALLBACK_PATTERN = r"^trip:(retry|change|new):[0-9a-f]{8}$"
KNOWN_ANY_RESULT_CALLBACK_PATTERN = (
    r"^trip:(?:(?:components|checkout|detail):[0-9a-f]{8}:\d+"
    r"(?::(?:outbound|return|hotel))?|(?:retry|change|new):[0-9a-f]{8})$"
)


class State(IntEnum):
    INTAKE = 1
    FORM = 2
    CONFIRM = 3
    RESULTS = 4


PROMPTS = {
    "origin": "Из какого города вы отправляетесь?",
    "destination": "В какой город хотите поехать?",
    "departure_date": "Дата отправления? Укажите в формате ДД.ММ.ГГГГ.",
    "return_date": "Дата возвращения? Укажите в формате ДД.ММ.ГГГГ.",
    "hotel_mode": "Нужен отель? Ответьте: да, нет или необязательно.",
    "adults": "Сколько взрослых едет? Укажите 1 или 2.",
    "budget": "Какой общий бюджет в рублях? Например: 25 000. Если ограничения нет — «нет».",
    "allowed_modes": ("Какой транспорт подходит: поезд, самолёт, автобус, электричка или любые?"),
    "sort": "Что важнее: дешевле, меньше времени в дороге или баланс цены и удобства?",
    "event_date": "Укажите дату события в формате ДД.ММ.ГГГГ.",
}

EDITABLE_FIELDS = (
    ("origin", "Город отправления"),
    ("destination", "Город назначения"),
    ("departure_date", "Дата отправления"),
    ("return_date", "Дата возвращения"),
    ("adults", "Пассажиры"),
    ("allowed_modes", "Транспорт"),
    ("hotel_mode", "Отель"),
    ("budget", "Бюджет"),
    ("sort", "Приоритет"),
)


class TripConversation:
    def __init__(
        self,
        parser: RequestParser,
        planner: TripPlanner,
        clock: Clock,
        *,
        timezone: str,
        handoff: TripHandoffService | None = None,
        analytics: ProductAnalytics | None = None,
        enabled: bool = True,
        voice: Voice | None = None,
    ) -> None:
        self._parser = parser
        self._planner = planner
        self._clock = clock
        self._timezone = timezone
        self._handoff = handoff
        self._analytics = analytics
        self._enabled = enabled
        self._voice = voice or Voice()
        self._safety_salt = secrets.token_bytes(32)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not self._enabled:
            await update.effective_message.reply_text(
                "Бот временно приостановлен для технических работ. Попробуйте позже."
            )
            return ConversationHandler.END
        context.user_data.clear()
        await self._track(
            "conversation_started",
            context,
            dimensions={"flow_type": "destination_known"},
        )
        await update.effective_message.reply_text(
            self._voice.known_start,
            parse_mode=ParseMode.HTML,
        )
        return State.INTAKE

    async def intake(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        message = update.effective_message
        source_text = message.text or ""
        if contains_sensitive_data(source_text):
            await message.reply_text(SENSITIVE_DATA_RESPONSE)
            return State.INTAKE
        if not self._allow(context, "llm", limit=8, within=timedelta(minutes=1)):
            await message.reply_text(
                "Слишком много запросов за короткое время. Подождите минуту и попробуйте снова."
            )
            return State.INTAKE
        conflicts = explicit_conflicts(source_text)
        progress = await message.reply_text(self._voice.progress("known_parse", context.user_data))
        started = time.perf_counter()
        parsed_successfully = False
        try:
            parsed = await self._parser.parse(
                source_text,
                now=self._clock.now(),
                timezone=self._timezone,
                safety_identifier=self._safety_identifier(update),
            )
            draft = parsed.draft
            parsed_successfully = True
            logger.info(
                "product_event",
                extra={
                    "event": "llm_parse_succeeded",
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                },
            )
        except (LlmParseError, LlmProviderError):
            draft = recover_known_draft(source_text, today=self._clock.now().date())
            logger.info(
                "product_event",
                extra={
                    "event": "llm_parse_fallback",
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                },
            )
            await progress.edit_text(
                "Не получилось разобрать весь запрос. Однозначные параметры сохранены; "
                "остальное уточню по шагам."
            )
        await self._track(
            "intent_classified",
            context,
            intent=TripIntent.DESTINATION_KNOWN,
            dimensions={"flow_type": "destination_known"},
        )
        context.user_data[_DRAFT] = draft
        if conflicts:
            updates = {}
            if any("отель" in item for item in conflicts):
                updates["hotel_mode"] = None
            if any("самолёт" in item for item in conflicts):
                updates["allowed_modes"] = frozenset()
            context.user_data[_DRAFT] = draft.model_copy(update=updates)
            await progress.edit_text(
                "В запросе есть противоречия:\n"
                + "\n".join(f"• {item}" for item in conflicts)
                + "\nОднозначные маршрут и даты сохранены."
            )
            field = "hotel_mode" if "hotel_mode" in updates else "allowed_modes"
            context.user_data[_PENDING_FIELD] = field
            await message.reply_text(
                "Уточните одним сообщением, нужен ли отель и какой транспорт можно использовать."
            )
            return State.FORM
        if parsed_successfully:
            await progress.edit_text("Вводные собраны. Проверяю, хватает ли данных.")
        return await self._continue_or_confirm(message, context)

    async def accept_draft(self, message, context, draft: ParsedTripDraft) -> int:
        """Continue known-trip flow from the shared intent router without a second LLM call."""
        if not self._enabled:
            await message.reply_text("Поиск временно приостановлен. Попробуйте позже.")
            return ConversationHandler.END
        context.user_data[_DRAFT] = draft
        return await self._continue_or_confirm(message, context)

    async def modify_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Apply a natural-language patch while preserving the current request."""
        message = update.effective_message
        if contains_sensitive_data(message.text or ""):
            await message.reply_text(SENSITIVE_DATA_RESPONSE)
            return State.CONFIRM
        current = context.user_data.get(_DRAFT)
        if not isinstance(current, ParsedTripDraft):
            return await self.intake(update, context)
        if not self._allow(context, "llm", limit=8, within=timedelta(minutes=1)):
            await message.reply_text("Подождите минуту перед следующим изменением.")
            return State.CONFIRM
        progress = await message.reply_text(self._voice.progress("known_patch", context.user_data))
        try:
            parsed = await self._parser.parse(
                message.text or "",
                now=self._clock.now(),
                timezone=self._timezone,
                safety_identifier=self._safety_identifier(update),
            )
        except (LlmParseError, LlmProviderError):
            await progress.edit_text(
                "Не получилось понять изменение. Выберите конкретное поле кнопкой ниже."
            )
            await message.reply_text("Что изменить?", reply_markup=self._edit_keyboard())
            return State.CONFIRM
        draft = self._merge_drafts(current, parsed.draft)
        context.user_data[_DRAFT] = draft
        context.user_data.pop(_RESULT, None)
        context.user_data.pop(_SEARCH_ID, None)
        await progress.edit_text("Готово. Проверьте обновлённые параметры.")
        return await self._continue_or_confirm(message, context)

    async def form_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        message = update.effective_message
        source_text = message.text or ""
        draft = context.user_data.get(_DRAFT)
        field = context.user_data.get(_PENDING_FIELD)
        if not isinstance(draft, ParsedTripDraft) or not isinstance(field, str):
            context.user_data[_DRAFT] = ParsedTripDraft()
            return await self._ask_field(message, context, "origin")
        if contains_sensitive_data(source_text):
            await message.reply_text(SENSITIVE_DATA_RESPONSE)
            return State.FORM
        conflicts = explicit_conflicts(source_text)
        if conflicts:
            await message.reply_text(
                "В ответе остались противоречия:\n"
                + "\n".join(f"• {item}" for item in conflicts)
                + "\nУточните условия без взаимоисключающих вариантов."
            )
            return State.FORM
        try:
            direct_draft = apply_form_answer(
                draft,
                field,
                source_text,
                today=self._clock.now().date(),
            )
        except DraftInputError:
            direct_draft = None
        if direct_draft is not None:
            context.user_data[_DRAFT] = direct_draft
            return await self._continue_or_confirm(message, context)
        parse = getattr(self._parser, "parse", None)
        if callable(parse):
            try:
                parsed = await parse(
                    source_text,
                    now=self._clock.now(),
                    timezone=self._timezone,
                    safety_identifier=self._safety_identifier(update),
                )
            except (LlmParseError, LlmProviderError):
                parsed = None
        else:
            parsed = None
        if parsed is not None:
            merged = self._merge_drafts(draft, parsed.draft)
            if merged != draft:
                context.user_data[_DRAFT] = merged
                return await self._continue_or_confirm(message, context)
        try:
            draft = apply_form_answer(
                draft,
                field,
                source_text,
                today=self._clock.now().date(),
            )
        except DraftInputError as error:
            await message.reply_text(str(error))
            return State.FORM
        context.user_data[_DRAFT] = draft
        return await self._continue_or_confirm(message, context)

    async def confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        await query.answer()
        action = query.data or ""
        if action == "trip:cancel":
            context.user_data.clear()
            await query.edit_message_text("Поиск отменён. Для новой поездки используйте /newtrip.")
            return ConversationHandler.END
        if action == "trip:edit":
            await query.message.reply_text(
                "Что изменить? Остальные параметры сохранятся.",
                reply_markup=self._edit_keyboard(),
            )
            return State.CONFIRM
        if action.startswith("trip:editfield:"):
            field = action.rsplit(":", 1)[-1]
            if field not in {item[0] for item in EDITABLE_FIELDS}:
                await query.message.reply_text("Это поле больше недоступно.")
                return State.CONFIRM
            draft = context.user_data.get(_DRAFT)
            if not isinstance(draft, ParsedTripDraft):
                await query.message.reply_text("Параметры устарели. Начните заново: /newtrip")
                return ConversationHandler.END
            empty_value = frozenset() if field == "allowed_modes" else None
            context.user_data[_DRAFT] = draft.model_copy(update={field: empty_value})
            return await self._ask_field(query.message, context, field)
        request = context.user_data.get(_REQUEST)
        if action != "trip:confirm" or not isinstance(request, TripRequest):
            await query.edit_message_text("Параметры устарели. Начните заново: /newtrip")
            context.user_data.clear()
            return ConversationHandler.END
        await query.edit_message_reply_markup(reply_markup=None)
        return await self._run_search(query.message, context)

    async def result_control(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        parts = (query.data or "").split(":")
        search_id = context.user_data.get(_SEARCH_ID)
        if len(parts) != 3 or parts[2] != search_id:
            await query.answer("Это кнопка от предыдущего поиска", show_alert=True)
            return State.RESULTS
        await query.answer()
        if parts[1] == "retry":
            return await self._run_search(query.message, context)
        if parts[1] == "change":
            await query.message.reply_text(
                "Что изменить? Остальные параметры сохранятся.",
                reply_markup=self._edit_keyboard(),
            )
            return State.CONFIRM
        if parts[1] == "new":
            context.user_data.clear()
            await query.message.reply_text("Опишите новую поездку одним сообщением.")
            return State.INTAKE
        return State.RESULTS

    async def result_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        result = context.user_data.get(_RESULT)
        search_id = context.user_data.get(_SEARCH_ID)
        if not isinstance(result, TripSearchResult) or self._handoff is None:
            await query.answer("Результаты устарели", show_alert=True)
            return State.RESULTS
        parts = (query.data or "").split(":")
        try:
            action = parts[1]
            if parts[2] != search_id:
                await query.answer("Это кнопка от предыдущего поиска", show_alert=True)
                return State.RESULTS
            index = int(parts[3])
            await query.answer()
            if action == "components":
                components = self._handoff.available_components(result, index)
                await query.message.reply_text(
                    "Что показать подробнее?",
                    reply_markup=self._components_keyboard(search_id, index, components),
                )
                return State.RESULTS
            if action == "detail":
                logger.info(
                    "product_event",
                    extra={"event": "offer_details_requested", "component": parts[4]},
                )
                await self._track(
                    "proposal_details_opened",
                    context,
                    dimensions={"flow_type": "destination_known"},
                )
                if not self._allow(context, "details", limit=15, within=timedelta(minutes=1)):
                    await query.message.reply_text(
                        "Слишком много запросов деталей. Попробуйте через минуту."
                    )
                    return State.RESULTS
                component = TripComponent(parts[4])
                cache = context.user_data.setdefault(_DETAILS_CACHE, {})
                cache_key = f"{index}:{component.value}"
                details = cache.get(cache_key)
                if details is None:
                    status = await query.message.reply_text("Загружаю подробности на Tutu…")
                    details = await self._handoff.get_details(result, index, component)
                    cache[cache_key] = details
                    await status.edit_text(
                        format_details(component, details), parse_mode=ParseMode.HTML
                    )
                else:
                    await query.message.reply_text(
                        format_details(component, details), parse_mode=ParseMode.HTML
                    )
                return State.RESULTS
            if action == "checkout":
                logger.info("product_event", extra={"event": "handoff_requested"})
                await self._track(
                    "handoff_requested",
                    context,
                    dimensions={"flow_type": "destination_known"},
                )
                status = await query.message.reply_text(
                    self._voice.progress("handoff", context.user_data)
                )
                items = await self._handoff.create_checkout_items(result, index)
                buttons = [
                    [
                        InlineKeyboardButton(
                            f"{TripHandoffService.button_prefix(item)}: "
                            f"{COMPONENT_LABELS[item.component]}",
                            url=str(item.link.url),
                        )
                    ]
                    for item in items
                ]
                has_schedule_link = any(item.link.kind == "schedule_url" for item in items)
                has_broad_link = any(
                    not TripHandoffService.is_offer_specific(item)
                    and item.link.kind != "schedule_url"
                    for item in items
                )
                link_scope = ""
                if has_schedule_link:
                    link_scope += (
                        " Для электричек Tutu открывает расписание на выбранную дату, "
                        "поэтому сверьте время из рекомендации."
                    )
                if has_broad_link:
                    link_scope += " Для ссылок без deeplink откроется список вариантов."
                if not has_schedule_link and not has_broad_link:
                    link_scope = " Кнопки ведут к выбранным билетам и отелю."
                await status.edit_text(
                    "Вы перейдёте на Tutu для проверки актуальной цены и оформления. "
                    "Транспорт и отель могут оформляться отдельно; переход по ссылке "
                    "не резервирует места, а стоимость и доступность могут измениться."
                    + link_scope,
                    reply_markup=InlineKeyboardMarkup(buttons),
                )
                await self._track(
                    "handoff_created",
                    context,
                    dimensions={"flow_type": "destination_known"},
                )
                return State.RESULTS
        except (IndexError, ValueError, TutuAssistantError):
            await query.message.reply_text(
                "Предложение устарело или временно недоступно. Выполните новый поиск: /newtrip"
            )
        return State.RESULTS

    async def stale_result_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Acknowledge known-flow buttons when their ConversationHandler state is gone."""
        del context
        query = update.callback_query
        if query is not None:
            await query.answer(
                "Это кнопка от предыдущего поиска. Запустите новый поиск командой /newtrip.",
                show_alert=True,
            )

    async def _run_search(self, message, context) -> int:
        request = context.user_data.get(_REQUEST)
        if not isinstance(request, TripRequest):
            await message.reply_text("Параметры устарели. Начните заново: /newtrip")
            return ConversationHandler.END
        if not self._allow(context, "search", limit=5, within=timedelta(minutes=5)):
            await message.reply_text(
                "Лимит поисков временно исчерпан. Подождите несколько минут; параметры сохранены."
            )
            return State.CONFIRM
        progress = await message.reply_text(self._voice.progress("known_search", context.user_data))
        await self._track(
            "verification_started",
            context,
            dimensions={"flow_type": "destination_known", "candidate_count": 1},
        )
        logger.info("product_event", extra={"event": "search_started"})
        started = time.perf_counter()
        try:
            result = await self._planner.search_trip(request)
        except TutuAssistantError:
            logger.info(
                "product_event",
                extra={"event": "search_failed", "category": "expected_error"},
            )
            await progress.edit_text(
                "Не удалось выполнить поиск. Параметры сохранены — "
                "можно повторить или изменить их.",
                reply_markup=self._search_recovery_keyboard(),
            )
            return State.CONFIRM
        except Exception:
            logger.exception("unexpected trip search failure")
            await progress.edit_text(
                "Поиск временно недоступен. Параметры сохранены — попробуйте повторить позже.",
                reply_markup=self._search_recovery_keyboard(),
            )
            return State.CONFIRM
        search_id = secrets.token_hex(4)
        context.user_data[_RESULT] = result
        context.user_data[_DETAILS_CACHE] = {}
        context.user_data[_SEARCH_ID] = search_id
        await self._track(
            "verification_completed",
            context,
            dimensions={
                "flow_type": "destination_known",
                "verified_count": int(bool(result.options)),
                "failure_category": result.failures[0].category if result.failures else "none",
            },
        )
        if result.options:
            await self._track(
                "proposals_shown",
                context,
                dimensions={
                    "flow_type": "destination_known",
                    "proposal_count": len(result.options),
                    "price_completeness": _known_price_completeness(result),
                },
            )
        logger.info(
            "product_event",
            extra={
                "event": "search_completed",
                "result_count": len(result.options),
                "category": result.failures[0].category if result.failures else "success",
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        await progress.edit_text(
            format_results(result),
            parse_mode=ParseMode.HTML,
            reply_markup=self._results_keyboard(result, search_id),
        )
        return State.RESULTS

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        logger.info("product_event", extra={"event": "conversation_cancelled"})
        context.user_data.clear()
        await update.effective_message.reply_text("Диалог и введённые параметры удалены.")
        return ConversationHandler.END

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            HELP,
            parse_mode=ParseMode.HTML,
        )

    async def privacy(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(PRIVACY)

    async def feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            "Если результат оказался неверным, сохраните номер варианта и кратко "
            "опишите проблему владельцу бота. Не прикладывайте чувствительные данные."
        )

    async def unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.effective_message.reply_text(
            "Такой команды нет. Доступны /newtrip, /ideas, /help, /privacy, /feedback и /cancel."
        )

    async def timeout(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        logger.info("product_event", extra={"event": "conversation_timed_out"})
        context.user_data.clear()
        if update.effective_message is not None:
            await update.effective_message.reply_text(
                "Прошло 30 минут бездействия, поэтому параметры удалены. Начните заново: /newtrip"
            )
        return ConversationHandler.END

    async def _continue_or_confirm(self, message, context) -> int:
        draft = context.user_data[_DRAFT]
        if (
            draft.departure_date is not None
            and draft.return_date is not None
            and draft.return_date < draft.departure_date
        ):
            context.user_data[_DRAFT] = draft.model_copy(update={"return_date": None})
            await message.reply_text(
                "Дата возвращения не может быть раньше даты отправления. "
                "Маршрут и дата отправления сохранены."
            )
            return await self._ask_field(message, context, "return_date")
        if draft.departure_date is not None and draft.departure_date < self._clock.now().date():
            context.user_data[_DRAFT] = draft.model_copy(update={"departure_date": None})
            await message.reply_text("Дата отправления уже прошла. Остальные параметры сохранены.")
            return await self._ask_field(message, context, "departure_date")
        missing = next_missing_field(draft)
        if missing is not None:
            return await self._ask_field(message, context, missing)
        try:
            request = build_trip_request(draft, today=self._clock.now().date())
        except DraftInputError as error:
            field = error.field if error.field in PROMPTS else "departure_date"
            updates = {field: None}
            if field == "adults":
                updates.update(children=0, rooms=1)
            context.user_data[_DRAFT] = draft.model_copy(update=updates)
            await message.reply_text(str(error))
            return await self._ask_field(message, context, field)
        except (ValidationError, ValueError):
            await message.reply_text(
                "Не удалось проверить один из параметров. Уже введённые данные сохранены."
            )
            context.user_data[_DRAFT] = draft.model_copy(update={"departure_date": None})
            return await self._ask_field(message, context, "departure_date")
        context.user_data.pop(_PENDING_FIELD, None)
        context.user_data[_REQUEST] = request
        logger.info("product_event", extra={"event": "confirmation_shown"})
        await message.reply_text(
            format_confirmation(request),
            parse_mode=ParseMode.HTML,
            reply_markup=self._confirmation_keyboard(),
        )
        return State.CONFIRM

    async def _ask_field(self, message, context, field: str) -> int:
        context.user_data[_PENDING_FIELD] = field
        await message.reply_text(PROMPTS[field])
        return State.FORM

    @staticmethod
    def _confirmation_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Найти варианты", callback_data="trip:confirm")],
                [InlineKeyboardButton("Изменить параметры", callback_data="trip:edit")],
                [InlineKeyboardButton("Отменить", callback_data="trip:cancel")],
            ]
        )

    @staticmethod
    def _results_keyboard(result: TripSearchResult, search_id: str) -> InlineKeyboardMarkup:
        if not result.options:
            return InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Повторить поиск", callback_data=f"trip:retry:{search_id}"
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "Изменить параметры", callback_data=f"trip:change:{search_id}"
                        )
                    ],
                    [InlineKeyboardButton("Новая поездка", callback_data=f"trip:new:{search_id}")],
                ]
            )
        rows: list[list[InlineKeyboardButton]] = []
        for index in range(len(result.options)):
            rows.extend(
                (
                    [
                        InlineKeyboardButton(
                            f"Посмотреть детали · вариант {index + 1}",
                            callback_data=f"trip:components:{search_id}:{index}",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            f"Перейти к оформлению · вариант {index + 1}",
                            callback_data=f"trip:checkout:{search_id}:{index}",
                        )
                    ],
                )
            )
        rows.append(
            [InlineKeyboardButton("Изменить параметры", callback_data=f"trip:change:{search_id}")]
        )
        return InlineKeyboardMarkup(rows)

    @staticmethod
    def _components_keyboard(
        search_id: str,
        index: int,
        components: tuple[TripComponent, ...],
    ) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        COMPONENT_LABELS[component],
                        callback_data=(f"trip:detail:{search_id}:{index}:{component.value}"),
                    )
                ]
                for component in components
            ]
        )

    @staticmethod
    def _edit_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(label, callback_data=f"trip:editfield:{field}")]
                for field, label in EDITABLE_FIELDS
            ]
        )

    @staticmethod
    def _search_recovery_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Повторить поиск", callback_data="trip:confirm")],
                [InlineKeyboardButton("Изменить параметры", callback_data="trip:edit")],
            ]
        )

    def _allow(self, context, key: str, *, limit: int, within: timedelta) -> bool:
        now = self._clock.now()
        events = context.user_data.setdefault(_RATE_EVENTS, {}).setdefault(key, [])
        threshold = now - within
        events[:] = [item for item in events if item > threshold]
        if len(events) >= limit:
            return False
        events.append(now)
        return True

    @staticmethod
    def _merge_drafts(current: ParsedTripDraft, patch: ParsedTripDraft) -> ParsedTripDraft:
        updates = {}
        for field in patch.__class__.model_fields:
            value = getattr(patch, field)
            if value is None:
                continue
            if isinstance(value, frozenset) and not value:
                continue
            if isinstance(value, bool) and value is False:
                continue
            updates[field] = value
        return current.model_copy(update=updates)

    def _safety_identifier(self, update: Update) -> str | None:
        user = getattr(update, "effective_user", None)
        user_id = getattr(user, "id", None)
        if user_id is None:
            return None
        return hmac.new(
            self._safety_salt,
            str(user_id).encode(),
            hashlib.sha256,
        ).hexdigest()

    async def _track(self, name, context, *, intent=None, dimensions=None) -> None:
        if self._analytics is None:
            return
        flow_id = context.user_data.get(_ANALYTICS_FLOW_ID)
        if not isinstance(flow_id, str) or len(flow_id) < 8:
            flow_id = secrets.token_hex(4)
            context.user_data[_ANALYTICS_FLOW_ID] = flow_id
        await self._analytics.track(
            name,
            flow_id=flow_id,
            intent=intent,
            dimensions={**self._voice.analytics_dimensions, **(dimensions or {})},
        )


def _known_price_completeness(result: TripSearchResult) -> str:
    prices = tuple(item.combination.price for item in result.options)
    if prices and all(
        not item.missing_components and item.known_total is not None for item in prices
    ):
        return "exact"
    if any(item.known_total is not None for item in prices):
        return "partial"
    return "unknown"


def build_conversation_handler(conversation: TripConversation) -> ConversationHandler:
    intake_text = MessageHandler(filters.TEXT & ~filters.COMMAND, conversation.intake)
    form_text = MessageHandler(filters.TEXT & ~filters.COMMAND, conversation.form_input)
    modify_text = MessageHandler(filters.TEXT & ~filters.COMMAND, conversation.modify_text)
    common_commands = [
        CommandHandler("help", conversation.help),
        CommandHandler("privacy", conversation.privacy),
        CommandHandler("feedback", conversation.feedback),
    ]
    # Conversation state is per chat/user, not per message. Versioned callbacks and the
    # per-chat update processor prevent stale-message and concurrent-state collisions.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="If 'per_message=False'.*",
            category=PTBUserWarning,
        )
        return ConversationHandler(
            entry_points=[
                CommandHandler("start", conversation.start),
                CommandHandler("newtrip", conversation.start),
                *common_commands,
            ],
            states={
                State.INTAKE: [intake_text],
                State.FORM: [form_text],
                State.CONFIRM: [
                    CallbackQueryHandler(
                        conversation.confirm,
                        pattern=r"^trip:(confirm|edit|cancel|editfield:[a-z_]+)$",
                    ),
                    modify_text,
                ],
                State.RESULTS: [
                    CallbackQueryHandler(
                        conversation.result_action,
                        pattern=KNOWN_RESULT_CALLBACK_PATTERN,
                    ),
                    CallbackQueryHandler(
                        conversation.result_control,
                        pattern=KNOWN_CONTROL_CALLBACK_PATTERN,
                    ),
                    modify_text,
                ],
                ConversationHandler.TIMEOUT: [MessageHandler(filters.ALL, conversation.timeout)],
            },
            fallbacks=[
                CommandHandler("cancel", conversation.cancel),
                CommandHandler("start", conversation.start),
                CommandHandler("newtrip", conversation.start),
                *common_commands,
                CallbackQueryHandler(
                    conversation.stale_result_callback,
                    pattern=KNOWN_ANY_RESULT_CALLBACK_PATTERN,
                ),
                MessageHandler(filters.COMMAND, conversation.unknown_command),
            ],
            allow_reentry=True,
            conversation_timeout=30 * 60,
            name="short_trip",
        )
