"""Shared Telegram intent router and combined known/discovery state machine."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import warnings
from enum import IntEnum

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

from app.bot.conversation import (
    KNOWN_CONTROL_CALLBACK_PATTERN,
    KNOWN_RESULT_CALLBACK_PATTERN,
    State,
    TripConversation,
)
from app.bot.discovery_conversation import DiscoveryConversation, DiscoveryState
from app.bot.feedback_conversation import FeedbackConversation, FeedbackState
from app.domain.discovery_models import TripIntent
from app.domain.errors import LlmParseError, LlmProviderError
from app.ports.clock import Clock
from app.ports.discovery_llm import ConversationContext, IntentExtractor
from app.services.input_safety import SENSITIVE_DATA_RESPONSE, contains_sensitive_data
from app.services.product_analytics import ProductAnalytics
from app.voice import Voice

_FLOW_ID = "flow_id"
_INTENT_TRACKED = "intent_analytics_tracked"
logger = logging.getLogger(__name__)


class RouterState(IntEnum):
    INTAKE = 10


class BotRouter:
    def __init__(
        self,
        extractor: IntentExtractor,
        known: TripConversation,
        discovery: DiscoveryConversation,
        clock: Clock,
        *,
        timezone: str,
        analytics: ProductAnalytics | None = None,
        enabled: bool = True,
        voice: Voice | None = None,
    ) -> None:
        self._extractor = extractor
        self._known = known
        self._discovery = discovery
        self._clock = clock
        self._timezone = timezone
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
        context.user_data[_FLOW_ID] = secrets.token_hex(4)
        await self._track(
            "conversation_started",
            context,
            dimensions={"flow_type": "router"},
        )
        await update.effective_message.reply_text(
            self._voice.router_start,
            parse_mode=ParseMode.HTML,
            reply_markup=self._route_keyboard(),
        )
        return RouterState.INTAKE

    async def intake(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        message = update.effective_message
        if contains_sensitive_data(message.text or ""):
            await message.reply_text(SENSITIVE_DATA_RESPONSE)
            return RouterState.INTAKE
        progress = await message.reply_text(self._voice.progress("route", context.user_data))
        try:
            parsed = await self._extractor.extract(
                message.text or "",
                context=ConversationContext(
                    timezone=self._timezone,
                    current_date=self._clock.now().date().isoformat(),
                ),
                safety_identifier=self._safety_identifier(update),
            )
        except LlmParseError:
            logger.warning(
                "router_intent_mapping_failed",
                extra={"event": "router_intent_mapping_failed"},
            )
            await progress.edit_text(
                "Произошла техническая ошибка при разборе запроса. Выберите сценарий вручную:",
                reply_markup=self._route_keyboard(),
            )
            return RouterState.INTAKE
        except LlmProviderError:
            await progress.edit_text(
                "Сервис распознавания временно недоступен. Выберите сценарий вручную:",
                reply_markup=self._route_keyboard(),
            )
            return RouterState.INTAKE
        await self._track(
            "intent_classified",
            context,
            intent=parsed.intent,
            dimensions={"flow_type": parsed.intent.value},
        )
        context.user_data[_INTENT_TRACKED] = True
        if parsed.intent is TripIntent.DESTINATION_KNOWN or (
            parsed.intent is TripIntent.EVENT_LED and parsed.known_draft is not None
        ):
            if parsed.known_draft is None:
                await progress.edit_text("Не удалось извлечь маршрут. Попробуйте /newtrip.")
                return RouterState.INTAKE
            await progress.edit_text("Маршрут вижу. Осталось проверить параметры перед поиском.")
            return await self._known.accept_draft(message, context, parsed.known_draft)
        await progress.edit_text("Город пока не выбран — соберу идеи под ваши пожелания.")
        return await self._discovery.accept_result(message, context, parsed)

    async def route_choice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        await query.answer()
        if query.data == "route:known":
            context.user_data.clear()
            await query.message.reply_text("Напишите маршрут, даты и ограничения одним сообщением.")
            return State.INTAKE
        return await self._discovery.start(update, context)

    async def orphan_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Give plain text outside an active conversation an explicit next step."""

        del context
        if contains_sensitive_data(update.effective_message.text or ""):
            await update.effective_message.reply_text(SENSITIVE_DATA_RESPONSE)
            return
        await update.effective_message.reply_text(
            "Активного диалога нет. Выберите, что хотите сделать:",
            reply_markup=self._route_keyboard(),
        )

    @staticmethod
    def _route_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Проверить готовый маршрут", callback_data="route:known")],
                [InlineKeyboardButton("Подобрать идею", callback_data="route:ideas")],
            ]
        )

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
        await self._analytics.track(
            name,
            flow_id=context.user_data[_FLOW_ID],
            intent=intent,
            dimensions={**self._voice.analytics_dimensions, **(dimensions or {})},
        )


def build_routed_conversation_handler(
    router: BotRouter,
    known: TripConversation,
    discovery: DiscoveryConversation,
    feedback: FeedbackConversation,
) -> ConversationHandler:
    text = filters.TEXT & ~filters.COMMAND
    common_commands = [
        CommandHandler("help", known.help),
        CommandHandler("privacy", known.privacy),
        CommandHandler("feedback", feedback.start),
        CommandHandler("deletefeedback", feedback.delete),
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="If 'per_message=False'.*",
            category=PTBUserWarning,
        )
        return ConversationHandler(
            entry_points=[
                CommandHandler("start", router.start),
                CommandHandler("newtrip", known.start),
                CommandHandler("ideas", discovery.start),
                *common_commands,
            ],
            states={
                RouterState.INTAKE: [
                    MessageHandler(text, router.intake),
                    CallbackQueryHandler(router.route_choice, pattern=r"^route:(known|ideas)$"),
                ],
                State.INTAKE: [MessageHandler(text, known.intake)],
                State.FORM: [MessageHandler(text, known.form_input)],
                State.CONFIRM: [
                    CallbackQueryHandler(
                        known.confirm,
                        pattern=r"^trip:(confirm|edit|cancel|editfield:[a-z_]+)$",
                    ),
                    MessageHandler(text, known.modify_text),
                ],
                State.RESULTS: [
                    CallbackQueryHandler(
                        known.result_action,
                        pattern=KNOWN_RESULT_CALLBACK_PATTERN,
                    ),
                    CallbackQueryHandler(
                        known.result_control,
                        pattern=KNOWN_CONTROL_CALLBACK_PATTERN,
                    ),
                    MessageHandler(text, known.modify_text),
                ],
                DiscoveryState.INTAKE: [MessageHandler(text, discovery.intake)],
                DiscoveryState.CLARIFY: [MessageHandler(text, discovery.clarification_input)],
                DiscoveryState.RESULTS: [
                    CallbackQueryHandler(discovery.callback, pattern=r"^d1:"),
                    MessageHandler(text, discovery.refine_input),
                ],
                DiscoveryState.REFINE: [MessageHandler(text, discovery.refine_input)],
                FeedbackState.REASON: [
                    CallbackQueryHandler(
                        feedback.choose_reason,
                        pattern=r"^feedback:reason:[a-z_]+$",
                    )
                ],
                FeedbackState.COMMENT: [
                    CallbackQueryHandler(feedback.skip_comment, pattern=r"^feedback:skip$"),
                    MessageHandler(text, feedback.comment),
                ],
                ConversationHandler.TIMEOUT: [MessageHandler(filters.ALL, known.timeout)],
            },
            fallbacks=[
                CommandHandler("cancel", known.cancel),
                CommandHandler("start", router.start),
                CommandHandler("newtrip", known.start),
                CommandHandler("ideas", discovery.start),
                *common_commands,
                CallbackQueryHandler(
                    known.stale_result_callback,
                    pattern=r"^trip:",
                ),
                CallbackQueryHandler(discovery.stale_callback, pattern=r"^d1:"),
                MessageHandler(filters.COMMAND, known.unknown_command),
            ],
            allow_reentry=True,
            conversation_timeout=30 * 60,
            name="short_trip",
        )
