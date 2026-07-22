"""Receipt-based Telegram feedback and deletion flow."""

from __future__ import annotations

import secrets
from enum import IntEnum

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes, ConversationHandler

from app.bot.conversation import State
from app.bot.discovery_conversation import DiscoveryState
from app.domain.product_models import FeedbackReason
from app.services.feedback_service import FeedbackService
from app.services.input_safety import SENSITIVE_DATA_RESPONSE, contains_sensitive_data

_REASON = "feedback_reason"
_FLOW_ID = "feedback_flow_id"
_RETURN_STATE = "feedback_return_state"


class FeedbackState(IntEnum):
    REASON = 201
    COMMENT = 202


REASON_LABELS = {
    FeedbackReason.IRRELEVANT: "Варианты не соответствуют запросу",
    FeedbackReason.TOO_EXPENSIVE: "Слишком дорого",
    FeedbackReason.BAD_LOGISTICS: "Неудобная дорога",
    FeedbackReason.UNINTERESTING: "Неинтересная программа",
    FeedbackReason.WRONG_DATA: "Неверные данные",
    FeedbackReason.TECHNICAL_PROBLEM: "Техническая проблема",
    FeedbackReason.OTHER: "Другое",
}


class FeedbackConversation:
    def __init__(
        self,
        service: FeedbackService,
        *,
        retention_days: int = 30,
        enabled: bool = True,
    ) -> None:
        if not 1 <= retention_days <= 365:
            raise ValueError("feedback retention must be between 1 and 365 days")
        self._service = service
        self._retention_days = retention_days
        self._enabled = enabled

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not self._enabled:
            await update.effective_message.reply_text(
                "Обратная связь в этой версии временно недоступна. Команда сохранена и "
                "заработает после подключения постоянного защищённого хранилища."
            )
            return ConversationHandler.END
        self._flow_id(context)
        context.user_data[_RETURN_STATE] = self._detect_return_state(context)
        context.user_data.pop(_REASON, None)
        await update.effective_message.reply_text(
            "Что пошло не так? Выберите основную причину. Обращение хранится "
            f"не более {self._retention_days} дней; не отправляйте паспортные или "
            "платёжные данные.",
            reply_markup=self._reason_keyboard(),
        )
        return FeedbackState.REASON

    async def choose_reason(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        await query.answer()
        raw_reason = (query.data or "").removeprefix("feedback:reason:")
        try:
            reason = FeedbackReason(raw_reason)
        except ValueError:
            await query.message.reply_text("Эта причина больше не поддерживается.")
            return FeedbackState.REASON
        context.user_data[_REASON] = reason
        await query.message.reply_text(
            "Добавьте короткий комментарий без личных данных или отправьте без комментария.",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Без комментария", callback_data="feedback:skip")]]
            ),
        )
        return FeedbackState.COMMENT

    async def comment(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if contains_sensitive_data(update.effective_message.text or ""):
            await update.effective_message.reply_text(SENSITIVE_DATA_RESPONSE)
            return FeedbackState.COMMENT
        return await self._submit(update.effective_message, context, update.effective_message.text)

    async def skip_comment(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        await query.answer()
        return await self._submit(query.message, context, None)

    async def delete(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        receipt_id = context.args[0].strip() if context.args else ""
        if not receipt_id:
            await update.effective_message.reply_text(
                "Укажите номер обращения: /deletefeedback fb_…"
            )
            return
        deleted = await self._service.delete(receipt_id)
        if deleted is None:
            text = "Сервис удаления временно недоступен. Попробуйте позже."
        elif deleted:
            text = "Обращение удалено. Восстановить его невозможно."
        else:
            text = "Обращение с таким номером не найдено или уже удалено."
        await update.effective_message.reply_text(text)

    async def _submit(self, message, context, comment: str | None) -> int:
        reason = context.user_data.get(_REASON)
        if not isinstance(reason, FeedbackReason):
            await message.reply_text("Причина не выбрана. Начните снова: /feedback")
            return FeedbackState.REASON
        receipt = await self._service.submit(
            flow_id=self._flow_id(context),
            reason=reason,
            comment=comment,
        )
        context.user_data.pop(_REASON, None)
        if receipt is None:
            await message.reply_text(
                "Не удалось сохранить обращение. Ничего не потеряно в текущей поездке; "
                "попробуйте /feedback позже."
            )
        else:
            await message.reply_text(
                "Спасибо, обращение сохранено.\n"
                f"Номер: {receipt}\n"
                f"Удалить раньше срока: /deletefeedback {receipt}"
            )
        return_state = context.user_data.pop(_RETURN_STATE, None)
        return return_state if isinstance(return_state, int) else ConversationHandler.END

    @staticmethod
    def _reason_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        label,
                        callback_data=f"feedback:reason:{reason.value}",
                    )
                ]
                for reason, label in REASON_LABELS.items()
            ]
        )

    @staticmethod
    def _flow_id(context) -> str:
        for key in ("discovery_flow_id", "flow_id", "trip_search_id", _FLOW_ID):
            value = context.user_data.get(key)
            if isinstance(value, str) and len(value) >= 8:
                context.user_data[_FLOW_ID] = value
                return value
        value = secrets.token_hex(4)
        context.user_data[_FLOW_ID] = value
        return value

    @staticmethod
    def _detect_return_state(context) -> int | None:
        if "discovery_proposals" in context.user_data:
            return DiscoveryState.RESULTS
        if "trip_result" in context.user_data:
            return State.RESULTS
        return None
