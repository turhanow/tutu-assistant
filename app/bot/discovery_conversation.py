"""Telegram application layer for the destination-unknown inspiration journey."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from enum import IntEnum

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes, ConversationHandler

from app.bot.callbacks import CallbackCodec, DiscoveryAction, DiscoveryCallback
from app.bot.discovery_formatters import (
    format_program,
    format_proposal_sections,
    format_shortlist,
    pack_html_sections,
)
from app.bot.formatters import COMPONENT_LABELS
from app.domain.discovery_models import (
    CandidateShortlist,
    DiscoveryDraft,
    DiscoveryFeasibilityResult,
    DiscoveryProposalResult,
    ExperienceProfile,
    IntentParseResult,
    RoadTolerance,
    TripIntent,
)
from app.domain.errors import (
    LlmParseError,
    LlmProviderError,
    TutuAssistantError,
    UnsupportedOriginError,
)
from app.domain.models import TripSearchResult
from app.ports.clock import Clock
from app.ports.discovery_llm import ConversationContext, IntentExtractor
from app.services.candidate_selector import CandidateSelector
from app.services.clarification_policy import plan_clarifications
from app.services.discovery_planner import DiscoveryPlanner
from app.services.discovery_request_builder import (
    DiscoveryInputError,
    build_discovery_request,
    missing_discovery_fields,
)
from app.services.product_analytics import ProductAnalytics
from app.services.proposal_builder import (
    GroundedProposalNarration,
    ProposalBuilder,
    fallback_proposal_copies,
    project_grounded_facts,
)
from app.services.trip_handoff import TripHandoffService
from app.voice import Voice

logger = logging.getLogger(__name__)

_DRAFT = "discovery_draft"
_FLOW_ID = "discovery_flow_id"
_REVISION = "discovery_revision"
_SHORTLIST = "discovery_shortlist"
_FEASIBILITY = "discovery_feasibility"
_PROPOSALS = "discovery_proposals"
_REJECTED_INDEX = "discovery_rejected_index"
_ANALYTICS_FLOW_ID = "flow_id"
_INTENT_TRACKED = "intent_analytics_tracked"
_REJECT_REASONS = frozenset({"too_expensive", "bad_logistics", "uninteresting", "other"})


class DiscoveryState(IntEnum):
    INTAKE = 101
    CLARIFY = 102
    RESULTS = 103
    REFINE = 104


class DiscoveryConversation:
    def __init__(
        self,
        extractor: IntentExtractor,
        selector: CandidateSelector,
        planner: DiscoveryPlanner,
        proposal_builder: ProposalBuilder,
        narration: GroundedProposalNarration,
        clock: Clock,
        *,
        timezone: str,
        handoff: TripHandoffService | None = None,
        analytics: ProductAnalytics | None = None,
        enabled: bool = True,
        voice: Voice | None = None,
    ) -> None:
        self._extractor = extractor
        self._selector = selector
        self._planner = planner
        self._proposal_builder = proposal_builder
        self._narration = narration
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
                "Подбор направлений временно отключён. Готовый маршрут можно проверить "
                "через /newtrip."
            )
            return ConversationHandler.END
        context.user_data.clear()
        self._initialize_flow(context)
        await self._track(
            "conversation_started",
            context,
            dimensions={"flow_type": "destination_unknown"},
        )
        await update.effective_message.reply_text(
            self._voice.discovery_start,
            parse_mode=ParseMode.HTML,
        )
        return DiscoveryState.INTAKE

    async def intake(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        message = update.effective_message
        progress = await message.reply_text(
            self._voice.progress("discovery_parse", context.user_data)
        )
        try:
            parsed = await self._extract(message.text or "", update)
        except LlmParseError:
            logger.warning(
                "discovery_intent_mapping_failed",
                extra={"event": "discovery_intent_mapping_failed"},
            )
            await progress.edit_text(
                "Произошла техническая ошибка при разборе запроса. Повторите его ещё раз; "
                "если ошибка сохранится, сообщите через /feedback."
            )
            return DiscoveryState.INTAKE
        except LlmProviderError:
            await progress.edit_text(
                "Сервис распознавания временно недоступен. Запрос можно повторить через минуту."
            )
            return DiscoveryState.INTAKE
        return await self.accept_result(message, context, parsed, progress=progress)

    async def accept_result(
        self,
        message,
        context,
        parsed: IntentParseResult,
        *,
        progress=None,
    ) -> int:
        if not self._enabled:
            text = "Подбор направлений временно отключён. Используйте /newtrip."
            if progress is not None:
                await progress.edit_text(text)
            else:
                await message.reply_text(text)
            return ConversationHandler.END
        if parsed.intent is TripIntent.DESTINATION_KNOWN or parsed.discovery_draft is None:
            text = (
                "В запросе уже выбрано направление. Для такого сценария используйте "
                "/newtrip — там я сравню варианты дороги и проживания."
            )
            if progress is not None:
                await progress.edit_text(text)
            else:
                await message.reply_text(text)
            return DiscoveryState.INTAKE
        self._ensure_flow(context)
        if not context.user_data.pop(_INTENT_TRACKED, False):
            await self._track(
                "intent_classified",
                context,
                intent=parsed.intent,
                dimensions={"flow_type": parsed.intent.value},
            )
        context.user_data[_DRAFT] = parsed.discovery_draft
        if progress is not None:
            await progress.edit_text("Понял настроение поездки. Проверяю, хватает ли вводных.")
        return await self._continue(message, context)

    async def clarification_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> int:
        message = update.effective_message
        progress = await message.reply_text(
            self._voice.progress("discovery_clarify", context.user_data)
        )
        try:
            parsed = await self._extract(message.text or "", update)
        except LlmParseError:
            logger.warning(
                "discovery_clarification_mapping_failed",
                extra={"event": "discovery_clarification_mapping_failed"},
            )
            await progress.edit_text(
                "Произошла техническая ошибка при обработке ответа. Повторите его ещё раз."
            )
            return DiscoveryState.CLARIFY
        except LlmProviderError:
            await progress.edit_text(
                "Сервис распознавания временно недоступен. Ответ можно повторить через минуту."
            )
            return DiscoveryState.CLARIFY
        patch = parsed.discovery_draft
        current = context.user_data.get(_DRAFT)
        if patch is None or not isinstance(current, DiscoveryDraft):
            await progress.edit_text("Не получилось связать ответы с запросом. Попробуйте ещё раз.")
            return DiscoveryState.CLARIFY
        context.user_data[_DRAFT] = _merge_discovery_drafts(current, patch)
        await self._track(
            "clarification_answered",
            context,
            dimensions={"flow_type": "destination_unknown"},
        )
        await progress.edit_text("Ответы добавлены к подборке.")
        return await self._continue(message, context)

    async def refine_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        message = update.effective_message
        current = context.user_data.get(_DRAFT)
        if not isinstance(current, DiscoveryDraft):
            await message.reply_text("Параметры устарели. Начните заново: /ideas")
            return ConversationHandler.END
        progress = await message.reply_text(
            self._voice.progress("discovery_refine", context.user_data)
        )
        try:
            parsed = await self._extract(message.text or "", update)
        except LlmParseError:
            logger.warning(
                "discovery_refinement_mapping_failed",
                extra={"event": "discovery_refinement_mapping_failed"},
            )
            await progress.edit_text(
                "Произошла техническая ошибка при обработке изменения. Повторите его ещё раз."
            )
            return DiscoveryState.REFINE
        except LlmProviderError:
            await progress.edit_text(
                "Сервис распознавания временно недоступен. Изменение можно повторить через минуту."
            )
            return DiscoveryState.REFINE
        if parsed.discovery_draft is None:
            await progress.edit_text("Не получилось применить изменение к текущей подборке.")
            return DiscoveryState.REFINE
        context.user_data[_DRAFT] = _merge_discovery_drafts(
            current,
            parsed.discovery_draft,
        )
        self._advance_revision(context)
        await progress.edit_text("Пожелания обновлены. Пересобираю направления.")
        return await self._continue(message, context)

    async def callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        try:
            callback = CallbackCodec.decode(query.data or "")
        except ValueError:
            await query.answer("Кнопка повреждена или больше не поддерживается", show_alert=True)
            return DiscoveryState.RESULTS
        if callback.flow_id != context.user_data.get(
            _FLOW_ID
        ) or callback.revision != context.user_data.get(_REVISION):
            await query.answer("Это кнопка от предыдущей версии подборки", show_alert=True)
            return DiscoveryState.RESULTS
        await query.answer()
        if callback.action is DiscoveryAction.DETAILS:
            recommendation = self._recommendation(context, callback.argument)
            if recommendation is None:
                await query.message.reply_text("Этот вариант больше недоступен.")
            else:
                await self._track(
                    "proposal_details_opened",
                    context,
                    dimensions={"flow_type": "destination_unknown"},
                )
                await query.message.reply_text(
                    format_program(recommendation),
                    parse_mode=ParseMode.HTML,
                )
            return DiscoveryState.RESULTS
        if callback.action is DiscoveryAction.REFINE:
            await query.message.reply_text(
                "Что изменить? Например: «дешевле», «меньше времени в дороге» или «больше природы»."
            )
            return DiscoveryState.REFINE
        if callback.action is DiscoveryAction.RECHECK:
            shortlist = context.user_data.get(_SHORTLIST)
            if not isinstance(shortlist, CandidateShortlist):
                await query.message.reply_text("Подборка устарела. Начните заново: /ideas")
                return ConversationHandler.END
            self._advance_revision(context)
            await self._track(
                "price_rechecked",
                context,
                dimensions={
                    "flow_type": "destination_unknown",
                    "revision": context.user_data[_REVISION],
                },
            )
            progress = await query.message.reply_text(
                self._voice.progress("discovery_recheck", context.user_data)
            )
            return await self._verify_shortlist(
                query.message,
                context,
                shortlist,
                progress,
                use_llm_narration=False,
            )
        if callback.action is DiscoveryAction.REJECT:
            recommendation = self._recommendation(context, callback.argument)
            if recommendation is None:
                await query.message.reply_text("Этот вариант больше недоступен.")
                return DiscoveryState.RESULTS
            context.user_data[_REJECTED_INDEX] = int(callback.argument or "-1")
            await query.message.reply_text(
                "Что именно не подошло?",
                reply_markup=self._reject_reason_keyboard(context),
            )
            return DiscoveryState.RESULTS
        if callback.action is DiscoveryAction.REJECT_REASON:
            if callback.argument not in _REJECT_REASONS:
                await query.message.reply_text("Эта причина больше не поддерживается.")
                return DiscoveryState.RESULTS
            logger.info(
                "product_event",
                extra={"event": "discovery_proposal_rejected", "category": callback.argument},
            )
            await self._track(
                "proposal_rejected",
                context,
                dimensions={
                    "flow_type": "destination_unknown",
                    "reason_code": callback.argument,
                },
            )
            context.user_data.pop(_REJECTED_INDEX, None)
            await query.message.reply_text(
                "Спасибо, учту причину в следующей итерации. Чтобы изменить подборку, "
                "нажмите «Уточнить пожелания»."
            )
            return DiscoveryState.RESULTS
        if callback.action is DiscoveryAction.HANDOFF:
            await self._track(
                "handoff_requested",
                context,
                dimensions={"flow_type": "destination_unknown"},
            )
            await self._handoff_option(query.message, context, callback.argument)
            return DiscoveryState.RESULTS
        if callback.action is DiscoveryAction.NEW:
            return await self.start(update, context)
        return DiscoveryState.RESULTS

    async def _continue(self, message, context) -> int:
        draft = context.user_data.get(_DRAFT)
        if not isinstance(draft, DiscoveryDraft):
            await message.reply_text("Опишите идею поездки заново: /ideas")
            return ConversationHandler.END
        missing = missing_discovery_fields(draft)
        if missing:
            questions = plan_clarifications(draft)
            lines = ["Чтобы подборка получилась точнее, уточните несколько вещей:"]
            lines.extend(f"{index}. {item.text}" for index, item in enumerate(questions, start=1))
            lines.append("Ответьте одним сообщением по этим пунктам.")
            await self._track(
                "clarification_shown",
                context,
                dimensions={"flow_type": "destination_unknown"},
            )
            await message.reply_text("\n".join(lines))
            return DiscoveryState.CLARIFY
        try:
            request = build_discovery_request(draft, today=self._clock.now().date())
        except DiscoveryInputError as error:
            await message.reply_text(f"Не удалось проверить параметры: {error}")
            return DiscoveryState.CLARIFY
        progress = await message.reply_text(
            self._voice.progress("discovery_select", context.user_data)
        )
        try:
            shortlist = await self._selector.select(request)
        except UnsupportedOriginError as error:
            await progress.edit_text(str(error))
            return DiscoveryState.RESULTS
        except Exception:
            logger.exception("destination shortlist failed")
            await progress.edit_text(
                "Не удалось собрать направления. Параметры сохранены — попробуйте позднее."
            )
            return DiscoveryState.RESULTS
        context.user_data[_SHORTLIST] = shortlist
        await self._track(
            "shortlist_generated" if shortlist.candidates else "shortlist_empty",
            context,
            dimensions={
                "flow_type": "destination_unknown",
                "catalog_version": shortlist.catalog_version,
                "candidate_count": len(shortlist.candidates),
            },
        )
        await progress.edit_text(format_shortlist(shortlist), parse_mode=ParseMode.HTML)
        if not shortlist.candidates:
            await message.reply_text(
                "Можно изменить пожелания или начать заново.",
                reply_markup=self._result_controls_keyboard(context),
            )
            return DiscoveryState.RESULTS
        verification = await message.reply_text(
            self._voice.progress("discovery_verify", context.user_data)
        )
        return await self._verify_shortlist(message, context, shortlist, verification)

    async def _verify_shortlist(
        self,
        message,
        context,
        shortlist,
        progress,
        *,
        use_llm_narration: bool = True,
    ) -> int:
        await self._track(
            "verification_started",
            context,
            dimensions={
                "flow_type": "destination_unknown",
                "candidate_count": len(shortlist.candidates),
            },
        )
        try:
            feasibility = await self._planner.verify(shortlist)
            proposals = await self._proposal_builder.build(feasibility)
        except Exception:
            logger.exception("discovery verification failed")
            await progress.edit_text(
                "Проверка направлений временно недоступна. Можно повторить её без ввода "
                "параметров заново.",
                reply_markup=self._result_controls_keyboard(context),
            )
            return DiscoveryState.RESULTS
        context.user_data[_FEASIBILITY] = feasibility
        context.user_data[_PROPOSALS] = proposals
        verified_count = sum(
            item.status.value in {"verified", "partial"} for item in feasibility.snapshots
        )
        for _ in range(verified_count):
            await self._track(
                "destination_verified",
                context,
                dimensions={"flow_type": "destination_unknown"},
            )
        await self._track(
            "verification_completed",
            context,
            dimensions={
                "flow_type": "destination_unknown",
                "verified_count": verified_count,
                "failure_category": "none" if proposals.recommendations else "no_proposals",
            },
        )
        if not proposals.recommendations:
            await progress.edit_text(
                "На эти даты не удалось подтвердить целостную поездку. Попробуйте изменить "
                "ограничения или повторить проверку позже.",
                reply_markup=self._result_controls_keyboard(context),
            )
            return DiscoveryState.RESULTS
        if use_llm_narration:
            copies = await self._narration.narrate(
                proposals.recommendations,
                context=ConversationContext(
                    timezone=self._timezone,
                    current_date=self._clock.now().date().isoformat(),
                ),
            )
        else:
            copies = fallback_proposal_copies(project_grounded_facts(proposals.recommendations))
        messages = pack_html_sections(format_proposal_sections(proposals.recommendations, copies))
        await self._track(
            "proposals_shown",
            context,
            dimensions={
                "flow_type": "destination_unknown",
                "proposal_count": len(proposals.recommendations),
                "price_completeness": _price_completeness(proposals),
            },
        )
        for index, text in enumerate(messages):
            keyboard = (
                self._proposals_keyboard(context, proposals) if index == len(messages) - 1 else None
            )
            if index == 0:
                await progress.edit_text(
                    text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=keyboard,
                )
            else:
                await message.reply_text(
                    text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=keyboard,
                )
        return DiscoveryState.RESULTS

    async def _handoff_option(self, message, context, argument: str | None) -> None:
        recommendation = self._recommendation(context, argument)
        feasibility = context.user_data.get(_FEASIBILITY)
        if (
            recommendation is None
            or not isinstance(feasibility, DiscoveryFeasibilityResult)
            or self._handoff is None
        ):
            await message.reply_text("Переход к оформлению для этого варианта недоступен.")
            return
        destination_id = recommendation.proposal.candidate.destination.destination_id
        snapshot = next(
            (item for item in feasibility.snapshots if item.destination_id == destination_id),
            None,
        )
        if snapshot is None or snapshot.trip_result is None:
            await message.reply_text("Данные предложения устарели. Выполните повторную проверку.")
            return
        selected = TripSearchResult(
            request=snapshot.trip_result.request,
            options=(recommendation.proposal.trip_option,),
            failures=snapshot.trip_result.failures,
            searched_at=snapshot.trip_result.searched_at,
        )
        status = await message.reply_text(self._voice.progress("handoff", context.user_data))
        try:
            items = await self._handoff.create_checkout_items(selected, 0)
        except (TutuAssistantError, ValueError, IndexError):
            await status.edit_text(
                "Ссылки уже неактуальны. Повторно проверьте вариант перед оформлением."
            )
            return
        buttons = [
            [
                InlineKeyboardButton(
                    f"Открыть на Tutu · {COMPONENT_LABELS[item.component]}",
                    url=str(item.link.url),
                )
            ]
            for item in items
        ]
        await status.edit_text(
            "Проверьте актуальную цену и условия на Tutu. Компоненты могут оформляться "
            "отдельно; переход не резервирует места.",
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        await self._track(
            "handoff_created",
            context,
            dimensions={"flow_type": "destination_unknown"},
        )

    def _recommendation(self, context, argument: str | None):
        proposals = context.user_data.get(_PROPOSALS)
        if not isinstance(proposals, DiscoveryProposalResult) or argument is None:
            return None
        try:
            index = int(argument)
        except (ValueError, IndexError):
            return None
        if index < 0:
            return None
        try:
            return proposals.recommendations[index]
        except IndexError:
            return None

    def _proposals_keyboard(self, context, result) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = []
        for index, recommendation in enumerate(result.recommendations):
            name = recommendation.proposal.candidate.destination.name[:24]
            rows.append(
                [
                    InlineKeyboardButton(
                        f"План на два дня · {name}",
                        callback_data=self._callback(context, DiscoveryAction.DETAILS, str(index)),
                    ),
                    InlineKeyboardButton(
                        "К оформлению",
                        callback_data=self._callback(context, DiscoveryAction.HANDOFF, str(index)),
                    ),
                ]
            )
            rows.append(
                [
                    InlineKeyboardButton(
                        f"Почему не подходит · {name}",
                        callback_data=self._callback(context, DiscoveryAction.REJECT, str(index)),
                    )
                ]
            )
        rows.extend(self._control_rows(context))
        return InlineKeyboardMarkup(rows)

    def _result_controls_keyboard(self, context) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(self._control_rows(context))

    def _control_rows(self, context) -> list[list[InlineKeyboardButton]]:
        return [
            [
                InlineKeyboardButton(
                    "Изменить пожелания",
                    callback_data=self._callback(context, DiscoveryAction.REFINE),
                ),
                InlineKeyboardButton(
                    "Обновить цены",
                    callback_data=self._callback(context, DiscoveryAction.RECHECK),
                ),
            ],
            [
                InlineKeyboardButton(
                    "Новая подборка",
                    callback_data=self._callback(context, DiscoveryAction.NEW),
                )
            ],
        ]

    def _reject_reason_keyboard(self, context) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        label,
                        callback_data=self._callback(
                            context,
                            DiscoveryAction.REJECT_REASON,
                            reason,
                        ),
                    )
                ]
                for reason, label in (
                    ("too_expensive", "Слишком дорого"),
                    ("bad_logistics", "Неудобная дорога"),
                    ("uninteresting", "Неинтересная программа"),
                    ("other", "Другая причина"),
                )
            ]
        )

    def _callback(
        self,
        context,
        action: DiscoveryAction,
        argument: str | None = None,
    ) -> str:
        return CallbackCodec.encode(
            DiscoveryCallback(
                flow_id=context.user_data[_FLOW_ID],
                revision=context.user_data[_REVISION],
                action=action,
                argument=argument,
            )
        )

    async def _extract(self, text: str, update: Update) -> IntentParseResult:
        return await self._extractor.extract(
            text,
            context=ConversationContext(
                timezone=self._timezone,
                current_date=self._clock.now().date().isoformat(),
            ),
            safety_identifier=self._safety_identifier(update),
        )

    def _initialize_flow(self, context) -> None:
        flow_id = context.user_data.get(_ANALYTICS_FLOW_ID)
        if not isinstance(flow_id, str) or len(flow_id) < 8:
            flow_id = secrets.token_hex(4)
        context.user_data[_ANALYTICS_FLOW_ID] = flow_id
        context.user_data[_FLOW_ID] = flow_id
        context.user_data[_REVISION] = 1

    def _ensure_flow(self, context) -> None:
        if _FLOW_ID not in context.user_data or _REVISION not in context.user_data:
            self._initialize_flow(context)

    @staticmethod
    def _advance_revision(context) -> None:
        context.user_data[_REVISION] = int(context.user_data.get(_REVISION, 0)) + 1
        context.user_data.pop(_PROPOSALS, None)
        context.user_data.pop(_FEASIBILITY, None)

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
        self._ensure_flow(context)
        await self._analytics.track(
            name,
            flow_id=context.user_data[_FLOW_ID],
            intent=intent,
            dimensions={**self._voice.analytics_dimensions, **(dimensions or {})},
        )


def _merge_discovery_drafts(current: DiscoveryDraft, patch: DiscoveryDraft) -> DiscoveryDraft:
    values = current.model_dump()
    for field in (
        "origin",
        "departure_date",
        "return_date",
        "date_flexibility",
        "adults",
        "children",
        "rooms",
        "budget",
        "currency",
        "hotel_mode",
        "allowed_modes",
    ):
        value = getattr(patch, field)
        if value is not None and not (field == "currency" and value == "RUB"):
            values[field] = value
    current_experience = current.experience
    patch_experience = patch.experience
    current_tolerance = current_experience.road_tolerance
    patch_tolerance = patch_experience.road_tolerance
    values["experience"] = ExperienceProfile(
        motives=patch_experience.motives or current_experience.motives,
        interests=patch_experience.interests or current_experience.interests,
        pace=patch_experience.pace or current_experience.pace,
        road_tolerance=RoadTolerance(
            max_one_way_duration=(
                patch_tolerance.max_one_way_duration or current_tolerance.max_one_way_duration
            ),
            max_transfers=(
                patch_tolerance.max_transfers
                if patch_tolerance.max_transfers is not None
                else current_tolerance.max_transfers
            ),
            allow_night_travel=(
                patch_tolerance.allow_night_travel
                if patch_tolerance.allow_night_travel is not None
                else current_tolerance.allow_night_travel
            ),
        ),
    )
    return DiscoveryDraft.model_validate(values)


def _price_completeness(result: DiscoveryProposalResult) -> str:
    costs = tuple(item.proposal.cost for item in result.recommendations)
    if costs and all(
        item.confirmed_total is not None and not item.unknown_components for item in costs
    ):
        return "exact"
    if any(item.confirmed_total is not None for item in costs):
        return "partial"
    return "unknown"
