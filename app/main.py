"""Composition root for wiring Telegram, OpenAI and Tutu adapters."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from telegram import BotCommand, Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from app.adapters.buffered_analytics import BufferedAnalyticsSink, NullAnalyticsSink
from app.adapters.clock import SystemClock
from app.adapters.file_catalog import (
    FileCatalogRepository,
    FileDestinationCatalog,
    FileTravelContentGateway,
)
from app.adapters.hybrid_catalog import HybridDestinationCatalog, HybridTravelContentGateway
from app.adapters.openai_destination_discovery import OpenAIDestinationDiscovery
from app.adapters.resilient_tutu import ProtectedTutuGateway
from app.adapters.sqlite_feedback import SqliteFeedbackSink
from app.adapters.tutu_mcp import TutuMcpPool
from app.bootstrap import build_llm_resources
from app.bot.conversation import TripConversation
from app.bot.discovery_conversation import DiscoveryConversation
from app.bot.feedback_conversation import FeedbackConversation
from app.bot.router import BotRouter, build_routed_conversation_handler
from app.bot.update_processor import ChatSerialUpdateProcessor
from app.config import Settings
from app.logging_config import configure_logging
from app.services.candidate_selector import CandidateSelector
from app.services.destination_feasibility import DestinationFeasibilityService
from app.services.discovery_planner import DiscoveryPlanner
from app.services.feedback_service import FeedbackService
from app.services.product_analytics import ProductAnalytics
from app.services.proposal_builder import GroundedProposalNarration, ProposalBuilder
from app.services.readiness import build_readiness_report
from app.services.trip_handoff import TripHandoffService
from app.services.trip_planner import TripPlanner
from app.voice import BOT_DESCRIPTION, BOT_SHORT_DESCRIPTION, BRAND_NAME, Voice
from app.webhook import run_webhook

logger = logging.getLogger(__name__)

BOT_COMMANDS = (
    BotCommand("start", "Начать планировать поездку"),
    BotCommand("newtrip", "Проверить готовый маршрут"),
    BotCommand("ideas", "Подобрать идею для выходных"),
    BotCommand("help", "Что умеет бот"),
    BotCommand("privacy", "Как обрабатываются данные"),
    BotCommand("feedback", "Сообщить о проблеме"),
    BotCommand("deletefeedback", "Удалить сохранённое обращение"),
    BotCommand("cancel", "Отменить диалог и удалить параметры"),
    BotCommand("reset", "Сбросить контекст и начать сначала"),
)


async def handle_update_error(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    error = context.error
    exc_info = None
    if isinstance(error, BaseException):
        exc_info = (type(error), error, error.__traceback__)
    logger.error(
        "telegram_update_failed",
        exc_info=exc_info,
        extra={"event": "telegram_update_failed", "category": "unexpected_error"},
    )
    if isinstance(update, Update) and update.effective_message is not None:
        with suppress(Exception):
            await update.effective_message.reply_text(
                "Не удалось обработать сообщение. Параметры сохранены; "
                "попробуйте ещё раз или используйте /newtrip."
            )


def build_application(settings: Settings) -> Application:
    settings.validate_runtime()
    llm = build_llm_resources(settings)
    raw_gateway = TutuMcpPool(
        str(settings.tutu_mcp_url),
        size=settings.tutu_pool_size,
        timeout_seconds=settings.tutu_timeout_seconds,
    )
    gateway = ProtectedTutuGateway(
        raw_gateway,
        max_inflight=settings.provider_max_inflight,
        calls_per_minute=settings.provider_calls_per_minute,
        failure_threshold=settings.provider_circuit_failure_threshold,
        recovery_seconds=settings.provider_circuit_recovery_seconds,
    )
    clock = SystemClock(settings.app_timezone)
    analytics = ProductAnalytics(
        (
            BufferedAnalyticsSink(
                queue_size=settings.analytics_queue_size,
                flush_timeout_seconds=settings.analytics_flush_timeout_seconds,
            )
            if settings.analytics_enabled
            else NullAnalyticsSink()
        ),
        clock,
    )
    planner = TripPlanner(
        gateway,
        clock,
        timeout_seconds=settings.search_timeout_seconds,
    )
    voice = Voice(
        tone_v2_enabled=settings.tone_of_voice_v2_enabled,
        controlled_delight_enabled=settings.controlled_delight_enabled,
    )
    handoff = TripHandoffService(gateway)
    known_conversation = TripConversation(
        llm.parser,
        planner,
        clock,
        timezone=settings.app_timezone,
        handoff=handoff,
        analytics=analytics,
        enabled=not settings.bot_kill_switch,
        voice=voice,
    )
    catalog_repository = FileCatalogRepository.from_path(settings.destination_catalog_path)
    file_catalog = FileDestinationCatalog(catalog_repository)
    file_content = FileTravelContentGateway(catalog_repository)
    content_gateway = file_content
    candidate_catalog = file_catalog
    if settings.dynamic_discovery_enabled:
        dynamic_discovery = OpenAIDestinationDiscovery(
            llm.client,
            clock,
            model=settings.openai_model,
            timeout_seconds=settings.dynamic_discovery_timeout_seconds,
        )
        candidate_catalog = HybridDestinationCatalog(dynamic_discovery, file_catalog)
        content_gateway = HybridTravelContentGateway(dynamic_discovery, file_content)
    selector = CandidateSelector(candidate_catalog)
    discovery_planner = DiscoveryPlanner(
        DestinationFeasibilityService(planner),
        clock,
        max_concurrency=settings.discovery_max_concurrency,
        timeout_seconds=settings.discovery_verification_timeout_seconds,
    )
    discovery_conversation = DiscoveryConversation(
        llm.intent_extractor,
        selector,
        discovery_planner,
        ProposalBuilder(content_gateway, clock),
        GroundedProposalNarration(llm.proposal_narrator),
        clock,
        timezone=settings.app_timezone,
        handoff=handoff,
        analytics=analytics,
        enabled=not settings.bot_kill_switch and settings.discovery_enabled,
        voice=voice,
    )
    router = BotRouter(
        llm.intent_extractor,
        known_conversation,
        discovery_conversation,
        clock,
        timezone=settings.app_timezone,
        analytics=analytics,
        enabled=not settings.bot_kill_switch,
        voice=voice,
    )
    feedback_conversation = FeedbackConversation(
        FeedbackService(
            SqliteFeedbackSink(
                settings.feedback_db_path,
                retention_days=settings.feedback_retention_days,
            ),
            clock,
            analytics=analytics,
            timeout_seconds=settings.feedback_timeout_seconds,
        ),
        retention_days=settings.feedback_retention_days,
        enabled=not settings.bot_kill_switch and settings.feedback_enabled,
    )

    async def warm_provider(application: Application) -> None:
        try:
            await raw_gateway.__aenter__()
            capabilities = await gateway.discover_capabilities()
        except Exception:
            logger.exception("provider_startup_warmup_failed")
            return
        application.bot_data["readiness"] = build_readiness_report(
            clock=clock,
            catalog_version=catalog_repository.version,
            destination_count=catalog_repository.destination_count,
            provider_schema_hash=capabilities.schema_hash,
            provider_circuit_state=gateway.circuit_state,
            kill_switch=settings.bot_kill_switch,
        ).model_dump(mode="json")
        logger.info("readiness_checked", extra={"schema_hash": capabilities.schema_hash})

    async def publish_bot_profile(application: Application) -> None:
        try:
            await asyncio.gather(
                application.bot.set_my_commands(BOT_COMMANDS),
                application.bot.set_my_name(BRAND_NAME),
                application.bot.set_my_short_description(BOT_SHORT_DESCRIPTION),
                application.bot.set_my_description(BOT_DESCRIPTION),
            )
        except Exception:
            logger.exception("telegram_profile_sync_failed")

    async def initialize_external_runtime(application: Application) -> None:
        await asyncio.gather(warm_provider(application), publish_bot_profile(application))

    async def post_init(application: Application) -> None:
        application.bot_data["health"] = {"status": "ok"}
        application.bot_data["readiness"] = build_readiness_report(
            clock=clock,
            catalog_version=catalog_repository.version,
            destination_count=catalog_repository.destination_count,
            provider_schema_hash=None,
            provider_circuit_state=gateway.circuit_state,
            kill_switch=settings.bot_kill_switch,
        ).model_dump(mode="json")
        application.bot_data["provider_metrics"] = gateway.metrics
        application.bot_data["startup_task"] = asyncio.create_task(
            initialize_external_runtime(application),
            name="external-runtime-warmup",
        )
        # Let in-memory/fake adapters complete in tests without waiting for real network I/O.
        await asyncio.sleep(0)

    async def post_shutdown(application: Application) -> None:
        startup_task = application.bot_data.pop("startup_task", None)
        if isinstance(startup_task, asyncio.Task) and not startup_task.done():
            startup_task.cancel()
            with suppress(asyncio.CancelledError):
                await startup_task
        await raw_gateway.close()
        await analytics.close()
        await llm.close()

    application = (
        Application.builder()
        .token(settings.require_bot_token().get_secret_value())
        .concurrent_updates(ChatSerialUpdateProcessor(settings.max_concurrent_updates))
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    application.add_handler(
        build_routed_conversation_handler(
            router,
            known_conversation,
            discovery_conversation,
            feedback_conversation,
        )
    )
    # This recovery handler must stay after ConversationHandler in the same group.
    # Making it a re-entrant conversation entry point intercepts every active-flow text.
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, router.orphan_text))
    application.add_error_handler(handle_update_error)
    return application


def main() -> None:
    settings = Settings()
    settings.validate_runtime()
    secrets = [
        settings.require_bot_token().get_secret_value(),
        settings.require_openai_api_key().get_secret_value(),
    ]
    if settings.telegram_webhook_secret is not None:
        secrets.append(settings.telegram_webhook_secret.get_secret_value())
    configure_logging(settings.log_level, secrets=tuple(secrets))
    application = build_application(settings)
    if settings.bot_transport == "webhook":
        asyncio.run(run_webhook(application, settings))
    else:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )


if __name__ == "__main__":
    main()
