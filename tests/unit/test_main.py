import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.ext import CallbackQueryHandler, ConversationHandler, MessageHandler

import app.main as main_module
from app.config import Settings


class FakeGateway:
    def __init__(self, endpoint: str, **_: object) -> None:
        self.endpoint = endpoint

    async def __aenter__(self):
        return self

    async def discover_capabilities(self):
        return SimpleNamespace(schema_hash="test-schema")

    async def close(self) -> None:
        return None


def test_application_wires_conversation_and_job_queue(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        main_module,
        "build_llm_resources",
        lambda _: SimpleNamespace(
            parser=SimpleNamespace(),
            intent_extractor=SimpleNamespace(),
            proposal_narrator=SimpleNamespace(),
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(main_module, "TutuMcpPool", FakeGateway)

    application = main_module.build_application(Settings(_env_file=None))

    assert application.job_queue is not None
    conversation_handler = next(
        handler
        for group in application.handlers.values()
        for handler in group
        if isinstance(handler, ConversationHandler)
    )
    stale_patterns = {
        handler.pattern.pattern
        for handler in conversation_handler.fallbacks
        if isinstance(handler, CallbackQueryHandler) and hasattr(handler.pattern, "pattern")
    }
    assert {"^trip:", "^d1:"}.issubset(stale_patterns)
    group_handlers = application.handlers[0]
    assert group_handlers[0] is conversation_handler
    assert isinstance(group_handlers[1], MessageHandler)
    assert not any(
        isinstance(handler, MessageHandler) for handler in conversation_handler.entry_points
    )
    assert application.error_handlers


def test_bot_command_catalog_matches_supported_handlers() -> None:
    commands = {item.command: item.description for item in main_module.BOT_COMMANDS}

    assert set(commands) == {
        "start",
        "newtrip",
        "ideas",
        "help",
        "privacy",
        "feedback",
        "deletefeedback",
        "cancel",
    }
    assert all(description for description in commands.values())


@pytest.mark.asyncio
async def test_post_init_publishes_brand_profile_and_commands(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        main_module,
        "build_llm_resources",
        lambda _: SimpleNamespace(
            parser=SimpleNamespace(),
            intent_extractor=SimpleNamespace(),
            proposal_narrator=SimpleNamespace(),
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(main_module, "TutuMcpPool", FakeGateway)
    application = main_module.build_application(Settings(_env_file=None))
    bot = SimpleNamespace(
        set_my_commands=AsyncMock(),
        set_my_name=AsyncMock(),
        set_my_short_description=AsyncMock(),
        set_my_description=AsyncMock(),
    )
    runtime = SimpleNamespace(bot=bot, bot_data={})

    await application.post_init(runtime)
    await runtime.bot_data["startup_task"]

    bot.set_my_commands.assert_awaited_once_with(main_module.BOT_COMMANDS)
    bot.set_my_name.assert_awaited_once_with("Ту-да и обратно")
    assert runtime.bot_data["readiness"]["status"] == "ok"


@pytest.mark.asyncio
async def test_post_init_does_not_block_static_commands_on_provider_cold_start(monkeypatch) -> None:
    release_provider = asyncio.Event()

    class SlowGateway(FakeGateway):
        async def __aenter__(self):
            await release_provider.wait()
            return self

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        main_module,
        "build_llm_resources",
        lambda _: SimpleNamespace(
            parser=SimpleNamespace(),
            intent_extractor=SimpleNamespace(),
            proposal_narrator=SimpleNamespace(),
            close=AsyncMock(),
        ),
    )
    monkeypatch.setattr(main_module, "TutuMcpPool", SlowGateway)
    application = main_module.build_application(Settings(_env_file=None))
    bot = SimpleNamespace(
        set_my_commands=AsyncMock(),
        set_my_name=AsyncMock(),
        set_my_short_description=AsyncMock(),
        set_my_description=AsyncMock(),
    )
    runtime = SimpleNamespace(bot=bot, bot_data={})

    await asyncio.wait_for(application.post_init(runtime), timeout=0.1)

    assert runtime.bot_data["health"] == {"status": "ok"}
    assert runtime.bot_data["readiness"]["status"] == "not_ready"
    assert not runtime.bot_data["startup_task"].done()

    release_provider.set()
    await runtime.bot_data["startup_task"]
    assert runtime.bot_data["readiness"]["status"] == "ok"


def test_analytics_feature_flag_selects_null_sink(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANALYTICS_ENABLED", "false")
    monkeypatch.setattr(
        main_module,
        "build_llm_resources",
        lambda _: SimpleNamespace(
            parser=SimpleNamespace(),
            intent_extractor=SimpleNamespace(),
            proposal_narrator=SimpleNamespace(),
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(main_module, "TutuMcpPool", FakeGateway)
    null_sink = object()
    monkeypatch.setattr(main_module, "NullAnalyticsSink", lambda: null_sink)
    monkeypatch.setattr(
        main_module,
        "BufferedAnalyticsSink",
        lambda **_: pytest.fail("buffered analytics must stay disabled"),
    )
    captured = {}

    class CapturingAnalytics:
        def __init__(self, sink, clock) -> None:
            captured["sink"] = sink
            captured["clock"] = clock

        async def track(self, *args, **kwargs) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(main_module, "ProductAnalytics", CapturingAnalytics)

    main_module.build_application(Settings(_env_file=None))

    assert captured["sink"] is null_sink


@pytest.mark.asyncio
async def test_global_update_error_handler_logs_without_serializing_update(caplog) -> None:
    error = RuntimeError("telegram failure")

    await main_module.handle_update_error(
        object(),
        SimpleNamespace(error=error),  # type: ignore[arg-type]
    )

    assert "telegram_update_failed" in caplog.text
