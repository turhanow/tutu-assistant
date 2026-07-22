from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from telegram import User
from telegram.ext import Application, ConversationHandler, MessageHandler, filters

import app.webhook as webhook_module
from app.bot.conversation import State
from app.bot.discovery_conversation import DiscoveryState
from app.bot.router import build_routed_conversation_handler
from app.config import Settings
from app.webhook import TELEGRAM_SECRET_HEADER, build_webhook_app, run_webhook


class FakeUpdateProcessor:
    async def process_update(self, update, coroutine) -> None:
        await coroutine


class FakeWebhookApplication:
    def __init__(self) -> None:
        self.bot = object()
        self.bot_data = {
            "health": {"status": "ok"},
            "readiness": {"status": "ok", "checks": {"catalog_loaded": True}},
        }
        self.update_processor = FakeUpdateProcessor()
        self.processed_updates = []

    async def process_update(self, update) -> None:
        self.processed_updates.append(update)


def _transport(application=None, *, secret="webhook_secret") -> httpx.ASGITransport:
    app = application or FakeWebhookApplication()
    return httpx.ASGITransport(
        app=build_webhook_app(
            app,  # type: ignore[arg-type]
            secret_token=secret,
            max_body_bytes=1_024,
        )
    )


class RecordingConversation:
    def __init__(self, *, start_state: int, intake_state: int) -> None:
        self.start_state = start_state
        self.intake_state = intake_state
        self.calls: list[tuple[str, str]] = []

    def __getattr__(self, name: str):
        async def callback(update, _context):
            self.calls.append((name, update.effective_message.text or ""))
            if name == "start":
                return self.start_state
            if name in {"intake", "clarification_input"}:
                return self.intake_state
            return ConversationHandler.END

        return callback


def _telegram_message(update_id: int, text: str, *, command: bool = False) -> dict:
    message = {
        "message_id": update_id,
        "date": int(datetime.now(UTC).timestamp()),
        "chat": {"id": 100, "type": "private"},
        "from": {"id": 42, "is_bot": False, "first_name": "QA"},
        "text": text,
    }
    if command:
        message["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text)}]
    return {"update_id": update_id, "message": message}


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:Ignoring `conversation_timeout`:telegram.warnings.PTBUserWarning"
)
@pytest.mark.parametrize(
    ("command", "payload", "flow_name"),
    [
        ("/newtrip", "Москва — Казань, 15–17 августа", "known"),
        ("/ideas", "Куда-нибудь на выходные без суеты", "discovery"),
    ],
)
async def test_webhook_keeps_conversation_state_between_command_and_payload(
    command: str,
    payload: str,
    flow_name: str,
) -> None:
    application = Application.builder().token("123456:test-token").build()
    object.__setattr__(
        application.bot,
        "_bot_user",
        User(id=999, first_name="Bot", is_bot=True, username="test_bot"),
    )
    object.__setattr__(application.bot, "_bot_initialized", True)
    application._initialized = True

    router = RecordingConversation(start_state=10, intake_state=10)
    known = RecordingConversation(start_state=State.INTAKE, intake_state=State.FORM)
    discovery = RecordingConversation(
        start_state=DiscoveryState.INTAKE,
        intake_state=DiscoveryState.CLARIFY,
    )
    feedback = RecordingConversation(start_state=201, intake_state=202)
    conversation_handler = build_routed_conversation_handler(
        router,  # type: ignore[arg-type]
        known,  # type: ignore[arg-type]
        discovery,  # type: ignore[arg-type]
        feedback,  # type: ignore[arg-type]
    )
    application.add_handler(conversation_handler)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, router.orphan_text))

    async with httpx.AsyncClient(
        transport=_transport(application), base_url="https://test"
    ) as client:
        first = await client.post(
            "/telegram/webhook",
            json=_telegram_message(1, command, command=True),
            headers={TELEGRAM_SECRET_HEADER: "webhook_secret"},
        )
        second = await client.post(
            "/telegram/webhook",
            json=_telegram_message(2, payload),
            headers={TELEGRAM_SECRET_HEADER: "webhook_secret"},
        )

    selected = known if flow_name == "known" else discovery
    assert first.status_code == second.status_code == 200
    assert selected.calls == [("start", command), ("intake", payload)]
    assert not any(name == "orphan_text" for name, _ in router.calls)


@pytest.mark.asyncio
async def test_webhook_rejects_missing_or_wrong_secret() -> None:
    application = FakeWebhookApplication()
    async with httpx.AsyncClient(
        transport=_transport(application), base_url="https://test"
    ) as client:
        missing = await client.post("/telegram/webhook", json={"update_id": 1})
        wrong = await client.post(
            "/telegram/webhook",
            json={"update_id": 1},
            headers={TELEGRAM_SECRET_HEADER: "wrong"},
        )

    assert missing.status_code == 403
    assert wrong.status_code == 403
    assert application.processed_updates == []


@pytest.mark.asyncio
async def test_webhook_processes_valid_update_before_acknowledging() -> None:
    application = FakeWebhookApplication()
    async with httpx.AsyncClient(
        transport=_transport(application), base_url="https://test"
    ) as client:
        response = await client.post(
            "/telegram/webhook",
            json={"update_id": 42},
            headers={TELEGRAM_SECRET_HEADER: "webhook_secret"},
        )

    assert response.status_code == 200
    assert [item.update_id for item in application.processed_updates] == [42]


@pytest.mark.asyncio
async def test_webhook_rejects_invalid_or_oversized_payload() -> None:
    headers = {TELEGRAM_SECRET_HEADER: "webhook_secret"}
    async with httpx.AsyncClient(transport=_transport(), base_url="https://test") as client:
        invalid = await client.post("/telegram/webhook", content=b"not-json", headers=headers)
        oversized = await client.post("/telegram/webhook", content=b"x" * 1_025, headers=headers)

    assert invalid.status_code == 400
    assert oversized.status_code == 413


@pytest.mark.asyncio
async def test_operational_endpoints_expose_only_bounded_status() -> None:
    async with httpx.AsyncClient(transport=_transport(), base_url="https://test") as client:
        health = await client.get("/health")
        legacy_health = await client.get("/healthz")
        readiness = await client.get("/readyz")

    assert health.json() == {"status": "ok"}
    assert legacy_health.json() == {"status": "ok"}
    assert readiness.json() == {
        "status": "ok",
        "checks": {"catalog_loaded": True},
    }


@pytest.mark.asyncio
async def test_webhook_runtime_registers_secret_and_closes_lifecycle(monkeypatch) -> None:
    events = []

    class FakeBot:
        async def set_webhook(self, **kwargs) -> None:
            events.append(("set_webhook", kwargs))

    async def callback(name, _application) -> None:
        events.append(name)

    class FakeApplication(FakeWebhookApplication):
        def __init__(self) -> None:
            super().__init__()
            self.bot = FakeBot()
            self.post_init = lambda app: callback("post_init", app)
            self.post_stop = None
            self.post_shutdown = lambda app: callback("post_shutdown", app)

        async def initialize(self) -> None:
            events.append("initialize")

        async def start(self) -> None:
            events.append("start")

        async def stop(self) -> None:
            events.append("stop")

        async def shutdown(self) -> None:
            events.append("shutdown")

    class FakeServer:
        def __init__(self, _config) -> None:
            pass

        async def serve(self) -> None:
            events.append("serve")

    monkeypatch.setattr(webhook_module.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(
        webhook_module.uvicorn,
        "Config",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    settings = Settings(
        _env_file=None,
        TELEGRAM_BOT_TOKEN="123:test",
        OPENAI_API_KEY="openai-test",
        BOT_TRANSPORT="webhook",
        PUBLIC_BASE_URL="https://bot.example.com",
        TELEGRAM_WEBHOOK_SECRET="runtime_secret",
    )

    await run_webhook(FakeApplication(), settings)  # type: ignore[arg-type]

    assert [item if isinstance(item, str) else item[0] for item in events] == [
        "initialize",
        "post_init",
        "start",
        "set_webhook",
        "serve",
        "stop",
        "shutdown",
        "post_shutdown",
    ]
    webhook_options = next(item[1] for item in events if not isinstance(item, str))
    assert webhook_options["url"] == "https://bot.example.com/telegram/webhook"
    assert webhook_options["secret_token"] == "runtime_secret"
    assert webhook_options["drop_pending_updates"] is False
