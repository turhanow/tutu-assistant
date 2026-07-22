from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import app.webhook as webhook_module
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
