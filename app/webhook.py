"""Cloud Run-compatible Telegram webhook transport and operational endpoints."""

from __future__ import annotations

import hmac
import json
import logging
from contextlib import suppress
from http import HTTPStatus

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from telegram import Update
from telegram.ext import Application

from app.config import Settings

logger = logging.getLogger(__name__)
TELEGRAM_SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"


def build_webhook_app(
    application: Application,
    *,
    secret_token: str,
    max_body_bytes: int,
) -> Starlette:
    """Build a small ASGI edge that validates and processes Telegram updates inline."""

    async def telegram_webhook(request: Request) -> Response:
        supplied_secret = request.headers.get(TELEGRAM_SECRET_HEADER, "")
        if not hmac.compare_digest(supplied_secret, secret_token):
            logger.warning(
                "telegram_webhook_rejected",
                extra={"event": "telegram_webhook_rejected", "category": "authentication"},
            )
            return Response(status_code=HTTPStatus.FORBIDDEN)

        content_length = request.headers.get("content-length")
        if content_length is not None:
            with suppress(ValueError):
                if int(content_length) > max_body_bytes:
                    return Response(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_body_bytes:
                return Response(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            body.extend(chunk)
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("Telegram update must be a JSON object")
            update = Update.de_json(payload, application.bot)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            logger.warning(
                "telegram_webhook_invalid_payload",
                extra={"event": "telegram_webhook_invalid_payload", "category": "validation"},
            )
            return Response(status_code=HTTPStatus.BAD_REQUEST)

        # Keep the HTTP request active while processing. Cloud Run request-based billing only
        # guarantees CPU during a request, so fire-and-forget processing could lose updates.
        await application.update_processor.process_update(
            update,
            application.process_update(update),
        )
        return Response(status_code=HTTPStatus.OK)

    async def health(_: Request) -> JSONResponse:
        health_state = application.bot_data.get("health", {"status": "starting"})
        status = (
            HTTPStatus.OK if health_state.get("status") == "ok" else HTTPStatus.SERVICE_UNAVAILABLE
        )
        return JSONResponse({"status": health_state.get("status", "starting")}, status_code=status)

    async def readiness(_: Request) -> JSONResponse:
        report = application.bot_data.get("readiness")
        if not isinstance(report, dict):
            return JSONResponse(
                {"status": "not_ready", "checks": {}},
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        ready = report.get("status") == "ok"
        return JSONResponse(
            {
                "status": report.get("status", "not_ready"),
                "checks": report.get("checks", {}),
            },
            status_code=HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
        )

    return Starlette(
        routes=[
            Route("/telegram/webhook", telegram_webhook, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
            Route("/healthz", health, methods=["GET"]),
            Route("/readyz", readiness, methods=["GET"]),
        ]
    )


async def run_webhook(application: Application, settings: Settings) -> None:
    """Run PTB and ASGI lifecycles together without deleting the shared webhook on shutdown."""

    secret = settings.require_webhook_secret().get_secret_value()
    web_app = build_webhook_app(
        application,
        secret_token=secret,
        max_body_bytes=settings.webhook_max_body_bytes,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app=web_app,
            host="0.0.0.0",
            port=settings.port,
            access_log=False,
            server_header=False,
            use_colors=False,
        )
    )

    initialized = False
    started = False
    try:
        await application.initialize()
        initialized = True
        if application.post_init is not None:
            await application.post_init(application)
        await application.start()
        started = True
        await application.bot.set_webhook(
            url=settings.webhook_url,
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=False,
            max_connections=settings.webhook_max_connections,
            secret_token=secret,
        )
        logger.info("telegram_webhook_registered", extra={"event": "telegram_webhook_registered"})
        await server.serve()
    finally:
        if started:
            await application.stop()
            if application.post_stop is not None:
                await application.post_stop(application)
        if initialized:
            await application.shutdown()
            if application.post_shutdown is not None:
                await application.post_shutdown(application)
