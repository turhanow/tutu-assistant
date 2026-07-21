"""Validate OpenAI parser connectivity without printing secrets or response content."""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from app.bootstrap import build_llm_resources
from app.config import Settings


async def main() -> None:
    settings = Settings()
    resources = build_llm_resources(settings)
    try:
        await resources.parser.parse(
            "Поездка из Москвы в Казань 21–23 августа 2026 года, нужен отель.",
            now=datetime.now(ZoneInfo(settings.app_timezone)),
            timezone=settings.app_timezone,
        )
    except Exception as error:
        cause = error.__cause__ or error
        body = getattr(cause, "body", None)
        provider_code = None
        if isinstance(body, dict):
            details = body.get("error", body)
            if isinstance(details, dict):
                provider_code = details.get("code") or details.get("type")
        print("result=error")
        print(f"application_error={type(error).__name__}")
        print(f"provider_error={type(cause).__name__}")
        print(f"http_status={getattr(cause, 'status_code', None)}")
        print(f"provider_code={provider_code}")
    else:
        print("result=ok")
    finally:
        await resources.close()


if __name__ == "__main__":
    asyncio.run(main())
