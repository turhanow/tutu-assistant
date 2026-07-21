"""Capture the live Tutu MCP capability contract without invoking tools.

This script is intentionally limited to MCP initialization and discovery. Search
smoke tests are performed only after a human or adapter has reviewed the
captured JSON Schemas and constructed arguments from those schemas.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

DEFAULT_URL = "https://mcp.tutu.ru/mcp"
DEFAULT_TIMEOUT_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("var/tutu_mcp"))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def schema_hash(pages: list[dict[str, Any]]) -> str:
    payload = json.dumps(pages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_markdown(
    *,
    endpoint: str,
    captured_at: str,
    digest: str,
    pages: list[dict[str, Any]],
) -> str:
    tools = [tool for page in pages for tool in page["tools"]]
    lines = [
        "# Tutu MCP tools",
        "",
        f"Captured at: `{captured_at}`  ",
        f"Endpoint: `{endpoint}`  ",
        f"Pages: `{len(pages)}`  ",
        f"Tools: `{len(tools)}`  ",
        f"SHA-256: `{digest}`",
        "",
    ]
    for tool in tools:
        lines.extend(
            [
                f"## `{tool['name']}`",
                "",
                tool.get("description") or "No description supplied.",
                "",
                "### Input schema",
                "",
                "```json",
                json.dumps(tool["inputSchema"], ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
                "### Output schema",
                "",
                "```json",
                json.dumps(tool.get("outputSchema"), ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


async def inspect(
    endpoint: str, timeout_seconds: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    timeout = timedelta(seconds=timeout_seconds)
    async with (
        streamable_http_client(endpoint) as (read_stream, write_stream, _),
        ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timeout,
        ) as session,
    ):
        initialized = await session.initialize()
        pages: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()

        while True:
            result = await session.list_tools(cursor=cursor)
            pages.append(result.model_dump(mode="json", by_alias=True, exclude_none=False))
            cursor = result.nextCursor
            if cursor is None:
                break
            if cursor in seen_cursors:
                raise RuntimeError(f"Tutu MCP returned a repeated pagination cursor: {cursor!r}")
            seen_cursors.add(cursor)

        return (
            initialized.model_dump(mode="json", by_alias=True, exclude_none=False),
            pages,
        )


async def async_main() -> None:
    args = parse_args()
    initialized, pages = await inspect(args.url, args.timeout)
    captured_at = datetime.now(UTC).isoformat()
    digest = schema_hash(pages)
    artifact = {
        "capturedAt": captured_at,
        "endpoint": args.url,
        "initialize": initialized,
        "pages": pages,
        "sha256": digest,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tools-list.json").write_text(canonical_json(artifact), encoding="utf-8")
    (args.output / "TOOLS.md").write_text(
        render_markdown(
            endpoint=args.url,
            captured_at=captured_at,
            digest=digest,
            pages=pages,
        ),
        encoding="utf-8",
    )

    tools = [tool for page in pages for tool in page["tools"]]
    names = [tool["name"] for tool in tools]
    if len(names) != len(set(names)):
        raise RuntimeError("Tutu MCP returned duplicate tool names")
    print(f"Captured {len(tools)} tools in {len(pages)} page(s); SHA-256={digest}")


if __name__ == "__main__":
    asyncio.run(async_main())
