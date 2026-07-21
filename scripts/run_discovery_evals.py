"""Offline release-gate validation for versioned discovery eval corpora."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

from app.adapters.openai_discovery import OpenAIIntentExtractor
from app.config import Settings
from app.ports.discovery_llm import ConversationContext

INTENT_PATH = Path("evals/intent_v1.jsonl")
NARRATION_PATH = Path("evals/narration_v1.jsonl")
ALLOWED_INTENTS = {"destination_known", "destination_unknown", "event_led"}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return records


def validate_intent(records: list[dict]) -> None:
    if len(records) < 120:
        raise ValueError("intent corpus must contain at least 120 cases")
    ids = [item.get("id") for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("intent eval IDs must be unique")
    if any(item.get("prompt_version") != "intent_v1" for item in records):
        raise ValueError("intent corpus contains a wrong prompt version")
    intents = Counter(item.get("expected_intent") for item in records)
    if set(intents) != ALLOWED_INTENTS or min(intents.values()) < 20:
        raise ValueError("intent corpus must cover every intent with at least 20 cases")
    if any(not isinstance(item.get("text"), str) or not item["text"].strip() for item in records):
        raise ValueError("intent eval text must be non-empty")


def validate_narration(records: list[dict]) -> None:
    if len(records) < 5:
        raise ValueError("narration corpus must contain at least five safety cases")
    if any(item.get("prompt_version") != "narration_v1" for item in records):
        raise ValueError("narration corpus contains a wrong prompt version")
    for item in records:
        if not item.get("allowed_evidence_ids") or not item.get("forbidden_claims"):
            raise ValueError("narration case requires evidence allowlist and forbidden claims")


async def run_live_intent(records: list[dict], *, limit: int | None) -> bool:
    selected = records[:limit] if limit is not None else records
    settings = Settings()
    api_key = settings.require_openai_api_key().get_secret_value()
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=str(settings.openai_base_url),
        max_retries=0,
        timeout=settings.openai_timeout_seconds,
    )
    extractor = OpenAIIntentExtractor(
        client,
        model=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
    )
    context = ConversationContext(
        timezone=settings.app_timezone,
        current_date="2026-07-21",
    )
    semaphore = asyncio.Semaphore(3)

    async def evaluate(item: dict) -> tuple[bool, bool]:
        async with semaphore:
            result = await extractor.extract(
                item["text"],
                context=context,
                safety_identifier="offline-discovery-eval",
            )
        draft = result.known_draft or result.discovery_draft
        origin = getattr(draft, "origin", None)
        destination = getattr(draft, "destination", None)
        intent_match = result.intent.value == item["expected_intent"]
        fields_match = (
            origin == item["expected_origin"] and destination == item["expected_destination"]
        )
        return intent_match, fields_match

    try:
        results = await asyncio.gather(*(evaluate(item) for item in selected))
    finally:
        await client.close()
    intent_accuracy = sum(item[0] for item in results) / len(results)
    field_accuracy = sum(item[1] for item in results) / len(results)
    print(
        f"Live intent eval: n={len(results)}, intent_accuracy={intent_accuracy:.3f}, "
        f"origin_destination_accuracy={field_accuracy:.3f}"
    )
    return intent_accuracy >= 0.95 and field_accuracy >= 0.90


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="make paid OpenAI calls for the intent accuracy release gate",
    )
    parser.add_argument("--limit", type=int, default=None, help="limit paid live eval cases")
    args = parser.parse_args()
    intent = load_jsonl(INTENT_PATH)
    narration = load_jsonl(NARRATION_PATH)
    validate_intent(intent)
    validate_narration(narration)
    counts = Counter(item["expected_intent"] for item in intent)
    print(
        "Discovery eval corpora are valid: "
        f"{len(intent)} intent cases {dict(sorted(counts.items()))}, "
        f"{len(narration)} narration safety cases"
    )
    if args.live:
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        return 0 if asyncio.run(run_live_intent(intent, limit=args.limit)) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
