"""Offline release-gate validation for versioned discovery eval corpora."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

from app.adapters.openai_discovery import OpenAIIntentExtractor, OpenAIProposalNarrator
from app.config import Settings
from app.domain.discovery_models import GroundedProposalFacts
from app.ports.discovery_llm import ConversationContext
from app.voice import FORBIDDEN_SLANG

INTENT_PATH = Path("evals/intent_v1.jsonl")
NARRATION_PATH = Path("evals/narration_v2.jsonl")
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
    if any(item.get("prompt_version") != "narration_v2" for item in records):
        raise ValueError("narration corpus contains a wrong prompt version")
    for item in records:
        if not item.get("allowed_evidence_ids") or not item.get("forbidden_claims"):
            raise ValueError("narration case requires evidence allowlist and forbidden claims")
        if not item.get("forbidden_style") or not isinstance(item.get("humor_allowed"), bool):
            raise ValueError("narration case requires tone constraints")


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


async def run_live_narration() -> bool:
    settings = Settings()
    client = AsyncOpenAI(
        api_key=settings.require_openai_api_key().get_secret_value(),
        base_url=str(settings.openai_base_url),
        max_retries=0,
        timeout=settings.openai_timeout_seconds,
    )
    narrator = OpenAIProposalNarrator(
        client,
        model=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
    )
    facts = GroundedProposalFacts(
        proposal_id="live_proposal_1",
        destination_name="Коломна",
        match_reasons=("Подходит по интересам: архитектура, история",),
        day_activity_names=(
            ("Прогулка по Коломенскому кремлю",),
            ("Знакомство с коломенской пастилой",),
        ),
        transport_facts=(
            "Туда: rail, 2026-08-15T08:00:00+03:00 — 2026-08-15T10:00:00+03:00",
            "Обратно: rail, 2026-08-16T18:00:00+03:00 — 2026-08-16T20:00:00+03:00",
        ),
        confirmed_cost="8200 RUB",
        unknown_components=("food", "local_transport"),
        trade_off="Питание и городской транспорт пока не включены в стоимость",
        evidence_ids={"kolomna_portal"},
    )
    try:
        result = await narrator.narrate(
            (facts,),
            context=ConversationContext(
                timezone=settings.app_timezone,
                current_date="2026-07-22",
            ),
        )
    finally:
        await client.close()
    copy = result[0]
    rendered = " ".join((copy.title, copy.reason, copy.trade_off)).casefold()
    safe = (
        copy.proposal_id == facts.proposal_id
        and copy.evidence_ids == facts.evidence_ids
        and "!" not in rendered
        and not any(word in rendered for word in FORBIDDEN_SLANG)
    )
    print(
        "Live narration eval: "
        f"schema_valid={bool(result)}, grounded={copy.evidence_ids == facts.evidence_ids}, "
        f"tone_safe={safe}"
    )
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="make paid OpenAI calls for the intent accuracy release gate",
    )
    parser.add_argument(
        "--live-narration",
        action="store_true",
        help="make one paid OpenAI call for the grounded tone-of-voice smoke test",
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
    success = True
    if args.live:
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        success = asyncio.run(run_live_intent(intent, limit=args.limit)) and success
    if args.live_narration:
        success = asyncio.run(run_live_narration()) and success
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
