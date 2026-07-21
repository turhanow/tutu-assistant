from pathlib import Path

from scripts.build_intent_eval_corpus import build
from scripts.run_discovery_evals import (
    load_jsonl,
    validate_intent,
    validate_narration,
)


def test_generated_intent_corpus_is_deterministic_and_balanced() -> None:
    records = build()

    assert len(records) == 120
    assert len({item["id"] for item in records}) == 120
    assert {item["expected_intent"] for item in records} == {
        "destination_known",
        "destination_unknown",
        "event_led",
    }
    validate_intent(records)


def test_checked_in_eval_files_pass_release_gate() -> None:
    intent = load_jsonl(Path("evals/intent_v1.jsonl"))
    narration = load_jsonl(Path("evals/narration_v1.jsonl"))

    validate_intent(intent)
    validate_narration(narration)
