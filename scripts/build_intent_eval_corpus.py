"""Build the deterministic 120-case discovery intent eval corpus."""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path("evals/intent_v1.jsonl")

BASE_CASES = (
    (
        "known_01",
        "Москва — Казань 21–23 августа, нужен отель",
        "destination_known",
        "Москва",
        "Казань",
    ),
    ("known_02", "Хочу из Москвы в Тулу на выходные", "destination_known", "Москва", "Тула"),
    ("known_03", "Подбери поезд из Твери в Ярославль", "destination_known", "Тверь", "Ярославль"),
    (
        "known_04",
        "Во Владимир из Москвы вдвоём до 30000",
        "destination_known",
        "Москва",
        "Владимир",
    ),
    ("known_05", "Еду в Суздаль на два дня", "destination_known", None, "Суздаль"),
    ("known_06", "Калуга из Москвы, без отеля", "destination_known", "Москва", "Калуга"),
    (
        "known_07",
        "Нижний Новгород на следующие выходные",
        "destination_known",
        None,
        "Нижний Новгород",
    ),
    ("known_08", "Из Москвы в Коломну одним днём", "destination_known", "Москва", "Коломна"),
    ("known_09", "Рязань 15–17 августа, только поезд", "destination_known", None, "Рязань"),
    ("known_10", "Сергиев Посад завтра, один взрослый", "destination_known", None, "Сергиев Посад"),
    ("unknown_01", "Куда уехать из Москвы на выходные", "destination_unknown", "Москва", None),
    ("unknown_02", "Хочу сменить обстановку на два дня", "destination_unknown", None, None),
    ("unknown_03", "Куда-нибудь с красивой архитектурой", "destination_unknown", None, None),
    (
        "unknown_04",
        "Посоветуй недорогой город рядом с Москвой",
        "destination_unknown",
        "Москва",
        None,
    ),
    (
        "unknown_05",
        "Хочу природу и спокойный темп без ночной дороги",
        "destination_unknown",
        None,
        None,
    ),
    ("unknown_06", "Куда съездить вдвоём до 25000 рублей", "destination_unknown", None, None),
    (
        "unknown_07",
        "Нужны музеи и прогулки, максимум три часа в пути",
        "destination_unknown",
        None,
        None,
    ),
    ("unknown_08", "Предложи гастрономические выходные", "destination_unknown", None, None),
    (
        "unknown_09",
        "Не знаю куда, хочу фотографии и старый город",
        "destination_unknown",
        None,
        None,
    ),
    ("unknown_10", "Куда поехать одному на следующие выходные", "destination_unknown", None, None),
    ("event_01", "Еду на концерт в Казань 22 августа", "event_led", None, "Казань"),
    ("event_02", "Нужно попасть на свадьбу в Туле", "event_led", None, "Тула"),
    ("event_03", "Хочу выбрать город с фестивалем на выходных", "event_led", None, None),
    ("event_04", "Из Москвы на матч в Нижний Новгород", "event_led", "Москва", "Нижний Новгород"),
    ("event_05", "Еду к друзьям в Ярославль", "event_led", None, "Ярославль"),
    ("event_06", "Подбери поездку ради выставки, город не выбрал", "event_led", None, None),
    ("event_07", "В Рязань на день рождения 16 августа", "event_led", None, "Рязань"),
    ("event_08", "Нужно успеть на спектакль в Твери к 19:00", "event_led", None, "Тверь"),
    ("event_09", "Куда поехать на интересный концерт", "event_led", None, None),
    ("event_10", "Встреча с друзьями в Калуге на выходных", "event_led", None, "Калуга"),
)

VARIANTS = (
    lambda text: text,
    lambda text: text.casefold(),
    lambda text: f"Помоги: {text}",
    lambda text: f"{text}, пожалуйста",
)


def build() -> list[dict[str, str | None]]:
    records: list[dict[str, str | None]] = []
    for case_id, text, intent, origin, destination in BASE_CASES:
        for variant_index, transform in enumerate(VARIANTS, start=1):
            records.append(
                {
                    "id": f"{case_id}_v{variant_index}",
                    "prompt_version": "intent_v1",
                    "text": transform(text),
                    "expected_intent": intent,
                    "expected_origin": origin,
                    "expected_destination": destination,
                }
            )
    return records


def main() -> None:
    records = build()
    if len(records) != 120:
        raise RuntimeError(f"expected 120 cases, got {len(records)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
