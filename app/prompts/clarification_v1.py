"""Versioned deterministic clarification policy shared with Telegram UX."""

PROMPT_VERSION = "clarification_v1"

FIELD_PRIORITY = (
    "origin",
    "dates",
    "budget",
    "motives",
    "hotel_mode",
    "road_tolerance",
    "travelers",
)

QUESTIONS = {
    "origin": "Из какого города вы хотите поехать?",
    "dates": "Какие выходные рассматриваете?",
    "budget": "Какой общий бюджет комфортен? Можно ответить «без лимита».",
    "motives": "Что хочется получить: смену обстановки, культуру, природу или гастрономию?",
    "hotel_mode": "Нужен отель: да, нет или можно показать оба варианта?",
    "road_tolerance": "Сколько времени в одну сторону допустимо и подходит ли ночная дорога?",
    "travelers": "Вы едете один или вдвоём?",
}


def select_questions(missing_fields: tuple[str, ...], *, limit: int = 3) -> tuple[str, ...]:
    if limit < 1:
        return ()
    missing = set(missing_fields)
    selected = [QUESTIONS[field] for field in FIELD_PRIORITY if field in missing]
    return tuple(selected[: min(limit, 3)])
