"""Central brand voice and bounded conversational copy for Telegram flows."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass

BRAND_NAME = "Ту-да и обратно"
VOICE_VERSION = "voice_v2"
BOT_SHORT_DESCRIPTION = "Соберу реальную поездку на выходные: дорога, отель, план и бюджет."
BOT_DESCRIPTION = (
    "Ту-да и обратно помогает выбрать направление или проверить готовый маршрут. "
    "Сравнивает дорогу и жильё, показывает программу короткой поездки, полную известную "
    "стоимость и следующий шаг к оформлению на Tutu."
)

ROUTER_START = (
    "<b>Ту-да и обратно</b>\nКуда на выходные?\n"
    "Если город уже есть — сравню дорогу и отель. Если хочется «куда-нибудь, только не "
    "дома» — предложу несколько проверенных идей.\n\n"
    "Подбор идей пока работает для выезда из Москвы; готовый маршрут можно проверить из "
    "другого города.\n\n"
    "Напишите одним сообщением город отправления, даты и чего хочется от поездки. "
    "Бюджет тоже пригодится.\n\n"
    "Например: «Из Москвы 15–16 августа, спокойно гулять и смотреть старый город, "
    "до 25 000 ₽».\n\n"
    "Для разбора текста используется OpenAI. Не присылайте паспортные и платёжные данные — "
    "подробнее в /privacy."
)

KNOWN_START = (
    "<b>Маршрут уже выбран — соберём поездку целиком.</b>\n"
    "Напишите одним сообщением города, даты, пассажиров, подходящий транспорт, нужен ли "
    "отель и общий бюджет.\n\n"
    "Например: «Москва — Казань, 25–27 июля, поезд, двое взрослых, нужен отель, "
    "до 30 000 ₽».\n\n"
    "Для разбора текста используется OpenAI. Не присылайте паспортные и платёжные данные. "
    "Незавершённый диалог удалится через 30 минут — подробнее в /privacy."
)

DISCOVERY_START = (
    "<b>Подберём идею для короткой поездки.</b>\n"
    "На этапе MVP подбор направлений работает для выезда из Москвы. Если город уже выбран, "
    "готовый маршрут из другого города можно проверить через /newtrip.\n\n"
    "Напишите, откуда и на какие даты хотите уехать и каких впечатлений не хватает: "
    "прогулок, истории, природы, гастрономии или просто смены обстановки. Бюджет и "
    "максимальное время в дороге можно указать сразу.\n\n"
    "Например: «Из Москвы 15–16 августа, спокойно гулять, смотреть архитектуру, "
    "до 25 000 ₽»."
)

HELP = (
    "<b>Что умеет «Ту-да и обратно»</b>\n"
    "/newtrip — проверить поездку в выбранный город\n"
    "/ideas — подобрать направление под даты, бюджет и настроение\n"
    "/cancel — удалить параметры текущего диалога\n"
    "/privacy — узнать, как обрабатываются данные\n\n"
    "Покажу до трёх проверенных вариантов с дорогой, жильём, временем в городе и "
    "известной стоимостью."
)

PRIVACY = (
    "Текст запроса передаётся OpenAI только для извлечения параметров поездки. "
    "Подтверждённые параметры используются для поиска на Tutu. Не присылайте паспортные, "
    "платёжные и другие чувствительные данные.\n\n"
    "Текущий диалог хранится в памяти до /cancel, перезапуска или 30 минут бездействия. "
    "Обращения /feedback хранятся отдельно в течение настроенного срока и удаляются по "
    "номеру через /deletefeedback."
)

PROGRESS_COPY = {
    "route": "Считываю вводные: маршрут, даты и настроение поездки…",
    "known_parse": "Разбираю маршрут, даты и ограничения…",
    "known_patch": "Обновляю параметры, остальное оставляю без изменений…",
    "known_search": "Ищу варианты на Tutu и сверяю поездку целиком…",
    "discovery_parse": "Считываю даты, бюджет и настроение поездки…",
    "discovery_clarify": "Добавляю ответы к вашей подборке…",
    "discovery_refine": "Обновляю пожелания и пересобираю подборку…",
    "discovery_select": "Подбираю направления под ваши интересы и ограничения…",
    "discovery_verify": "Идеи есть. Проверяю дорогу, жильё и время в городе…",
    "discovery_recheck": "Обновляю цены и доступность для этой подборки…",
    "handoff": "Готовлю ссылки на Tutu для проверки условий и оформления…",
}

LEGACY_PROGRESS_COPY = {
    "route": "Определяю сценарий и параметры поездки…",
    "known_parse": "Разбираю маршрут и ограничения…",
    "known_patch": "Применяю изменения…",
    "known_search": "Ищу транспорт на Tutu…",
    "discovery_parse": "Понимаю пожелания и ограничения…",
    "discovery_clarify": "Учитываю ответы…",
    "discovery_refine": "Применяю изменения к подборке…",
    "discovery_select": "Подбираю направления по интересам и ограничениям…",
    "discovery_verify": "Проверяю расписание и жильё для лучших направлений…",
    "discovery_recheck": "Повторно проверяю цены и доступность…",
    "handoff": "Готовлю безопасные ссылки для оформления…",
}

DELIGHT_COPY = {
    "known_search": "Проверяю, чтобы выходные не превратились в экскурсию по пересадкам.",
    "discovery_verify": "Красивой идеи мало — она ещё должна нормально складываться по логистике.",
}

FORBIDDEN_SLANG = frozenset(
    {
        "вайб",
        "вайбовый",
        "имба",
        "краш",
        "кринж",
        "топчик",
        "чилл",
        "чилловый",
    }
)

_USED_DELIGHT_KEY = "voice_used_delight"


@dataclass(frozen=True, slots=True)
class Voice:
    """Render consistent copy while keeping delight deterministic and non-repeating."""

    tone_v2_enabled: bool = True
    controlled_delight_enabled: bool = True

    @property
    def version(self) -> str:
        return VOICE_VERSION if self.tone_v2_enabled else "voice_v1"

    @property
    def router_start(self) -> str:
        return (
            ROUTER_START
            if self.tone_v2_enabled
            else (
                "Помогу и с готовым маршрутом, и с выбором направления. Опишите поездку "
                "одним сообщением — я определю сценарий. Подробнее: /privacy"
            )
        )

    @property
    def known_start(self) -> str:
        return (
            KNOWN_START
            if self.tone_v2_enabled
            else (
                "Напишите маршрут, даты, пассажиров, транспорт, отель и бюджет одним сообщением. "
                "Подробнее: /privacy"
            )
        )

    @property
    def discovery_start(self) -> str:
        return (
            DISCOVERY_START
            if self.tone_v2_enabled
            else (
                "Напишите город отправления, даты и что хочется получить от поездки. "
                "Бюджет и допустимое время в дороге можно добавить сразу."
            )
        )

    def progress(
        self,
        key: str,
        user_data: MutableMapping[str, object] | None = None,
    ) -> str:
        text = (PROGRESS_COPY if self.tone_v2_enabled else LEGACY_PROGRESS_COPY)[key]
        if (
            not self.tone_v2_enabled
            or not self.controlled_delight_enabled
            or key not in DELIGHT_COPY
            or user_data is None
        ):
            return text
        used = user_data.get(_USED_DELIGHT_KEY)
        used_keys = set(used) if isinstance(used, (set, frozenset, tuple, list)) else set()
        if used_keys:
            return text
        used_keys.add(key)
        user_data[_USED_DELIGHT_KEY] = frozenset(used_keys)
        return f"{text}\n{DELIGHT_COPY[key]}"

    @property
    def analytics_dimensions(self) -> dict[str, str]:
        return {"voice_version": self.version}
