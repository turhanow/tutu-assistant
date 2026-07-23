# Техническая документация «Ту-да и обратно»

Актуально для функционального MVP на Python `>=3.11,<3.15`. Источник истины — код,
`app/config.py`, locked dependencies и GitHub workflows; документация должна обновляться
вместе с изменением публичного поведения.

Версия документации: `0.2.x`; последнее ревью — 23 июля 2026 года.

## Навигация

- [Архитектура](#архитектура)
- [Роль LLM](#роль-llm-и-границы-доверия)
- [Конфигурация и запуск](#конфигурация-и-локальный-запуск)
- [Тестирование](#тестирование)
- [Безопасность и данные](#безопасность-и-данные)
- [CI/CD и эксплуатация](#cicd-и-эксплуатация)
- [Как расширять](#как-расширять-систему)

## Стек

| Задача | Технология |
|---|---|
| Telegram | `python-telegram-bot`, polling/webhook, conversations, callbacks |
| AI | OpenAI Responses API, Structured Outputs |
| Модели и настройки | Pydantic v2, pydantic-settings |
| Travel-поиск | MCP client и Tutu MCP |
| Webhook | Starlette + Uvicorn |
| Локальный feedback | SQLite |
| Качество | Pytest, Ruff, coverage 80% |
| Поставка | Docker, GitHub Actions, GHCR/GAR, Cloud Run |

Зависимости закреплены в `requirements.lock` и `requirements.runtime.lock`, пакет описан
в `pyproject.toml`.

Термины: **draft** — ещё не подтверждённые параметры; **fallback** — безопасный запасной
путь; **gateway** — граница внешнего сервиса; **handoff** — переход к оформлению;
**grounded facts** — факты из разрешённых и проверенных данных.

## Архитектура

Проект — модульный монолит с портами и адаптерами:

| Каталог | Ответственность |
|---|---|
| `app/domain` | Pydantic-модели и продуктовые инварианты |
| `app/services` | Планирование, совместимость, бюджет, ранжирование, resilience |
| `app/ports` | Контракты LLM, каталога, Tutu, аналитики и feedback |
| `app/adapters` | OpenAI, MCP, файловый/AI-каталог, SQLite, clock |
| `app/bot` | Telegram UX, routing, состояния, callbacks, formatters |
| `app/prompts` | Версионируемые инструкции отдельных LLM-задач |
| `app/main.py`, `app/bootstrap.py` | Composition root, сборка LLM и lifecycle ресурсов |

```mermaid
flowchart TD
    TG[Telegram] --> R[BotRouter]
    R --> K[/newtrip]
    R --> D[/ideas]
    K --> IX[OpenAI Structured Outputs]
    D --> IX
    K --> TP[TripPlanner]
    D --> CS[CandidateSelector]
    CS --> HC[AI + файловый каталог]
    D --> DP[DiscoveryPlanner]
    TP --> GW[Protected Tutu gateway]
    DP --> TP
    GW --> MCP[Tutu MCP]
    D --> PB[ProposalBuilder]
    PB --> N[Grounded narration]
    K --> F[Telegram formatters]
    N --> F
    F --> TG
    F --> H[Trip handoff]
    H --> GW
```

### `/newtrip`: известный маршрут

1. LLM преобразует текст в `ParsedTripDraft`.
2. Request builder проверяет маршрут, даты, состав, отель и событие.
3. `TripPlanner` получает транспорт и отели, строит совместимые комбинации.
4. Ranking выделяет бюджетный, быстрый и сбалансированный варианты.
5. Formatter показывает факты и только offer-specific ссылки Tutu.

### `/ideas`: неизвестное направление

1. Intent extractor создаёт `DiscoveryDraft`; follow-up дополняют сохранённый контекст.
2. Clarification policy спрашивает критичные недостающие поля.
3. Hybrid catalog объединяет AI-гипотезы и файловый fallback.
4. Candidate selector формирует разнообразный shortlist.
5. Discovery planner ограниченно-параллельно проверяет логистику через Tutu.
6. Itinerary builder распределяет уникальные активности по дням.
7. Proposal builder разделяет подтверждённую цену, оценки и неизвестные расходы.
8. Narration переформулирует allowlist фактов или использует deterministic fallback.

Состояние conversation хранится в памяти. `ChatSerialUpdateProcessor` сохраняет порядок
сообщений одного чата, одновременно позволяя ограниченную обработку разных чатов.

## Роль LLM и границы доверия

LLM обязателен для понимания свободного текста и AI-discovery, но не управляет ценой,
расписанием или совместимостью:

- ответы модели проходят через Structured Outputs и Pydantic;
- сокращения, разговорные названия и падежные формы городов нормализуются моделью с
  контекстом текущего вопроса; неоднозначный ответ возвращается на уточнение;
- известные даты дополнительно восстанавливаются детерминированно;
- нестандартные даты разбираются с текущей локальной датой;
- деньги, длительность, отель, уникальность мест и ranking проверяет доменный слой;
- AI предлагает гипотезы городов и активностей, Tutu проверяет только логистику;
- Яндекс-ссылки строятся приложением как поиск по названию/адресу, без выдуманных org ID;
- `store=False`, output limits и timeout ограничивают хранение, задержку и стоимость.

Fallback зависит от этапа:

| Сбой | Поведение |
|---|---|
| Обязательный intent parsing | Пользователю предлагают повторить запрос |
| Dynamic discovery | Используется файловый каталог, если он подходит |
| Narration | Используется детерминированный текст |
| Tutu MCP | Возвращается безопасная ошибка без выдуманных вариантов |

`OPENAI_MODEL` должен быть доступен используемой учётной записи. При замене модели нужно
прогнать structured-output tests, offline evals и отдельный opt-in live eval.

## Tone of Voice

Голос бренда: компетентный travel-приятель — современный, спокойный и честный об
ограничениях. Формула: **70% ясности, 20% вдохновения, 10% тонкой самоиронии**.

- обращение на «вы», короткие предложения, один следующий шаг в сообщении;
- не более одной лёгкой реплики за диалог;
- юмор запрещён в ошибках, цене, privacy, feedback и оформлении;
- без FOMO, рекламных обещаний и шуток о бюджете или ограничениях пользователя;
- запрещён искусственный сленг: «вайб», «имба», «кринж», «топчик», «чилл»;
- факты, кнопки, деньги и предупреждения формируются детерминированно.

Реализация управляется `TONE_OF_VOICE_V2_ENABLED` и
`CONTROLLED_DELIGHT_ENABLED`; версия голоса попадает в безопасную аналитику.

## Конфигурация и локальный запуск

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pip check
cp .env.example .env
```

Минимальный локальный `.env`:

```dotenv
TELEGRAM_BOT_TOKEN=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.6-sol
BOT_TRANSPORT=polling
```

Не коммитьте `.env`. Запуск:

```bash
python -m app.main
```

Polling снимает активный webhook и запускается с `drop_pending_updates=True`: старые
необработанные updates будут удалены. Нельзя одновременно использовать второй polling
process или production webhook с тем же token.

Основные настройки:

| Группа | Переменные |
|---|---|
| Telegram | `BOT_TRANSPORT`, `PUBLIC_BASE_URL`, `TELEGRAM_WEBHOOK_SECRET` |
| OpenAI | `OPENAI_MODEL`, `OPENAI_TIMEOUT_SECONDS`, `OPENAI_BASE_URL` |
| Tutu | `TUTU_MCP_URL`, `TUTU_TIMEOUT_SECONDS`, `TUTU_POOL_SIZE` |
| Concurrency | `MAX_CONCURRENT_UPDATES`, `DISCOVERY_MAX_CONCURRENCY` |
| Resilience | `PROVIDER_MAX_INFLIGHT`, rate budget, circuit breaker settings |
| Features | `DISCOVERY_ENABLED`, `DYNAMIC_DISCOVERY_ENABLED`, `FEEDBACK_ENABLED` |
| Storage | `FEEDBACK_DB_PATH`, `FEEDBACK_RETENTION_DAYS` |

Внешние URL обязаны использовать HTTPS. OpenAI key нельзя передавать на custom host без
явного `OPENAI_ALLOW_CUSTOM_BASE_URL=true`.

### Docker и webhook

```bash
docker build -t tutu-assistant:local .
docker run --rm --env-file .env tutu-assistant:local
```

Команда предполагает `BOT_TRANSPORT=polling` в `.env`. Сам production image по умолчанию
настроен на webhook. Для него обязательны HTTPS `PUBLIC_BASE_URL` и случайный
`TELEGRAM_WEBHOOK_SECRET`; Telegram endpoint — `/telegram/webhook`.

## Тестирование

```bash
python -m ruff check app tests scripts
python -m ruff format --check app tests scripts
python -m pytest --cov=app --cov-report=term-missing
python scripts/validate_destination_catalog.py
python scripts/validate_cicd.py
python scripts/check_secrets.py
```

Основные тесты offline: LLM и MCP заменяются fake/contract adapters. Каждый исправленный
пользовательский дефект должен получить тест на наблюдаемое поведение Telegram, а при
изменении parser — ещё и unit-тест входной формулировки.

AI-проверки:

| Уровень | Команда | Внешний вызов |
|---|---|---|
| Offline regression evals | `python scripts/run_discovery_evals.py` | Нет |
| Live narration | `python scripts/run_discovery_evals.py --live-narration` | OpenAI, платно |
| Tutu smoke | `RUN_TUTU_INTEGRATION=1 python -m pytest tests/integration/test_tutu_smoke.py -q` | Tutu |

Coverage gate — 80%. Live-команды запускаются только с локальными credentials и не
должны выводить ключи.

## Безопасность и данные

| Данные | Куда передаются | Хранение приложением |
|---|---|---|
| Текст запроса | OpenAI | `store=False`; в conversation остаются структурированные параметры |
| Параметры поиска | Tutu | Только на время поиска |
| Контекст | Память процесса | До `/reset`, `/cancel`, перезапуска или 30 минут |
| Геопозиция | Telegram update | Координаты не сохраняются |
| Feedback | SQLite/будущий внешний store | До `FEEDBACK_RETENTION_DAYS`, удаление по номеру |

Дополнительные меры:

- секреты — `SecretStr`, Google Secret Manager и GitHub OIDC без JSON keys;
- sensitive-data guard блокирует паспортные и платёжные данные до LLM;
- структурированные product events используют только разрешённые dimensions;
- circuit breaker, semaphore, rate budget и timeout защищают внешние вызовы;
- callback содержит flow/revision и отклоняет устаревшие кнопки;
- webhook проверяет secret header и ограничивает размер тела;
- Telegram HTML делится по видимой длине, не обрезая checkout URL;
- доменный инвариант запрещает повтор одного места в поездке.

Прикладные логи и аналитика не должны включать текст запроса. При добавлении нового
exception logging отдельно проверяйте, что стороннее исключение не раскрывает входные
данные или секреты.

## CI/CD и эксплуатация

`ci.yml` для `main` и pull request выполняет locked install, lint/format, тесты с
coverage, проверки каталога и контрактов, secret scan, dependency audit и container build.

Тег `vX.Y.Z` запускает `release.yml`:

1. повторяет quality gate;
2. публикует multi-platform OCI image в GHCR и GAR;
3. создаёт SBOM, provenance и attestation;
4. деплоит точный digest в production Cloud Run через OIDC;
5. проверяет `/health` и `/readyz`.

Необходимые GitHub Variables и Secret Manager resources перечислены непосредственно в
`release.yml`. Production Environment рекомендуется защитить обязательным approval.

Текущий Cloud Run: `min-instances=0`, `max-instances=1`, concurrency 16. Возможен cold
start. Один instance обязателен, пока conversation state находится в памяти. Feedback в
workflow отключён (`FEEDBACK_ENABLED=false`), поскольку локальный SQLite неустойчив в
serverless deployment.

### Health и восстановление

- `/health` — процесс запущен;
- `/readyz` — каталог загружен, MCP contract получен, circuit закрыт, kill switch выключен;
- `/healthz` — совместимый alias, который может перехватываться Google Frontend.

Если `/readyz` не готов, проверьте provider warmup, schema hash, circuit state и kill
switch. Ошибка синхронизации имени/команд Telegram из-за flood control не останавливает
обработку сообщений.

`promote.yml` проверяет digest и переставляет environment-tag в GHCR, но пока не
передеплоит Cloud Run. Production rollback выполняется авторизованным развёртыванием
ранее рабочего GAR digest; отдельный one-click rollback workflow остаётся задачей после
MVP. Секреты в `release.yml` привязаны к конкретным версиям: после ротации обновите
ссылки на версии и выпустите новый релиз.

## Как расширять систему

| Изменение | Где начинать |
|---|---|
| Telegram-текст | `app/voice.py` или formatter + conversation test |
| Prompt | `app/prompts` + Structured Output schema + eval |
| Параметр поездки | draft → request builder → provider mapping → formatter |
| Каталог/источник | `DestinationCatalog` или `TravelContentGateway` adapter |
| Travel provider | реализация gateway port |
| Analytics/feedback | новый sink adapter |

Перед commit: добавьте регрессионный тест, выполните lint, format check, pytest, catalog
validation и secret scan. Не меняйте Telegram, LLM и provider одновременно без отдельного
контрактного теста на каждой границе.

После MVP для горизонтального масштабирования нужны общий conversation store,
распределённые блокировки/rate limits и внешняя БД для feedback. Следующие продуктовые
расширения — новые города отправления, события, уведомления о цене и booking attribution.
