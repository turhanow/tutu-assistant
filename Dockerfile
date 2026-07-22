# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements.runtime.lock ./
RUN python -m pip wheel --wheel-dir /wheels -r requirements.runtime.lock

COPY pyproject.toml README.md ./
COPY app ./app
RUN python -m pip wheel --no-deps --wheel-dir /wheels .


FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93 AS runtime

LABEL org.opencontainers.image.title="Ту-да и обратно" \
      org.opencontainers.image.description="Telegram assistant for short-trip planning"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    BOT_TRANSPORT=webhook \
    PORT=8080 \
    DESTINATION_CATALOG_PATH=/app/data/destinations/v1/catalog.json \
    FEEDBACK_DB_PATH=/app/var/feedback.sqlite3

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --home-dir /home/app app

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels tutu-assistant \
    && rm -rf /wheels

COPY --chown=10001:10001 data ./data
RUN mkdir -p /app/var && chown 10001:10001 /app/var

USER 10001:10001

VOLUME ["/app/var"]

EXPOSE 8080

CMD ["python", "-m", "app.main"]
