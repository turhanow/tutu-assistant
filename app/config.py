"""Environment-backed application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with secrets kept out of repr and logs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=True,
    )

    telegram_bot_token: SecretStr | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    bot_transport: Literal["polling", "webhook"] = Field(default="polling", alias="BOT_TRANSPORT")
    public_base_url: AnyHttpUrl | None = Field(default=None, alias="PUBLIC_BASE_URL")
    telegram_webhook_secret: SecretStr | None = Field(
        default=None,
        alias="TELEGRAM_WEBHOOK_SECRET",
    )
    port: int = Field(default=8080, alias="PORT", ge=1, le=65_535)
    webhook_max_connections: int = Field(
        default=16,
        alias="WEBHOOK_MAX_CONNECTIONS",
        ge=1,
        le=100,
    )
    webhook_max_body_bytes: int = Field(
        default=1_000_000,
        alias="WEBHOOK_MAX_BODY_BYTES",
        ge=1_024,
        le=4_000_000,
    )
    tutu_mcp_url: AnyHttpUrl = Field(default="https://mcp.tutu.ru/mcp", alias="TUTU_MCP_URL")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: AnyHttpUrl = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    openai_model: str = Field(default="gpt-5.6-sol", alias="OPENAI_MODEL", min_length=1)
    openai_timeout_seconds: float = Field(default=25, alias="OPENAI_TIMEOUT_SECONDS", ge=1, le=30)
    openai_allow_custom_base_url: bool = Field(default=False, alias="OPENAI_ALLOW_CUSTOM_BASE_URL")
    tutu_timeout_seconds: int = Field(default=12, alias="TUTU_TIMEOUT_SECONDS", ge=3, le=60)
    tutu_pool_size: int = Field(default=5, alias="TUTU_POOL_SIZE", ge=1, le=8)
    search_timeout_seconds: int = Field(default=25, alias="SEARCH_TIMEOUT_SECONDS", ge=5, le=60)
    discovery_verification_timeout_seconds: int = Field(
        default=18,
        alias="DISCOVERY_VERIFICATION_TIMEOUT_SECONDS",
        ge=5,
        le=25,
    )
    discovery_max_concurrency: int = Field(
        default=5,
        alias="DISCOVERY_MAX_CONCURRENCY",
        ge=1,
        le=5,
    )
    max_concurrent_updates: int = Field(default=16, alias="MAX_CONCURRENT_UPDATES", ge=1, le=100)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    app_timezone: str = Field(default="Europe/Moscow", alias="APP_TIMEZONE")
    destination_catalog_path: Path = Field(
        default=Path(__file__).resolve().parent.parent / "data/destinations/v1/catalog.json",
        alias="DESTINATION_CATALOG_PATH",
    )
    analytics_queue_size: int = Field(
        default=256,
        alias="ANALYTICS_QUEUE_SIZE",
        ge=1,
        le=10_000,
    )
    analytics_flush_timeout_seconds: float = Field(
        default=2,
        alias="ANALYTICS_FLUSH_TIMEOUT_SECONDS",
        gt=0,
        le=30,
    )
    feedback_db_path: Path = Field(
        default=Path(__file__).resolve().parent.parent / "var/feedback.sqlite3",
        alias="FEEDBACK_DB_PATH",
    )
    feedback_retention_days: int = Field(
        default=30,
        alias="FEEDBACK_RETENTION_DAYS",
        ge=1,
        le=365,
    )
    feedback_timeout_seconds: float = Field(
        default=2,
        alias="FEEDBACK_TIMEOUT_SECONDS",
        gt=0,
        le=10,
    )
    bot_kill_switch: bool = Field(default=False, alias="BOT_KILL_SWITCH")
    tone_of_voice_v2_enabled: bool = Field(default=True, alias="TONE_OF_VOICE_V2_ENABLED")
    controlled_delight_enabled: bool = Field(default=True, alias="CONTROLLED_DELIGHT_ENABLED")
    discovery_enabled: bool = Field(default=True, alias="DISCOVERY_ENABLED")
    dynamic_discovery_enabled: bool = Field(default=True, alias="DYNAMIC_DISCOVERY_ENABLED")
    dynamic_discovery_timeout_seconds: float = Field(
        default=30,
        alias="DYNAMIC_DISCOVERY_TIMEOUT_SECONDS",
        ge=5,
        le=30,
    )
    feedback_enabled: bool = Field(default=True, alias="FEEDBACK_ENABLED")
    analytics_enabled: bool = Field(default=True, alias="ANALYTICS_ENABLED")
    provider_max_inflight: int = Field(default=8, alias="PROVIDER_MAX_INFLIGHT", ge=1, le=32)
    provider_calls_per_minute: int = Field(
        default=120, alias="PROVIDER_CALLS_PER_MINUTE", ge=1, le=10_000
    )
    provider_circuit_failure_threshold: int = Field(
        default=3, alias="PROVIDER_CIRCUIT_FAILURE_THRESHOLD", ge=1, le=20
    )
    provider_circuit_recovery_seconds: float = Field(
        default=30, alias="PROVIDER_CIRCUIT_RECOVERY_SECONDS", gt=0, le=600
    )

    @field_validator("tutu_mcp_url", "openai_base_url", "public_base_url")
    @classmethod
    def require_https(cls, value: AnyHttpUrl | None) -> AnyHttpUrl | None:
        if value is not None and value.scheme != "https":
            raise ValueError("external service URLs must use HTTPS")
        return value

    @field_validator("app_timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("APP_TIMEZONE must be a valid IANA timezone") from error
        return value

    @model_validator(mode="after")
    def protect_openai_credentials(self) -> Settings:
        host = (self.openai_base_url.host or "").casefold()
        if not self.openai_allow_custom_base_url and host != "api.openai.com":
            raise ValueError(
                "OPENAI_BASE_URL must use api.openai.com unless "
                "OPENAI_ALLOW_CUSTOM_BASE_URL=true is explicitly set"
            )
        return self

    @model_validator(mode="after")
    def validate_webhook_configuration(self) -> Settings:
        if self.bot_transport != "webhook":
            return self
        if self.public_base_url is None:
            raise ValueError("PUBLIC_BASE_URL is required when BOT_TRANSPORT=webhook")
        if self.public_base_url.path not in {"", "/"}:
            raise ValueError("PUBLIC_BASE_URL must not contain a path")
        if self.public_base_url.query or self.public_base_url.fragment:
            raise ValueError("PUBLIC_BASE_URL must not contain a query or fragment")
        secret = self.require_webhook_secret().get_secret_value()
        if not 1 <= len(secret) <= 256 or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in secret
        ):
            raise ValueError(
                "TELEGRAM_WEBHOOK_SECRET must contain 1-256 characters from A-Z, a-z, 0-9, _ and -"
            )
        return self

    def require_bot_token(self) -> SecretStr:
        if self.telegram_bot_token is None or not self.telegram_bot_token.get_secret_value():
            raise ValueError("TELEGRAM_BOT_TOKEN is required to start the bot")
        return self.telegram_bot_token

    def require_openai_api_key(self) -> SecretStr:
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value():
            raise ValueError("OPENAI_API_KEY is required to start the bot")
        return self.openai_api_key

    def require_webhook_secret(self) -> SecretStr:
        if (
            self.telegram_webhook_secret is None
            or not self.telegram_webhook_secret.get_secret_value()
        ):
            raise ValueError("TELEGRAM_WEBHOOK_SECRET is required when BOT_TRANSPORT=webhook")
        return self.telegram_webhook_secret

    @property
    def webhook_url(self) -> str:
        if self.public_base_url is None:
            raise ValueError("PUBLIC_BASE_URL is required to build the webhook URL")
        return f"{str(self.public_base_url).rstrip('/')}/telegram/webhook"

    def validate_runtime(self) -> None:
        """Fail closed before starting a transport without exposing credential values."""
        self.require_bot_token()
        self.require_openai_api_key()
        if self.bot_transport == "webhook":
            self.require_webhook_secret()
