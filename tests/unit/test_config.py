import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults_do_not_require_secrets() -> None:
    settings = Settings(_env_file=None)

    assert str(settings.tutu_mcp_url) == "https://mcp.tutu.ru/mcp"
    assert settings.app_timezone == "Europe/Moscow"
    assert settings.openai_model == "gpt-5.6-sol"
    assert str(settings.openai_base_url) == "https://api.openai.com/v1"
    assert "telegram_bot_token" not in repr(settings) or "SecretStr" not in repr(settings)


def test_bot_token_is_validated_only_at_runtime() -> None:
    settings = Settings(_env_file=None)

    try:
        settings.require_bot_token()
    except ValueError as error:
        assert "TELEGRAM_BOT_TOKEN" in str(error)
    else:
        raise AssertionError("missing bot token must fail runtime validation")


def test_openai_key_is_required_and_never_exposed(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-secret")
    settings = Settings(_env_file=None)

    try:
        settings.validate_runtime()
    except ValueError as error:
        assert str(error) == "OPENAI_API_KEY is required to start the bot"
        assert "telegram-secret" not in str(error)
    else:
        raise AssertionError("missing OpenAI key must block polling")


def test_complete_runtime_secrets_are_masked(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    settings = Settings(_env_file=None)

    settings.validate_runtime()
    rendered = repr(settings)
    assert "telegram-secret" not in rendered
    assert "openai-secret" not in rendered


def test_external_urls_require_https(monkeypatch) -> None:
    monkeypatch.setenv("TUTU_MCP_URL", "http://example.com/mcp")

    try:
        Settings(_env_file=None)
    except ValidationError as error:
        assert "HTTPS" in str(error)
    else:
        raise AssertionError("insecure provider URL must be rejected")


def test_openai_key_cannot_be_sent_to_untrusted_host_by_misconfiguration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://evil.example/v1")

    with pytest.raises(ValidationError, match=r"api\.openai\.com"):
        Settings(_env_file=None)


def test_custom_openai_gateway_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://trusted.internal/v1")
    monkeypatch.setenv("OPENAI_ALLOW_CUSTOM_BASE_URL", "true")

    settings = Settings(_env_file=None)

    assert settings.openai_base_url.host == "trusted.internal"


def test_invalid_application_timezone_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("APP_TIMEZONE", "Mars/Olympus")

    with pytest.raises(ValidationError, match="IANA timezone"):
        Settings(_env_file=None)


def test_webhook_mode_requires_public_url_and_secret(monkeypatch) -> None:
    monkeypatch.setenv("BOT_TRANSPORT", "webhook")

    with pytest.raises(ValidationError, match="PUBLIC_BASE_URL"):
        Settings(_env_file=None)


def test_webhook_configuration_builds_fixed_https_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("BOT_TRANSPORT", "webhook")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://bot.example.com")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "safe_webhook-secret_123")

    settings = Settings(_env_file=None)

    assert settings.webhook_url == "https://bot.example.com/telegram/webhook"
    assert "safe_webhook-secret_123" not in repr(settings)


@pytest.mark.parametrize("secret", ["contains space", "bad/slash", "x" * 257])
def test_webhook_secret_rejects_telegram_incompatible_values(monkeypatch, secret) -> None:
    monkeypatch.setenv("BOT_TRANSPORT", "webhook")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://bot.example.com")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", secret)

    with pytest.raises(ValidationError, match="TELEGRAM_WEBHOOK_SECRET"):
        Settings(_env_file=None)
