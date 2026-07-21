import json
import logging

from app.logging_config import RedactingFormatter, StructuredRedactingFormatter


def test_formatter_redacts_secrets_from_message_and_exception() -> None:
    secret = "123456:telegram-secret"
    formatter = RedactingFormatter((secret,))
    try:
        raise RuntimeError(f"request failed at /bot{secret}/getMe")
    except RuntimeError:
        record = logging.LogRecord(
            "test",
            logging.ERROR,
            __file__,
            1,
            "provider error: %s",
            (secret,),
            exc_info=__import__("sys").exc_info(),
        )

    rendered = formatter.format(record)

    assert secret not in rendered
    assert rendered.count("[REDACTED]") == 2


def test_structured_formatter_allowlists_product_metrics_and_redacts() -> None:
    secret = "openai-secret"
    formatter = StructuredRedactingFormatter((secret,))
    record = logging.LogRecord(
        "app.product",
        logging.INFO,
        __file__,
        1,
        "completed %s",
        (secret,),
        exc_info=None,
    )
    record.event = "search_completed"
    record.result_count = 3
    record.raw_user_text = "Москва — Казань"

    payload = json.loads(formatter.format(record))

    assert payload["event"] == "search_completed"
    assert payload["result_count"] == 3
    assert "raw_user_text" not in payload
    assert secret not in payload["message"]
