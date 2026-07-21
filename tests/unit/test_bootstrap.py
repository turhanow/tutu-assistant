import pytest

from app.bootstrap import build_llm_resources
from app.config import Settings


@pytest.mark.asyncio
async def test_llm_resources_do_not_expose_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-openai-secret")
    settings = Settings(_env_file=None)

    resources = build_llm_resources(settings)
    try:
        assert "test-only-openai-secret" not in repr(settings)
        assert "test-only-openai-secret" not in repr(resources)
        assert resources.parser is not None
        assert resources.intent_extractor is not None
        assert resources.proposal_narrator is not None
    finally:
        await resources.close()
