import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm_service import (
    call_openai_json,
    call_openai_text,
    get_anthropic_client,
)


@pytest.mark.asyncio
async def test_call_openai_json_returns_parsed_dict(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    with patch("app.services.llm_service.AsyncOpenAI") as MockClient:
        mock_resp = MagicMock(choices=[MagicMock(message=MagicMock(
            content=json.dumps({"foo": "bar", "n": 1})
        ))])
        MockClient.return_value.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await call_openai_json("gpt-4.1-mini", "system", "user", temperature=0)
    assert result == {"foo": "bar", "n": 1}


@pytest.mark.asyncio
async def test_call_openai_text_returns_stripped_string(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    with patch("app.services.llm_service.AsyncOpenAI") as MockClient:
        mock_resp = MagicMock(choices=[MagicMock(message=MagicMock(content="  hello  "))])
        MockClient.return_value.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await call_openai_text("gpt-4.1-mini", "system", "user", temperature=0)
    assert result == "hello"


@pytest.mark.asyncio
async def test_call_openai_json_raises_when_key_missing(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "openai_api_key", "")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        await call_openai_json("gpt-4.1-mini", "system", "user")


def test_get_anthropic_client_raises_when_key_missing(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_anthropic_client()


def test_get_anthropic_client_returns_async_client(monkeypatch):
    from app.config import settings
    from anthropic import AsyncAnthropic
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    client = get_anthropic_client()
    assert isinstance(client, AsyncAnthropic)
