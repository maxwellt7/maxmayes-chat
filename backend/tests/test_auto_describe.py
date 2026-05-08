import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.auto_describe import generate_index_description


def _patch_settings(monkeypatch, **keys: str) -> None:
    from app.config import settings
    for attr, value in keys.items():
        monkeypatch.setattr(settings, attr, value)


@pytest.mark.asyncio
async def test_generate_index_description_calls_llm_with_samples(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="test-key")

    with patch("app.services.auto_describe.generate_embedding") as mock_embed, \
         patch("app.services.auto_describe.retrieve") as mock_retrieve, \
         patch("app.services.auto_describe.AsyncOpenAI") as MockOpenAI:
        mock_embed.return_value = [0.1] * 1024
        mock_retrieve.return_value = [
            {"text": "Sample copywriting framework chunk", "score": 0.9},
            {"text": "Email subject line formula", "score": 0.85},
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "domain_description": "Email copywriting frameworks",
                "sample_queries": ["q1", "q2", "q3"],
            })))]
        ))
        MockOpenAI.return_value = mock_client

        result = await generate_index_description("max-copywriting-voice", "1", 1024)

        assert result["domain_description"] == "Email copywriting frameworks"
        assert len(result["sample_queries"]) == 3


@pytest.mark.asyncio
async def test_generate_returns_fallback_when_no_chunks(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="test-key")

    with patch("app.services.auto_describe.generate_embedding") as mock_embed, \
         patch("app.services.auto_describe.retrieve") as mock_retrieve:
        mock_embed.return_value = [0.1] * 1024
        mock_retrieve.return_value = []

        result = await generate_index_description("empty-index", "1", 1024)

        assert "No sample content" in result["domain_description"]
        assert result["sample_queries"] == []


@pytest.mark.asyncio
async def test_generate_raises_when_openai_key_missing(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="")

    with patch("app.services.auto_describe.generate_embedding") as mock_embed, \
         patch("app.services.auto_describe.retrieve") as mock_retrieve:
        mock_embed.return_value = [0.1] * 1024
        mock_retrieve.return_value = [{"text": "some content", "score": 0.9}]

        with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
            await generate_index_description("test-index", "1", 1024)
