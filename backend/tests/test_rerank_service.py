from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.rerank_service import rerank


@pytest.mark.asyncio
async def test_rerank_returns_reordered_chunks(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test-cohere")
    chunks = [{"text": "first"}, {"text": "second"}, {"text": "third"}]

    with patch("app.services.rerank_service.cohere.AsyncClientV2") as MockClient:
        mock_results = [MagicMock(index=2), MagicMock(index=0), MagicMock(index=1)]
        MockClient.return_value.rerank = AsyncMock(
            return_value=MagicMock(results=mock_results)
        )
        result = await rerank("query", chunks, top_n=3)

    assert [c["text"] for c in result] == ["third", "first", "second"]


@pytest.mark.asyncio
async def test_rerank_falls_back_to_input_order_on_no_key(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "")
    chunks = [{"text": "a"}, {"text": "b"}]
    result = await rerank("query", chunks, top_n=2)
    assert [c["text"] for c in result] == ["a", "b"]


@pytest.mark.asyncio
async def test_rerank_falls_back_on_api_error(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test-cohere")
    chunks = [{"text": "a"}, {"text": "b"}]
    with patch("app.services.rerank_service.cohere.AsyncClientV2") as MockClient:
        MockClient.return_value.rerank = AsyncMock(side_effect=Exception("API down"))
        result = await rerank("query", chunks, top_n=2)
    assert [c["text"] for c in result] == ["a", "b"]


@pytest.mark.asyncio
async def test_rerank_empty_chunks_returns_empty():
    result = await rerank("query", [], top_n=5)
    assert result == []
