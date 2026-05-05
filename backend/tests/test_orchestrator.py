import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.orchestrator import optimize_query, route_query, verify_accuracy


def _patch_settings(monkeypatch, **keys: str) -> None:
    from app.config import settings
    for attr, value in keys.items():
        monkeypatch.setattr(settings, attr, value)


async def test_optimize_query_returns_string(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="test-openai")
    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="onboarding email sequence automation"))]
        ))
        MockOpenAI.return_value = mock_client

        result = await optimize_query("hey whats the deal with that email onboarding thing")

        assert isinstance(result, str)
        assert len(result) > 0


async def test_route_query_returns_index_and_project(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="test-openai")
    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "index_name": "my-index",
                "project_id": "1",
                "confidence": 0.9,
                "reasoning": "Query matches domain"
            })))]
        ))
        MockOpenAI.return_value = mock_client

        catalog = [{"index_name": "my-index", "project_id": "1", "domain_description": "test", "sample_queries": []}]
        result = await route_query("test query", catalog)

        assert result["index_name"] == "my-index"
        assert result["project_id"] == "1"
        assert result["confidence"] == 0.9


async def test_verify_accuracy_proceed(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="test-openai")
    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "score": 0.92,
                "recommendation": "proceed",
                "reasoning": "Context directly answers the query"
            })))]
        ))
        MockOpenAI.return_value = mock_client

        result = await verify_accuracy("test query", [{"text": "relevant content", "score": 0.9}])

        assert result["recommendation"] == "proceed"
        assert result["score"] == 0.92


async def test_verify_accuracy_insufficient(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="test-openai")
    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "score": 0.2,
                "recommendation": "insufficient_context",
                "reasoning": "No relevant content found"
            })))]
        ))
        MockOpenAI.return_value = mock_client

        result = await verify_accuracy("test query", [])

        assert result["recommendation"] == "insufficient_context"
