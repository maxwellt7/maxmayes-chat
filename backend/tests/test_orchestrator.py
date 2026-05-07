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


async def test_run_pipeline_deactivates_broken_index(monkeypatch):
    """If retrieve raises (e.g., 404), the index should be deactivated and a friendly message yielded."""
    _patch_settings(monkeypatch, openai_api_key="test-openai")

    from unittest.mock import MagicMock
    from app.services.orchestrator import run_pipeline

    fake_entry = MagicMock()
    fake_entry.index_name = "broken-index"
    fake_entry.project_id = "1"
    fake_entry.dimension = 1024
    fake_entry.is_active = True
    fake_entry.domain_description = "test"
    fake_entry.sample_queries = []

    fake_db = MagicMock()
    # First call: list of active indexes — returns one entry
    # Second call: fetch single registry entry by name+project — also returns the entry
    fake_db.query.return_value.filter.return_value.all.return_value = [fake_entry]
    fake_db.query.return_value.filter.return_value.first.return_value = fake_entry

    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI, \
         patch("app.services.orchestrator.generate_embedding") as mock_embed, \
         patch("app.services.orchestrator.retrieve") as mock_retrieve, \
         patch("app.services.orchestrator.IndexRegistry") as MockRegistry:

        # Make IndexRegistry.is_active behave like a real column for filter()
        MockRegistry.is_active = MagicMock()
        MockRegistry.index_name = MagicMock()
        MockRegistry.project_id = MagicMock()

        # DB query chain: .filter().all() → active list; .filter().first() → single entry
        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = [fake_entry]
        mock_query.filter.return_value.first.return_value = fake_entry
        fake_db.query.return_value = mock_query

        # optimize_query and route_query use OpenAI
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            MagicMock(choices=[MagicMock(message=MagicMock(content="optimized"))]),  # optimize
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
                "index_name": "broken-index", "project_id": "1", "confidence": 0.95
            })))]),  # route
        ])
        MockOpenAI.return_value = mock_client

        mock_embed.return_value = [0.1] * 1024
        mock_retrieve.side_effect = Exception("Pinecone 404 NOT_FOUND")

        chunks = []
        async for chunk in run_pipeline("test query", fake_db):
            chunks.append(chunk)

        assert any("moved or removed" in c for c in chunks), f"Expected friendly fallback, got: {chunks}"
        assert fake_entry.is_active is False, "Broken index should have been deactivated"
        fake_db.commit.assert_called()
