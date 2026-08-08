import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.orchestrator import (
    _normalize_router_response,
    _rrf_merge,
    optimize_query,
    route_query,
    verify_accuracy,
)


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


async def test_route_query_returns_normalized_candidates(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="test-openai")
    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "candidates": [
                    {"index_name": "my-index", "project_id": "1", "confidence": 0.9, "reasoning": "match"},
                    {"index_name": "alt-index", "project_id": "2", "confidence": 0.5, "reasoning": "adjacent"},
                ],
                "out_of_domain": False,
            })))]
        ))
        MockOpenAI.return_value = mock_client

        catalog = [{"index_name": "my-index", "project_id": "1", "domain_description": "test", "sample_queries": []}]
        result = await route_query("test query", catalog)

        assert result["out_of_domain"] is False
        assert len(result["candidates"]) == 2
        assert result["candidates"][0]["index_name"] == "my-index"
        assert result["candidates"][0]["confidence"] == 0.9


def test_normalize_router_accepts_legacy_shape():
    """A router that returns the old flat shape shouldn't break the pipeline."""
    legacy = {"index_name": "x", "project_id": "1", "confidence": 0.8}
    result = _normalize_router_response(legacy)
    assert result["candidates"] == [
        {"index_name": "x", "project_id": "1", "confidence": 0.8, "reasoning": ""}
    ]
    assert result["out_of_domain"] is False


def test_normalize_router_enforces_out_of_domain_when_all_low():
    """If max confidence < 0.4 the gate fires even when the model forgot to set the flag."""
    raw = {
        "candidates": [
            {"index_name": "a", "project_id": "1", "confidence": 0.2},
            {"index_name": "b", "project_id": "1", "confidence": 0.3},
        ],
        "out_of_domain": False,  # router lied
    }
    result = _normalize_router_response(raw)
    assert result["out_of_domain"] is True


def test_normalize_router_filters_invalid_candidates():
    raw = {
        "candidates": [
            {"index_name": "good", "project_id": "1", "confidence": 0.9},
            {"index_name": "", "project_id": "1"},  # empty name
            {"project_id": "2"},  # missing name
            "not-a-dict",
        ],
        "out_of_domain": False,
    }
    result = _normalize_router_response(raw)
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["index_name"] == "good"


def test_rrf_merge_prefers_chunks_ranked_high_in_multiple_lists():
    """A chunk that appears at rank 1 in two lists should beat a chunk at rank 0
    in only one list, because RRF rewards consistency across sources."""
    list_a = [
        {"text": "shared chunk"},
        {"text": "a-only"},
    ]
    list_b = [
        {"text": "b-only"},
        {"text": "shared chunk"},
    ]
    result = _rrf_merge([list_a, list_b])
    assert result[0]["text"] == "shared chunk"


def test_rrf_merge_dedups_identical_text():
    list_a = [{"text": "same"}, {"text": "different-a"}]
    list_b = [{"text": "same"}, {"text": "different-b"}]
    result = _rrf_merge([list_a, list_b])
    texts = [c["text"] for c in result]
    assert texts.count("same") == 1


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


def _register_index(db, index_name: str, project_id: str = "1", dimension: int = 1024):
    """A real `index_registry` row in the throwaway test database.

    Fixture rows only — the production registry is deliberately empty and must
    stay that way; see the privacy gate in docs/KICKOFF-PROMPT.md §4a.
    """
    from app.models.index_registry import IndexRegistry

    entry = IndexRegistry(
        index_name=index_name,
        project_id=project_id,
        api_key_env_var="FIXTURE_KEY",
        dimension=dimension,
        embedding_model="fixture-embed",
        metric="cosine",
        domain_description="fixture domain",
        sample_queries=[],
        namespaces={},
        is_active=True,
    )
    db.add(entry)
    db.commit()
    return entry


def _router_replies(*payloads: str):
    return AsyncMock(side_effect=[
        MagicMock(choices=[MagicMock(message=MagicMock(content=payload))])
        for payload in payloads
    ])


async def test_run_pipeline_out_of_domain_skips_retrieval(monkeypatch, db_session):
    """When the router flags out_of_domain, the pipeline should refuse without
    embedding or retrieving — that's the whole point of the gate."""
    _patch_settings(monkeypatch, openai_api_key="test-openai")
    from app.db.database import SessionLocal
    from app.services.orchestrator import run_pipeline

    _register_index(db_session, "local-business")

    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI, \
         patch("app.services.orchestrator.generate_embedding") as mock_embed, \
         patch("app.services.orchestrator.retrieve") as mock_retrieve:

        mock_client = MagicMock()
        mock_client.chat.completions.create = _router_replies(
            "optimized",
            json.dumps({
                "candidates": [
                    {"index_name": "local-business", "project_id": "1", "confidence": 0.25}
                ],
                "out_of_domain": True,
            }),
        )
        MockOpenAI.return_value = mock_client

        chunks = []
        async for chunk in run_pipeline("D2C marketing strategy", SessionLocal):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert "isn't in my knowledge base" in full_response or "Wrong map" in full_response
        mock_embed.assert_not_called()
        mock_retrieve.assert_not_called()


async def test_run_pipeline_deactivates_broken_index(monkeypatch, db_session):
    """If retrieve raises (e.g., 404), the index should be deactivated.

    Asserted against a real row rather than a mock, so this also covers the
    deactivation happening in its own short-lived session now that the pipeline
    no longer borrows the request's.
    """
    _patch_settings(monkeypatch, openai_api_key="test-openai")
    from app.db.database import SessionLocal
    from app.models.index_registry import IndexRegistry
    from app.services.orchestrator import run_pipeline

    _register_index(db_session, "broken-index")

    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI, \
         patch("app.services.orchestrator.generate_embedding") as mock_embed, \
         patch("app.services.orchestrator.retrieve") as mock_retrieve:

        mock_client = MagicMock()
        mock_client.chat.completions.create = _router_replies(
            "optimized",
            json.dumps({
                "candidates": [
                    {"index_name": "broken-index", "project_id": "1", "confidence": 0.95}
                ],
                "out_of_domain": False,
            }),
        )
        MockOpenAI.return_value = mock_client

        mock_embed.return_value = [0.1] * 1024
        mock_retrieve.side_effect = Exception("Pinecone 404 NOT_FOUND")

        chunks = []
        async for chunk in run_pipeline("test query", SessionLocal):
            chunks.append(chunk)

    full_response = "".join(chunks)
    assert "couldn't find anything relevant" in full_response
    # The refusal must not name the index, the project, or Pinecone.
    assert "broken-index" not in full_response
    assert "Pinecone" not in full_response
    assert "index" not in full_response.lower()

    db_session.expire_all()
    stored = (
        db_session.query(IndexRegistry)
        .filter(IndexRegistry.index_name == "broken-index")
        .one()
    )
    assert stored.is_active is False
