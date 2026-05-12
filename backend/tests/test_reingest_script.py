"""Tests for app.scripts.reingest primitives."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.scripts.reingest import (
    enrich_metadata,
    fetch_source_chunks,
    reconstruct_documents,
)


@pytest.mark.asyncio
async def test_fetch_source_chunks_paginates_and_dedupes():
    fake_idx = MagicMock()
    # Two pages of IDs from list_paginated
    fake_idx.list_paginated.side_effect = [
        MagicMock(
            vectors=[MagicMock(id="a"), MagicMock(id="b")],
            pagination=MagicMock(next="cursor1"),
        ),
        MagicMock(
            vectors=[MagicMock(id="c")],
            pagination=None,
        ),
    ]
    fake_idx.fetch.return_value.vectors = {
        "a": MagicMock(metadata={"text": "alpha"}),
        "b": MagicMock(metadata={"text": "beta"}),
        "c": MagicMock(metadata={"text": "gamma"}),
    }
    factory = MagicMock()
    factory.get_client.return_value.Index.return_value = fake_idx

    with patch("app.scripts.reingest.get_factory", return_value=factory):
        chunks = await fetch_source_chunks("idx", "1")

    texts = sorted(c["text"] for c in chunks)
    assert texts == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
async def test_fetch_source_chunks_handles_empty_index():
    fake_idx = MagicMock()
    fake_idx.list_paginated.return_value = MagicMock(
        vectors=[], pagination=None
    )
    factory = MagicMock()
    factory.get_client.return_value.Index.return_value = fake_idx
    with patch("app.scripts.reingest.get_factory", return_value=factory):
        chunks = await fetch_source_chunks("empty", "1")
    assert chunks == []


def test_reconstruct_documents_groups_by_source_id_and_orders_by_position():
    chunks = [
        {"text": "p2", "metadata": {"source_id": "doc1", "chunk_position": 1}},
        {"text": "p1", "metadata": {"source_id": "doc1", "chunk_position": 0}},
        {"text": "lone", "metadata": {"source_id": "doc2", "chunk_position": 0}},
    ]
    docs = reconstruct_documents(chunks)
    assert docs["doc1"] == "p1\n\np2"
    assert docs["doc2"] == "lone"


def test_reconstruct_documents_keeps_orphans_as_standalone():
    chunks = [
        {"text": "orphan-a", "metadata": {}},
        {"text": "orphan-b", "metadata": None},
        {"text": "grouped", "metadata": {"source_id": "doc1", "chunk_position": 0}},
    ]
    docs = reconstruct_documents(chunks)
    assert "doc1" in docs
    orphan_texts = [v for k, v in docs.items() if k.startswith("_orphan_")]
    assert "orphan-a" in orphan_texts
    assert "orphan-b" in orphan_texts


@pytest.mark.asyncio
async def test_enrich_metadata_returns_parsed_dict(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "openai_api_key", "test")
    with patch(
        "app.scripts.reingest.call_openai_json",
        new=AsyncMock(return_value={
            "topic_tags": ["email", "B2B"],
            "source_type": "email",
            "public_safe": True,
        }),
    ):
        meta = await enrich_metadata("Some chunk content here.")
    assert meta["source_type"] == "email"
    assert meta["public_safe"] is True
