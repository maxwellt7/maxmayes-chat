from unittest.mock import MagicMock, patch

import pytest

from app.services.chunking_service import semantic_chunk_text


def test_semantic_chunk_returns_list_of_strings(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test")
    text = "First sentence. Second sentence. Third unrelated thought. " * 30

    with patch("app.services.chunking_service.SemanticChunker") as MockChunker:
        instance = MockChunker.return_value
        instance.split_text.return_value = ["chunk one " * 50, "chunk two " * 50]
        chunks = semantic_chunk_text(text)

    assert len(chunks) == 2
    assert all(isinstance(c, str) for c in chunks)


def test_semantic_chunk_returns_single_chunk_for_short_text(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test")
    with patch("app.services.chunking_service.SemanticChunker") as MockChunker:
        MockChunker.return_value.split_text.return_value = ["short"]
        chunks = semantic_chunk_text("short")
    assert chunks == ["short"]


def test_semantic_chunk_raises_when_key_missing(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "")
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        semantic_chunk_text("anything")
