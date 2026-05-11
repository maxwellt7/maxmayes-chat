from unittest.mock import MagicMock, patch

import pytest

from app.services.embedding_service import generate_embedding, generate_embedding_cohere, generate_embeddings_cohere_batch


def _patch_settings(monkeypatch, **keys: str) -> None:
    from app.config import settings
    for attr, value in keys.items():
        monkeypatch.setattr(settings, attr, value)


def test_1536_dim_uses_openai_small(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="test-openai")
    with patch("app.services.embedding_service.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )
        MockOpenAI.return_value = mock_client

        result = generate_embedding("test query", 1536)

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="test query"
        )
        assert len(result) == 1536


def test_2048_dim_uses_openai_large(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="test-openai")
    with patch("app.services.embedding_service.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 2048)]
        )
        MockOpenAI.return_value = mock_client

        result = generate_embedding("test query", 2048)

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large", input="test query", dimensions=2048
        )
        assert len(result) == 2048


def test_1024_dim_uses_cohere(monkeypatch):
    _patch_settings(monkeypatch, cohere_api_key="test-cohere")
    with patch("app.services.embedding_service.cohere") as mock_cohere_module:
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
        mock_cohere_module.Client.return_value = mock_client

        result = generate_embedding("test query", 1024)

        mock_client.embed.assert_called_once_with(
            texts=["test query"],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        assert len(result) == 1024


def test_unsupported_dimension_raises():
    with pytest.raises(ValueError, match="Unsupported embedding dimension"):
        generate_embedding("test query", 512)


def test_missing_openai_key_raises(monkeypatch):
    _patch_settings(monkeypatch, openai_api_key="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
        generate_embedding("test query", 1536)


def test_missing_cohere_key_raises(monkeypatch):
    _patch_settings(monkeypatch, cohere_api_key="")
    with pytest.raises(ValueError, match="COHERE_API_KEY is not configured"):
        generate_embedding("test query", 1024)


def test_generate_embedding_cohere_returns_list_of_floats(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test-cohere")
    with patch("app.services.embedding_service.cohere.ClientV2") as MockClient:
        mock_resp = MagicMock()
        mock_resp.embeddings.float = [[0.1] * 1024]
        MockClient.return_value.embed.return_value = mock_resp
        result = generate_embedding_cohere("test text", input_type="search_query")
        assert isinstance(result, list)
        assert len(result) == 1024


def test_generate_embeddings_cohere_batch_returns_list_of_embeddings(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test-cohere")
    with patch("app.services.embedding_service.cohere.ClientV2") as MockClient:
        mock_resp = MagicMock()
        mock_resp.embeddings.float = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        MockClient.return_value.embed.return_value = mock_resp
        result = generate_embeddings_cohere_batch(["a", "b", "c"])
        assert len(result) == 3
        assert all(len(emb) == 1024 for emb in result)


def test_generate_embeddings_cohere_batch_rejects_oversized_batch(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test-cohere")
    with pytest.raises(ValueError, match="96"):
        generate_embeddings_cohere_batch(["x"] * 97)
