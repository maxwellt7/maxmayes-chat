from unittest.mock import MagicMock, patch

import pytest

from app.services.pinecone_service import PineconeClientFactory, retrieve


def _patch_settings(monkeypatch, **keys: str) -> None:
    """Set Pinecone API keys on the settings singleton."""
    from app.config import settings
    for attr, value in keys.items():
        monkeypatch.setattr(settings, attr, value)


def test_factory_returns_client_for_project_1(monkeypatch):
    _patch_settings(monkeypatch, pinecone_api_key_1="test-key-1")
    with patch("app.services.pinecone_service.Pinecone") as MockPinecone:
        mock_client = MagicMock()
        MockPinecone.return_value = mock_client

        factory = PineconeClientFactory()
        client = factory.get_client("1")

        MockPinecone.assert_called_once_with(api_key="test-key-1")
        assert client is mock_client


def test_factory_caches_client(monkeypatch):
    _patch_settings(monkeypatch, pinecone_api_key_1="test-key-1")
    with patch("app.services.pinecone_service.Pinecone") as MockPinecone:
        factory = PineconeClientFactory()
        c1 = factory.get_client("1")
        c2 = factory.get_client("1")
        assert c1 is c2
        assert MockPinecone.call_count == 1


def test_factory_raises_for_unknown_project():
    factory = PineconeClientFactory()
    with pytest.raises(ValueError, match="Unknown project_id"):
        factory.get_client("99")


def test_factory_raises_when_api_key_missing(monkeypatch):
    _patch_settings(monkeypatch, pinecone_api_key_1="")
    factory = PineconeClientFactory()
    with pytest.raises(ValueError, match="not configured"):
        factory.get_client("1")


def test_retrieve_calls_correct_index(monkeypatch):
    _patch_settings(monkeypatch, pinecone_api_key_1="test-key-1")
    with patch("app.services.pinecone_service.Pinecone") as MockPinecone:
        mock_index = MagicMock()
        mock_index.query.return_value = MagicMock(
            matches=[
                MagicMock(metadata={"text": "chunk1"}, score=0.9),
                MagicMock(metadata={"text": "chunk2"}, score=0.8),
            ]
        )
        MockPinecone.return_value.Index.return_value = mock_index

        factory = PineconeClientFactory()
        results = retrieve(factory, "my-index", "1", [0.1, 0.2, 0.3], top_k=2)

        mock_index.query.assert_called_once_with(vector=[0.1, 0.2, 0.3], top_k=2, include_metadata=True)
        assert results == [{"text": "chunk1", "score": 0.9}, {"text": "chunk2", "score": 0.8}]


def test_retrieve_handles_none_metadata(monkeypatch):
    _patch_settings(monkeypatch, pinecone_api_key_1="test-key-1")
    with patch("app.services.pinecone_service.Pinecone") as MockPinecone:
        mock_index = MagicMock()
        mock_index.query.return_value = MagicMock(
            matches=[MagicMock(metadata=None, score=0.5)]
        )
        MockPinecone.return_value.Index.return_value = mock_index

        factory = PineconeClientFactory()
        results = retrieve(factory, "my-index", "1", [0.1], top_k=1)
        assert results == [{"text": "", "score": 0.5}]


def test_retrieve_raises_on_empty_index_name(monkeypatch):
    _patch_settings(monkeypatch, pinecone_api_key_1="test-key-1")
    with patch("app.services.pinecone_service.Pinecone"):
        factory = PineconeClientFactory()
        with pytest.raises(ValueError, match="index_name"):
            retrieve(factory, "", "1", [0.1])
