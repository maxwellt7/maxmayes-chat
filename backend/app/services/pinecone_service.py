from typing import Any

from pinecone import Pinecone

from app.config import settings


class PineconeClientFactory:
    # Keys must stay in sync with the pinecone_api_key_N fields in app/config.py Settings.
    _PROJECT_KEY_ATTR_MAP = {
        "1": "pinecone_api_key_1",
        "2": "pinecone_api_key_2",
        "3": "pinecone_api_key_3",
    }

    def __init__(self) -> None:
        self._cache: dict[str, Pinecone] = {}

    def _key_for_project(self, project_id: str) -> str:
        attr = self._PROJECT_KEY_ATTR_MAP.get(project_id)
        if not attr:
            raise ValueError(f"Unknown project_id: {project_id}")
        api_key = getattr(settings, attr, "")
        if not api_key:
            raise ValueError(f"Pinecone API key for project_id {project_id} is not configured")
        return api_key

    def get_client(self, project_id: str) -> Pinecone:
        if project_id not in self._cache:
            self._cache[project_id] = Pinecone(api_key=self._key_for_project(project_id))
        return self._cache[project_id]


def retrieve(
    factory: PineconeClientFactory,
    index_name: str,
    project_id: str,
    vector: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if not index_name:
        raise ValueError("index_name must be a non-empty string")
    client = factory.get_client(project_id)
    index = client.Index(index_name)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [
        {
            "text": _extract_text_from_metadata(match.metadata or {}),
            "score": match.score,
            "metadata": match.metadata or {},
        }
        for match in results.matches
    ]


_TEXT_FIELD_CANDIDATES = (
    "text",
    "content",
    "chunk_text",
    "chunk",
    "body",
    "page_content",
    "passage",
    "transcript",
    "summary",
    "raw_text",
    "_node_content",
)


def _extract_text_from_metadata(metadata: dict[str, Any]) -> str:
    """Try common text field names first, then fall back to any string value > 40 chars."""
    for key in _TEXT_FIELD_CANDIDATES:
        val = metadata.get(key)
        if isinstance(val, str) and val.strip():
            # _node_content is sometimes JSON — extract the inner text if so
            if key == "_node_content":
                try:
                    import json as _json
                    parsed = _json.loads(val)
                    if isinstance(parsed, dict):
                        for inner_key in ("text", "content", "page_content"):
                            inner = parsed.get(inner_key)
                            if isinstance(inner, str) and inner.strip():
                                return inner
                except Exception:
                    pass
            return val
    # Fallback: any string field longer than 40 chars
    for key, val in metadata.items():
        if isinstance(val, str) and len(val.strip()) > 40:
            return val
    return ""


# Lazy singleton — constructed on first call to get_factory()
_factory: PineconeClientFactory | None = None


def get_factory() -> PineconeClientFactory:
    global _factory
    if _factory is None:
        _factory = PineconeClientFactory()
    return _factory
