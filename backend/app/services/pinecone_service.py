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
        {"text": (match.metadata or {}).get("text", ""), "score": match.score}
        for match in results.matches
    ]


# Lazy singleton — constructed on first call to get_factory()
_factory: PineconeClientFactory | None = None


def get_factory() -> PineconeClientFactory:
    global _factory
    if _factory is None:
        _factory = PineconeClientFactory()
    return _factory
