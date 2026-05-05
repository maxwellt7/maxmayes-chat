import os
from typing import Any

from pinecone import Pinecone


class PineconeClientFactory:
    _PROJECT_KEY_MAP = {
        "1": "PINECONE_API_KEY_1",
        "2": "PINECONE_API_KEY_2",
        "3": "PINECONE_API_KEY_3",
    }

    def __init__(self) -> None:
        self._cache: dict[str, Pinecone] = {}

    def _key_for_project(self, project_id: str) -> str:
        env_var = self._PROJECT_KEY_MAP.get(project_id)
        if not env_var:
            raise ValueError(f"Unknown project_id: {project_id}")
        return os.environ[env_var]

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
    client = factory.get_client(project_id)
    index = client.Index(index_name)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [
        {"text": match.metadata.get("text", ""), "score": match.score}
        for match in results.matches
    ]


_factory = PineconeClientFactory()


def get_factory() -> PineconeClientFactory:
    return _factory
