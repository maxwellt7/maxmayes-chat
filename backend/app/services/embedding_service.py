import cohere
from openai import OpenAI

from app.config import settings

_MODEL_OPENAI_SMALL = "text-embedding-3-small"
_MODEL_OPENAI_LARGE = "text-embedding-3-large"
_MODEL_COHERE = "embed-english-v3.0"

_OPENAI_DIMENSION_MODEL_MAP = {
    1536: _MODEL_OPENAI_SMALL,
    2048: _MODEL_OPENAI_LARGE,
}


def generate_embedding(query: str, dimension: int) -> list[float]:
    if dimension in _OPENAI_DIMENSION_MODEL_MAP:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=settings.openai_api_key)
        kwargs: dict = {
            "model": _OPENAI_DIMENSION_MODEL_MAP[dimension],
            "input": query,
        }
        if dimension == 2048:
            kwargs["dimensions"] = 2048
        response = client.embeddings.create(**kwargs)
        return response.data[0].embedding

    if dimension == 1024:
        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY is not configured")
        co = cohere.Client(api_key=settings.cohere_api_key)
        response = co.embed(
            texts=[query],
            model=_MODEL_COHERE,
            input_type="search_query",
        )
        return response.embeddings[0]

    raise ValueError(f"Unsupported embedding dimension: {dimension}. Supported: 1024, 1536, 2048")


def generate_embedding_cohere(text: str, input_type: str = "search_query", model: str = _MODEL_COHERE) -> list[float]:
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY is not configured")
    client = cohere.ClientV2(api_key=settings.cohere_api_key)
    resp = client.embed(texts=[text], model=model, input_type=input_type, embedding_types=["float"])
    return resp.embeddings.float[0]


def generate_embeddings_cohere_batch(texts: list[str], input_type: str = "search_document", model: str = _MODEL_COHERE) -> list[list[float]]:
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY is not configured")
    if len(texts) > 96:
        raise ValueError(f"Cohere embed batch limit is 96, got {len(texts)}")
    client = cohere.ClientV2(api_key=settings.cohere_api_key)
    resp = client.embed(texts=texts, model=model, input_type=input_type, embedding_types=["float"])
    return resp.embeddings.float
