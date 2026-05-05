import cohere
from openai import OpenAI

from app.config import settings


def generate_embedding(query: str, dimension: int) -> list[float]:
    if dimension == 1536:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(model="text-embedding-3-small", input=query)
        return response.data[0].embedding

    if dimension == 2048:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large", input=query, dimensions=2048
        )
        return response.data[0].embedding

    if dimension == 1024:
        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY is not configured")
        co = cohere.Client(api_key=settings.cohere_api_key)
        response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        return response.embeddings[0]

    raise ValueError(f"Unsupported embedding dimension: {dimension}. Supported: 1024, 1536, 2048")
