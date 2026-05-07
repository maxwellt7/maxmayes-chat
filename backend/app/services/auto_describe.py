import asyncio
import json
import logging
from string import Template

from openai import AsyncOpenAI

from app.config import settings
from app.services.embedding_service import generate_embedding
from app.services.pinecone_service import get_factory, retrieve

logger = logging.getLogger(__name__)

_MODEL = "gpt-4.1-mini"

_PROMPT = Template("""Look at these sample chunks from a Pinecone vector index.

Index name: $index_name

Sample chunks:
$samples

Generate:
1. A 2-3 sentence domain_description explaining what topics, content, and use cases this index covers. Be concrete — name the kinds of documents, frameworks, or domains present.
2. 3 sample_queries: representative natural-language questions a user might ask that should route to this index.

Return strict JSON:
{
  "domain_description": "<2-3 sentences>",
  "sample_queries": ["<question 1>", "<question 2>", "<question 3>"]
}
""")


def _require_openai_key() -> str:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    return settings.openai_api_key


async def generate_index_description(
    index_name: str, project_id: str, dimension: int
) -> dict:
    """Sample the index and use an LLM to write a description + sample queries."""
    factory = get_factory()

    # Generate an embedding for a generic exploratory query, then pull top chunks.
    vector = await asyncio.to_thread(
        generate_embedding,
        "introduction overview summary main topic key concept",
        dimension,
    )
    chunks = await asyncio.to_thread(retrieve, factory, index_name, project_id, vector, 8)

    if not chunks:
        return {
            "domain_description": f"Pinecone index '{index_name}' (project {project_id}). No sample content retrieved.",
            "sample_queries": [],
        }

    samples_text = "\n\n---\n\n".join(
        (c["text"] or "")[:600] for c in chunks if c.get("text")
    )
    if not samples_text.strip():
        return {
            "domain_description": f"Pinecone index '{index_name}' (project {project_id}). Chunks found but no text metadata available.",
            "sample_queries": [],
        }

    prompt = _PROMPT.safe_substitute(index_name=index_name, samples=samples_text)

    client = AsyncOpenAI(api_key=_require_openai_key())
    response = await client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)
