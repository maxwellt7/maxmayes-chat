import json
from pathlib import Path
from typing import Any, AsyncGenerator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.config import settings
from app.models.index_registry import IndexRegistry
from app.services.embedding_service import generate_embedding
from app.services.pinecone_service import get_factory, retrieve

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def _require_openai_key() -> str:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    return settings.openai_api_key


def _require_anthropic_key() -> str:
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not configured")
    return settings.anthropic_api_key


async def optimize_query(raw_query: str) -> str:
    client = AsyncOpenAI(api_key=_require_openai_key())
    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": _load_prompt("optimizer.txt")},
            {"role": "user", "content": raw_query},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


async def route_query(
    optimized_query: str, catalog: list[dict[str, Any]]
) -> dict[str, Any]:
    catalog_text = "\n".join(
        f"- index_name: {e['index_name']} | project_id: {e['project_id']}\n"
        f"  description: {e['domain_description']}\n"
        f"  sample_queries: {', '.join(e.get('sample_queries', [])[:3])}"
        for e in catalog
    )
    prompt = (
        _load_prompt("router.txt")
        .replace("{catalog}", catalog_text)
        .replace("{query}", optimized_query)
    )

    client = AsyncOpenAI(api_key=_require_openai_key())
    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


async def verify_accuracy(
    query: str, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    chunks_text = "\n\n".join(
        f"[{i+1}] (score: {c['score']:.2f})\n{c['text']}" for i, c in enumerate(chunks)
    )
    prompt = (
        _load_prompt("verifier.txt")
        .replace("{query}", query)
        .replace("{chunks}", chunks_text or "(no chunks retrieved)")
    )

    client = AsyncOpenAI(api_key=_require_openai_key())
    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


async def _fetch_voice_profile(top_k: int = 5) -> str:
    factory = get_factory()
    try:
        vector = generate_embedding("writing style tone voice communication", 1536)
        chunks = retrieve(
            factory,
            settings.voice_index_name,
            settings.voice_index_project,
            vector,
            top_k=top_k,
        )
        return "\n\n".join(c["text"] for c in chunks)
    except Exception:
        return ""


async def run_pipeline(
    raw_query: str, db: Session
) -> AsyncGenerator[str, None]:
    # Step A: optimize
    optimized = await optimize_query(raw_query)

    # Step B: route — load active catalog from registry
    active_indexes = (
        db.query(IndexRegistry)
        .filter(IndexRegistry.is_active == True)  # noqa: E712
        .all()
    )
    catalog = [
        {
            "index_name": idx.index_name,
            "project_id": idx.project_id,
            "domain_description": idx.domain_description,
            "sample_queries": idx.sample_queries,
        }
        for idx in active_indexes
    ]

    if not catalog:
        yield "I don't have any indexes configured yet. Please add indexes in the admin dashboard."
        return

    route_result = await route_query(optimized, catalog)
    index_name = route_result["index_name"]
    project_id = route_result["project_id"]

    registry_entry = (
        db.query(IndexRegistry)
        .filter(
            IndexRegistry.index_name == index_name,
            IndexRegistry.project_id == project_id,
        )
        .first()
    )
    if not registry_entry:
        yield "I couldn't find the target index in the registry. Please check the admin dashboard."
        return

    # Step C: retrieve
    vector = generate_embedding(optimized, registry_entry.dimension)
    factory = get_factory()
    chunks = retrieve(factory, index_name, project_id, vector, top_k=10)

    # Multi-index fallback for low confidence
    if route_result.get("confidence", 1.0) < 0.7 and route_result.get("candidates"):
        for candidate in route_result["candidates"][:2]:
            candidate_entry = (
                db.query(IndexRegistry)
                .filter(
                    IndexRegistry.index_name == candidate["index_name"],
                    IndexRegistry.project_id == candidate["project_id"],
                )
                .first()
            )
            if candidate_entry:
                extra_vector = generate_embedding(optimized, candidate_entry.dimension)
                extra_chunks = retrieve(
                    factory,
                    candidate["index_name"],
                    candidate["project_id"],
                    extra_vector,
                    top_k=5,
                )
                chunks = sorted(chunks + extra_chunks, key=lambda c: c["score"], reverse=True)[:10]

    # Step D: verify
    verification = await verify_accuracy(raw_query, chunks)

    if verification["recommendation"] == "insufficient_context":
        yield "I don't have enough information in my knowledge base to answer that confidently."
        return

    # Step E: synthesize with voice
    context_text = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    )
    voice_profile = await _fetch_voice_profile()

    caveat = (
        "\nNote: Some parts of this answer have limited supporting context — indicate any uncertainty naturally."
        if verification["recommendation"] == "proceed_with_caveat"
        else ""
    )

    prompt_template = _load_prompt("voice_synthesizer.txt")
    system_prompt = (
        prompt_template
        .replace("{voice_profile}", voice_profile or "Write clearly and conversationally.")
        .replace("{context}", context_text)
        .replace("{caveat_instruction}", caveat)
    )

    client = AsyncAnthropic(api_key=_require_anthropic_key())
    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": raw_query}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
