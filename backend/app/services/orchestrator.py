import asyncio
import json
import logging
from pathlib import Path
from string import Template
from typing import Any, AsyncGenerator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.config import settings
from app.models.index_registry import IndexRegistry
from app.services.embedding_service import generate_embedding
from app.services.pinecone_service import get_factory, retrieve

_log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_MODEL_OPTIMIZER = "gpt-4.1-mini"
_MODEL_ROUTER = "gpt-4.1-mini"
_MODEL_VERIFIER = "gpt-4.1-mini"
_MODEL_SYNTHESIZER = "claude-sonnet-4-6"


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text()


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
        model=_MODEL_OPTIMIZER,
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
    prompt = Template(_load_prompt("router.txt")).safe_substitute(
        catalog=catalog_text,
        query=optimized_query,
    )

    client = AsyncOpenAI(api_key=_require_openai_key())
    response = await client.chat.completions.create(
        model=_MODEL_ROUTER,
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
    prompt = Template(_load_prompt("verifier.txt")).safe_substitute(
        query=query,
        chunks=chunks_text or "(no chunks retrieved)",
    )

    client = AsyncOpenAI(api_key=_require_openai_key())
    response = await client.chat.completions.create(
        model=_MODEL_VERIFIER,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


async def _fetch_voice_profile(top_k: int = 5) -> str:
    factory = get_factory()
    try:
        vector = await asyncio.to_thread(
            generate_embedding,
            "writing style tone voice communication",
            settings.voice_index_dimension,
        )
        chunks = await asyncio.to_thread(
            retrieve,
            factory,
            settings.voice_index_name,
            settings.voice_index_project,
            vector,
            top_k,
        )
        return "\n\n".join(c["text"] for c in chunks)
    except Exception as exc:
        _log.warning("_fetch_voice_profile failed, using fallback voice: %s", exc)
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
    index_name = route_result.get("index_name")
    project_id = route_result.get("project_id")

    if not index_name or not project_id:
        yield "I couldn't determine which knowledge base to search. Please rephrase your question."
        return

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
    factory = get_factory()
    vector = await asyncio.to_thread(generate_embedding, optimized, registry_entry.dimension)
    chunks = await asyncio.to_thread(retrieve, factory, index_name, project_id, vector, 10)

    # Multi-index fallback for low confidence
    if route_result.get("confidence", 1.0) < 0.7 and route_result.get("candidates"):
        for candidate in route_result["candidates"][:2]:
            cand_index = candidate.get("index_name")
            cand_project = candidate.get("project_id")
            if not cand_index or not cand_project:
                continue
            candidate_entry = (
                db.query(IndexRegistry)
                .filter(
                    IndexRegistry.index_name == cand_index,
                    IndexRegistry.project_id == cand_project,
                )
                .first()
            )
            if candidate_entry:
                extra_vector = await asyncio.to_thread(
                    generate_embedding, optimized, candidate_entry.dimension
                )
                extra_chunks = await asyncio.to_thread(
                    retrieve, factory, cand_index, cand_project, extra_vector, 5
                )
                chunks = sorted(chunks + extra_chunks, key=lambda c: c["score"], reverse=True)[:10]

    # Step D: verify
    verification = await verify_accuracy(raw_query, chunks)

    if verification.get("recommendation") == "insufficient_context":
        yield "I don't have enough information in my knowledge base to answer that confidently."
        return

    # Step E: synthesize with voice
    context_text = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    )
    voice_profile = await _fetch_voice_profile()

    caveat = (
        "\nNote: Some parts of this answer have limited supporting context — indicate any uncertainty naturally."
        if verification.get("recommendation") == "proceed_with_caveat"
        else ""
    )

    system_prompt = Template(_load_prompt("voice_synthesizer.txt")).safe_substitute(
        voice_profile=voice_profile or "Write clearly and conversationally.",
        context=context_text,
        caveat_instruction=caveat,
    )

    client = AsyncAnthropic(api_key=_require_anthropic_key())
    async with client.messages.stream(
        model=_MODEL_SYNTHESIZER,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": raw_query}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
