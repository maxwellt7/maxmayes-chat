import asyncio
import hashlib
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

# Reciprocal Rank Fusion constant — 60 is the value from the original Cormack et al. paper
# and works well across most heterogeneous-source ranking scenarios.
_RRF_K = 60

# How many indexes to fan retrieval out to. Higher than 3 yields diminishing returns
# and inflates Pinecone cost.
_TOP_INDEXES = 3

# Per-index top_k for retrieval — first/second/third by router confidence.
_PER_INDEX_TOP_K = (10, 5, 5)

# Final chunks fed into the synthesizer after rerank.
_FINAL_TOP_N = 8


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
    raw = json.loads(response.choices[0].message.content)
    return _normalize_router_response(raw)


def _normalize_router_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept both the new {candidates, out_of_domain} schema and the legacy
    {index_name, project_id, confidence, candidates?} shape so a router that
    momentarily ignores the schema doesn't crash the pipeline."""
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        # Legacy shape — promote the top-level fields to a single candidate.
        if raw.get("index_name") and raw.get("project_id"):
            candidates = [
                {
                    "index_name": raw["index_name"],
                    "project_id": raw["project_id"],
                    "confidence": float(raw.get("confidence", 0.0) or 0.0),
                    "reasoning": raw.get("reasoning", ""),
                }
            ]
        else:
            candidates = []

    cleaned: list[dict[str, Any]] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        name = c.get("index_name")
        project = c.get("project_id")
        if not name or not project:
            continue
        cleaned.append(
            {
                "index_name": name,
                "project_id": str(project),
                "confidence": float(c.get("confidence", 0.0) or 0.0),
                "reasoning": c.get("reasoning", ""),
            }
        )

    out_of_domain = bool(raw.get("out_of_domain", False))
    if not out_of_domain and cleaned:
        # Defensive: enforce the rule from the prompt server-side.
        out_of_domain = max(c["confidence"] for c in cleaned) < 0.4

    return {"candidates": cleaned, "out_of_domain": out_of_domain}


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


def _chunk_key(chunk: dict[str, Any]) -> str:
    """Stable identity for RRF dedup — Pinecone matches don't carry IDs through
    `retrieve`, so hash the chunk text. Texts collide only when they're truly
    identical, which is the deduplication we want anyway."""
    text = chunk.get("text") or ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _rrf_merge(
    ranked_lists: list[list[dict[str, Any]]], k: int = _RRF_K
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    Pinecone scores aren't comparable across indexes — different embedding spaces,
    sometimes different dimensions. RRF discards the raw scores and uses only
    rank position, which is what makes it robust for cross-source merging.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            # Keep the earlier-seen copy so source attribution stays stable.
            chunks.setdefault(key, chunk)
    return sorted(chunks.values(), key=lambda c: scores[_chunk_key(c)], reverse=True)


async def _rerank_chunks(
    query: str, chunks: list[dict[str, Any]], top_n: int = _FINAL_TOP_N
) -> list[dict[str, Any]]:
    """Cross-encoder rerank via Cohere. Falls back to the input ordering if the
    API key is missing or the call fails — RRF-only is still a usable signal."""
    if not chunks:
        return []
    if not settings.cohere_api_key:
        return chunks[:top_n]
    try:
        import cohere

        client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
        response = await client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[c["text"] for c in chunks],
            top_n=min(top_n, len(chunks)),
        )
        return [chunks[result.index] for result in response.results]
    except Exception as exc:
        _log.warning("rerank failed, falling back to RRF order: %s", exc)
        return chunks[:top_n]


async def _retrieve_one(
    factory: Any,
    db: Session,
    candidate: dict[str, Any],
    optimized: str,
    top_k: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Retrieve top_k chunks from a single candidate index, handling the same
    auto-deactivate-on-failure path the original pipeline used."""
    registry_entry = (
        db.query(IndexRegistry)
        .filter(
            IndexRegistry.index_name == candidate["index_name"],
            IndexRegistry.project_id == candidate["project_id"],
        )
        .first()
    )
    if not registry_entry:
        return None, []

    try:
        vector = await asyncio.to_thread(
            generate_embedding, optimized, registry_entry.dimension
        )
        ns_dict = registry_entry.namespaces or {}
        namespaces = list(ns_dict.keys()) if ns_dict else None
        chunks = await asyncio.to_thread(
            retrieve,
            factory,
            registry_entry.index_name,
            registry_entry.project_id,
            vector,
            top_k,
            namespaces,
        )
        # Tag each chunk with its source so downstream stages can keep attribution.
        for c in chunks:
            c["source_index"] = registry_entry.index_name
            c["source_project"] = registry_entry.project_id
        return registry_entry, chunks
    except Exception as exc:
        _log.warning(
            "retrieve failed for %s/%s: %s",
            registry_entry.project_id,
            registry_entry.index_name,
            exc,
        )
        registry_entry.is_active = False
        db.commit()
        return registry_entry, []


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
    candidates = route_result["candidates"][:_TOP_INDEXES]

    if not candidates:
        yield "I couldn't determine which knowledge base to search. Please rephrase your question."
        return

    # Step B.5: pre-retrieval out-of-domain gate
    # If the router judged none of the indexes plausibly contain relevant material,
    # skip retrieval entirely and refuse honestly. This avoids the failure mode
    # where the verifier later synthesizes a hedged-but-wrong answer from
    # adjacent-but-irrelevant chunks.
    if route_result["out_of_domain"]:
        yield (
            "That topic isn't in my knowledge base — what I have is centered on "
            "different domains. Wrong map for the territory. If you can rephrase "
            "around something I do cover, I'm happy to dig in."
        )
        return

    # Step C: retrieve from top-N candidates in parallel
    factory = get_factory()
    per_index_k = list(_PER_INDEX_TOP_K)
    while len(per_index_k) < len(candidates):
        per_index_k.append(per_index_k[-1])

    retrieval_tasks = [
        _retrieve_one(factory, db, candidate, optimized, per_index_k[i])
        for i, candidate in enumerate(candidates)
    ]
    retrieval_results = await asyncio.gather(*retrieval_tasks)

    ranked_lists = [chunks for _, chunks in retrieval_results if chunks]
    if not ranked_lists:
        yield (
            "I tried to consult my indexes, but none of the candidate sources "
            "returned results. Please try rephrasing your question."
        )
        return

    # Step C.5: merge candidate result lists with Reciprocal Rank Fusion.
    # Raw Pinecone scores aren't comparable across indexes (different embedding
    # models, sometimes different dimensions), so we use rank-only fusion.
    fused = _rrf_merge(ranked_lists)

    # Step C.6: cross-encoder rerank — biggest single precision lever.
    chunks = await _rerank_chunks(optimized, fused, top_n=_FINAL_TOP_N)

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
