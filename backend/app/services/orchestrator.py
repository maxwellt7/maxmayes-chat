import asyncio
import hashlib
import json
import logging
from pathlib import Path
from string import Template
from typing import Any, AsyncGenerator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import settings
from app.db.database import SessionFactory, session_scope
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


IndexKey = tuple[str, str]


def _index_key(index_name: Any, project_id: Any) -> IndexKey:
    """Registry identity as a hashable pair.

    `project_id` is a `String(50)` column but arrives from the router as whatever
    JSON the model produced, so both halves are coerced. The old code compared
    these in SQL, which coerced for us.
    """
    return (str(index_name), str(project_id))


def _load_active_catalog(session_factory: SessionFactory) -> list[dict[str, Any]]:
    """Snapshot the active registry into plain dicts.

    Plain dicts rather than ORM instances on purpose: the rows outlive the
    session they were loaded in, and a detached instance would raise on the first
    lazy attribute access. Everything retrieval needs is read here, while the
    session is open, and the connection goes back to the pool immediately after.
    """
    with session_scope(session_factory) as db:
        rows = (
            db.query(IndexRegistry)
            .filter(IndexRegistry.is_active == True)  # noqa: E712
            .all()
        )
        return [
            {
                "index_name": row.index_name,
                "project_id": row.project_id,
                "domain_description": row.domain_description,
                "sample_queries": row.sample_queries,
                "dimension": row.dimension,
                "namespaces": row.namespaces,
            }
            for row in rows
        ]


def _deactivate_indexes(
    session_factory: SessionFactory, keys: list[IndexKey]
) -> None:
    """Flip `is_active` off for indexes that failed to retrieve.

    Same footgun as before — any exception deactivates, including a rate limit or
    a stale key — but batched into one short session after retrieval rather than
    committing from inside the fan-out. Left as-is deliberately; narrowing which
    exceptions deactivate is defect I2 and is not this change.
    """
    if not keys:
        return
    with session_scope(session_factory) as db:
        for index_name, project_id in keys:
            entry = (
                db.query(IndexRegistry)
                .filter(
                    IndexRegistry.index_name == index_name,
                    IndexRegistry.project_id == project_id,
                )
                .first()
            )
            if entry is not None:
                entry.is_active = False
        db.commit()


async def _retrieve_one(
    factory: Any,
    entry: dict[str, Any],
    optimized: str,
    top_k: int,
) -> tuple[IndexKey | None, list[dict[str, Any]]]:
    """Retrieve top_k chunks from a single registry entry.

    Returns the entry's key when retrieval failed, so the caller can deactivate
    it in one batch. Holds no database session: the whole fan-out runs with the
    connection already back in the pool.
    """
    try:
        vector = await asyncio.to_thread(
            generate_embedding, optimized, entry["dimension"]
        )
        ns_dict = entry.get("namespaces") or {}
        namespaces = list(ns_dict.keys()) if ns_dict else None
        chunks = await asyncio.to_thread(
            retrieve,
            factory,
            entry["index_name"],
            entry["project_id"],
            vector,
            top_k,
            namespaces,
        )
        # Tag each chunk with its source so downstream stages can keep attribution.
        for c in chunks:
            c["source_index"] = entry["index_name"]
            c["source_project"] = entry["project_id"]
        return None, chunks
    except Exception as exc:
        _log.warning(
            "retrieve failed for %s/%s: %s",
            entry["project_id"],
            entry["index_name"],
            exc,
        )
        return _index_key(entry["index_name"], entry["project_id"]), []


async def run_pipeline(
    raw_query: str, session_factory: SessionFactory
) -> AsyncGenerator[str, None]:
    """Run the retrieval pipeline and stream the synthesized answer.

    Takes a session factory rather than a session. The caller streams this for as
    long as the model takes to answer — up to the ~120s timeout budget — and a
    session passed in here would keep a pooled connection checked out for all of
    it. Instead every database touch below opens its own short-lived session, so
    the pool sees three brief checkouts rather than one long one.
    """
    # Step A: optimize
    optimized = await optimize_query(raw_query)

    # Step B: route — load active catalog from registry
    catalog = _load_active_catalog(session_factory)

    if not catalog:
        # An operational problem, not something the caller did or can act on.
        # The instruction to "add indexes in the admin dashboard" was reaching
        # end users, telling strangers an admin dashboard exists.
        _log.error("pipeline_no_active_indexes: index_registry has no active rows")
        yield (
            "I'm not able to answer questions right now. Please try again shortly."
        )
        return

    route_result = await route_query(optimized, catalog)
    candidates = route_result["candidates"][:_TOP_INDEXES]

    if not candidates:
        yield (
            "I couldn't find a good match for that question. Try rephrasing it, "
            "or asking something more specific."
        )
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

    # Step C: retrieve from top-N candidates in parallel.
    # The registry rows were snapshotted above, so a candidate the router
    # hallucinated — or one deactivated since — simply has no entry and is
    # skipped, exactly as the per-candidate lookup used to do.
    by_key = {
        _index_key(entry["index_name"], entry["project_id"]): entry
        for entry in catalog
    }
    entries = [
        entry
        for entry in (
            by_key.get(_index_key(c["index_name"], c["project_id"]))
            for c in candidates
        )
        if entry is not None
    ]

    factory = get_factory()
    per_index_k = list(_PER_INDEX_TOP_K)
    while len(per_index_k) < len(entries):
        per_index_k.append(per_index_k[-1])

    retrieval_tasks = [
        _retrieve_one(factory, entry, optimized, per_index_k[i])
        for i, entry in enumerate(entries)
    ]
    retrieval_results = await asyncio.gather(*retrieval_tasks)

    # One short session for the deactivations, after the fan-out rather than
    # during it.
    _deactivate_indexes(
        session_factory, [key for key, _ in retrieval_results if key is not None]
    )

    ranked_lists = [chunks for _, chunks in retrieval_results if chunks]
    if not ranked_lists:
        # Could be a genuine miss or could be every candidate index erroring.
        # Which one it was is a server-side concern; the caller gets the same
        # message either way, with no mention of indexes or sources.
        _log.warning(
            "pipeline_no_results candidates=%d", len(candidates)
        )
        yield (
            "I couldn't find anything relevant for that. Try rephrasing your "
            "question, or asking about something else."
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
