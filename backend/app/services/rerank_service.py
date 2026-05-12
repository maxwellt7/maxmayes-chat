"""Cohere Rerank-v3.5 wrapper with graceful fallback.

Falls back to the input order when the API key is missing or the call fails,
so callers can rely on a non-empty result for non-empty input.
"""
import logging
from typing import Any

import cohere

from app.config import settings

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "rerank-v3.5"


async def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int,
    model: str = _DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Cross-encoder rerank via Cohere.

    Each chunk must have a ``text`` field. Returns up to ``top_n`` chunks in
    relevance order. On missing API key or upstream failure, returns the first
    ``top_n`` chunks in original order so the pipeline always has a usable
    ranked list to work with (per NFR-REL-3).
    """
    if not chunks:
        return []
    if not settings.cohere_api_key:
        return chunks[:top_n]
    try:
        client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
        response = await client.rerank(
            model=model,
            query=query,
            documents=[c["text"] for c in chunks],
            top_n=min(top_n, len(chunks)),
        )
        return [chunks[r.index] for r in response.results]
    except Exception as exc:
        _log.warning("rerank failed, falling back to input order: %s", exc)
        return chunks[:top_n]
