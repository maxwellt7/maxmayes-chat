"""Semantic chunker for the re-ingest pipeline.

Wraps LangChain's ``SemanticChunker`` with Cohere embeddings. The chunker
finds natural breakpoints by sentence-pair similarity rather than fixed
token windows, which produces tighter chunks for heterogeneous content
(per spec §7.4).
"""
from __future__ import annotations

from langchain_cohere import CohereEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from app.config import settings

_DEFAULT_MODEL = "embed-english-v3.0"


def semantic_chunk_text(
    text: str,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 90.0,
    model: str = _DEFAULT_MODEL,
) -> list[str]:
    """Split ``text`` into semantically-coherent chunks (~512 token target).

    Args:
        text: Raw document text.
        breakpoint_threshold_type: Strategy for choosing breakpoints. See
            LangChain docs; 'percentile' is the default and most reliable.
        breakpoint_threshold_amount: Higher = fewer, larger chunks. 90 is a
            sane default for prose; lower for code or dialogue.
        model: Cohere embedding model to use for similarity scoring.

    Returns:
        Ordered list of chunk strings. May return a single-element list for
        short inputs that don't trigger a breakpoint.
    """
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY is not configured")
    embeddings = CohereEmbeddings(
        cohere_api_key=settings.cohere_api_key,
        model=model,
    )
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    return chunker.split_text(text)
