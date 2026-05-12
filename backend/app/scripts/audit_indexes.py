"""Phase-0 audit script.

Walks every active row in ``index_registry``, fetches Pinecone stats, samples
8 chunks per index, classifies via LLM, proposes a disposition, and writes
the result into the ``index_audits`` table.

Usage:
    python -m app.scripts.audit_indexes [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
from datetime import date
from pathlib import Path
from string import Template
from typing import Any

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.index_audit import IndexAudit
from app.models.index_registry import IndexRegistry
from app.services.llm_service import call_openai_json
from app.services.pinecone_service import get_factory

_log = logging.getLogger(__name__)

_SAMPLE_SIZE = 8
_NS_SAMPLE_CAP = 5  # cap on namespaces we'll sample per index
_AUDIT_MODEL = "gpt-4.1-mini"
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "audit_classify.txt"


async def _fetch_stats(index_name: str, project_id: str) -> dict[str, Any]:
    factory = get_factory()
    client = factory.get_client(project_id)
    index = client.Index(index_name)
    stats = index.describe_index_stats()
    return {
        "total_vector_count": stats.total_vector_count,
        "dimension": stats.dimension,
        "namespaces": {
            ns: {"vector_count": v.vector_count}
            for ns, v in (stats.namespaces or {}).items()
        },
    }


async def _sample_chunks(
    index_name: str,
    project_id: str,
    dimension: int,
    namespaces: list[str] | None,
    n: int = _SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """Random-vector query to get a sample of chunks. Loops over namespaces
    when present, capped at ``_NS_SAMPLE_CAP`` to keep latency bounded."""
    factory = get_factory()
    client = factory.get_client(project_id)
    index = client.Index(index_name)
    vector = [random.uniform(-1, 1) for _ in range(dimension)]

    if namespaces:
        all_matches: list[Any] = []
        for ns in namespaces[:_NS_SAMPLE_CAP]:
            res = index.query(
                vector=vector, top_k=n, include_metadata=True, namespace=ns
            )
            all_matches.extend(res.matches)
        matches = all_matches[:n]
    else:
        res = index.query(vector=vector, top_k=n, include_metadata=True)
        matches = res.matches

    return [
        {
            "text": (m.metadata or {}).get("text", "")[:400],
            "metadata": m.metadata or {},
        }
        for m in matches
    ]


async def audit_one_index(
    index_name: str, project_id: str, audit_date: date,
) -> dict[str, Any]:
    """Audit a single Pinecone index. Returns a dict suitable for IndexAudit."""
    stats = await _fetch_stats(index_name, project_id)
    namespaces = list(stats["namespaces"].keys())
    sample = await _sample_chunks(
        index_name, project_id, stats["dimension"], namespaces or None
    )

    sample_text = "\n\n---\n\n".join(c["text"] for c in sample if c["text"])
    prompt = Template(_PROMPT_PATH.read_text()).safe_substitute(
        index_name=index_name,
        record_count=stats["total_vector_count"],
        embedding_model=f"dim={stats['dimension']}",
        sample_chunks=sample_text or "(no sample available)",
    )
    classification = await call_openai_json(
        model=_AUDIT_MODEL, system="", user=prompt, temperature=0,
    )

    return {
        "audit_date": audit_date,
        "index_name": index_name,
        "project_id": project_id,
        "record_count": stats["total_vector_count"],
        "embedding_model": f"dim={stats['dimension']}",
        "dominant_domain": classification.get("dominant_domain"),
        "topic_tags": classification.get("topic_tags", []),
        "sample_chunks": sample,
        "proposed_disposition": classification.get("proposed_disposition", "KEEP"),
        "proposed_target_index": classification.get("proposed_target_index"),
    }


async def run_audit(dry_run: bool = False) -> list[dict[str, Any]]:
    """Audit every active index in the registry. Returns the list of audit
    rows. Persists to ``index_audits`` table unless ``dry_run`` is set."""
    db: Session = SessionLocal()
    try:
        active = (
            db.query(IndexRegistry)
            .filter(IndexRegistry.is_active == True)  # noqa: E712
            .all()
        )
        audit_date = date.today()
        rows: list[dict[str, Any]] = []
        for entry in active:
            try:
                row = await audit_one_index(
                    entry.index_name, entry.project_id, audit_date
                )
                rows.append(row)
                if not dry_run:
                    db.add(IndexAudit(**row))
                _log.info(
                    "audited %s: %s",
                    entry.index_name,
                    row["proposed_disposition"],
                )
            except Exception as exc:
                _log.error(
                    "audit failed for %s/%s: %s",
                    entry.project_id,
                    entry.index_name,
                    exc,
                )
        if not dry_run:
            db.commit()
        return rows
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_audit(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
