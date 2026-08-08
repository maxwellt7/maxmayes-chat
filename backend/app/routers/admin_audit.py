"""Admin endpoints for Phase-0 audit + disposition approval + ingest jobs.

All endpoints require the ``owner`` role, enforced via :func:`require_owner`,
which resolves the role from the ``user_accounts`` table after verifying the
Clerk token's signature. The role is never read from a token claim: these
endpoints return raw corpus sample chunks, so a caller who could assert their own
role could exfiltrate the knowledge base.

The audit endpoint synchronously runs the audit job inside the request — fine at
Phase-0 scale (~40 indexes, ~30 sec). If scale grows, this should move to
BackgroundTasks.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.index_audit import IndexAudit
from app.models.ingest_job import IngestJob
from app.scripts.audit_indexes import run_audit
from app.scripts.reingest import run_ingest_job
from app.security.dependencies import require_owner
from app.security.principal import Principal

router = APIRouter(prefix="/api/admin", tags=["admin", "audit"])


class AuditRequest(BaseModel):
    mode: Literal["dry_run", "execute"]


class AuditResponse(BaseModel):
    audit_id: str
    row_count: int
    audit_date: date


class DispositionUpdate(BaseModel):
    audit_row_id: uuid.UUID
    approved_disposition: Literal[
        "KEEP", "MERGE", "RE-INGEST", "ARCHIVE", "SPLIT"
    ]
    approved_target_index: str | None = None


class DispositionResult(BaseModel):
    created_jobs: list[str]
    updated_rows: int


@router.post("/audit", response_model=AuditResponse)
async def kick_off_audit(
    body: AuditRequest,
    _owner: Principal = Depends(require_owner),
) -> AuditResponse:
    """Run the index audit. ``mode='dry_run'`` returns rows without persisting."""
    rows = await run_audit(dry_run=(body.mode == "dry_run"))
    return AuditResponse(
        audit_id=date.today().isoformat(),
        row_count=len(rows),
        audit_date=date.today(),
    )


@router.get("/audit/latest")
def get_latest_audit(
    db: Session = Depends(get_db),
    _owner: Principal = Depends(require_owner),
) -> list[dict[str, Any]]:
    """Return all rows from the most-recent audit (by audit_date)."""
    latest_date = (
        db.query(IndexAudit.audit_date)
        .order_by(IndexAudit.audit_date.desc())
        .limit(1)
        .scalar()
    )
    if not latest_date:
        return []
    rows = (
        db.query(IndexAudit).filter(IndexAudit.audit_date == latest_date).all()
    )
    return [
        {
            "id": str(r.id),
            "index_name": r.index_name,
            "project_id": r.project_id,
            "record_count": r.record_count,
            "embedding_model": r.embedding_model,
            "dominant_domain": r.dominant_domain,
            "topic_tags": r.topic_tags,
            "sample_chunks": r.sample_chunks,
            "proposed_disposition": r.proposed_disposition,
            "proposed_target_index": r.proposed_target_index,
            "approved_disposition": r.approved_disposition,
        }
        for r in rows
    ]


@router.post(
    "/audit/{audit_date}/dispositions", response_model=DispositionResult
)
def approve_dispositions(
    audit_date: str,
    updates: list[DispositionUpdate],
    db: Session = Depends(get_db),
    _owner: Principal = Depends(require_owner),
) -> DispositionResult:
    """Bulk approve dispositions and auto-create ingest_jobs for RE-INGEST/MERGE.

    Per-row override of target index is supported via ``approved_target_index``;
    when omitted, the audit row's ``proposed_target_index`` is used.
    """
    created_jobs: list[str] = []
    for upd in updates:
        row = (
            db.query(IndexAudit).filter(IndexAudit.id == upd.audit_row_id).first()
        )
        if not row:
            continue
        row.approved_disposition = upd.approved_disposition
        row.approved_at = datetime.utcnow()
        target_index = (
            upd.approved_target_index or row.proposed_target_index
        )
        if upd.approved_target_index:
            row.proposed_target_index = upd.approved_target_index

        if upd.approved_disposition in ("RE-INGEST", "MERGE"):
            if not target_index:
                raise HTTPException(
                    400,
                    f"target_index required for {upd.approved_disposition} "
                    f"(audit_row_id={upd.audit_row_id})",
                )
            job = IngestJob(
                source_index=row.index_name,
                target_index=target_index,
                target_namespace=row.dominant_domain or "default",
                status="pending",
                config={"source_project_id": row.project_id},
            )
            db.add(job)
            db.flush()
            created_jobs.append(str(job.id))

    db.commit()
    return DispositionResult(
        created_jobs=created_jobs,
        updated_rows=len(updates),
    )


@router.post("/ingest/{job_id}/run", status_code=202)
async def trigger_ingest(
    job_id: uuid.UUID,
    background: BackgroundTasks,
    _owner: Principal = Depends(require_owner),
) -> dict[str, Any]:
    """Kick off (or resume) an ingest job in the background.

    Returns 202 immediately; long-running work proceeds asynchronously. The
    job's status can be polled via ``GET /api/admin/ingest/{job_id}`` (not
    in v1 — clients can query the DB or watch logs).
    """
    background.add_task(run_ingest_job, str(job_id))
    return {"job_id": str(job_id), "status": "started"}
