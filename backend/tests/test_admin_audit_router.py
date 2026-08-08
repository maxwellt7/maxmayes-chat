"""Tests for the admin audit + dispositions endpoints.

Rewritten. The previous version built `hdr.<base64>.sig` strings with a
`publicMetadata.role` of `admin` and asserted 200 — it was a test that the auth
bypass worked. Routes now require a cryptographically verified token *and* an
`owner` row in `user_accounts`, and these tests assert both halves.
"""
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.models.index_audit import IndexAudit
from app.models.user_account import ROLE_MEMBER, ROLE_OWNER
from tests.authhelpers import bearer, claims, shaped_but_unsigned_token


@pytest.fixture
def owner_token(signing_key, make_account):
    make_account("u_owner", role=ROLE_OWNER)
    return signing_key.sign(claims("u_owner"))


@pytest.fixture
def member_token(signing_key, make_account):
    make_account("u_member", role=ROLE_MEMBER)
    return signing_key.sign(claims("u_member"))


def test_audit_routes_are_registered():
    """Guards against a missing `include_router`.

    Builds the schema in-process rather than fetching `/openapi.json`, which is
    no longer served: the schema is a complete map of the admin surface and does
    not belong on the public internet. Generating it here still catches an
    unregistered router.
    """
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/api/admin/audit" in paths
    assert "/api/admin/audit/latest" in paths
    assert "/api/admin/audit/{audit_date}/dispositions" in paths
    assert "/api/admin/ingest/{job_id}/run" in paths


def test_post_audit_runs_for_owner(client, owner_token):
    fake_row = {
        "audit_date": date.today(),
        "index_name": "x",
        "project_id": "1",
        "record_count": 0,
        "embedding_model": "dim=1024",
        "dominant_domain": "marketing",
        "topic_tags": [],
        "sample_chunks": [],
        "proposed_disposition": "KEEP",
        "proposed_target_index": None,
    }
    with patch(
        "app.routers.admin_audit.run_audit",
        new=AsyncMock(return_value=[fake_row]),
    ):
        resp = client.post(
            "/api/admin/audit",
            json={"mode": "execute"},
            headers=bearer(owner_token),
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "audit_id" in data
    assert data["row_count"] == 1


def test_post_audit_rejects_member(client, member_token):
    resp = client.post(
        "/api/admin/audit",
        json={"mode": "dry_run"},
        headers=bearer(member_token),
    )
    assert resp.status_code == 403


def test_post_audit_rejects_unauthenticated(client):
    assert client.post("/api/admin/audit", json={"mode": "dry_run"}).status_code == 401


def test_post_audit_rejects_forged_admin_token(client):
    """The exact request the old test suite asserted should succeed."""
    forged = shaped_but_unsigned_token(
        {"sub": "u_attacker", "publicMetadata": {"role": "admin"}}
    )
    resp = client.post(
        "/api/admin/audit", json={"mode": "dry_run"}, headers=bearer(forged)
    )
    assert resp.status_code == 401


def test_get_latest_rejects_member(client, member_token):
    assert client.get(
        "/api/admin/audit/latest", headers=bearer(member_token)
    ).status_code == 403


def test_get_latest_returns_rows_for_owner(client, owner_token, db_session):
    """Also confirms the endpoint really does return raw corpus sample chunks,
    which is why it is owner-only."""
    db_session.add(
        IndexAudit(
            audit_date=date(2026, 5, 11),
            index_name="personal-knowledge-base",
            project_id="1",
            record_count=3,
            embedding_model="dim=1024",
            dominant_domain="personal",
            topic_tags=["private"],
            sample_chunks=[{"text": "a private excerpt", "metadata": {}}],
            proposed_disposition="KEEP",
        )
    )
    db_session.commit()

    resp = client.get("/api/admin/audit/latest", headers=bearer(owner_token))
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["index_name"] == "personal-knowledge-base"
    assert rows[0]["sample_chunks"][0]["text"] == "a private excerpt"


def test_get_latest_is_empty_when_no_audits_exist(client, owner_token):
    resp = client.get("/api/admin/audit/latest", headers=bearer(owner_token))
    assert resp.status_code == 200
    assert resp.json() == []


def test_dispositions_rejects_member(client, member_token):
    resp = client.post(
        "/api/admin/audit/2026-05-11/dispositions",
        json=[],
        headers=bearer(member_token),
    )
    assert resp.status_code == 403


def test_dispositions_creates_an_ingest_job_for_reingest(
    client, owner_token, db_session
):
    from app.models.ingest_job import IngestJob

    row = IndexAudit(
        audit_date=date(2026, 5, 11),
        index_name="src-index",
        project_id="1",
        record_count=10,
        embedding_model="dim=1024",
        dominant_domain="marketing",
        topic_tags=[],
        sample_chunks=[],
        proposed_disposition="RE-INGEST",
        proposed_target_index="max-marketing",
    )
    db_session.add(row)
    db_session.commit()

    resp = client.post(
        "/api/admin/audit/2026-05-11/dispositions",
        json=[
            {
                "audit_row_id": str(row.id),
                "approved_disposition": "RE-INGEST",
            }
        ],
        headers=bearer(owner_token),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["updated_rows"] == 1
    assert len(body["created_jobs"]) == 1

    job = db_session.query(IngestJob).one()
    assert job.source_index == "src-index"
    assert job.target_index == "max-marketing"
    assert job.status == "pending"


def test_dispositions_rejects_reingest_without_a_target(
    client, owner_token, db_session
):
    row = IndexAudit(
        audit_date=date(2026, 5, 11),
        index_name="src-index",
        project_id="1",
        record_count=10,
        embedding_model="dim=1024",
        dominant_domain=None,
        topic_tags=[],
        sample_chunks=[],
        proposed_disposition="RE-INGEST",
        proposed_target_index=None,
    )
    db_session.add(row)
    db_session.commit()

    resp = client.post(
        "/api/admin/audit/2026-05-11/dispositions",
        json=[
            {"audit_row_id": str(row.id), "approved_disposition": "RE-INGEST"}
        ],
        headers=bearer(owner_token),
    )
    assert resp.status_code == 400


def test_disposition_approval_is_recorded(client, owner_token, db_session):
    row = IndexAudit(
        audit_date=date(2026, 5, 11),
        index_name="src-index",
        project_id="1",
        record_count=10,
        embedding_model="dim=1024",
        dominant_domain="marketing",
        topic_tags=[],
        sample_chunks=[],
        proposed_disposition="KEEP",
    )
    db_session.add(row)
    db_session.commit()
    row_id = str(row.id)

    before = datetime.now(timezone.utc)
    resp = client.post(
        "/api/admin/audit/2026-05-11/dispositions",
        json=[{"audit_row_id": row_id, "approved_disposition": "ARCHIVE"}],
        headers=bearer(owner_token),
    )
    assert resp.status_code == 200

    db_session.expire_all()
    stored = db_session.query(IndexAudit).one()
    assert stored.approved_disposition == "ARCHIVE"
    assert stored.approved_at is not None
    assert stored.approved_at.replace(tzinfo=timezone.utc) >= before.replace(
        microsecond=0
    )
