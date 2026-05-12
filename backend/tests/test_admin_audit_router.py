"""Tests for the admin audit + dispositions endpoints.

Focuses on auth and routing. Full DB integration is validated by the
manual checkpoint after migration (T12) since the SQLAlchemy session is
mocked globally in conftest.py and round-tripping IndexAudit through a
mocked Base produces brittle assertions.
"""
import base64
import json
from datetime import date
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def _make_jwt(role: str) -> str:
    payload = {"sub": "u_test", "publicMetadata": {"role": role}}
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode())
        .rstrip(b"=")
        .decode()
    )
    return f"hdr.{encoded}.sig"


def _admin_token() -> str:
    return _make_jwt("admin")


def _user_token() -> str:
    return _make_jwt("user")


def test_audit_endpoint_registered_in_openapi():
    """Routes must appear in OpenAPI — guards against missing include_router."""
    from app.main import app
    client = TestClient(app)
    spec = client.get("/openapi.json").json()
    assert "/api/admin/audit" in spec["paths"]
    assert "/api/admin/audit/latest" in spec["paths"]
    assert "/api/admin/audit/{audit_date}/dispositions" in spec["paths"]


def test_post_audit_runs_when_admin():
    from app.main import app

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
        client = TestClient(app)
        resp = client.post(
            "/api/admin/audit",
            json={"mode": "execute"},
            headers={"Authorization": f"Bearer {_admin_token()}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "audit_id" in data
    assert data["row_count"] == 1


def test_post_audit_rejects_non_admin():
    from app.main import app
    client = TestClient(app)
    resp = client.post(
        "/api/admin/audit",
        json={"mode": "dry_run"},
        headers={"Authorization": f"Bearer {_user_token()}"},
    )
    assert resp.status_code == 403


def test_post_audit_rejects_unauthenticated():
    from app.main import app
    client = TestClient(app)
    resp = client.post("/api/admin/audit", json={"mode": "dry_run"})
    assert resp.status_code == 401


def test_get_latest_rejects_non_admin():
    from app.main import app
    client = TestClient(app)
    resp = client.get(
        "/api/admin/audit/latest",
        headers={"Authorization": f"Bearer {_user_token()}"},
    )
    assert resp.status_code == 403


def test_dispositions_rejects_non_admin():
    from app.main import app
    client = TestClient(app)
    resp = client.post(
        "/api/admin/audit/2026-05-11/dispositions",
        json=[],
        headers={"Authorization": f"Bearer {_user_token()}"},
    )
    assert resp.status_code == 403
