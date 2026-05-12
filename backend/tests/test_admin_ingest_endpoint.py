"""Tests for the admin ingest run endpoint."""
import base64
import json
import uuid
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def _admin_token() -> str:
    payload = {"sub": "u_admin", "publicMetadata": {"role": "admin"}}
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode())
        .rstrip(b"=")
        .decode()
    )
    return f"hdr.{encoded}.sig"


def _user_token() -> str:
    payload = {"sub": "u_user", "publicMetadata": {"role": "user"}}
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode())
        .rstrip(b"=")
        .decode()
    )
    return f"hdr.{encoded}.sig"


def test_ingest_run_endpoint_registered():
    from app.main import app
    client = TestClient(app)
    spec = client.get("/openapi.json").json()
    assert "/api/admin/ingest/{job_id}/run" in spec["paths"]


def test_ingest_run_returns_202_for_admin():
    from app.main import app

    job_id = str(uuid.uuid4())
    with patch(
        "app.routers.admin_audit.run_ingest_job",
        new=AsyncMock(return_value=None),
    ):
        client = TestClient(app)
        resp = client.post(
            f"/api/admin/ingest/{job_id}/run",
            headers={"Authorization": f"Bearer {_admin_token()}"},
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["status"] == "started"


def test_ingest_run_rejects_non_admin():
    from app.main import app
    client = TestClient(app)
    resp = client.post(
        f"/api/admin/ingest/{uuid.uuid4()}/run",
        headers={"Authorization": f"Bearer {_user_token()}"},
    )
    assert resp.status_code == 403


def test_ingest_run_rejects_unauthenticated():
    from app.main import app
    client = TestClient(app)
    resp = client.post(f"/api/admin/ingest/{uuid.uuid4()}/run")
    assert resp.status_code == 401
