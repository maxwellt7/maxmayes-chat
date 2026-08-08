"""Tests for the admin ingest run endpoint.

Rewritten: the previous version authenticated with an unsigned
`hdr.<base64>.sig` string carrying `publicMetadata.role = admin`, which is the
bypass rather than a test of it.
"""
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.models.user_account import ROLE_MEMBER, ROLE_OWNER
from tests.authhelpers import bearer, claims, shaped_but_unsigned_token, unsigned_token


@pytest.fixture
def owner_token(signing_key, make_account):
    make_account("u_admin", role=ROLE_OWNER)
    return signing_key.sign(claims("u_admin"))


@pytest.fixture
def member_token(signing_key, make_account):
    make_account("u_user", role=ROLE_MEMBER)
    return signing_key.sign(claims("u_user"))


def test_ingest_run_route_is_registered():
    from app.main import app

    assert "/api/admin/ingest/{job_id}/run" in app.openapi()["paths"]


def test_ingest_run_returns_202_for_owner(client, owner_token):
    job_id = str(uuid.uuid4())
    with patch(
        "app.routers.admin_audit.run_ingest_job",
        new=AsyncMock(return_value=None),
    ):
        resp = client.post(
            f"/api/admin/ingest/{job_id}/run", headers=bearer(owner_token)
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["status"] == "started"


def test_ingest_run_rejects_member(client, member_token):
    resp = client.post(
        f"/api/admin/ingest/{uuid.uuid4()}/run", headers=bearer(member_token)
    )
    assert resp.status_code == 403


def test_ingest_run_rejects_unauthenticated(client):
    resp = client.post(f"/api/admin/ingest/{uuid.uuid4()}/run")
    assert resp.status_code == 401


@pytest.mark.parametrize("forge", [shaped_but_unsigned_token, unsigned_token])
def test_ingest_run_rejects_forged_tokens(client, forge):
    """Ingest writes to Pinecone, so a forged token here is not just a read leak.

    Both forgeries carry `role: admin`; neither is signed by a published key.
    """
    forged = forge({"sub": "u_attacker", "publicMetadata": {"role": "admin"}})
    resp = client.post(
        f"/api/admin/ingest/{uuid.uuid4()}/run", headers=bearer(forged)
    )
    assert resp.status_code == 401


def test_ingest_run_does_not_dispatch_work_for_an_unauthorized_caller(
    client, member_token
):
    """A 403 must happen before the background task is queued."""
    with patch(
        "app.routers.admin_audit.run_ingest_job",
        new=AsyncMock(return_value=None),
    ) as runner:
        client.post(
            f"/api/admin/ingest/{uuid.uuid4()}/run", headers=bearer(member_token)
        )
    runner.assert_not_called()
