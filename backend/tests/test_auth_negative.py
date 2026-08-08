"""Negative authentication and authorization tests against the live app.

`test_clerk_jwt_verification.py` tests the verifier in isolation. This file tests
the thing that actually matters: that HTTP requests to real routes are refused.
The distinction is not academic — before this change the verifier existed, was
correct, and was imported by nothing, so every one of the requests below
succeeded.

Every test drives a `TestClient` through the real dependency graph: signature
verification against a loopback JWKS, then role resolution from the
`user_accounts` table. No Clerk credentials, no network, no mocked auth.
"""
from __future__ import annotations

import jwt
import pytest

from app.models.chat import ChatMessage
from app.models.user_account import ROLE_MEMBER, ROLE_OWNER
from tests.authhelpers import (
    FRONTEND,
    ISSUER,
    bearer,
    claims,
    shaped_but_unsigned_token,
    unsigned_token,
)

# Endpoints that must never serve an unauthenticated or forged caller. Chosen to
# cover all three routers, since each previously had its own auth mistake.
PROTECTED = [
    ("get", "/api/chat/sessions"),
    ("get", "/api/chat/history/some-session"),
    ("get", "/api/admin/indexes"),
    ("post", "/api/admin/discover"),
    ("get", "/api/admin/audit/latest"),
]

OWNER_ONLY = [
    ("get", "/api/admin/indexes"),
    ("post", "/api/admin/discover"),
    ("get", "/api/admin/audit/latest"),
    ("post", "/api/admin/indexes/health-check"),
]


def _call(client, method: str, path: str, headers: dict | None = None):
    return getattr(client, method)(path, headers=headers or {})


# --- no credentials --------------------------------------------------------


@pytest.mark.parametrize("method,path", PROTECTED)
def test_missing_authorization_header_is_rejected(client, method, path):
    assert _call(client, method, path).status_code == 401


@pytest.mark.parametrize(
    "header",
    [
        "",
        "Bearer",
        "Bearer ",
        "Basic dXNlcjpwYXNz",
        "token abc",
        "abc",
    ],
)
def test_malformed_authorization_headers_are_rejected(client, header):
    resp = client.get("/api/chat/sessions", headers={"Authorization": header})
    assert resp.status_code == 401


# --- forged credentials ----------------------------------------------------


def test_shaped_but_unsigned_token_is_rejected(client):
    """`hdr.<base64>.sig` — the exact string the deleted middleware trusted.

    The old `tests/test_admin_role_middleware.py` built this shape and asserted a
    200. It is the whole bypass.
    """
    forged = shaped_but_unsigned_token(
        {"sub": "user_attacker", "publicMetadata": {"role": "admin"}}
    )
    for method, path in PROTECTED:
        assert _call(client, method, path, bearer(forged)).status_code == 401


def test_alg_none_token_is_rejected(client):
    forged = unsigned_token(claims("user_attacker"))
    for method, path in PROTECTED:
        assert _call(client, method, path, bearer(forged)).status_code == 401


def test_token_signed_with_unpublished_key_is_rejected(client, attacker_key):
    """A structurally perfect RS256 token signed by a key not in the JWKS."""
    token = attacker_key.sign(claims("user_attacker"))
    for method, path in PROTECTED:
        assert _call(client, method, path, bearer(token)).status_code == 401


def test_expired_token_is_rejected(client, signing_key):
    token = signing_key.sign(claims("user_legit", lifetime=60, issued_offset=-7200))
    for method, path in PROTECTED:
        assert _call(client, method, path, bearer(token)).status_code == 401


def test_not_yet_valid_token_is_rejected(client, signing_key):
    token = signing_key.sign(claims("user_legit", issued_offset=3600))
    assert client.get(
        "/api/chat/sessions", headers=bearer(token)
    ).status_code == 401


def test_token_from_another_issuer_is_rejected(client, signing_key):
    token = signing_key.sign(
        claims("user_legit", issuer="https://evil.clerk.accounts.dev")
    )
    assert client.get(
        "/api/chat/sessions", headers=bearer(token)
    ).status_code == 401


def test_token_for_unauthorized_frontend_is_rejected(client, signing_key):
    token = signing_key.sign(claims("user_legit", azp="https://evil.example.test"))
    assert client.get(
        "/api/chat/sessions", headers=bearer(token)
    ).status_code == 401


def test_hs256_token_signed_with_kid_is_rejected(client, signing_key):
    """Algorithm confusion: a symmetric token that names a real `kid`.

    The verifier restricts algorithms to RS256, so the HMAC branch is never
    entered regardless of what the header asks for.
    """
    token = jwt.encode(
        claims("user_attacker"),
        key="anything",
        algorithm="HS256",
        headers={"kid": signing_key.kid},
    )
    assert client.get(
        "/api/chat/sessions", headers=bearer(token)
    ).status_code == 401


def test_token_with_unknown_kid_is_rejected(client, signing_key):
    token = jwt.encode(
        claims("user_attacker"),
        signing_key.private_pem,
        algorithm="RS256",
        headers={"kid": "no-such-kid"},
    )
    assert client.get(
        "/api/chat/sessions", headers=bearer(token)
    ).status_code == 401


# --- privilege escalation --------------------------------------------------


@pytest.mark.parametrize("method,path", OWNER_ONLY)
def test_validly_signed_member_cannot_reach_owner_routes(
    client, signing_key, make_account, method, path
):
    """A real user, correctly authenticated, is still not an admin."""
    make_account("user_member", role=ROLE_MEMBER)
    token = signing_key.sign(claims("user_member"))
    assert _call(client, method, path, bearer(token)).status_code == 403


@pytest.mark.parametrize(
    "forged_claims",
    [
        {"publicMetadata": {"role": "admin"}},
        {"public_metadata": {"role": "owner"}},
        {"role": "admin"},
        {"persona": "owner"},
        {"metadata": {"role": "owner"}},
        {"org_role": "admin"},
    ],
    ids=["publicMetadata", "public_metadata", "role", "persona", "metadata", "org_role"],
)
def test_role_claims_in_a_valid_token_confer_nothing(
    client, signing_key, make_account, forged_claims
):
    """Signature validity is not authority.

    A user can ask Clerk for a token; on some configurations they can influence
    their own metadata. So the only safe source for a role is our own table,
    which is what these assertions pin down. Each of these claim shapes would
    have granted admin under the deleted middleware.
    """
    make_account("user_member", role=ROLE_MEMBER)
    token = signing_key.sign(claims("user_member", **forged_claims))
    assert client.get(
        "/api/admin/audit/latest", headers=bearer(token)
    ).status_code == 403


def test_role_claim_does_not_upgrade_an_account_in_the_database(
    client, signing_key, make_account, db_session
):
    from app.models.user_account import UserAccount

    make_account("user_member", role=ROLE_MEMBER)
    token = signing_key.sign(
        claims("user_member", publicMetadata={"role": "admin"})
    )
    client.get("/api/chat/sessions", headers=bearer(token))

    db_session.expire_all()
    account = (
        db_session.query(UserAccount)
        .filter(UserAccount.clerk_user_id == "user_member")
        .one()
    )
    assert account.role == ROLE_MEMBER


def test_unknown_subject_is_created_as_member_not_owner(
    client, signing_key, db_session
):
    """First sight of a new user must not be a promotion."""
    from app.models.user_account import UserAccount

    token = signing_key.sign(claims("user_brand_new"))
    assert client.get("/api/chat/sessions", headers=bearer(token)).status_code == 200

    account = (
        db_session.query(UserAccount)
        .filter(UserAccount.clerk_user_id == "user_brand_new")
        .one()
    )
    assert account.role == ROLE_MEMBER

    assert client.get(
        "/api/admin/audit/latest", headers=bearer(token)
    ).status_code == 403


def test_deactivated_account_is_refused(client, signing_key, make_account):
    make_account("user_banned", role=ROLE_MEMBER, is_active=False)
    token = signing_key.sign(claims("user_banned"))
    assert client.get(
        "/api/chat/sessions", headers=bearer(token)
    ).status_code == 403


def test_owner_account_reaches_owner_routes(client, signing_key, make_account):
    """The positive control. Without it the 403s above could be a broken route."""
    make_account("user_owner", role=ROLE_OWNER)
    token = signing_key.sign(claims("user_owner"))
    assert client.get(
        "/api/admin/audit/latest", headers=bearer(token)
    ).status_code == 200


def test_database_refuses_to_store_an_unrecognised_role(db_session):
    """The first line of defence against a bogus role is that it cannot be
    written at all.

    `persona_for_role` failing closed (see `test_authorization_rules.py`) is the
    second: constraints can be dropped by a careless migration, and the code must
    still refuse.
    """
    from sqlalchemy.exc import IntegrityError

    from app.models.user_account import UserAccount

    db_session.add(
        UserAccount(clerk_user_id="user_weird", email=None, role="superadmin")
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


# --- tenant isolation ------------------------------------------------------


def _seed_history(db, user_id: str, session_id: str, text: str) -> None:
    db.add(
        ChatMessage(
            session_id=session_id, user_id=user_id, role="user", content=text
        )
    )
    db.add(
        ChatMessage(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=f"reply to {text}",
        )
    )
    db.commit()


def test_user_cannot_read_another_users_history(
    client, signing_key, make_account, db_session
):
    make_account("user_a", role=ROLE_MEMBER)
    make_account("user_b", role=ROLE_MEMBER)
    _seed_history(db_session, "user_b", "session-of-b", "b private question")

    token_a = signing_key.sign(claims("user_a"))
    resp = client.get(
        "/api/chat/history/session-of-b", headers=bearer(token_a)
    )

    # 200 with nothing in it: the session ID is not a secret, so revealing
    # whether it exists would itself be a leak.
    assert resp.status_code == 200
    assert resp.json()["messages"] == []
    assert "b private question" not in resp.text


def test_owner_cannot_read_another_users_history_either(
    client, signing_key, make_account, db_session
):
    """Owner is an admin role, not a licence to read member transcripts through
    the member-facing endpoint."""
    make_account("user_owner", role=ROLE_OWNER)
    make_account("user_b", role=ROLE_MEMBER)
    _seed_history(db_session, "user_b", "session-of-b", "b private question")

    token = signing_key.sign(claims("user_owner"))
    resp = client.get("/api/chat/history/session-of-b", headers=bearer(token))
    assert resp.json()["messages"] == []


def test_session_list_is_scoped_to_the_caller(
    client, signing_key, make_account, db_session
):
    make_account("user_a", role=ROLE_MEMBER)
    make_account("user_b", role=ROLE_MEMBER)
    _seed_history(db_session, "user_a", "session-of-a", "a question")
    _seed_history(db_session, "user_b", "session-of-b", "b question")

    token_a = signing_key.sign(claims("user_a"))
    body = client.get("/api/chat/sessions", headers=bearer(token_a)).json()

    assert [s["session_id"] for s in body["sessions"]] == ["session-of-a"]
    assert "b question" not in str(body)


def test_history_is_readable_by_its_owner(
    client, signing_key, make_account, db_session
):
    """Positive control for the isolation tests above."""
    make_account("user_a", role=ROLE_MEMBER)
    _seed_history(db_session, "user_a", "session-of-a", "a question")

    token_a = signing_key.sign(claims("user_a"))
    body = client.get(
        "/api/chat/history/session-of-a", headers=bearer(token_a)
    ).json()
    assert [m["content"] for m in body["messages"]] == [
        "a question",
        "reply to a question",
    ]


# --- configuration failure modes -------------------------------------------


def test_unconfigured_clerk_rejects_everything(monkeypatch, signing_key):
    """With no issuer, no JWKS URL and no PEM, the API must refuse, not allow.

    A misconfigured auth layer failing open is how the bypass would come back.
    """
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.main import app
    from app.security.dependencies import reset_verifier_cache

    monkeypatch.setattr(settings, "clerk_jwks_url", "")
    monkeypatch.setattr(settings, "clerk_issuer", "")
    monkeypatch.setattr(settings, "clerk_jwt_public_key", "")
    reset_verifier_cache()

    token = signing_key.sign(claims("user_legit"))
    with TestClient(app) as unconfigured:
        resp = unconfigured.get("/api/chat/sessions", headers=bearer(token))
    assert resp.status_code == 503
    reset_verifier_cache()


def test_jwks_url_is_derived_from_the_issuer(monkeypatch):
    """Env-driven config: setting only `CLERK_ISSUER` must work, for a dev
    instance and a live one alike."""
    from app.config import settings

    for issuer in (
        "https://literate-jaguar-27.clerk.accounts.dev",
        "https://clerk.example.io",
    ):
        monkeypatch.setattr(settings, "clerk_jwks_url", "")
        monkeypatch.setattr(settings, "clerk_issuer", issuer)
        assert (
            settings.resolved_clerk_jwks_url
            == f"{issuer}/.well-known/jwks.json"
        )


def test_explicit_jwks_url_overrides_the_derived_one(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "clerk_issuer", "https://clerk.example.io")
    monkeypatch.setattr(
        settings, "clerk_jwks_url", "https://other.example/jwks.json"
    )
    assert settings.resolved_clerk_jwks_url == "https://other.example/jwks.json"


def test_openapi_schema_is_not_served(client):
    """The generated schema maps every admin endpoint. It stays off the network."""
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert client.get(path).status_code == 404


def test_authorized_parties_accepts_multiple_frontends(monkeypatch):
    """Dev and live frontends coexist without a code change."""
    from app.config import settings

    monkeypatch.setattr(
        settings,
        "clerk_authorized_parties",
        f" {FRONTEND} , https://chat.maxmayes.io ",
    )
    assert settings.authorized_parties == [FRONTEND, "https://chat.maxmayes.io"]


def test_issuer_check_is_enforced_when_configured(client, signing_key):
    """Guard against the check silently becoming optional: the same token that
    works with the right issuer must fail with the wrong one."""
    good = signing_key.sign(claims("user_legit", issuer=ISSUER))
    bad = signing_key.sign(claims("user_legit", issuer=f"{ISSUER}.evil.test"))
    assert client.get("/api/chat/sessions", headers=bearer(good)).status_code == 200
    assert client.get("/api/chat/sessions", headers=bearer(bad)).status_code == 401
