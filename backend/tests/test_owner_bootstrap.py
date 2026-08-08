"""Tests for the owner lockout recovery paths.

The scenario being defended against: the owner signs up before
`OWNER_CLERK_USER_IDS` is set, is created as a member, and can no longer reach the
admin surface that would let them fix it. Because Clerk session tokens carry no
email claim by default, `OWNER_EMAILS` cannot be relied on to save them.
"""
from app.models.user_account import ROLE_MEMBER, ROLE_OWNER, UserAccount
from app.security.owner_bootstrap import reconcile_owner_allowlist
from tests.authhelpers import bearer, claims


def _account(db, clerk_user_id: str, role: str = ROLE_MEMBER, email=None):
    account = UserAccount(clerk_user_id=clerk_user_id, email=email, role=role)
    db.add(account)
    db.commit()
    return account


def _role_of(db, clerk_user_id: str) -> str:
    db.expire_all()
    return (
        db.query(UserAccount)
        .filter(UserAccount.clerk_user_id == clerk_user_id)
        .one()
        .role
    )


def test_reconciliation_promotes_an_already_registered_owner(
    db_session, monkeypatch
):
    """The core recovery: the account exists as a member, the env var is set
    afterwards, and a restart fixes it."""
    from app.config import settings

    _account(db_session, "user_max", role=ROLE_MEMBER)
    monkeypatch.setattr(settings, "owner_clerk_user_ids", "user_max")

    assert reconcile_owner_allowlist(db_session) == 1
    assert _role_of(db_session, "user_max") == ROLE_OWNER


def test_reconciliation_promotes_by_email_when_the_claim_was_stored(
    db_session, monkeypatch
):
    from app.config import settings

    _account(db_session, "user_max", role=ROLE_MEMBER, email="max@example.com")
    monkeypatch.setattr(settings, "owner_emails", "MAX@example.com")

    assert reconcile_owner_allowlist(db_session) == 1
    assert _role_of(db_session, "user_max") == ROLE_OWNER


def test_reconciliation_is_idempotent(db_session, monkeypatch):
    from app.config import settings

    _account(db_session, "user_max", role=ROLE_MEMBER)
    monkeypatch.setattr(settings, "owner_clerk_user_ids", "user_max")

    assert reconcile_owner_allowlist(db_session) == 1
    assert reconcile_owner_allowlist(db_session) == 0
    assert _role_of(db_session, "user_max") == ROLE_OWNER


def test_reconciliation_leaves_non_allowlisted_accounts_alone(
    db_session, monkeypatch
):
    from app.config import settings

    _account(db_session, "user_max", role=ROLE_MEMBER)
    _account(db_session, "user_stranger", role=ROLE_MEMBER)
    monkeypatch.setattr(settings, "owner_clerk_user_ids", "user_max")

    reconcile_owner_allowlist(db_session)
    assert _role_of(db_session, "user_stranger") == ROLE_MEMBER


def test_reconciliation_never_demotes(db_session, monkeypatch):
    """An allowlist shortened by accident must not strip the only owner.

    Revoking access is a deliberate act against the account — see
    `app.scripts.grant_owner --revoke` — not a side effect of editing config.
    """
    from app.config import settings

    _account(db_session, "user_max", role=ROLE_OWNER)
    monkeypatch.setattr(settings, "owner_clerk_user_ids", "someone_else")

    reconcile_owner_allowlist(db_session)
    assert _role_of(db_session, "user_max") == ROLE_OWNER


def test_empty_allowlist_promotes_nobody(db_session, monkeypatch):
    """In particular it must not promote the first account to sign up, which on a
    public product would hand admin to whichever stranger registered first."""
    from app.config import settings

    monkeypatch.setattr(settings, "owner_clerk_user_ids", "")
    monkeypatch.setattr(settings, "owner_emails", "")
    _account(db_session, "user_stranger", role=ROLE_MEMBER)

    assert reconcile_owner_allowlist(db_session) == 0
    assert _role_of(db_session, "user_stranger") == ROLE_MEMBER


def test_allowlisted_owner_is_promoted_on_next_login(
    client, signing_key, db_session, monkeypatch
):
    """The other half of the recovery: no restart needed either.

    An existing member account on the allowlist is repaired the next time its
    owner makes a request.
    """
    from app.config import settings

    _account(db_session, "user_max", role=ROLE_MEMBER)
    monkeypatch.setattr(settings, "owner_clerk_user_ids", "user_max")

    token = signing_key.sign(claims("user_max"))
    assert client.get(
        "/api/admin/audit/latest", headers=bearer(token)
    ).status_code == 200
    assert _role_of(db_session, "user_max") == ROLE_OWNER


def test_allowlisted_owner_is_created_as_owner_on_first_login(
    client, signing_key, db_session, monkeypatch
):
    from app.config import settings

    monkeypatch.setattr(settings, "owner_clerk_user_ids", "user_max")

    token = signing_key.sign(claims("user_max"))
    assert client.get(
        "/api/admin/audit/latest", headers=bearer(token)
    ).status_code == 200
    assert _role_of(db_session, "user_max") == ROLE_OWNER


def test_grant_owner_cli_promotes_an_existing_account(db_session, capsys):
    from app.scripts.grant_owner import main

    _account(db_session, "user_max", role=ROLE_MEMBER)

    assert main(["user_max"]) == 0
    assert _role_of(db_session, "user_max") == ROLE_OWNER
    assert "member -> owner" in capsys.readouterr().out


def test_grant_owner_cli_can_revoke(db_session):
    from app.scripts.grant_owner import main

    _account(db_session, "user_max", role=ROLE_OWNER)

    assert main(["--revoke", "user_max"]) == 0
    assert _role_of(db_session, "user_max") == ROLE_MEMBER


def test_grant_owner_cli_reports_a_missing_account(db_session, capsys):
    from app.scripts.grant_owner import main

    assert main(["user_nobody"]) == 1
    assert "must sign in once" in capsys.readouterr().err
