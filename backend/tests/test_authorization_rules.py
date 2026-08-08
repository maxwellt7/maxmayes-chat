"""Unit tests for the authorization primitives.

`persona_for_role` is the single function that converts a stored string into
privilege. Its previous implementation returned `MEMBER` for any non-`None`
string, which contradicted its own docstring and meant a typo'd or legacy role
granted member-level corpus access. These tests pin the fail-closed contract so
it cannot regress quietly.
"""
import pytest

from app.models.user_account import ROLE_MEMBER, ROLE_OWNER
from app.security.owner_bootstrap import email_from_claims, is_allowlisted_owner
from app.security.principal import (
    Persona,
    Principal,
    anonymous_principal,
    persona_for_role,
)


def test_owner_role_maps_to_owner_persona():
    assert persona_for_role(ROLE_OWNER) is Persona.OWNER


def test_member_role_maps_to_member_persona():
    assert persona_for_role(ROLE_MEMBER) is Persona.MEMBER


@pytest.mark.parametrize(
    "role",
    [
        None,
        "",
        "  ",
        "admin",          # the old middleware's magic word
        "superadmin",
        "Owner",          # case matters; a near-miss is still a miss
        "OWNER",
        "member ",        # trailing whitespace from a hand-written migration
        "owner,member",
        "user",
        "public",
        "root",
    ],
)
def test_unrecognised_roles_fail_closed_to_public(role):
    """Anything not exactly `owner` or `member` gets the least privilege.

    Notably `"admin"` — the role name the deleted middleware looked for — confers
    nothing, so a half-finished migration from the old scheme cannot leave
    accounts silently authorized.
    """
    assert persona_for_role(role) is Persona.PUBLIC


def test_public_persona_is_not_a_member():
    assert not Principal("u", Persona.PUBLIC).at_least(Persona.MEMBER)


def test_persona_ordering_is_monotonic():
    public = Principal("u", Persona.PUBLIC)
    member = Principal("u", Persona.MEMBER)
    owner = Principal("u", Persona.OWNER)

    assert owner.at_least(Persona.OWNER)
    assert owner.at_least(Persona.MEMBER)
    assert owner.at_least(Persona.PUBLIC)
    assert member.at_least(Persona.MEMBER)
    assert not member.at_least(Persona.OWNER)
    assert public.at_least(Persona.PUBLIC)
    assert not public.at_least(Persona.OWNER)


def test_only_owner_persona_is_owner():
    assert Principal("u", Persona.OWNER).is_owner
    assert not Principal("u", Persona.MEMBER).is_owner
    assert not Principal("u", Persona.PUBLIC).is_owner


def test_anonymous_principal_is_least_privileged():
    principal = anonymous_principal("ig:12345")
    assert principal.persona is Persona.PUBLIC
    assert not principal.is_owner


def test_principal_is_immutable():
    """Persona must not be widenable by a downstream pipeline stage."""
    principal = Principal("u", Persona.MEMBER)
    with pytest.raises(Exception):
        principal.persona = Persona.OWNER  # type: ignore[misc]


# --- owner allowlist -------------------------------------------------------


def test_allowlist_matches_on_clerk_user_id(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "owner_clerk_user_ids", "user_abc, user_def")
    assert is_allowlisted_owner("user_abc", None)
    assert is_allowlisted_owner("user_def", None)
    assert not is_allowlisted_owner("user_xyz", None)


def test_allowlist_matches_email_case_insensitively(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "owner_emails", "Max@Example.com")
    assert is_allowlisted_owner("user_xyz", "max@example.com")
    assert is_allowlisted_owner("user_xyz", "MAX@EXAMPLE.COM")
    assert not is_allowlisted_owner("user_xyz", "other@example.com")


def test_empty_allowlist_promotes_nobody(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "owner_clerk_user_ids", "")
    monkeypatch.setattr(settings, "owner_emails", "")
    assert not is_allowlisted_owner("user_abc", "max@example.com")
    assert not is_allowlisted_owner("", None)


def test_blank_allowlist_entries_are_ignored(monkeypatch):
    """`OWNER_CLERK_USER_IDS=","` must not make the empty string an owner."""
    from app.config import settings

    monkeypatch.setattr(settings, "owner_clerk_user_ids", " , ,, ")
    assert settings.owner_user_id_allowlist == set()
    assert not is_allowlisted_owner("", None)


# --- email claim extraction ------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "max@example.com"},
        {"email_address": "max@example.com"},
        {"primary_email_address": "max@example.com"},
        {"user": {"email": "max@example.com"}},
        {"publicMetadata": {"email": "max@example.com"}},
    ],
)
def test_email_is_found_in_any_supported_claim_shape(payload):
    assert email_from_claims(payload) == "max@example.com"


def test_missing_email_claim_is_normal_not_an_error():
    """A default Clerk session token has no email claim at all.

    This is the case that broke owner bootstrap: the code read `claims["email"]`,
    got `None`, and fell through to member.
    """
    default_clerk_claims = {
        "sub": "user_2abc",
        "sid": "sess_1",
        "iss": "https://clerk.example.io",
        "azp": "https://chat.example.io",
        "iat": 1,
        "nbf": 1,
        "exp": 2,
        "v": 2,
    }
    assert email_from_claims(default_clerk_claims) is None


def test_non_email_values_are_not_accepted():
    assert email_from_claims({"email": "not-an-email"}) is None
    assert email_from_claims({"email": 12345}) is None
    assert email_from_claims({"email": None}) is None
