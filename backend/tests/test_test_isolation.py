"""Guards on the test harness itself.

A real `.env` sits beside these tests pointing at the deployed database. The
suite now creates and truncates tables for real, so if that URL ever won over
the override in `conftest.py`, running `pytest` would delete production data.
These assertions make that a test failure rather than an incident.
"""
import os

from app.config import settings
from app.db.database import engine

LOCAL_HOSTS = {None, "", "localhost", "127.0.0.1", "::1"}


def test_the_test_database_is_not_remote():
    url = engine.url
    assert url.host in LOCAL_HOSTS or str(url.host).startswith("/"), (
        f"tests are pointed at a remote database host {url.host!r} — refusing. "
        "conftest.py must override DATABASE_URL before app.config is imported."
    )


def test_the_env_override_beat_the_dotenv_file():
    """pydantic-settings ranks environment variables above `.env`. If that ever
    changed, this is where it would be noticed."""
    assert settings.database_url == os.environ["DATABASE_URL"]


def test_the_test_database_is_sqlite_or_an_explicit_test_postgres():
    name = engine.dialect.name
    assert name in {"sqlite", "postgresql"}
    if name == "postgresql":
        assert "TEST_DATABASE_URL" in os.environ, (
            "a Postgres test database must be opted into explicitly via "
            "TEST_DATABASE_URL, never inherited from the ambient DATABASE_URL"
        )
        assert "test" in str(engine.url.database or "").lower()


def test_clerk_is_not_configured_against_a_real_instance_by_default():
    """Individual tests install a loopback JWKS; nothing should reach real Clerk."""
    for value in (
        settings.clerk_issuer,
        settings.clerk_jwks_url,
        settings.clerk_jwt_public_key,
    ):
        assert "clerk.accounts.dev" not in value
        assert "clerk.maxmayes.io" not in value


def test_no_owner_allowlist_leaks_in_from_the_environment():
    """Otherwise a stray env var would make the privilege-escalation tests pass
    by accident."""
    assert settings.owner_user_id_allowlist == set()
    assert settings.owner_email_allowlist == set()
