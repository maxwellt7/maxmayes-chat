"""Shared test configuration.

This file previously replaced `app.db.database` with a `MagicMock()` — including
`Base` — for the whole process. The effect was that no test ever touched
SQLAlchemy: every model was a mock attribute, every query returned a mock, and a
model change could not fail a test. That is why the suite reported green while
the auth bypass was live.

Instead, the real schema is created from `Base.metadata` against a throwaway
database, so models are exercised for real and a broken column definition fails
collection.

Backend selection:

- default: in-memory SQLite. Zero setup, works in CI. The models declare
  dialect-portable types (`app.db.types`) so the same definitions produce
  `uuid`/`jsonb`/`text[]` on Postgres and workable equivalents here.
- `TEST_DATABASE_URL=postgresql+psycopg://...`: run against a real Postgres
  instead, which is worth doing before trusting anything about `FOR UPDATE`
  locking or numeric precision.
"""
import os

# Must precede any `app.*` import: `app.config.Settings` is instantiated at
# import time and reads DATABASE_URL. A real `.env` sits next to these tests and
# points at a deployed database — pointing the tests at it would be catastrophic,
# so the override is unconditional and happens first.
_TEST_DB_URL = os.environ.get("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ["DATABASE_URL"] = _TEST_DB_URL

# Keep tests off any real Clerk instance. Individual tests that exercise
# verification install their own issuer/JWKS.
os.environ.setdefault("CLERK_ISSUER", "")
os.environ.setdefault("CLERK_JWKS_URL", "")
os.environ.setdefault("CLERK_JWT_PUBLIC_KEY", "")
os.environ.setdefault("CLERK_AUTHORIZED_PARTIES", "")
os.environ.setdefault("OWNER_CLERK_USER_IDS", "")
os.environ.setdefault("OWNER_EMAILS", "")

import pytest  # noqa: E402
from sqlalchemy import event  # noqa: E402

import app.models  # noqa: E402,F401  — registers every table on Base.metadata
from app.db.database import Base, SessionLocal, engine  # noqa: E402


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record):
    """Make SQLite behave enough like Postgres to be worth testing against.

    Foreign keys are off by default in SQLite, which would let a test pass on a
    constraint violation that Postgres would reject.
    """
    if engine.dialect.name != "sqlite":
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@pytest.fixture(scope="session", autouse=True)
def _schema():
    """Create the real schema once for the test session."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _clean_tables(_schema):
    """Truncate between tests.

    Per-test transactional rollback would be faster, but the code under test
    calls `db.commit()` in several places — the spend reservation and the rate
    limiter both must commit to release their row locks — and a wrapping
    transaction makes those commits meaningless. Deleting rows keeps what the
    code actually does intact.
    """
    yield
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture
def db_session():
    """A real Session against the test database."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(autouse=True)
def _reset_verifier():
    """Clear the cached Clerk verifier so per-test settings take effect.

    The verifier is a process-wide singleton built on first use; without this a
    test that configures a JWKS URL would inherit whichever verifier an earlier
    test happened to build.
    """
    from app.security.dependencies import reset_verifier_cache

    reset_verifier_cache()
    yield
    reset_verifier_cache()


# --- auth fixtures ---------------------------------------------------------
# These stand up a complete, self-contained Clerk substitute: a signing key whose
# public half is published as JWKS over loopback, and an attacker key that is
# never published. Tests can then mint both legitimate and forged tokens.


@pytest.fixture(scope="session")
def signing_key():
    from tests.authhelpers import KeyPair

    return KeyPair("test-signing-kid")


@pytest.fixture(scope="session")
def attacker_key():
    """A well-formed RSA key that is deliberately absent from the JWKS."""
    from tests.authhelpers import KeyPair

    return KeyPair("attacker-kid")


@pytest.fixture(scope="session")
def jwks_server(signing_key):
    from tests.authhelpers import JWKSServer

    with JWKSServer([signing_key]) as server:
        yield server


@pytest.fixture
def clerk_configured(monkeypatch, jwks_server):
    """Point the app's verifier at the fake JWKS.

    Uses the same settings production uses, so this also demonstrates the
    configuration is environment-driven rather than tied to any one Clerk
    instance.
    """
    from app.config import settings
    from app.security.dependencies import reset_verifier_cache
    from tests.authhelpers import FRONTEND, ISSUER

    monkeypatch.setattr(settings, "clerk_jwks_url", jwks_server.url)
    monkeypatch.setattr(settings, "clerk_issuer", ISSUER)
    monkeypatch.setattr(settings, "clerk_jwt_public_key", "")
    monkeypatch.setattr(settings, "clerk_authorized_parties", FRONTEND)
    reset_verifier_cache()
    yield
    reset_verifier_cache()


@pytest.fixture
def client(clerk_configured):
    """A TestClient wired to the real dependency graph and the test database."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def make_account(db_session):
    """Insert a `user_accounts` row, which is where authorization actually lives."""
    from app.models.user_account import ROLE_MEMBER, UserAccount

    def _make(clerk_user_id: str, role: str = ROLE_MEMBER, is_active: bool = True):
        account = UserAccount(
            clerk_user_id=clerk_user_id,
            email=None,
            role=role,
            is_active=is_active,
        )
        db_session.add(account)
        db_session.commit()
        return account

    return _make
