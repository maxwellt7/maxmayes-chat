"""End-to-end tests for the chat endpoint's cost and disclosure guards.

The pipeline itself is patched out — no Pinecone, no providers, no registry reads.
What is under test is the order of the gates in front of it: authentication, rate
limit, spend ceiling, and what the client is told when something fails.
"""
from decimal import Decimal
from unittest.mock import patch

import pytest

from app.models.chat import ChatMessage
from app.models.schemas import MAX_CHAT_MESSAGE_CHARS
from app.models.usage import DailySpend
from app.models.user_account import ROLE_MEMBER
from app.services import spend_guard
from tests.authhelpers import bearer, claims


@pytest.fixture
def member(signing_key, make_account):
    make_account("user_chat", role=ROLE_MEMBER)
    return signing_key.sign(claims("user_chat"))


@pytest.fixture
def generous_limits(monkeypatch):
    """Limits high enough not to interfere with tests about other things."""
    from app.config import settings

    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 1000.0)
    monkeypatch.setattr(settings, "chat_rate_limit_per_account_per_hour", 10_000)
    monkeypatch.setattr(settings, "chat_rate_limit_per_ip_per_hour", 10_000)
    monkeypatch.setattr(settings, "chat_rate_limit_burst", 500)


def _fake_pipeline(*chunks: str):
    async def _gen(_query, _db):
        for chunk in chunks:
            yield chunk

    return _gen


def _boom(exc: Exception):
    async def _gen(_query, _db):
        yield ""  # start the stream, then fail mid-flight
        raise exc

    return _gen


def _post(client, token, message="hello", session_id="s1"):
    return client.post(
        "/api/chat",
        json={"message": message, "session_id": session_id},
        headers=bearer(token),
    )


# --- input bounds ----------------------------------------------------------


def test_message_longer_than_the_maximum_is_rejected(client, member, generous_limits):
    resp = _post(client, member, message="x" * (MAX_CHAT_MESSAGE_CHARS + 1))
    assert resp.status_code == 422


def test_a_maximum_length_message_is_accepted(client, member, generous_limits):
    with patch(
        "app.routers.chat.run_pipeline", new=_fake_pipeline("ok")
    ):
        resp = _post(client, member, message="x" * MAX_CHAT_MESSAGE_CHARS)
    assert resp.status_code == 200


def test_empty_message_is_rejected(client, member, generous_limits):
    assert _post(client, member, message="").status_code == 422


def test_a_megabyte_message_is_rejected_before_any_provider_call(
    client, member, generous_limits
):
    """The cost-amplification case: without a maximum this reached the optimizer,
    the router, the verifier and Claude."""
    with patch("app.routers.chat.run_pipeline") as pipeline:
        resp = _post(client, member, message="x" * 1_000_000)
    assert resp.status_code == 422
    pipeline.assert_not_called()


def test_unauthenticated_chat_is_rejected(client):
    resp = client.post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 401


# --- the spend ceiling in the request path ---------------------------------


def test_chat_is_refused_once_the_daily_ceiling_is_reached(
    client, member, db_session, monkeypatch
):
    from app.config import settings

    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 1.0)
    monkeypatch.setattr(settings, "chat_rate_limit_per_account_per_hour", 10_000)
    monkeypatch.setattr(settings, "chat_rate_limit_burst", 500)

    spend_guard.reserve(
        db_session, Decimal("1.00"), user_id="someone_else", request_id="r0"
    )

    with patch("app.routers.chat.run_pipeline") as pipeline:
        resp = _post(client, member)

    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "daily_capacity_reached"
    # The point of checking before synthesis: the expensive call never happens.
    pipeline.assert_not_called()


def test_the_ceiling_is_global_not_per_user(client, signing_key, make_account, monkeypatch):
    """A second free signup must not get its own fresh budget."""
    from app.config import settings

    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 1000.0)
    monkeypatch.setattr(settings, "chat_rate_limit_per_account_per_hour", 10_000)
    monkeypatch.setattr(settings, "chat_rate_limit_burst", 500)

    make_account("user_a", role=ROLE_MEMBER)
    make_account("user_b", role=ROLE_MEMBER)
    token_a = signing_key.sign(claims("user_a"))
    token_b = signing_key.sign(claims("user_b"))

    with patch("app.routers.chat.run_pipeline", new=_fake_pipeline("ok")):
        _post(client, token_a, session_id="sa")
        _post(client, token_b, session_id="sb")

    from app.db.database import SessionLocal

    session = SessionLocal()
    try:
        rows = session.query(DailySpend).all()
        assert len(rows) == 1
        assert rows[0].request_count == 2
    finally:
        session.close()


def test_a_successful_request_records_its_cost(client, member, generous_limits):
    with patch("app.routers.chat.run_pipeline", new=_fake_pipeline("hello")):
        assert _post(client, member).status_code == 200

    from app.db.database import SessionLocal

    session = SessionLocal()
    try:
        assert spend_guard.current_spend_usd(session) > Decimal("0")
    finally:
        session.close()


def test_a_failed_request_returns_its_reservation(client, member, generous_limits):
    """A pipeline failure produced no answer, so it should not consume budget —
    otherwise a broken provider burns the day's allowance."""
    with patch(
        "app.routers.chat.run_pipeline", new=_boom(RuntimeError("provider exploded"))
    ):
        resp = _post(client, member)
    assert resp.status_code == 200  # SSE stream opened, then reported the error

    from app.db.database import SessionLocal

    session = SessionLocal()
    try:
        assert spend_guard.current_spend_usd(session) == Decimal("0")
    finally:
        session.close()


def test_a_partially_streamed_answer_is_not_refunded(client, member, generous_limits):
    """Tokens that reached the user were really paid for. Refunding them would
    understate spend, which is the direction that loses money."""

    async def _partial(_query, _db):
        yield "here is the start of an answ"
        raise RuntimeError("provider died mid-stream")

    with patch("app.routers.chat.run_pipeline", new=_partial):
        _post(client, member)

    from app.db.database import SessionLocal

    session = SessionLocal()
    try:
        assert spend_guard.current_spend_usd(session) > Decimal("0")
    finally:
        session.close()


def test_a_partially_streamed_answer_is_persisted(client, member, generous_limits):
    """The user saw these tokens, so the transcript must not silently lose them."""

    async def _partial(_query, _db):
        yield "here is the start of an answ"
        raise RuntimeError("provider died mid-stream")

    with patch("app.routers.chat.run_pipeline", new=_partial):
        _post(client, member, session_id="sPartial")

    from app.db.database import SessionLocal

    session = SessionLocal()
    try:
        reply = (
            session.query(ChatMessage)
            .filter(
                ChatMessage.session_id == "sPartial",
                ChatMessage.role == "assistant",
            )
            .one()
        )
        assert reply.content == "here is the start of an answ"
    finally:
        session.close()


# --- rate limiting in the request path -------------------------------------


def test_a_burst_beyond_the_limit_is_throttled(client, member, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 1000.0)
    monkeypatch.setattr(settings, "chat_rate_limit_per_account_per_hour", 60)
    monkeypatch.setattr(settings, "chat_rate_limit_per_ip_per_hour", 10_000)
    monkeypatch.setattr(settings, "chat_rate_limit_burst", 3)

    with patch("app.routers.chat.run_pipeline", new=_fake_pipeline("ok")):
        codes = [_post(client, member, session_id=f"s{i}").status_code for i in range(6)]

    assert codes[:3] == [200, 200, 200]
    assert codes[3:] == [429, 429, 429]


def test_a_throttled_request_says_when_to_retry(client, member, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 1000.0)
    monkeypatch.setattr(settings, "chat_rate_limit_per_account_per_hour", 60)
    monkeypatch.setattr(settings, "chat_rate_limit_burst", 1)

    with patch("app.routers.chat.run_pipeline", new=_fake_pipeline("ok")):
        _post(client, member, session_id="s0")
        resp = _post(client, member, session_id="s1")

    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "rate_limited"
    assert resp.headers["Retry-After"] == "60"


def test_throttling_happens_before_the_message_is_persisted(
    client, member, monkeypatch
):
    """A throttled request should cost one row lock, not a write plus a pipeline."""
    from app.config import settings

    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 1000.0)
    monkeypatch.setattr(settings, "chat_rate_limit_per_account_per_hour", 60)
    monkeypatch.setattr(settings, "chat_rate_limit_burst", 1)

    with patch("app.routers.chat.run_pipeline", new=_fake_pipeline("ok")):
        _post(client, member, session_id="s0")
        _post(client, member, session_id="throttled")

    from app.db.database import SessionLocal

    session = SessionLocal()
    try:
        stored = {m.session_id for m in session.query(ChatMessage).all()}
        assert "throttled" not in stored
    finally:
        session.close()


def test_rate_limits_are_per_account(client, signing_key, make_account, monkeypatch):
    """One account exhausting its bucket must not throttle a different account.

    Distinct forwarded IPs isolate the per-IP bucket so this test is only about
    the per-account one.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 1000.0)
    monkeypatch.setattr(settings, "chat_rate_limit_per_account_per_hour", 60)
    monkeypatch.setattr(settings, "chat_rate_limit_per_ip_per_hour", 10_000)
    monkeypatch.setattr(settings, "chat_rate_limit_ip_burst", 500)
    monkeypatch.setattr(settings, "chat_rate_limit_burst", 1)

    make_account("user_a", role=ROLE_MEMBER)
    make_account("user_b", role=ROLE_MEMBER)

    def _send(subject: str, session_id: str, ip: str):
        return client.post(
            "/api/chat",
            json={"message": "hi", "session_id": session_id},
            headers={
                **bearer(signing_key.sign(claims(subject))),
                "X-Forwarded-For": ip,
            },
        )

    with patch("app.routers.chat.run_pipeline", new=_fake_pipeline("ok")):
        first = _send("user_a", "sa", "203.0.113.1")
        second = _send("user_a", "sa2", "203.0.113.2")
        other = _send("user_b", "sb", "203.0.113.3")

    assert first.status_code == 200
    assert second.status_code == 429
    assert other.status_code == 200


def test_a_shared_ip_does_not_throttle_unrelated_accounts_at_default_settings(
    client, signing_key, make_account, monkeypatch
):
    """Office and mobile-carrier NAT put many people on one address. At default
    settings a handful of colleagues must not lock each other out."""
    from app.config import Settings, settings

    defaults = Settings()
    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 1000.0)
    monkeypatch.setattr(
        settings,
        "chat_rate_limit_per_ip_per_hour",
        defaults.chat_rate_limit_per_ip_per_hour,
    )
    monkeypatch.setattr(
        settings, "chat_rate_limit_ip_burst", defaults.chat_rate_limit_ip_burst
    )
    monkeypatch.setattr(
        settings, "chat_rate_limit_burst", defaults.chat_rate_limit_burst
    )

    shared = {"X-Forwarded-For": "203.0.113.50"}
    codes = []
    with patch("app.routers.chat.run_pipeline", new=_fake_pipeline("ok")):
        for i in range(4):
            make_account(f"user_n{i}", role=ROLE_MEMBER)
            codes.append(
                client.post(
                    "/api/chat",
                    json={"message": "hi", "session_id": f"s{i}"},
                    headers={
                        **bearer(signing_key.sign(claims(f"user_n{i}"))),
                        **shared,
                    },
                ).status_code
            )

    assert codes == [200, 200, 200, 200]


def test_per_ip_limit_applies_across_accounts(
    client, signing_key, make_account, monkeypatch
):
    """Otherwise creating accounts is free and per-account limiting is moot."""
    from app.config import settings

    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 1000.0)
    monkeypatch.setattr(settings, "chat_rate_limit_per_account_per_hour", 10_000)
    monkeypatch.setattr(settings, "chat_rate_limit_per_ip_per_hour", 60)
    monkeypatch.setattr(settings, "chat_rate_limit_ip_burst", 1)
    monkeypatch.setattr(settings, "chat_rate_limit_burst", 500)
    monkeypatch.setattr(settings, "trust_forwarded_for", True)

    make_account("user_a", role=ROLE_MEMBER)
    make_account("user_b", role=ROLE_MEMBER)
    same_ip = {"X-Forwarded-For": "203.0.113.9"}

    with patch("app.routers.chat.run_pipeline", new=_fake_pipeline("ok")):
        first = client.post(
            "/api/chat",
            json={"message": "hi", "session_id": "sa"},
            headers={**bearer(signing_key.sign(claims("user_a"))), **same_ip},
        )
        second = client.post(
            "/api/chat",
            json={"message": "hi", "session_id": "sb"},
            headers={**bearer(signing_key.sign(claims("user_b"))), **same_ip},
        )

    assert first.status_code == 200
    assert second.status_code == 429


# --- what the client is told when things fail ------------------------------


class FakePineconeError(Exception):
    """Shaped like a real provider error, which quotes the request back.

    This is the actual leak: `str(exc)` on a Pinecone failure carries the index
    and namespace it was called against.
    """


LEAKY_MESSAGE = (
    "404 NOT_FOUND for index 'personal-knowledge-base' namespace "
    "'therapy-sessions' (project 1, host https://pkb-abc123.svc.pinecone.io)"
)


def test_provider_exception_text_is_not_streamed_to_the_client(
    client, member, generous_limits
):
    with patch(
        "app.routers.chat.run_pipeline", new=_boom(FakePineconeError(LEAKY_MESSAGE))
    ):
        body = _post(client, member).text

    for secret in (
        "personal-knowledge-base",
        "therapy-sessions",
        "pinecone",
        "NOT_FOUND",
        "project 1",
        "FakePineconeError",
    ):
        assert secret.lower() not in body.lower(), f"leaked {secret!r}"


def test_a_stream_failure_reports_a_taxonomy_code_and_request_id(
    client, member, generous_limits
):
    with patch(
        "app.routers.chat.run_pipeline", new=_boom(FakePineconeError(LEAKY_MESSAGE))
    ):
        body = _post(client, member).text

    assert '"code": "internal_error"' in body
    assert "request_id" in body


def test_the_full_exception_is_logged_server_side(
    client, member, generous_limits, caplog
):
    """Generic to the client, complete in the log: the detail must not be lost,
    only relocated."""
    import logging

    with caplog.at_level(logging.ERROR):
        with patch(
            "app.routers.chat.run_pipeline",
            new=_boom(FakePineconeError(LEAKY_MESSAGE)),
        ):
            _post(client, member).text

    logged = caplog.text
    assert "FakePineconeError" in logged
    assert "therapy-sessions" in logged


def test_a_successful_stream_persists_both_messages(client, member, generous_limits):
    with patch(
        "app.routers.chat.run_pipeline", new=_fake_pipeline("hel", "lo")
    ):
        resp = _post(client, member, message="a question", session_id="sX")
    assert resp.status_code == 200
    assert "[DONE]" in resp.text

    from app.db.database import SessionLocal

    session = SessionLocal()
    try:
        rows = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == "sX")
            .order_by(ChatMessage.role)
            .all()
        )
        assert [(r.role, r.content) for r in rows] == [
            ("assistant", "hello"),
            ("user", "a question"),
        ]
    finally:
        session.close()


def test_messages_are_attributed_to_the_verified_subject(
    client, member, generous_limits
):
    """Ownership comes from the verified token, not from anything in the body."""
    with patch("app.routers.chat.run_pipeline", new=_fake_pipeline("ok")):
        _post(client, member, session_id="sY")

    from app.db.database import SessionLocal

    session = SessionLocal()
    try:
        user_ids = {
            m.user_id
            for m in session.query(ChatMessage).filter(
                ChatMessage.session_id == "sY"
            )
        }
        assert user_ids == {"user_chat"}
    finally:
        session.close()
