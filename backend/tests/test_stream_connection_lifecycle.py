"""Defect T1: the request's DB session was held for the whole SSE stream.

The pool is 10 connections plus 20 overflow and a stream can run for the ~120s
timeout budget, so a few dozen concurrent chats took every connection and
everything else — including unrelated endpoints — blocked on `pool_timeout` and
failed. The recently added auth path made it worse: `resolve_account` runs a
SELECT and sometimes an INSERT plus commit on every authenticated request, on
that same session.

Three things are asserted here:

1. no connection is checked out while tokens are being streamed;
2. more simultaneous streams than the pool can hold are all served;
3. a control showing the pool really is the binding constraint, so (2) is not
   passing because the pool is secretly unbounded.
"""
import asyncio
import threading
from contextlib import contextmanager
from unittest.mock import patch

import httpx
import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import TimeoutError as SQLATimeoutError
from sqlalchemy.orm import sessionmaker

import app.db.database as database
from app.models.index_registry import IndexRegistry
from app.models.user_account import ROLE_MEMBER, UserAccount
from tests.authhelpers import bearer, claims

postgres_only = pytest.mark.skipif(
    database.engine.dialect.name != "postgresql",
    reason="needs a real QueuePool; SQLite tests run on a single StaticPool connection",
)


class PoolWatcher:
    """Counts connections checked out of a pool, via pool events.

    Events rather than `QueuePool.checkedout()` so the same instrument works on
    SQLite's `StaticPool`, which has no such method.
    """

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0
        self._lock = threading.Lock()

    def _on_checkout(self, *_args) -> None:
        with self._lock:
            self.live += 1
            self.peak = max(self.peak, self.live)

    def _on_checkin(self, *_args) -> None:
        with self._lock:
            self.live -= 1


@contextmanager
def watching(engine):
    watcher = PoolWatcher()
    event.listen(engine, "checkout", watcher._on_checkout)
    event.listen(engine, "checkin", watcher._on_checkin)
    try:
        yield watcher
    finally:
        event.remove(engine, "checkout", watcher._on_checkout)
        event.remove(engine, "checkin", watcher._on_checkin)


def _catalog_read(handle) -> None:
    """The registry read the real pipeline performs before synthesis.

    Accepts either shape deliberately. If the endpoint ever goes back to handing
    the pipeline its request-scoped `Session`, this read leaves a transaction —
    and therefore a pooled connection — open for the rest of the stream, and the
    assertions below fail. That is the defect, reproduced.
    """
    from sqlalchemy.orm import Session

    from app.services.orchestrator import _load_active_catalog

    if isinstance(handle, Session):
        handle.query(IndexRegistry).all()
    else:
        _load_active_catalog(handle)


@pytest.fixture
def member(signing_key, make_account):
    make_account("user_stream", role=ROLE_MEMBER)
    return signing_key.sign(claims("user_stream"))


@pytest.fixture
def generous_limits(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 1000.0)
    monkeypatch.setattr(settings, "chat_rate_limit_per_account_per_hour", 10_000)
    monkeypatch.setattr(settings, "chat_rate_limit_per_ip_per_hour", 10_000)
    monkeypatch.setattr(settings, "chat_rate_limit_burst", 500)
    monkeypatch.setattr(settings, "chat_rate_limit_ip_burst", 500)


# --- 1. nothing is held while streaming ------------------------------------


def test_no_connection_is_held_while_tokens_are_streaming(
    client, member, generous_limits, db_session
):
    """The core invariant. Sampled from inside the pipeline, which is exactly
    where the old code was sitting on the request's session."""
    db_session.close()  # don't let the fixture's own session confuse the count
    during_stream = []

    with watching(database.engine) as watcher:
        baseline = watcher.live

        async def _pipeline(_query, handle):
            _catalog_read(handle)
            during_stream.append(watcher.live)
            yield "first"
            during_stream.append(watcher.live)
            yield "second"

        with patch("app.routers.chat.run_pipeline", new=_pipeline):
            resp = client.post(
                "/api/chat",
                json={"message": "hello", "session_id": "s-lifecycle"},
                headers=bearer(member),
            )

        # The instrument works: the pre-stream phase (auth resolution, rate
        # limits, spend reservation, the user's message) did check connections
        # out. Without this the assertion below could pass on a broken watcher.
        assert watcher.peak > baseline

    assert resp.status_code == 200
    assert "[DONE]" in resp.text
    assert during_stream == [baseline, baseline], (
        f"a connection was held across the stream: {during_stream} "
        f"(baseline {baseline})"
    )


def test_the_stream_still_persists_both_messages_from_its_own_session(
    client, member, generous_limits
):
    """Releasing the request session early must not cost us the transcript."""

    async def _pipeline(_query, _factory):
        yield "ans"
        yield "wer"

    with patch("app.routers.chat.run_pipeline", new=_pipeline):
        client.post(
            "/api/chat",
            json={"message": "a question", "session_id": "s-persist"},
            headers=bearer(member),
        )

    from app.models.chat import ChatMessage

    with database.session_scope() as session:
        rows = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == "s-persist")
            .order_by(ChatMessage.role)
            .all()
        )
    assert [(r.role, r.content) for r in rows] == [
        ("assistant", "answer"),
        ("user", "a question"),
    ]


async def _open_stream(session_id: str, user_id: str = "user_stream"):
    """Invoke the endpoint directly and hand back its SSE body iterator.

    `TestClient` runs the app through a blocking portal and, on this version,
    drains the response rather than propagating a hangup, so a disconnect cannot
    be reproduced through it. Calling the endpoint and closing the async
    generator ourselves *is* what Starlette does when a client goes away, and it
    is deterministic.
    """
    from starlette.requests import Request

    from app.models.schemas import ChatRequest
    from app.routers.chat import chat_endpoint
    from app.security.principal import Persona, Principal

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [],
            "client": ("203.0.113.77", 1234),
        }
    )
    session = database.SessionLocal()
    response = await chat_endpoint(
        payload=ChatRequest(message="hi", session_id=session_id),
        request=request,
        principal=Principal(user_id=user_id, persona=Persona.MEMBER),
        db=session,
    )
    return response.body_iterator


async def test_a_client_disconnect_keeps_the_partial_answer(
    member, generous_limits
):
    """`GeneratorExit`/`CancelledError` are not `Exception`, so the error handler
    never saw a hangup: the turn was lost and the reservation leaked, ratcheting
    the daily ceiling closed against everyone. The generator's `finally` now
    settles it.

    The reply must be exactly the delivered prefix — if the pipeline were allowed
    to run to completion the full text would be stored, and this would be
    asserting nothing.
    """
    from app.models.chat import ChatMessage

    async def _pipeline(_query, _factory):
        yield "the part they saw"
        await asyncio.sleep(30)  # never reached; the hangup lands here
        yield "never delivered"

    with patch("app.routers.chat.run_pipeline", new=_pipeline):
        body = await _open_stream("s-hangup")
        first = await body.__anext__()
        assert "the part they saw" in first
        await body.aclose()  # the client goes away

    with database.session_scope() as session:
        reply = (
            session.query(ChatMessage)
            .filter(
                ChatMessage.session_id == "s-hangup",
                ChatMessage.role == "assistant",
            )
            .one_or_none()
        )
    assert reply is not None, "the turn the user partly saw was dropped"
    assert reply.content == "the part they saw"


async def test_a_disconnect_before_any_token_returns_the_reservation(
    member, generous_limits
):
    """Nothing was delivered, so nothing was paid for. Leaked reservations are
    how a flaky client walks the global ceiling down to zero."""
    from decimal import Decimal

    from app.models.chat import ChatMessage
    from app.services import spend_guard

    async def _pipeline(_query, _factory):
        await asyncio.sleep(30)  # never reached; the hangup lands here
        yield "never delivered"

    with patch("app.routers.chat.run_pipeline", new=_pipeline):
        body = await _open_stream("s-empty")
        # Starlette runs the body iterator in a task and cancels it when the
        # client vanishes. Cancelling mid-`__anext__` is that, exactly: the
        # `CancelledError` lands on the pipeline's `await`, not on a `yield`.
        pending = asyncio.ensure_future(body.__anext__())
        await asyncio.sleep(0.05)  # let the endpoint reach the pipeline
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    with database.session_scope() as session:
        assert spend_guard.current_spend_usd(session) == Decimal("0")
        assert (
            session.query(ChatMessage)
            .filter(
                ChatMessage.session_id == "s-empty",
                ChatMessage.role == "assistant",
            )
            .count()
            == 0
        ), "an empty assistant turn was written for an answer nobody received"


# --- 2 & 3. concurrency against a deliberately tiny pool --------------------

POOL_SIZE = 3
POOL_TIMEOUT = 2
CONCURRENT_STREAMS = 8


@pytest.fixture
def tiny_pool(monkeypatch):
    """Point the app at a pool small enough to exhaust deliberately.

    Smaller than production's 10+20 only so the test is fast; the failure mode is
    identical, just reached at 31 concurrent streams instead of 4.
    """
    url = database.engine.url.render_as_string(hide_password=False)
    small_engine = create_engine(
        url,
        pool_size=POOL_SIZE,
        max_overflow=0,
        pool_timeout=POOL_TIMEOUT,
        pool_pre_ping=True,
    )
    factory = sessionmaker(bind=small_engine, autoflush=False, autocommit=False)
    # `get_db`, `session_scope` and `get_session_factory` all read this global at
    # call time, so one patch redirects every session the request path opens.
    monkeypatch.setattr(database, "SessionLocal", factory)
    try:
        yield small_engine
    finally:
        small_engine.dispose()


@postgres_only
async def test_more_concurrent_streams_than_the_pool_are_all_served(
    clerk_configured, signing_key, generous_limits, tiny_pool, db_session
):
    """Eight simultaneous streams against a three-connection pool.

    Each stream does a real registry read through the pipeline's session factory
    and then parks until every other stream has arrived, so all eight are
    genuinely mid-flight at the same moment. If any of them retained its
    connection, the fourth would block for `pool_timeout` and fail.
    """
    from app.main import app

    for i in range(CONCURRENT_STREAMS):
        db_session.add(
            UserAccount(clerk_user_id=f"user_c{i}", email=None, role=ROLE_MEMBER)
        )
    db_session.add(_fixture_index())
    db_session.commit()
    db_session.close()

    all_arrived = asyncio.Event()
    arrived = 0

    async def _pipeline(_query, handle):
        nonlocal arrived
        # The same registry read the real pipeline does before synthesis.
        _catalog_read(handle)
        yield "start"
        arrived += 1
        if arrived >= CONCURRENT_STREAMS:
            all_arrived.set()
        await asyncio.wait_for(all_arrived.wait(), timeout=20)
        yield "end"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as http:

        async def one(i: int) -> str:
            async with http.stream(
                "POST",
                "/api/chat",
                json={"message": "hi", "session_id": f"s-conc-{i}"},
                headers={
                    **bearer(signing_key.sign(claims(f"user_c{i}"))),
                    "X-Forwarded-For": f"203.0.113.{i + 1}",
                },
                timeout=60.0,
            ) as resp:
                assert resp.status_code == 200, f"stream {i} was refused"
                return "".join([chunk async for chunk in resp.aiter_text()])

        with patch("app.routers.chat.run_pipeline", new=_pipeline):
            bodies = await asyncio.gather(
                *(one(i) for i in range(CONCURRENT_STREAMS))
            )

    assert arrived == CONCURRENT_STREAMS, (
        f"only {arrived} of {CONCURRENT_STREAMS} streams were concurrently "
        "in flight — the rest never got a connection"
    )
    for i, body in enumerate(bodies):
        assert "[DONE]" in body, f"stream {i} did not complete: {body!r}"
        assert "internal_error" not in body, f"stream {i} errored: {body!r}"


@postgres_only
def test_control_holding_sessions_across_a_stream_does_exhaust_the_pool(tiny_pool):
    """The control for the test above.

    Holds `POOL_SIZE` sessions with an open read transaction — precisely the
    state the old code left the request session in after the pipeline's catalog
    query — and shows the next checkout times out. Without this, the test above
    could be passing because the pool is not actually bounded.
    """
    held = [database.SessionLocal() for _ in range(POOL_SIZE)]
    try:
        for session in held:
            # A read with no commit: SQLAlchemy keeps the transaction, and
            # therefore the connection, until the session is closed.
            session.execute(select(IndexRegistry.id)).all()

        overflow = database.SessionLocal()
        try:
            with pytest.raises(SQLATimeoutError):
                overflow.execute(select(IndexRegistry.id)).all()
        finally:
            overflow.close()
    finally:
        for session in held:
            session.close()


def _fixture_index() -> IndexRegistry:
    """A registry row for the test database only.

    The production registry being empty is a load-bearing safety property and
    must not be seeded; see docs/KICKOFF-PROMPT.md §4a.
    """
    return IndexRegistry(
        index_name="fixture-index",
        project_id="1",
        api_key_env_var="FIXTURE_KEY",
        dimension=1024,
        embedding_model="fixture-embed",
        metric="cosine",
        domain_description="fixture domain",
        sample_queries=[],
        namespaces={},
        is_active=True,
    )
