from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import normalized_database_url


class Base(DeclarativeBase):
    pass


def _engine_options(url: str) -> dict[str, Any]:
    """Pooling options appropriate to the backend named by `url`.

    Postgres — the production target — gets a bounded pool. SQLite, which the
    test suite uses, rejects `pool_size`/`max_overflow` for in-memory databases
    and needs a single shared connection so every session sees the same schema.
    """
    if url.startswith("sqlite"):
        return {
            "connect_args": {"check_same_thread": False},
            "poolclass": StaticPool,
        }
    return {
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 10,
        "pool_recycle": 1800,
    }


_url = normalized_database_url()
engine = create_engine(_url, **_engine_options(_url))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# A callable that hands back a fresh Session. Passing this around, rather than a
# live Session, is what lets a long-running caller decide for itself how briefly
# to hold a pooled connection.
SessionFactory = Callable[[], Session]


def get_session_factory() -> SessionFactory:
    """The process-wide factory, resolved at call time.

    Callers that need to hand a factory to something long-running should go
    through this rather than importing `SessionLocal` into their own module
    namespace, so a test that swaps the factory swaps it everywhere.
    """
    return SessionLocal


def get_db() -> Iterator[Session]:
    """Request-scoped session.

    FastAPI keeps a `yield` dependency open until the response is *finished*,
    which for a `StreamingResponse` means the whole life of the stream. Anything
    that streams for longer than a moment must not rely on this — see
    `session_scope`.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope(factory: SessionFactory | None = None) -> Iterator[Session]:
    """A session whose pooled connection is returned as soon as the block ends.

    The pool is 10 connections plus 20 overflow. A request that holds one for the
    duration of a two-minute stream removes it from that budget for two minutes,
    so a few dozen concurrent readers exhaust the pool and everyone else waits out
    `pool_timeout` and fails. Work that happens around a stream uses this instead,
    holding a connection for the milliseconds it takes to run the statement.

    `factory` exists so callers that are handed a session factory rather than a
    session — the pipeline, chiefly — can use the same scoping discipline.
    """
    session = (factory or SessionLocal)()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
