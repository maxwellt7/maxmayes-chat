from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import normalized_database_url


class Base(DeclarativeBase):
    pass


def _engine_options(url: str) -> dict:
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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
