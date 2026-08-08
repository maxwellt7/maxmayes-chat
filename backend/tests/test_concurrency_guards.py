"""Concurrency tests for the two controls that must hold across workers.

These are the tests that justify keeping the spend ledger and the rate-limit
buckets in Postgres. They need real row locking, so they are skipped on SQLite
(which serialises writers at the database level and would pass for the wrong
reason). Run with:

    TEST_DATABASE_URL=postgresql+psycopg://... pytest tests/test_concurrency_guards.py
"""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from app.core.errors import SpendCeilingExceededError
from app.db.database import SessionLocal, engine
from app.services import rate_limit, spend_guard
from app.services.rate_limit import BucketPolicy

pytestmark = pytest.mark.skipif(
    engine.dialect.name != "postgresql",
    reason="needs a real Postgres to exercise SELECT ... FOR UPDATE",
)

WORKERS = 40


def test_the_spend_ceiling_is_not_overrun_by_concurrent_requests(monkeypatch):
    """The failure this rules out: every worker reads the same under-ceiling
    total, all conclude there is room, and the day's budget is blown by 4x.

    A ceiling of 10 with a unit cost of 1 must admit exactly 10 of 40 racers.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 10.0)

    def attempt(i: int) -> bool:
        session = SessionLocal()
        try:
            spend_guard.reserve(
                session, Decimal("1.0"), user_id=f"u{i}", request_id=f"r{i}"
            )
            return True
        except SpendCeilingExceededError:
            return False
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(attempt, range(WORKERS)))

    assert sum(results) == 10
    session = SessionLocal()
    try:
        assert spend_guard.current_spend_usd(session) == Decimal("10.000000")
    finally:
        session.close()


def test_the_token_bucket_is_not_overrun_by_concurrent_requests():
    """Capacity 5 with no refill must admit exactly 5 of 40 racers on one key."""
    policy = BucketPolicy(capacity=5.0, refill_per_second=0.0)

    def attempt(_i: int) -> bool:
        session = SessionLocal()
        try:
            return rate_limit.try_consume(session, "acct:shared", policy)
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(attempt, range(WORKERS)))

    assert sum(results) == 5


def test_concurrent_first_requests_of_the_day_do_not_duplicate_the_row():
    """`daily_spend` is keyed on the date, so the race to create the day's first
    row must resolve to one row, not an IntegrityError surfaced to a caller."""
    from app.models.usage import DailySpend

    def attempt(i: int) -> bool:
        session = SessionLocal()
        try:
            spend_guard.reserve(
                session, Decimal("0.001"), user_id=f"u{i}", request_id=f"r{i}"
            )
            return True
        except SpendCeilingExceededError:
            return False
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(attempt, range(WORKERS)))

    session = SessionLocal()
    try:
        assert session.query(DailySpend).count() == 1
    finally:
        session.close()
