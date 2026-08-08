"""Tests for the Postgres-backed token bucket."""
from datetime import datetime, timedelta, timezone

import pytest

from app.core.errors import RATE_LIMITED, RateLimitedError
from app.models.usage import RateLimitBucket
from app.services import rate_limit
from app.services.rate_limit import BucketPolicy


# A bucket that holds 3 and refills at 1/second, so the arithmetic in the
# assertions below is obvious.
FAST = BucketPolicy(capacity=3.0, refill_per_second=1.0)
# Refills so slowly that no test wall-clock time can top it up.
FROZEN = BucketPolicy(capacity=2.0, refill_per_second=0.0)


def test_a_fresh_bucket_allows_up_to_its_capacity(db_session):
    policy = BucketPolicy(capacity=3.0, refill_per_second=0.0)
    for _ in range(3):
        assert rate_limit.try_consume(db_session, "k", policy)
    assert not rate_limit.try_consume(db_session, "k", policy)


def test_burst_is_bounded_by_capacity_not_by_the_hourly_rate(db_session):
    """60/hour with a burst of 5 must not permit 60 back-to-back requests."""
    policy = BucketPolicy(capacity=5.0, refill_per_second=60 / 3600)
    allowed = sum(
        1 for _ in range(60) if rate_limit.try_consume(db_session, "k", policy)
    )
    assert allowed == 5


def test_tokens_refill_over_time(db_session):
    for _ in range(2):
        assert rate_limit.try_consume(db_session, "k", FROZEN)
    assert not rate_limit.try_consume(db_session, "k", FROZEN)

    # Rewind the bucket's clock rather than sleeping.
    row = db_session.query(RateLimitBucket).filter_by(bucket_key="k").one()
    row.updated_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    db_session.commit()

    assert rate_limit.try_consume(db_session, "k", FAST)


def test_refill_is_capped_at_capacity(db_session):
    """An idle bucket must not accumulate unlimited credit."""
    assert rate_limit.try_consume(db_session, "k", FAST)
    row = db_session.query(RateLimitBucket).filter_by(bucket_key="k").one()
    row.updated_at = datetime.now(timezone.utc) - timedelta(days=30)
    db_session.commit()

    allowed = sum(
        1 for _ in range(50) if rate_limit.try_consume(db_session, "k", FAST)
    )
    assert allowed == 3


def test_buckets_are_independent(db_session):
    policy = BucketPolicy(capacity=1.0, refill_per_second=0.0)
    assert rate_limit.try_consume(db_session, "acct:a", policy)
    assert not rate_limit.try_consume(db_session, "acct:a", policy)
    # A different subject is unaffected.
    assert rate_limit.try_consume(db_session, "acct:b", policy)


def test_hammering_a_drained_bucket_does_not_prevent_refill(db_session):
    """Regression guard: if a denied request left `updated_at` untouched the
    bucket would refill; if it advanced `updated_at` without crediting the
    accrual, a caller who kept retrying could never recover.
    """
    policy = BucketPolicy(capacity=1.0, refill_per_second=1.0)
    assert rate_limit.try_consume(db_session, "k", policy)

    row = db_session.query(RateLimitBucket).filter_by(bucket_key="k").one()
    row.updated_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    db_session.commit()

    for _ in range(20):
        rate_limit.try_consume(db_session, "k", policy)

    row = db_session.query(RateLimitBucket).filter_by(bucket_key="k").one()
    row.updated_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    db_session.commit()
    assert rate_limit.try_consume(db_session, "k", policy)


def test_a_cost_above_capacity_can_never_be_afforded(db_session):
    policy = BucketPolicy(capacity=2.0, refill_per_second=1.0)
    assert not rate_limit.try_consume(db_session, "k", policy, cost=5.0)


def test_state_is_persisted_not_in_memory(db_session):
    """The whole point of the table: a second worker sees the first's spend."""
    policy = BucketPolicy(capacity=1.0, refill_per_second=0.0)
    assert rate_limit.try_consume(db_session, "k", policy)

    from app.db.database import SessionLocal

    other_worker = SessionLocal()
    try:
        assert not rate_limit.try_consume(other_worker, "k", policy)
    finally:
        other_worker.close()


def test_enforce_raises_with_a_retry_after(db_session):
    policy = BucketPolicy(capacity=1.0, refill_per_second=1 / 60)
    rate_limit.enforce(db_session, "k", policy)
    with pytest.raises(RateLimitedError) as exc:
        rate_limit.enforce(db_session, "k", policy)
    assert exc.value.retry_after == 60
    assert exc.value.client_error is RATE_LIMITED


def test_naive_timestamps_are_treated_as_utc(db_session):
    """SQLite returns naive datetimes and Postgres returns aware ones. Neither
    may raise, and neither may be read as a refill of thousands of tokens."""
    assert rate_limit.try_consume(db_session, "k", FAST)
    row = db_session.query(RateLimitBucket).filter_by(bucket_key="k").one()
    row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db_session.commit()
    assert rate_limit.try_consume(db_session, "k", FAST)


def test_bucket_keys_longer_than_the_column_are_truncated(db_session):
    policy = BucketPolicy(capacity=1.0, refill_per_second=0.0)
    long_key = "acct:" + "x" * 500
    assert rate_limit.try_consume(db_session, long_key, policy)
    assert not rate_limit.try_consume(db_session, long_key, policy)


# --- policy construction ---------------------------------------------------


def test_policies_are_built_from_settings(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "chat_rate_limit_per_account_per_hour", 120)
    monkeypatch.setattr(settings, "chat_rate_limit_per_ip_per_hour", 30)
    monkeypatch.setattr(settings, "chat_rate_limit_burst", 7)
    monkeypatch.setattr(settings, "chat_rate_limit_ip_burst", 11)

    account = rate_limit.account_policy()
    assert account.capacity == 7
    assert account.refill_per_second == pytest.approx(120 / 3600)

    ip = rate_limit.ip_policy()
    assert ip.capacity == 11
    assert ip.refill_per_second == pytest.approx(30 / 3600)


def test_the_ip_allowance_is_looser_than_the_account_one_by_default():
    """Addresses are shared; accounts are not. A per-IP bucket as tight as the
    per-account one throttles unrelated people behind the same NAT."""
    from app.config import Settings

    defaults = Settings()
    assert (
        defaults.chat_rate_limit_ip_burst > defaults.chat_rate_limit_burst
    )
    assert (
        defaults.chat_rate_limit_per_ip_per_hour
        >= defaults.chat_rate_limit_per_account_per_hour
    )


def test_a_zero_hourly_rate_never_refills(monkeypatch, db_session):
    """`..._PER_HOUR=0` is a per-subject kill switch, and must not divide by
    zero when computing Retry-After."""
    from app.config import settings

    monkeypatch.setattr(settings, "chat_rate_limit_per_account_per_hour", 0)
    monkeypatch.setattr(settings, "chat_rate_limit_burst", 1)
    policy = rate_limit.account_policy()
    assert policy.seconds_per_token == 3600

    assert rate_limit.try_consume(db_session, "k", policy)
    assert not rate_limit.try_consume(db_session, "k", policy)


def test_capacity_is_at_least_one(monkeypatch):
    """A burst of 0 would deny every request forever, including the owner's."""
    from app.config import settings

    monkeypatch.setattr(settings, "chat_rate_limit_burst", 0)
    assert rate_limit.account_policy().capacity >= 1


# --- IP hashing ------------------------------------------------------------


def test_ip_hash_is_stable_and_not_reversible():
    first = rate_limit.hash_ip("203.0.113.7")
    assert first == rate_limit.hash_ip("203.0.113.7")
    assert "203.0.113.7" not in first
    assert first != rate_limit.hash_ip("203.0.113.8")


def test_ip_hash_depends_on_the_salt(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ip_hash_salt", "salt-a")
    a = rate_limit.hash_ip("203.0.113.7")
    monkeypatch.setattr(settings, "ip_hash_salt", "salt-b")
    assert rate_limit.hash_ip("203.0.113.7") != a


def test_ip_hash_fits_the_bucket_key_column():
    assert len(rate_limit.hash_ip("203.0.113.7")) <= 32
