"""Token-bucket rate limiting, stored in Postgres.

Kept out of process memory for the same reason as the spend ledger: an
in-process bucket is per-worker, so the effective limit is silently multiplied by
the worker count and resets on every deploy.

A token bucket rather than a fixed window because a fixed window lets a caller
spend the whole allowance in the last second of one window and again in the first
second of the next. The bucket stores a balance and a timestamp; the refill owed
since that timestamp is computed on read, so no background job is needed.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import RateLimitedError
from app.models.usage import RateLimitBucket

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BucketPolicy:
    capacity: float
    refill_per_second: float

    @property
    def seconds_per_token(self) -> int:
        if self.refill_per_second <= 0:
            return 3600
        return max(1, int(1.0 / self.refill_per_second))


def _per_hour_policy(per_hour: int, burst: int) -> BucketPolicy:
    return BucketPolicy(
        capacity=float(max(1, burst)),
        refill_per_second=float(max(0, per_hour)) / 3600.0,
    )


def account_policy() -> BucketPolicy:
    return _per_hour_policy(
        settings.chat_rate_limit_per_account_per_hour,
        settings.chat_rate_limit_burst,
    )


def ip_policy() -> BucketPolicy:
    return _per_hour_policy(
        settings.chat_rate_limit_per_ip_per_hour,
        settings.chat_rate_limit_ip_burst,
    )


def hash_ip(ip: str) -> str:
    """Salted hash, so the buckets table is not a log of who visited from where."""
    digest = hashlib.sha256(f"{settings.ip_hash_salt}:{ip}".encode()).hexdigest()
    return digest[:32]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _elapsed_seconds(updated_at: datetime) -> float:
    # SQLite hands back naive datetimes; Postgres hands back aware ones. Treat a
    # naive value as UTC rather than letting the subtraction raise.
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return max(0.0, (_now() - updated_at).total_seconds())


def try_consume(
    db: Session, bucket_key: str, policy: BucketPolicy, cost: float = 1.0
) -> bool:
    """Take `cost` tokens from `bucket_key`, or return False if unaffordable.

    The read takes a row lock so two workers cannot both spend the last token.
    Commits on both paths so the lock is not held across the caller's subsequent
    work.
    """
    key = bucket_key[:128]
    row = db.execute(
        select(RateLimitBucket)
        .where(RateLimitBucket.bucket_key == key)
        .with_for_update()
    ).scalar_one_or_none()

    if row is None:
        if cost > policy.capacity:
            return False
        row = RateLimitBucket(
            bucket_key=key, tokens=policy.capacity - cost, updated_at=_now()
        )
        db.add(row)
        try:
            db.commit()
            return True
        except IntegrityError:
            # Lost the race to create the bucket; fall through and treat it as
            # an existing bucket on the retry below.
            db.rollback()
            row = db.execute(
                select(RateLimitBucket)
                .where(RateLimitBucket.bucket_key == key)
                .with_for_update()
            ).scalar_one()

    available = min(
        policy.capacity,
        row.tokens + _elapsed_seconds(row.updated_at) * policy.refill_per_second,
    )
    if available < cost:
        # Persist the refill anyway: not doing so would let a caller who keeps
        # hammering the endpoint hold `updated_at` still and never accrue tokens.
        row.tokens = available
        row.updated_at = _now()
        db.commit()
        return False

    row.tokens = available - cost
    row.updated_at = _now()
    db.commit()
    return True


def enforce(db: Session, bucket_key: str, policy: BucketPolicy) -> None:
    """`try_consume`, raising `RateLimitedError` instead of returning False."""
    if not try_consume(db, bucket_key, policy):
        _log.warning("rate_limited bucket=%s", bucket_key)
        raise RateLimitedError(
            f"bucket={bucket_key}", retry_after=policy.seconds_per_token
        )
