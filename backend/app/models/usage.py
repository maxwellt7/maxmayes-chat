"""Tables backing the spend ceiling and the rate limiter.

Both live in Postgres rather than process memory on purpose. In-process counters
are per-worker, so with N uvicorn workers a "100 requests/hour" limit is really
100N, and a restart resets the ledger to zero — which is precisely the moment an
abuse spike is most likely to be underway.
"""
import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, Float, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.db.types import money_type, uuid_type


class DailySpend(Base):
    """One row per UTC day holding the reserved-and-settled spend total.

    The row is the concurrency control point: a request reserves its estimated
    cost with a locking read plus an update in a single transaction, so two
    workers cannot both observe "under the ceiling" and both proceed.
    """

    __tablename__ = "daily_spend"

    spend_date: Mapped[date] = mapped_column(Date, primary_key=True)
    total_usd: Mapped[Decimal] = mapped_column(
        money_type(), nullable=False, server_default="0"
    )
    request_count: Mapped[int] = mapped_column(
        nullable=False, server_default="0", default=0
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class UsageEvent(Base):
    """Per-request cost record. Exists so the aggregate above can be audited and
    so abuse can be attributed after the fact."""

    __tablename__ = "usage_events"
    __table_args__ = (
        Index("ix_usage_events_date_user", "spend_date", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        uuid_type(), primary_key=True, default=uuid.uuid4
    )
    spend_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    estimated_usd: Mapped[Decimal] = mapped_column(money_type(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RateLimitBucket(Base):
    """A leaky token bucket, one row per subject.

    `tokens` is the balance as of `updated_at`; the refill between then and now
    is computed on read. Storing the balance rather than a request log keeps this
    O(1) in space and avoids the periodic-cleanup job a sliding-window log needs.
    """

    __tablename__ = "rate_limit_buckets"

    bucket_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    tokens: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
