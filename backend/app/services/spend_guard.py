"""Global daily spend ceiling, enforced in Postgres.

The design is reserve-then-spend, not check-then-spend. Checking first leaves a
window in which every concurrent request reads the same under-ceiling total and
proceeds, which with a 20-connection pool is a 20x overrun. Instead each request
atomically adds its estimate to the day's total inside a transaction that holds a
row lock, and reads back the post-increment figure. If that figure is over the
ceiling the reservation is released and the request is refused, so at most one
request crosses the line.

The ceiling is global rather than per-user because the threat is a single free
signup, and a per-user ceiling generous enough to be usable multiplied by an
unbounded number of signups is not a ceiling.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import SpendCeilingExceededError
from app.models.usage import DailySpend, UsageEvent

_log = logging.getLogger(__name__)


def _today() -> date:
    """UTC, so the ceiling does not reset twice a year on a DST boundary and
    means the same thing regardless of where the server runs."""
    return datetime.now(timezone.utc).date()


def _locked_row(db: Session, day: date) -> DailySpend:
    """Fetch today's row for update, creating it if this is the day's first
    request.

    `with_for_update` serialises concurrent reservations on Postgres. On SQLite
    it is a no-op, but SQLite serialises writers at the database level anyway, so
    the invariant holds on both backends the test suite runs against.
    """
    row = db.execute(
        select(DailySpend).where(DailySpend.spend_date == day).with_for_update()
    ).scalar_one_or_none()
    if row is not None:
        return row

    row = DailySpend(spend_date=day, total_usd=Decimal("0"), request_count=0)
    db.add(row)
    try:
        db.flush()
    except IntegrityError:
        # Another worker created the row between our read and our insert.
        db.rollback()
        row = db.execute(
            select(DailySpend).where(DailySpend.spend_date == day).with_for_update()
        ).scalar_one()
    return row


def current_spend_usd(db: Session) -> Decimal:
    """Today's committed total. Read-only; for reporting, not for gating."""
    row = db.execute(
        select(DailySpend).where(DailySpend.spend_date == _today())
    ).scalar_one_or_none()
    return Decimal(row.total_usd) if row is not None else Decimal("0")


def ceiling_usd() -> Decimal:
    return Decimal(str(settings.daily_spend_ceiling_usd))


def reserve(
    db: Session,
    amount_usd: Decimal,
    *,
    user_id: str,
    request_id: str,
    kind: str = "chat",
) -> Decimal:
    """Reserve `amount_usd` against today's ceiling.

    Returns the day's total after the reservation. Raises
    `SpendCeilingExceededError` — and leaves the total unchanged — if the
    reservation would cross the ceiling.

    Fails closed in every failure mode: a ceiling of zero or less refuses
    everything, and a database error propagates rather than being swallowed into
    an allow.
    """
    limit = ceiling_usd()
    day = _today()

    if limit <= 0:
        raise SpendCeilingExceededError(
            f"daily_spend_ceiling_usd={limit} disables all spend"
        )

    row = _locked_row(db, day)
    projected = Decimal(row.total_usd) + amount_usd

    if projected > limit:
        db.rollback()
        _log.error(
            "spend_ceiling_reached day=%s total=%s requested=%s ceiling=%s "
            "user_id=%s request_id=%s",
            day,
            row.total_usd,
            amount_usd,
            limit,
            user_id,
            request_id,
        )
        raise SpendCeilingExceededError(
            f"day={day} total={row.total_usd} requested={amount_usd} ceiling={limit}"
        )

    row.total_usd = projected
    row.request_count = row.request_count + 1
    row.updated_at = datetime.now(timezone.utc)
    db.add(
        UsageEvent(
            spend_date=day,
            user_id=user_id[:255],
            request_id=request_id[:64],
            kind=kind,
            estimated_usd=amount_usd,
        )
    )
    db.commit()

    remaining = limit - projected
    if remaining <= limit / 10:
        _log.warning(
            "spend_ceiling_nearly_reached day=%s total=%s ceiling=%s remaining=%s",
            day,
            projected,
            limit,
            remaining,
        )
    return projected


def release(db: Session, amount_usd: Decimal) -> None:
    """Return an unused reservation to the day's budget.

    Called when a request is refused or fails before the expensive call, so a
    burst of failures does not consume the day's allowance. Never lets the total
    go negative, which would otherwise be a way to mint budget by repeatedly
    failing.
    """
    if amount_usd <= 0:
        return
    day = _today()
    try:
        row = _locked_row(db, day)
        row.total_usd = max(Decimal("0"), Decimal(row.total_usd) - amount_usd)
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
    except Exception:
        # A failed refund overstates spend, which is the safe direction. It must
        # not turn into a client-visible error on a path that is already failing.
        db.rollback()
        _log.warning("spend_release_failed amount=%s", amount_usd, exc_info=True)
