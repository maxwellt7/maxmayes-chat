"""Tests for the daily spend ceiling.

The property that matters is that it fails closed: when the day's budget is gone,
or when the configuration is nonsense, the expensive call does not happen. A
ceiling that is merely usually right is not a cost control.
"""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.core.errors import DAILY_CAPACITY_REACHED, SpendCeilingExceededError
from app.models.usage import DailySpend, UsageEvent
from app.services import spend_guard
from app.services.cost import estimate_chat_request_usd
from app.models.schemas import MAX_CHAT_MESSAGE_CHARS


@pytest.fixture
def ceiling(monkeypatch):
    def _set(value: float):
        from app.config import settings

        monkeypatch.setattr(settings, "daily_spend_ceiling_usd", value)

    return _set


def _reserve(db, amount, user_id="u1", request_id="r1"):
    return spend_guard.reserve(
        db, Decimal(str(amount)), user_id=user_id, request_id=request_id
    )


# --- the ceiling holds -----------------------------------------------------


def test_first_request_of_the_day_is_allowed(db_session, ceiling):
    ceiling(1.0)
    assert _reserve(db_session, "0.10") == Decimal("0.10")


def test_reservations_accumulate(db_session, ceiling):
    ceiling(1.0)
    _reserve(db_session, "0.30")
    _reserve(db_session, "0.30")
    assert spend_guard.current_spend_usd(db_session) == Decimal("0.60")


def test_request_that_would_cross_the_ceiling_is_refused(db_session, ceiling):
    ceiling(1.0)
    _reserve(db_session, "0.90")
    with pytest.raises(SpendCeilingExceededError):
        _reserve(db_session, "0.20")


def test_a_refused_request_does_not_consume_budget(db_session, ceiling):
    """The check must not itself spend, or a stream of oversized requests would
    lock out the small ones behind them."""
    ceiling(1.0)
    _reserve(db_session, "0.90")
    for _ in range(5):
        with pytest.raises(SpendCeilingExceededError):
            _reserve(db_session, "0.50")
    assert spend_guard.current_spend_usd(db_session) == Decimal("0.90")
    # And a request that does fit still fits.
    _reserve(db_session, "0.05")
    assert spend_guard.current_spend_usd(db_session) == Decimal("0.95")


def test_exactly_hitting_the_ceiling_is_allowed(db_session, ceiling):
    ceiling(1.0)
    assert _reserve(db_session, "1.00") == Decimal("1.00")
    with pytest.raises(SpendCeilingExceededError):
        _reserve(db_session, "0.000001")


def test_zero_ceiling_refuses_everything(db_session, ceiling):
    """A kill switch: setting the ceiling to 0 stops all spend immediately."""
    ceiling(0.0)
    with pytest.raises(SpendCeilingExceededError):
        _reserve(db_session, "0.000001")


def test_negative_ceiling_refuses_everything(db_session, ceiling):
    """A typo'd env var must fail closed, not open."""
    ceiling(-5.0)
    with pytest.raises(SpendCeilingExceededError):
        _reserve(db_session, "0.01")


def test_single_request_larger_than_the_whole_ceiling_is_refused(
    db_session, ceiling
):
    ceiling(0.01)
    with pytest.raises(SpendCeilingExceededError):
        _reserve(db_session, "5.00")
    assert spend_guard.current_spend_usd(db_session) == Decimal("0")


# --- ledger bookkeeping ----------------------------------------------------


def test_each_reservation_writes_an_auditable_event(db_session, ceiling):
    ceiling(10.0)
    _reserve(db_session, "0.10", user_id="user_a", request_id="req-1")
    _reserve(db_session, "0.20", user_id="user_b", request_id="req-2")

    events = db_session.query(UsageEvent).order_by(UsageEvent.request_id).all()
    assert [(e.user_id, e.request_id) for e in events] == [
        ("user_a", "req-1"),
        ("user_b", "req-2"),
    ]
    assert events[0].estimated_usd == Decimal("0.100000")


def test_refused_requests_are_not_recorded_as_spend(db_session, ceiling):
    ceiling(0.05)
    with pytest.raises(SpendCeilingExceededError):
        _reserve(db_session, "1.00")
    assert db_session.query(UsageEvent).count() == 0


def test_request_count_is_tracked(db_session, ceiling):
    ceiling(10.0)
    _reserve(db_session, "0.01")
    _reserve(db_session, "0.01")
    _reserve(db_session, "0.01")
    row = db_session.query(DailySpend).one()
    assert row.request_count == 3


def test_release_returns_unused_budget(db_session, ceiling):
    ceiling(1.0)
    _reserve(db_session, "0.80")
    spend_guard.release(db_session, Decimal("0.80"))
    assert spend_guard.current_spend_usd(db_session) == Decimal("0")
    _reserve(db_session, "0.90")


def test_release_cannot_mint_budget(db_session, ceiling):
    """Releasing more than was reserved must not drive the total negative, which
    would otherwise be a way to earn budget by failing repeatedly."""
    ceiling(1.0)
    _reserve(db_session, "0.10")
    spend_guard.release(db_session, Decimal("100.00"))
    assert spend_guard.current_spend_usd(db_session) == Decimal("0")


def test_release_of_zero_is_a_noop(db_session, ceiling):
    ceiling(1.0)
    _reserve(db_session, "0.10")
    spend_guard.release(db_session, Decimal("0"))
    assert spend_guard.current_spend_usd(db_session) == Decimal("0.10")


def test_yesterdays_spend_does_not_count_against_today(db_session, ceiling):
    """The ceiling is daily, so a spent yesterday must not block today."""
    ceiling(1.0)
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    db_session.add(
        DailySpend(spend_date=yesterday, total_usd=Decimal("1.00"), request_count=1)
    )
    db_session.commit()

    assert _reserve(db_session, "0.90") == Decimal("0.90")


def test_spend_is_keyed_on_utc_date(db_session, ceiling):
    ceiling(1.0)
    _reserve(db_session, "0.10")
    row = db_session.query(DailySpend).one()
    assert row.spend_date == datetime.now(timezone.utc).date()


def test_current_spend_is_zero_before_any_request(db_session):
    assert spend_guard.current_spend_usd(db_session) == Decimal("0")


def test_decimal_precision_is_preserved(db_session, ceiling):
    """Summing floats to enforce a ceiling accumulates error in the direction of
    overspending; the column is numeric for that reason."""
    ceiling(1.0)
    for _ in range(10):
        _reserve(db_session, "0.1")
    assert spend_guard.current_spend_usd(db_session) == Decimal("1.000000")
    with pytest.raises(SpendCeilingExceededError):
        _reserve(db_session, "0.000001")


# --- cost estimation -------------------------------------------------------


def test_cost_grows_with_message_length():
    small = estimate_chat_request_usd("hi")
    large = estimate_chat_request_usd("x" * MAX_CHAT_MESSAGE_CHARS)
    assert large > small


def test_cost_estimate_is_a_plausible_per_request_figure():
    """Sanity bounds. Not an exact assertion — prices move — but a regression that
    made this a hundredth of a cent would silently disable the ceiling."""
    estimate = estimate_chat_request_usd("what did I write about onboarding emails?")
    assert Decimal("0.01") < estimate < Decimal("1.00")


def test_a_max_length_message_still_costs_less_than_the_default_ceiling():
    """Otherwise the very first request of the day would be refused."""
    from app.config import settings

    worst_case = estimate_chat_request_usd("x" * MAX_CHAT_MESSAGE_CHARS)
    assert worst_case < Decimal(str(settings.daily_spend_ceiling_usd))


def test_unknown_model_is_priced_at_the_most_expensive_rate():
    """An unrecognised model name must not slip under the ceiling as free."""
    known = estimate_chat_request_usd("hello", synthesizer_model="claude-sonnet-4-6")
    unknown = estimate_chat_request_usd("hello", synthesizer_model="some-new-model")
    assert unknown >= known


def test_client_message_for_the_ceiling_names_no_internals():
    assert "spend" not in DAILY_CAPACITY_REACHED.message.lower()
    assert "$" not in DAILY_CAPACITY_REACHED.message
    assert DAILY_CAPACITY_REACHED.status_code == 429


def test_ceiling_reads_from_settings(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "daily_spend_ceiling_usd", 42.5)
    assert spend_guard.ceiling_usd() == Decimal("42.5")


def test_default_ceiling_is_conservative():
    """A default measured in tens of dollars, not thousands: an unnoticed
    overspend is not recoverable, whereas raising the limit is one env var."""
    from app.config import Settings

    assert Settings().daily_spend_ceiling_usd <= 25.0


def test_reserve_survives_a_day_boundary_row_already_existing(db_session, ceiling):
    ceiling(1.0)
    today = datetime.now(timezone.utc).date()
    db_session.add(
        DailySpend(spend_date=today, total_usd=Decimal("0.50"), request_count=2)
    )
    db_session.commit()
    assert _reserve(db_session, "0.25") == Decimal("0.75")


def test_spend_date_type_is_a_date(db_session, ceiling):
    ceiling(1.0)
    _reserve(db_session, "0.01")
    row = db_session.query(DailySpend).one()
    assert isinstance(row.spend_date, date)
