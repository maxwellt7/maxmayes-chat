"""spend ledger, usage events and rate-limit buckets

Revision ID: 004_usage_rate_limits
Revises: 003_user_accounts
Create Date: 2026-08-08

Backing store for the daily spend ceiling and the token-bucket rate limiter.

These live in Postgres rather than process memory because both controls must hold
across uvicorn workers and survive restarts. An in-process counter multiplies the
effective limit by the worker count and resets to zero on deploy, which is the
moment an abuse spike is most likely to be in progress.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004_usage_rate_limits"
down_revision: Union[str, None] = "003_user_accounts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # One row per UTC day. The primary key is the date, which makes the row the
    # natural lock target for reserve-then-spend.
    op.create_table(
        "daily_spend",
        sa.Column("spend_date", sa.Date(), primary_key=True),
        sa.Column(
            "total_usd",
            sa.Numeric(precision=12, scale=6),
            nullable=False,
            server_default="0",
        ),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "usage_events",
        sa.Column(
            "id",
            sa.Uuid(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("spend_date", sa.Date(), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("estimated_usd", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_usage_events_spend_date", "usage_events", ["spend_date"])
    op.create_index(
        "ix_usage_events_date_user", "usage_events", ["spend_date", "user_id"]
    )

    op.create_table(
        "rate_limit_buckets",
        sa.Column("bucket_key", sa.String(128), primary_key=True),
        sa.Column("tokens", sa.Float(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    op.drop_table("rate_limit_buckets")
    op.drop_index("ix_usage_events_date_user", table_name="usage_events")
    op.drop_index("ix_usage_events_spend_date", table_name="usage_events")
    op.drop_table("usage_events")
    op.drop_table("daily_spend")
