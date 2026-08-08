"""user accounts and chat_messages composite indexes

Revision ID: 003_user_accounts
Revises: 002_phase0_audit_ingest
Create Date: 2026-08-08

Creates user_accounts, which becomes the authorization source of truth now that
roles are no longer read from Clerk token claims.

Also adds the composite indexes the chat history and session-list queries
actually need. The existing single-column indexes on session_id and user_id
cannot serve `WHERE user_id = ? AND session_id = ?` or the session list's
`GROUP BY session_id ORDER BY max(created_at)` efficiently.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003_user_accounts"
down_revision: Union[str, None] = "002_phase0_audit_ingest"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_accounts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("clerk_user_id", sa.String(255), nullable=False),
        sa.Column("email", sa.String(320), nullable=True),
        sa.Column("role", sa.String(20), nullable=False, server_default="member"),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "role IN ('owner', 'member')", name="ck_user_accounts_role"
        ),
    )
    op.create_index(
        "ix_user_accounts_clerk_user_id",
        "user_accounts",
        ["clerk_user_id"],
        unique=True,
    )

    op.create_index(
        "ix_chat_messages_user_session",
        "chat_messages",
        ["user_id", "session_id", "created_at"],
    )
    op.create_index(
        "ix_chat_messages_user_created",
        "chat_messages",
        ["user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_chat_messages_user_created", table_name="chat_messages")
    op.drop_index("ix_chat_messages_user_session", table_name="chat_messages")
    op.drop_index("ix_user_accounts_clerk_user_id", table_name="user_accounts")
    op.drop_table("user_accounts")
