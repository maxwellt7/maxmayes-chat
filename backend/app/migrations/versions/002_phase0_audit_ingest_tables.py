"""phase0 audit ingest tables

Revision ID: 002_phase0_audit_ingest
Revises: 001_initial_schema
Create Date: 2026-05-11

Creates the index_audits and ingest_jobs tables and extends index_registry
with the new domain / agent_module_path / status / public_safe_default
columns needed by Phase 0.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "002_phase0_audit_ingest"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "index_audits",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("audit_date", sa.Date(), nullable=False),
        sa.Column("index_name", sa.String(255), nullable=False),
        sa.Column("project_id", sa.String(50), nullable=False),
        sa.Column("record_count", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(255), nullable=True),
        sa.Column("dominant_domain", sa.String(100), nullable=True),
        sa.Column(
            "topic_tags",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("sample_chunks", postgresql.JSONB(), nullable=False),
        sa.Column("proposed_disposition", sa.String(50), nullable=False),
        sa.Column("proposed_target_index", sa.String(255), nullable=True),
        sa.Column("approved_disposition", sa.String(50), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "audit_date",
            "index_name",
            "project_id",
            name="index_audits_unique",
        ),
    )

    op.create_table(
        "ingest_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("source_index", sa.String(255), nullable=False),
        sa.Column("target_index", sa.String(255), nullable=False),
        sa.Column("target_namespace", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("total_chunks", sa.Integer(), nullable=True),
        sa.Column(
            "processed_chunks",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "failed_chunks",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "config",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ingest_jobs_status_idx", "ingest_jobs", ["status"])

    op.add_column(
        "index_registry",
        sa.Column("domain", sa.String(100), nullable=True),
    )
    op.add_column(
        "index_registry",
        sa.Column(
            "public_safe_default",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )
    op.add_column(
        "index_registry",
        sa.Column("agent_module_path", sa.String(255), nullable=True),
    )
    op.add_column(
        "index_registry",
        sa.Column(
            "status",
            sa.String(50),
            nullable=False,
            server_default="active",
        ),
    )


def downgrade() -> None:
    op.drop_column("index_registry", "status")
    op.drop_column("index_registry", "agent_module_path")
    op.drop_column("index_registry", "public_safe_default")
    op.drop_column("index_registry", "domain")
    op.drop_index("ingest_jobs_status_idx", table_name="ingest_jobs")
    op.drop_table("ingest_jobs")
    op.drop_table("index_audits")
