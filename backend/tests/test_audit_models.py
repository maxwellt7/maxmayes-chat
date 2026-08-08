"""Schema tests for IndexAudit and IngestJob.

These used to read the model source with `inspect.getsource` and assert that
column names appeared as substrings, because `conftest.py` replaced `Base` with a
`MagicMock` and real introspection was unavailable. A string search cannot tell a
column from a comment mentioning one, and passes for a column that is declared
with the wrong type or nullability.

`Base` is real now, so these assert against the mapped table.
"""
from sqlalchemy import inspect as sa_inspect

from app.db.database import Base
from app.models.index_audit import IndexAudit
from app.models.ingest_job import IngestJob


def _columns(model) -> dict:
    return {c.name: c for c in sa_inspect(model).columns}


def test_index_audit_declares_expected_columns():
    columns = _columns(IndexAudit)
    expected = {
        "id", "audit_date", "index_name", "project_id", "record_count",
        "embedding_model", "dominant_domain", "topic_tags", "sample_chunks",
        "proposed_disposition", "proposed_target_index", "approved_disposition",
        "approved_at", "executed_at", "created_at",
    }
    assert expected <= set(columns)


def test_index_audit_has_unique_constraint_on_date_index_project():
    constraints = {
        tuple(sorted(c.columns.keys()))
        for c in IndexAudit.__table__.constraints
        if c.__class__.__name__ == "UniqueConstraint"
    }
    assert ("audit_date", "index_name", "project_id") in constraints


def test_index_audit_required_columns_are_not_nullable():
    columns = _columns(IndexAudit)
    for name in ("audit_date", "index_name", "project_id", "record_count"):
        assert not columns[name].nullable, f"{name} should be NOT NULL"


def test_ingest_job_declares_expected_columns():
    columns = _columns(IngestJob)
    expected = {
        "id", "source_index", "target_index", "target_namespace", "status",
        "total_chunks", "processed_chunks", "failed_chunks", "error_message",
        "config", "started_at", "completed_at", "created_at",
    }
    assert expected <= set(columns)


def test_ingest_job_status_is_indexed_under_the_deployed_name():
    """The index name must match migration 002, or autogenerate proposes dropping
    the real index and creating an identical one on every run."""
    indexes = {idx.name: tuple(idx.columns.keys()) for idx in IngestJob.__table__.indexes}
    assert indexes.get("ingest_jobs_status_idx") == ("status",)


def test_models_package_registers_every_table():
    """Importing `app.models` must register all tables.

    Alembic's autogenerate diffs `Base.metadata` against the database; a model
    the package does not import looks like a table that should be dropped.
    """
    import app.models  # noqa: F401

    registered = set(Base.metadata.tables)
    assert {
        "chat_messages",
        "index_audits",
        "index_registry",
        "ingest_jobs",
        "user_accounts",
        "daily_spend",
        "usage_events",
        "rate_limit_buckets",
    } <= registered
