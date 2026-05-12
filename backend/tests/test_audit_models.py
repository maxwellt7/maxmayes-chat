"""Smoke tests for IndexAudit and IngestJob models.

conftest.py mocks Base globally so live SQLAlchemy introspection isn't
available here. We verify the modules import cleanly, define the expected
classes, and declare the expected column names (verified via source).
Full schema correctness is enforced by the Alembic migration in T7.
"""
import importlib
import inspect


def test_index_audit_module_imports_and_defines_class():
    mod = importlib.import_module("app.models.index_audit")
    assert hasattr(mod, "IndexAudit")


def test_index_audit_source_declares_expected_columns():
    from app.models import index_audit
    source = inspect.getsource(index_audit)
    for name in [
        "audit_date", "index_name", "project_id", "record_count",
        "embedding_model", "dominant_domain", "topic_tags", "sample_chunks",
        "proposed_disposition", "proposed_target_index", "approved_disposition",
        "approved_at", "executed_at", "created_at",
    ]:
        assert f"{name}:" in source, f"IndexAudit source missing column {name}"
    assert 'unique_constraint' in source.lower() or 'UniqueConstraint' in source


def test_ingest_job_module_imports_and_defines_class():
    mod = importlib.import_module("app.models.ingest_job")
    assert hasattr(mod, "IngestJob")


def test_ingest_job_source_declares_expected_columns():
    from app.models import ingest_job
    source = inspect.getsource(ingest_job)
    for name in [
        "source_index", "target_index", "target_namespace", "status",
        "total_chunks", "processed_chunks", "failed_chunks", "error_message",
        "config", "started_at", "completed_at", "created_at",
    ]:
        assert f"{name}:" in source, f"IngestJob source missing column {name}"


def test_models_init_exports_new_modules():
    """models/__init__.py must import the new modules so SQLAlchemy can
    see them when Alembic introspects metadata."""
    import sys
    # Clear any cached imports to make this test deterministic.
    for mod_name in [
        "app.models", "app.models.index_audit", "app.models.ingest_job",
    ]:
        sys.modules.pop(mod_name, None)
    import app.models  # noqa: F401
    assert "app.models.index_audit" in sys.modules
    assert "app.models.ingest_job" in sys.modules
