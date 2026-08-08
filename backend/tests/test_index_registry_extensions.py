"""Verify IndexRegistry has the columns introduced in Phase 0."""
from sqlalchemy import inspect as sa_inspect

from app.models.index_registry import IndexRegistry


def test_index_registry_has_phase0_columns():
    columns = {c.name for c in sa_inspect(IndexRegistry).columns}
    assert {"domain", "public_safe_default", "agent_module_path", "status"} <= columns


def test_public_safe_default_defaults_to_false():
    """Privacy defaults must fail closed: an index is not public until said so."""
    column = IndexRegistry.__table__.columns["public_safe_default"]
    assert column.default is not None
    assert column.default.arg is False
    assert not column.nullable
