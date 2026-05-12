"""Verify IndexRegistry has the new columns introduced in Phase 0."""
import inspect


def test_index_registry_source_has_new_columns():
    from app.models import index_registry
    source = inspect.getsource(index_registry)
    for name in ["domain", "public_safe_default", "agent_module_path", "status"]:
        assert f"{name}:" in source, f"IndexRegistry source missing column {name}"


def test_index_registry_imports_cleanly():
    import importlib
    mod = importlib.import_module("app.models.index_registry")
    assert hasattr(mod, "IndexRegistry")
