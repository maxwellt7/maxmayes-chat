"""
Shared test configuration.

Stubs out psycopg and SQLAlchemy engine creation so that modules which
import app.db.database (e.g. models, orchestrator) can be collected without
a live Postgres connection.
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock

# --- stub psycopg before anything tries to import it ---
_psycopg = ModuleType("psycopg")
sys.modules.setdefault("psycopg", _psycopg)

# --- stub the database module so engine creation is skipped ---
_db_mod = ModuleType("app.db.database")
_db_mod.Base = MagicMock()
_db_mod.SessionLocal = MagicMock()
_db_mod.engine = MagicMock()
_db_mod.get_db = MagicMock()
sys.modules["app.db.database"] = _db_mod

# Also stub app.db so the sub-module lookup works
_db_pkg = sys.modules.get("app.db") or ModuleType("app.db")
_db_pkg.database = _db_mod  # type: ignore[attr-defined]
sys.modules.setdefault("app.db", _db_pkg)
