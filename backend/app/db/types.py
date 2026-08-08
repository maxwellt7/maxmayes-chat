"""Column types that render as native Postgres types in production but still
work on SQLite.

Production is Postgres and the DDL emitted there is unchanged: `uuid`, `jsonb`,
`text[]`. The variants exist so the test suite can create the real schema from
`Base.metadata` on an ephemeral SQLite database — which is what makes model
changes able to break a test instead of silently passing against a `MagicMock`.

Each helper returns a fresh type instance rather than exposing a module-level
constant, because a type object carries per-column state (`as_uuid`, comparator
caches) and sharing one across every table invites action-at-a-distance bugs.
"""
from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


def uuid_type() -> sa.types.TypeEngine[Any]:
    """`uuid` on Postgres, `CHAR(32)` on SQLite, `uuid.UUID` in Python either way."""
    return sa.Uuid(as_uuid=True)


def json_type() -> sa.types.TypeEngine[Any]:
    """`jsonb` on Postgres, `JSON` (text-backed) on SQLite."""
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def text_array_type() -> sa.types.TypeEngine[Any]:
    """`text[]` on Postgres, a JSON array on SQLite.

    SQLite has no array type. Storing a JSON array keeps the Python-side value a
    `list[str]` on both backends, so application code and tests agree.
    """
    return sa.JSON().with_variant(postgresql.ARRAY(sa.Text()), "postgresql")


def money_type() -> sa.types.TypeEngine[Any]:
    """`numeric(12, 6)` — exact decimal, because summing floats to enforce a
    spend ceiling accumulates error in the direction of overspending."""
    return sa.Numeric(precision=12, scale=6, asdecimal=True)
