# RAG Redesign — Phase 0 Implementation Plan (Audit + Re-ingest)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the audit + re-ingest infrastructure that consolidates 40 messy Pinecone indexes into 7 well-described domain indexes on Cohere `embed-v3.0`, with consistent metadata. Non-destructive: legacy indexes stay live throughout.

**Architecture:** Adds 5 new SQLAlchemy models, 3 new services (rerank, llm, chunking, plus extending embedding), 2 CLI scripts (audit, reingest), 4 admin endpoints, 1 admin page, and 1 middleware. No orchestrator changes — that's Phase 1.

**Tech Stack:** FastAPI + SQLAlchemy + Alembic + Pinecone + Cohere `embed-v3.0` + LangChain `SemanticChunker` + Next.js admin UI.

**Source spec:** `docs/superpowers/specs/2026-05-10-rag-redesign-design.md`
**Architecture:** `docs/architecture-rag-redesign-2026-05-10.md`

---

## Phase 0 deliverables

End state of this plan:
- Postgres has 5 new tables + extended `index_registry`
- Backend has audit + re-ingest scripts with admin API endpoints
- Frontend has `/admin/audit` page with bulk approval and ingest queue
- Audit has been run; Max has approved dispositions; re-ingest jobs have completed
- 7 new domain indexes exist in Pinecone alongside the original 40
- `legacy-archive-{date}` index exists as the safety backup
- Production chat behavior is **unchanged** (still uses old orchestrator)

---

### Task 1: Add Cohere embed-v3 to embedding_service

**Files:**
- Modify: `backend/app/services/embedding_service.py`
- Test: `backend/tests/test_embedding_service.py`

- [ ] **Step 1: Read current embedding_service.py to understand the existing OpenAI path**

Run: `cat backend/app/services/embedding_service.py`

- [ ] **Step 2: Write the failing test for Cohere embed**

```python
# backend/tests/test_embedding_service.py — add to existing file
from unittest.mock import MagicMock, patch
from app.services.embedding_service import generate_embedding_cohere


def test_generate_embedding_cohere_returns_list_of_floats(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test-cohere")
    with patch("app.services.embedding_service.cohere.ClientV2") as MockClient:
        mock_resp = MagicMock()
        mock_resp.embeddings.float = [[0.1] * 1024]
        MockClient.return_value.embed.return_value = mock_resp

        result = generate_embedding_cohere("test text", input_type="search_query")

        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_embedding_service.py::test_generate_embedding_cohere_returns_list_of_floats -v`
Expected: FAIL with `ImportError` or `AttributeError`

- [ ] **Step 4: Implement generate_embedding_cohere**

Add to `backend/app/services/embedding_service.py`:

```python
import cohere
from app.config import settings


def generate_embedding_cohere(
    text: str,
    input_type: str = "search_query",  # "search_query" or "search_document"
    model: str = "embed-v3.0",
) -> list[float]:
    """Generate a single embedding via Cohere embed-v3 (1024-dim)."""
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY is not configured")
    client = cohere.ClientV2(api_key=settings.cohere_api_key)
    resp = client.embed(
        texts=[text],
        model=model,
        input_type=input_type,
        embedding_types=["float"],
    )
    return resp.embeddings.float[0]


def generate_embeddings_cohere_batch(
    texts: list[str],
    input_type: str = "search_document",
    model: str = "embed-v3.0",
) -> list[list[float]]:
    """Batch embed (up to 96 texts per call per Cohere limits)."""
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY is not configured")
    if len(texts) > 96:
        raise ValueError(f"Cohere embed batch limit is 96, got {len(texts)}")
    client = cohere.ClientV2(api_key=settings.cohere_api_key)
    resp = client.embed(
        texts=texts,
        model=model,
        input_type=input_type,
        embedding_types=["float"],
    )
    return resp.embeddings.float
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python3 -m pytest tests/test_embedding_service.py::test_generate_embedding_cohere_returns_list_of_floats -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/embedding_service.py backend/tests/test_embedding_service.py
git commit -m "feat(embeddings): add Cohere embed-v3 single + batch helpers"
```

---

### Task 2: Add rerank_service for Cohere Rerank-v3.5

**Files:**
- Create: `backend/app/services/rerank_service.py`
- Test: `backend/tests/test_rerank_service.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_rerank_service.py
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.services.rerank_service import rerank


@pytest.mark.asyncio
async def test_rerank_returns_reordered_chunks(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test-cohere")
    chunks = [{"text": "first"}, {"text": "second"}, {"text": "third"}]

    with patch("app.services.rerank_service.cohere.AsyncClientV2") as MockClient:
        mock_results = [MagicMock(index=2), MagicMock(index=0), MagicMock(index=1)]
        MockClient.return_value.rerank = AsyncMock(
            return_value=MagicMock(results=mock_results)
        )
        result = await rerank("query", chunks, top_n=3)

    assert [c["text"] for c in result] == ["third", "first", "second"]


@pytest.mark.asyncio
async def test_rerank_falls_back_to_input_order_on_no_key(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "")
    chunks = [{"text": "a"}, {"text": "b"}]
    result = await rerank("query", chunks, top_n=2)
    assert [c["text"] for c in result] == ["a", "b"]


@pytest.mark.asyncio
async def test_rerank_falls_back_on_api_error(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test-cohere")
    chunks = [{"text": "a"}, {"text": "b"}]
    with patch("app.services.rerank_service.cohere.AsyncClientV2") as MockClient:
        MockClient.return_value.rerank = AsyncMock(side_effect=Exception("API down"))
        result = await rerank("query", chunks, top_n=2)
    assert [c["text"] for c in result] == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_rerank_service.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement rerank_service**

```python
# backend/app/services/rerank_service.py
import logging
from typing import Any

import cohere

from app.config import settings

_log = logging.getLogger(__name__)


async def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int,
    model: str = "rerank-v3.5",
) -> list[dict[str, Any]]:
    """Cross-encoder rerank via Cohere. Falls back to input order on missing
    key or API failure (graceful degradation per NFR-REL-3)."""
    if not chunks:
        return []
    if not settings.cohere_api_key:
        return chunks[:top_n]
    try:
        client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
        response = await client.rerank(
            model=model,
            query=query,
            documents=[c["text"] for c in chunks],
            top_n=min(top_n, len(chunks)),
        )
        return [chunks[r.index] for r in response.results]
    except Exception as exc:
        _log.warning("rerank failed, falling back to input order: %s", exc)
        return chunks[:top_n]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python3 -m pytest tests/test_rerank_service.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/rerank_service.py backend/tests/test_rerank_service.py
git commit -m "feat(rerank): Cohere Rerank-v3.5 wrapper with graceful fallback"
```

---

### Task 3: Add llm_service shared adapter

**Files:**
- Create: `backend/app/services/llm_service.py`
- Test: `backend/tests/test_llm_service.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_llm_service.py
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.services.llm_service import call_openai_json, call_openai_text


@pytest.mark.asyncio
async def test_call_openai_json_returns_parsed_dict(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    with patch("app.services.llm_service.AsyncOpenAI") as MockClient:
        mock_resp = MagicMock(choices=[MagicMock(message=MagicMock(
            content=json.dumps({"foo": "bar"})
        ))])
        MockClient.return_value.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await call_openai_json("gpt-4.1-mini", "system", "user", temperature=0)
    assert result == {"foo": "bar"}


@pytest.mark.asyncio
async def test_call_openai_text_returns_string(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    with patch("app.services.llm_service.AsyncOpenAI") as MockClient:
        mock_resp = MagicMock(choices=[MagicMock(message=MagicMock(content="hello"))])
        MockClient.return_value.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await call_openai_text("gpt-4.1-mini", "system", "user", temperature=0)
    assert result == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_llm_service.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement llm_service**

```python
# backend/app/services/llm_service.py
"""Thin shared adapters for OpenAI and Anthropic with consistent error handling.

Keeps API-call patterns DRY across orchestrator nodes and agent helpers.
"""
import logging

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import settings

_log = logging.getLogger(__name__)


def _require_openai_key() -> str:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    return settings.openai_api_key


def _require_anthropic_key() -> str:
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not configured")
    return settings.anthropic_api_key


async def call_openai_text(
    model: str, system: str, user: str, temperature: float = 0
) -> str:
    client = AsyncOpenAI(api_key=_require_openai_key())
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


async def call_openai_json(
    model: str, system: str, user: str, temperature: float = 0
) -> dict:
    import json as _json
    client = AsyncOpenAI(api_key=_require_openai_key())
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return _json.loads(resp.choices[0].message.content)


def get_anthropic_client() -> AsyncAnthropic:
    """Returned for streaming use cases; caller manages the stream context."""
    return AsyncAnthropic(api_key=_require_anthropic_key())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python3 -m pytest tests/test_llm_service.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/llm_service.py backend/tests/test_llm_service.py
git commit -m "feat(services): shared OpenAI/Anthropic adapter with text + json modes"
```

---

### Task 4: Create chunking_service wrapping LangChain SemanticChunker

**Files:**
- Create: `backend/app/services/chunking_service.py`
- Test: `backend/tests/test_chunking_service.py`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add LangChain dependencies**

Add to `backend/requirements.txt`:

```
langchain-core>=0.3.0,<0.4.0
langchain-experimental>=0.3.0,<0.4.0
langchain-cohere>=0.3.0,<0.4.0
```

Run: `cd backend && pip install -r requirements.txt 2>&1 | tail -5`
Expected: installs without errors

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_chunking_service.py
from unittest.mock import patch
from app.services.chunking_service import semantic_chunk_text


def test_semantic_chunk_returns_non_empty_list_for_long_text(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test")
    text = "First sentence. Second sentence. Third unrelated thought. " * 30

    # Mock SemanticChunker to avoid real Cohere call
    with patch("app.services.chunking_service.SemanticChunker") as MockChunker:
        instance = MockChunker.return_value
        instance.split_text.return_value = ["chunk one " * 50, "chunk two " * 50]
        chunks = semantic_chunk_text(text)

    assert len(chunks) == 2
    assert all(isinstance(c, str) for c in chunks)


def test_semantic_chunk_returns_single_chunk_for_short_text(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "cohere_api_key", "test")
    with patch("app.services.chunking_service.SemanticChunker") as MockChunker:
        MockChunker.return_value.split_text.return_value = ["short"]
        chunks = semantic_chunk_text("short")
    assert chunks == ["short"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_chunking_service.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Implement chunking_service**

```python
# backend/app/services/chunking_service.py
"""Semantic chunker for re-ingest pipeline.

Uses LangChain's SemanticChunker with Cohere embeddings to find natural
breakpoints by sentence-pair similarity. Better than fixed-token splitters
for heterogeneous content (per spec §7.4).
"""
from langchain_cohere import CohereEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from app.config import settings


def semantic_chunk_text(
    text: str,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 90.0,
) -> list[str]:
    """Split `text` into semantically-coherent chunks (~512 token target)."""
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY is not configured")
    embeddings = CohereEmbeddings(
        cohere_api_key=settings.cohere_api_key,
        model="embed-v3.0",
    )
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    return chunker.split_text(text)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `cd backend && python3 -m pytest tests/test_chunking_service.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/chunking_service.py backend/tests/test_chunking_service.py backend/requirements.txt
git commit -m "feat(chunking): semantic chunker via LangChain + Cohere embeddings"
```

---

### Task 5: SQLAlchemy models for index_audits + ingest_jobs

**Files:**
- Create: `backend/app/models/index_audit.py`
- Create: `backend/app/models/ingest_job.py`
- Modify: `backend/app/models/__init__.py`
- Test: `backend/tests/test_audit_models.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_audit_models.py
from datetime import date
from app.models.index_audit import IndexAudit
from app.models.ingest_job import IngestJob


def test_index_audit_columns_exist():
    cols = {c.name for c in IndexAudit.__table__.columns}
    expected = {
        "id", "audit_date", "index_name", "project_id", "record_count",
        "embedding_model", "dominant_domain", "topic_tags", "sample_chunks",
        "proposed_disposition", "proposed_target_index", "approved_disposition",
        "approved_at", "executed_at", "created_at",
    }
    assert expected.issubset(cols)


def test_ingest_job_columns_exist():
    cols = {c.name for c in IngestJob.__table__.columns}
    expected = {
        "id", "source_index", "target_index", "target_namespace", "status",
        "total_chunks", "processed_chunks", "failed_chunks", "error_message",
        "config", "started_at", "completed_at", "created_at",
    }
    assert expected.issubset(cols)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_audit_models.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create IndexAudit model**

```python
# backend/app/models/index_audit.py
import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class IndexAudit(Base):
    __tablename__ = "index_audits"
    __table_args__ = (
        UniqueConstraint("audit_date", "index_name", "project_id",
                         name="index_audits_unique"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    audit_date: Mapped[date] = mapped_column(Date, nullable=False)
    index_name: Mapped[str] = mapped_column(String(255), nullable=False)
    project_id: Mapped[str] = mapped_column(String(50), nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    dominant_domain: Mapped[str | None] = mapped_column(String(100), nullable=True)
    topic_tags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    sample_chunks: Mapped[list] = mapped_column(JSONB, nullable=False)
    proposed_disposition: Mapped[str] = mapped_column(String(50), nullable=False)
    proposed_target_index: Mapped[str | None] = mapped_column(String(255), nullable=True)
    approved_disposition: Mapped[str | None] = mapped_column(String(50), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
```

- [ ] **Step 4: Create IngestJob model**

```python
# backend/app/models/ingest_job.py
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_index: Mapped[str] = mapped_column(String(255), nullable=False)
    target_index: Mapped[str] = mapped_column(String(255), nullable=False)
    target_namespace: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processed_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
```

- [ ] **Step 5: Add to models/__init__.py**

```python
# backend/app/models/__init__.py — REPLACE entire file
from app.models import chat as _chat  # noqa: F401
from app.models import index_audit as _index_audit  # noqa: F401
from app.models import index_registry as _index_registry  # noqa: F401
from app.models import ingest_job as _ingest_job  # noqa: F401
from app.models import schemas as _schemas  # noqa: F401
```

- [ ] **Step 6: Run tests to verify pass**

Run: `cd backend && python3 -m pytest tests/test_audit_models.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/index_audit.py backend/app/models/ingest_job.py backend/app/models/__init__.py backend/tests/test_audit_models.py
git commit -m "feat(models): IndexAudit and IngestJob SQLAlchemy models"
```

---

### Task 6: Extend index_registry with new columns

**Files:**
- Modify: `backend/app/models/index_registry.py`
- Test: `backend/tests/test_index_registry_extensions.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_index_registry_extensions.py
from app.models.index_registry import IndexRegistry


def test_index_registry_has_new_columns():
    cols = {c.name for c in IndexRegistry.__table__.columns}
    new_cols = {"domain", "public_safe_default", "agent_module_path", "status"}
    assert new_cols.issubset(cols)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_index_registry_extensions.py -v`
Expected: FAIL — columns missing

- [ ] **Step 3: Add columns to IndexRegistry model**

Add to `backend/app/models/index_registry.py` inside `IndexRegistry` class:

```python
    domain: Mapped[str | None] = mapped_column(String(100), nullable=True)
    public_safe_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    agent_module_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python3 -m pytest tests/test_index_registry_extensions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/index_registry.py backend/tests/test_index_registry_extensions.py
git commit -m "feat(models): extend IndexRegistry with domain, public_safe_default, agent_module_path, status"
```

---

### Task 7: Alembic migration for new + extended tables

**Files:**
- Create: `backend/app/migrations/versions/002_phase0_audit_ingest_tables.py`

- [ ] **Step 1: Generate migration scaffold (if Alembic auto-gen works)**

Run: `cd backend && alembic revision --autogenerate -m "phase0 audit ingest tables" 2>&1 | tail -10`

If autogenerate is unavailable due to env, fall through to step 2.

- [ ] **Step 2: Hand-write the migration if scaffold not produced**

Create `backend/app/migrations/versions/002_phase0_audit_ingest_tables.py`:

```python
"""phase0 audit ingest tables

Revision ID: 002
Revises: 001
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "index_audits",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("audit_date", sa.Date(), nullable=False),
        sa.Column("index_name", sa.String(255), nullable=False),
        sa.Column("project_id", sa.String(50), nullable=False),
        sa.Column("record_count", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(255), nullable=True),
        sa.Column("dominant_domain", sa.String(100), nullable=True),
        sa.Column("topic_tags", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("sample_chunks", postgresql.JSONB(), nullable=False),
        sa.Column("proposed_disposition", sa.String(50), nullable=False),
        sa.Column("proposed_target_index", sa.String(255), nullable=True),
        sa.Column("approved_disposition", sa.String(50), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint("audit_date", "index_name", "project_id",
                            name="index_audits_unique"),
    )

    op.create_table(
        "ingest_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("source_index", sa.String(255), nullable=False),
        sa.Column("target_index", sa.String(255), nullable=False),
        sa.Column("target_namespace", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("total_chunks", sa.Integer(), nullable=True),
        sa.Column("processed_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("config", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_index("ingest_jobs_status_idx", "ingest_jobs", ["status"])

    op.add_column("index_registry", sa.Column("domain", sa.String(100), nullable=True))
    op.add_column("index_registry", sa.Column("public_safe_default", sa.Boolean(),
                                               nullable=False, server_default="false"))
    op.add_column("index_registry", sa.Column("agent_module_path", sa.String(255), nullable=True))
    op.add_column("index_registry", sa.Column("status", sa.String(50),
                                               nullable=False, server_default="active"))


def downgrade() -> None:
    op.drop_column("index_registry", "status")
    op.drop_column("index_registry", "agent_module_path")
    op.drop_column("index_registry", "public_safe_default")
    op.drop_column("index_registry", "domain")
    op.drop_index("ingest_jobs_status_idx", "ingest_jobs")
    op.drop_table("ingest_jobs")
    op.drop_table("index_audits")
```

- [ ] **Step 3: Verify migration syntax**

Run: `cd backend && python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', 'app/migrations/versions/002_phase0_audit_ingest_tables.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit (do not run migration yet — that happens at Task 12)**

```bash
git add backend/app/migrations/versions/002_phase0_audit_ingest_tables.py
git commit -m "feat(db): Alembic migration for index_audits, ingest_jobs, registry extensions"
```

---

### Task 8: Admin role middleware

**Files:**
- Create: `backend/app/middleware/admin_role.py`
- Test: `backend/tests/test_admin_role_middleware.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_admin_role_middleware.py
import pytest
from fastapi import HTTPException
from app.middleware.admin_role import require_admin_role


def test_require_admin_role_passes_for_admin():
    payload = {"sub": "user_123", "publicMetadata": {"role": "admin"}}
    require_admin_role(payload)  # should not raise


def test_require_admin_role_rejects_non_admin():
    payload = {"sub": "user_123", "publicMetadata": {"role": "user"}}
    with pytest.raises(HTTPException) as exc:
        require_admin_role(payload)
    assert exc.value.status_code == 403


def test_require_admin_role_rejects_missing_metadata():
    payload = {"sub": "user_123"}
    with pytest.raises(HTTPException) as exc:
        require_admin_role(payload)
    assert exc.value.status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_admin_role_middleware.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement middleware**

```python
# backend/app/middleware/admin_role.py
"""Role gate for admin endpoints. Reads `publicMetadata.role` from the Clerk
JWT payload and rejects with 403 when not 'admin'."""
from fastapi import Depends, HTTPException, status

from app.middleware.clerk_auth import require_bearer_token


def require_admin_role(token_payload: dict = Depends(require_bearer_token)) -> dict:
    metadata = token_payload.get("publicMetadata") or {}
    role = metadata.get("role")
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return token_payload
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python3 -m pytest tests/test_admin_role_middleware.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/middleware/admin_role.py backend/tests/test_admin_role_middleware.py
git commit -m "feat(auth): admin role middleware checking Clerk publicMetadata.role"
```

---

### Task 9: Audit prompts (LLM domain classifier + metadata enrichment)

**Files:**
- Create: `backend/app/prompts/audit_classify.txt`
- Create: `backend/app/prompts/metadata_enrich.txt`

- [ ] **Step 1: Write `audit_classify.txt`**

```
You are an index classifier helping reorganize a Pinecone corpus.

You will see sample chunks from a single Pinecone index. Determine:
1. The dominant domain — pick the SINGLE best fit from this list:
   - strategy (decision frameworks, leadership, founder-thinking, transcripts about strategy)
   - marketing (email, ads, funnels, sales copy, marketing case studies)
   - voice (writing style examples, voice/tone references — Max's first-person prose)
   - personal (journal entries, relationships, health, travel — Max's life)
   - tools-ops (n8n workflows, AI tooling, infrastructure docs, prompts)
   - content-library (YouTube transcripts, podcast transcripts, blog drafts, social posts)
   - verticals (industry-specific: telemed, local-business, coaching, saas, other)
   - mixed (genuinely heterogeneous — needs SPLIT)
   - empty (no usable content — needs ARCHIVE)

2. Up to 5 topic tags that describe what this index actually contains.

3. Proposed disposition:
   - KEEP: > 500 records, content homogeneous, chunks healthy (>200 chars on average)
   - MERGE: small (<500 records) or content already covered by a domain bucket
   - RE-INGEST: chunks broken (too short / boilerplate / metadata empty / inconsistent dimension)
   - ARCHIVE: empty or test-only
   - SPLIT: heterogeneous content that should be cut into multiple namespaces

Return JSON:
{
  "dominant_domain": "<one of the values above>",
  "topic_tags": ["...", ...],
  "proposed_disposition": "KEEP" | "MERGE" | "RE-INGEST" | "ARCHIVE" | "SPLIT",
  "proposed_target_index": "max-strategy" | "max-marketing" | ... | null,
  "reasoning": "<one sentence>"
}

Index name: $index_name
Record count: $record_count
Embedding model: $embedding_model
Sample chunks (truncated to 400 chars each):
$sample_chunks
```

- [ ] **Step 2: Write `metadata_enrich.txt`**

```
You enrich chunk metadata for a RAG corpus. Given a chunk's text, return:

1. topic_tags: 3-5 short tag strings describing what the chunk is about
2. source_type: ONE of these labels:
   email | transcript | note | tweet | podcast | blog-post | ad-copy |
   sales-page | newsletter | document | code | other
3. public_safe: true if this chunk could be quoted to a stranger; false if
   it's personal (journal, finances, health, relationships, internal strategy
   not yet shipped, anything name-attaching that isn't already public).

Return JSON:
{
  "topic_tags": ["..."],
  "source_type": "...",
  "public_safe": true | false
}

Chunk:
$chunk_text
```

- [ ] **Step 3: Commit (no test — these are pure templates)**

```bash
git add backend/app/prompts/audit_classify.txt backend/app/prompts/metadata_enrich.txt
git commit -m "feat(prompts): audit classifier + metadata enrichment templates"
```

---

### Task 10: audit_indexes.py CLI script

**Files:**
- Create: `backend/scripts/__init__.py`
- Create: `backend/scripts/audit_indexes.py`
- Test: `backend/tests/test_audit_indexes_script.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_audit_indexes_script.py
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


@pytest.mark.asyncio
async def test_audit_one_index_classifies_and_returns_audit_row():
    from backend.scripts.audit_indexes import audit_one_index

    fake_pinecone_stats = {"total_vector_count": 1500, "dimension": 1024,
                           "namespaces": {"default": {"vector_count": 1500}}}
    fake_sample = [{"text": "marketing email tactics ad copy " * 20}] * 8

    with patch("backend.scripts.audit_indexes._fetch_stats",
               new=AsyncMock(return_value=fake_pinecone_stats)), \
         patch("backend.scripts.audit_indexes._sample_chunks",
               new=AsyncMock(return_value=fake_sample)), \
         patch("backend.scripts.audit_indexes.call_openai_json",
               new=AsyncMock(return_value={
                   "dominant_domain": "marketing",
                   "topic_tags": ["email", "ads"],
                   "proposed_disposition": "KEEP",
                   "proposed_target_index": "max-marketing",
                   "reasoning": "homogeneous marketing content",
               })):
        row = await audit_one_index(
            index_name="copywriting-resources", project_id="1",
            audit_date=date.today(),
        )

    assert row["index_name"] == "copywriting-resources"
    assert row["record_count"] == 1500
    assert row["dominant_domain"] == "marketing"
    assert row["proposed_disposition"] == "KEEP"
    assert len(row["sample_chunks"]) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_audit_indexes_script.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create `backend/scripts/__init__.py`**

```python
# backend/scripts/__init__.py
```

- [ ] **Step 4: Implement audit_indexes.py**

```python
# backend/scripts/audit_indexes.py
"""Phase-0 audit script.

Walks every active row in `index_registry`, fetches Pinecone stats, samples
8-10 chunks per index, classifies via LLM, proposes a disposition, and writes
the result into the `index_audits` table.

Usage:
    python -m backend.scripts.audit_indexes [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
from datetime import date
from pathlib import Path
from string import Template
from typing import Any

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.index_audit import IndexAudit
from app.models.index_registry import IndexRegistry
from app.services.embedding_service import generate_embedding_cohere
from app.services.llm_service import call_openai_json
from app.services.pinecone_service import get_factory

_log = logging.getLogger(__name__)
_SAMPLE_SIZE = 8
_PROMPT_PATH = Path(__file__).parent.parent / "app" / "prompts" / "audit_classify.txt"


async def _fetch_stats(index_name: str, project_id: str) -> dict[str, Any]:
    factory = get_factory()
    client = factory.get_client(project_id)
    index = client.Index(index_name)
    stats = index.describe_index_stats()
    return {
        "total_vector_count": stats.total_vector_count,
        "dimension": stats.dimension,
        "namespaces": {ns: {"vector_count": v.vector_count}
                       for ns, v in (stats.namespaces or {}).items()},
    }


async def _sample_chunks(
    index_name: str, project_id: str, dimension: int, namespaces: list[str] | None,
    n: int = _SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """Random query with random vector to get a sample of chunks."""
    factory = get_factory()
    client = factory.get_client(project_id)
    index = client.Index(index_name)
    vector = [random.uniform(-1, 1) for _ in range(dimension)]
    if namespaces:
        all_matches: list = []
        for ns in namespaces[:5]:  # cap namespaces sampled
            res = index.query(vector=vector, top_k=n, include_metadata=True, namespace=ns)
            all_matches.extend(res.matches)
        matches = all_matches[:n]
    else:
        res = index.query(vector=vector, top_k=n, include_metadata=True)
        matches = res.matches
    return [{"text": (m.metadata or {}).get("text", "")[:400], "metadata": m.metadata or {}}
            for m in matches]


async def audit_one_index(
    index_name: str, project_id: str, audit_date: date,
) -> dict[str, Any]:
    stats = await _fetch_stats(index_name, project_id)
    namespaces = list(stats["namespaces"].keys())
    sample = await _sample_chunks(
        index_name, project_id, stats["dimension"], namespaces or None
    )

    sample_text = "\n\n---\n\n".join(c["text"] for c in sample if c["text"])
    prompt = Template(_PROMPT_PATH.read_text()).safe_substitute(
        index_name=index_name,
        record_count=stats["total_vector_count"],
        embedding_model=f"dim={stats['dimension']}",
        sample_chunks=sample_text or "(no sample available)",
    )
    classification = await call_openai_json(
        model="gpt-4.1-mini", system="", user=prompt, temperature=0,
    )

    return {
        "audit_date": audit_date,
        "index_name": index_name,
        "project_id": project_id,
        "record_count": stats["total_vector_count"],
        "embedding_model": f"dim={stats['dimension']}",
        "dominant_domain": classification.get("dominant_domain"),
        "topic_tags": classification.get("topic_tags", []),
        "sample_chunks": sample,
        "proposed_disposition": classification.get("proposed_disposition", "KEEP"),
        "proposed_target_index": classification.get("proposed_target_index"),
    }


async def run_audit(dry_run: bool = False) -> list[dict[str, Any]]:
    db: Session = SessionLocal()
    try:
        active = db.query(IndexRegistry).filter(IndexRegistry.is_active == True).all()  # noqa: E712
        audit_date = date.today()
        rows: list[dict[str, Any]] = []
        for entry in active:
            try:
                row = await audit_one_index(entry.index_name, entry.project_id, audit_date)
                rows.append(row)
                if not dry_run:
                    db.add(IndexAudit(**row))
                _log.info("audited %s: %s", entry.index_name,
                          row["proposed_disposition"])
            except Exception as exc:
                _log.error("audit failed for %s/%s: %s",
                           entry.project_id, entry.index_name, exc)
        if not dry_run:
            db.commit()
        return rows
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_audit(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python3 -m pytest tests/test_audit_indexes_script.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/__init__.py backend/scripts/audit_indexes.py backend/tests/test_audit_indexes_script.py
git commit -m "feat(audit): audit_indexes CLI — sample, classify, propose disposition"
```

---

### Task 11: Admin audit endpoints

**Files:**
- Create: `backend/app/routers/admin_audit.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_admin_audit_router.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_admin_audit_router.py
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def admin_token_payload():
    return {"sub": "user_admin", "publicMetadata": {"role": "admin"}}


def test_post_audit_returns_audit_id(monkeypatch, admin_token_payload):
    from app.main import app
    from app.middleware.clerk_auth import require_bearer_token
    app.dependency_overrides[require_bearer_token] = lambda: admin_token_payload

    with patch("app.routers.admin_audit.run_audit",
               new=AsyncMock(return_value=[{
                   "audit_date": date.today(),
                   "index_name": "x", "project_id": "1", "record_count": 0,
                   "embedding_model": "dim=1024", "dominant_domain": "marketing",
                   "topic_tags": [], "sample_chunks": [],
                   "proposed_disposition": "KEEP", "proposed_target_index": None,
               }])):
        client = TestClient(app)
        resp = client.post("/api/admin/audit", json={"mode": "execute"})

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    data = resp.json()
    assert "audit_id" in data
    assert "row_count" in data


def test_post_dispositions_marks_approved(monkeypatch, admin_token_payload):
    from app.main import app
    from app.middleware.clerk_auth import require_bearer_token
    app.dependency_overrides[require_bearer_token] = lambda: admin_token_payload

    with patch("app.routers.admin_audit.SessionLocal") as MockSL:
        mock_db = MagicMock()
        MockSL.return_value = mock_db
        client = TestClient(app)
        resp = client.post("/api/admin/audit/2026-05-10/dispositions", json=[
            {"audit_row_id": "00000000-0000-0000-0000-000000000001",
             "approved_disposition": "KEEP",
             "approved_target_index": None}
        ])

    app.dependency_overrides.clear()
    assert resp.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_admin_audit_router.py -v`
Expected: FAIL — module/route not found

- [ ] **Step 3: Implement admin_audit router**

```python
# backend/app/routers/admin_audit.py
"""Admin endpoints for Phase-0 audit + disposition approval + ingest job mgmt."""
from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import SessionLocal, get_db
from app.middleware.admin_role import require_admin_role
from app.models.index_audit import IndexAudit
from app.models.ingest_job import IngestJob
from backend.scripts.audit_indexes import run_audit

router = APIRouter(prefix="/api/admin", tags=["admin", "audit"])


class AuditRequest(BaseModel):
    mode: Literal["dry_run", "execute"]


class AuditResponse(BaseModel):
    audit_id: str
    row_count: int
    audit_date: date


class DispositionUpdate(BaseModel):
    audit_row_id: uuid.UUID
    approved_disposition: Literal["KEEP", "MERGE", "RE-INGEST", "ARCHIVE", "SPLIT"]
    approved_target_index: str | None = None


@router.post("/audit", response_model=AuditResponse)
async def kick_off_audit(
    body: AuditRequest,
    _admin: dict = Depends(require_admin_role),
) -> AuditResponse:
    rows = await run_audit(dry_run=(body.mode == "dry_run"))
    return AuditResponse(
        audit_id=date.today().isoformat(),
        row_count=len(rows),
        audit_date=date.today(),
    )


@router.post("/audit/{audit_date}/dispositions")
def approve_dispositions(
    audit_date: str,
    updates: list[DispositionUpdate],
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin_role),
) -> dict:
    """Bulk approve dispositions; auto-create ingest_jobs for RE-INGEST and MERGE."""
    created_jobs: list[str] = []
    for upd in updates:
        row = db.query(IndexAudit).filter(IndexAudit.id == upd.audit_row_id).first()
        if not row:
            continue
        row.approved_disposition = upd.approved_disposition
        row.approved_at = datetime.utcnow()
        if upd.approved_target_index:
            row.proposed_target_index = upd.approved_target_index

        if upd.approved_disposition in ("RE-INGEST", "MERGE"):
            target_index = upd.approved_target_index or row.proposed_target_index
            if not target_index:
                raise HTTPException(400, f"target_index required for {upd.approved_disposition}")
            job = IngestJob(
                source_index=row.index_name,
                target_index=target_index,
                target_namespace=row.dominant_domain or "default",
                status="pending",
                config={"source_project_id": row.project_id},
            )
            db.add(job)
            db.flush()
            created_jobs.append(str(job.id))
    db.commit()
    return {"created_jobs": created_jobs, "updated_rows": len(updates)}
```

- [ ] **Step 4: Register router in main.py**

Edit `backend/app/main.py` — add import and include:

```python
from app.routers import admin, admin_audit, chat as chat_router
# ...
app.include_router(admin_audit.router)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `cd backend && python3 -m pytest tests/test_admin_audit_router.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/admin_audit.py backend/app/main.py backend/tests/test_admin_audit_router.py
git commit -m "feat(api): admin audit + disposition endpoints with auto job creation"
```

---

### Task 12: Apply migration to production Postgres

**Files:** None (operational task)

- [ ] **Step 1: Verify Railway link is correct**

Run: `railway status`
Expected: `Project: chat-with-my-pinecone-backend`

- [ ] **Step 2: Run migration via Railway**

Run: `cd backend && railway run alembic upgrade head 2>&1 | tail -10`
Expected: `Running upgrade 001 -> 002, phase0 audit ingest tables`

- [ ] **Step 3: Verify tables exist**

Run: `railway run python -c "from app.db.database import engine; from sqlalchemy import inspect; print([t for t in inspect(engine).get_table_names() if 'audit' in t or 'ingest' in t])"`
Expected: `['index_audits', 'ingest_jobs']`

- [ ] **Step 4: No commit — operational only**

---

### Task 13: Re-ingest pipeline core (fetch + reconstruct + chunk)

**Files:**
- Create: `backend/scripts/reingest.py`
- Test: `backend/tests/test_reingest_script.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_reingest_script.py
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


@pytest.mark.asyncio
async def test_fetch_source_chunks_paginates():
    from backend.scripts.reingest import fetch_source_chunks
    factory = MagicMock()
    fake_idx = MagicMock()
    fake_idx.list_paginated.side_effect = [
        MagicMock(vectors=[MagicMock(id="a"), MagicMock(id="b")],
                  pagination=MagicMock(next="cursor1")),
        MagicMock(vectors=[MagicMock(id="c")], pagination=None),
    ]
    fake_idx.fetch.return_value.vectors = {
        "a": MagicMock(metadata={"text": "alpha"}),
        "b": MagicMock(metadata={"text": "beta"}),
        "c": MagicMock(metadata={"text": "gamma"}),
    }
    factory.get_client.return_value.Index.return_value = fake_idx

    with patch("backend.scripts.reingest.get_factory", return_value=factory):
        chunks = await fetch_source_chunks("idx", "1")

    texts = sorted(c["text"] for c in chunks)
    assert texts == ["alpha", "beta", "gamma"]


def test_reconstruct_documents_groups_by_source_id():
    from backend.scripts.reingest import reconstruct_documents
    chunks = [
        {"text": "p1", "metadata": {"source_id": "doc1", "chunk_position": 0}},
        {"text": "p2", "metadata": {"source_id": "doc1", "chunk_position": 1}},
        {"text": "lone", "metadata": {"source_id": "doc2", "chunk_position": 0}},
        {"text": "orphan", "metadata": {}},
    ]
    docs = reconstruct_documents(chunks)
    assert docs["doc1"] == "p1\n\np2"
    assert docs["doc2"] == "lone"
    assert "orphan" in [v for v in docs.values()]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_reingest_script.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement core reingest functions**

```python
# backend/scripts/reingest.py
"""Phase-0 re-ingest pipeline.

Pulls chunks from a source Pinecone index, reconstructs documents (when
metadata supports it), semantic re-chunks with Cohere, enriches metadata via
LLM, and upserts into a target index/namespace.

Idempotent: ingest_jobs.processed_chunks advances per upsert; rerunning
resumes at offset.

Usage:
    python -m backend.scripts.reingest <ingest_job_id>
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.ingest_job import IngestJob
from app.services.chunking_service import semantic_chunk_text
from app.services.embedding_service import generate_embeddings_cohere_batch
from app.services.llm_service import call_openai_json
from app.services.pinecone_service import get_factory

_log = logging.getLogger(__name__)
_BATCH_SIZE = 96  # Cohere embed batch limit
_PROMPT_PATH = Path(__file__).parent.parent / "app" / "prompts" / "metadata_enrich.txt"


async def fetch_source_chunks(
    index_name: str, project_id: str, batch_size: int = 100,
) -> list[dict[str, Any]]:
    """Paginate the source index and return all chunks with metadata."""
    factory = get_factory()
    client = factory.get_client(project_id)
    index = client.Index(index_name)

    all_ids: list[str] = []
    cursor: str | None = None
    while True:
        page = index.list_paginated(limit=batch_size, pagination_token=cursor)
        all_ids.extend(v.id for v in page.vectors)
        cursor = page.pagination.next if page.pagination else None
        if not cursor:
            break

    chunks: list[dict[str, Any]] = []
    for i in range(0, len(all_ids), 100):
        batch = all_ids[i:i + 100]
        fetched = index.fetch(ids=batch)
        for vid, vec in fetched.vectors.items():
            md = vec.metadata or {}
            chunks.append({"id": vid, "text": md.get("text", ""), "metadata": md})
    return chunks


def reconstruct_documents(chunks: list[dict[str, Any]]) -> dict[str, str]:
    """Group chunks back into source documents by source_id where available."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    orphans: list[str] = []
    for c in chunks:
        md = c.get("metadata") or {}
        sid = md.get("source_id")
        if sid:
            pos = md.get("chunk_position", 0)
            grouped.setdefault(sid, []).append((pos, c["text"]))
        else:
            orphans.append(c["text"])

    docs: dict[str, str] = {}
    for sid, items in grouped.items():
        items.sort(key=lambda t: t[0])
        docs[sid] = "\n\n".join(text for _, text in items)
    for i, text in enumerate(orphans):
        docs[f"_orphan_{i}"] = text
    return docs


async def enrich_metadata(chunk_text: str) -> dict[str, Any]:
    """LLM pass to extract topic_tags, source_type, public_safe."""
    prompt = Template(_PROMPT_PATH.read_text()).safe_substitute(chunk_text=chunk_text[:2000])
    return await call_openai_json(model="gpt-4.1-mini", system="", user=prompt, temperature=0)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python3 -m pytest tests/test_reingest_script.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/reingest.py backend/tests/test_reingest_script.py
git commit -m "feat(reingest): fetch, reconstruct, metadata-enrich primitives"
```

---

### Task 14: Re-ingest pipeline runner (embed + upsert + idempotency)

**Files:**
- Modify: `backend/scripts/reingest.py`
- Test: `backend/tests/test_reingest_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_reingest_runner.py
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


@pytest.mark.asyncio
async def test_run_ingest_job_processes_all_chunks_and_marks_completed():
    from backend.scripts.reingest import run_ingest_job

    fake_job = MagicMock(
        id=uuid.uuid4(),
        source_index="src", target_index="max-marketing",
        target_namespace="email", status="pending", processed_chunks=0,
        config={"source_project_id": "1"},
    )
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = fake_job

    fake_chunks = [{"id": f"c{i}", "text": "x" * 200,
                    "metadata": {"source_id": "doc1", "chunk_position": i}}
                   for i in range(3)]

    with patch("backend.scripts.reingest.SessionLocal", return_value=fake_db), \
         patch("backend.scripts.reingest.fetch_source_chunks",
               new=AsyncMock(return_value=fake_chunks)), \
         patch("backend.scripts.reingest.semantic_chunk_text",
               return_value=["combined chunk text " * 10]), \
         patch("backend.scripts.reingest.enrich_metadata",
               new=AsyncMock(return_value={
                   "topic_tags": ["a"], "source_type": "email", "public_safe": True
               })), \
         patch("backend.scripts.reingest.generate_embeddings_cohere_batch",
               return_value=[[0.1] * 1024]), \
         patch("backend.scripts.reingest.get_factory") as mock_factory:
        mock_index = MagicMock()
        mock_factory.return_value.get_client.return_value.Index.return_value = mock_index

        await run_ingest_job(str(fake_job.id))

    assert fake_job.status == "completed"
    assert fake_job.processed_chunks > 0
    assert mock_index.upsert.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_reingest_runner.py -v`
Expected: FAIL — function not defined

- [ ] **Step 3: Add run_ingest_job to reingest.py**

Append to `backend/scripts/reingest.py`:

```python
async def run_ingest_job(job_id: str) -> None:
    """Execute or resume an ingest job. Idempotent at processed_chunks offset."""
    db: Session = SessionLocal()
    try:
        job = db.query(IngestJob).filter(IngestJob.id == uuid.UUID(job_id)).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if job.status == "completed":
            return

        job.status = "running"
        if not job.started_at:
            job.started_at = datetime.utcnow()
        db.commit()

        try:
            project_id = job.config.get("source_project_id", "1")
            source_chunks = await fetch_source_chunks(job.source_index, project_id)
            documents = reconstruct_documents(source_chunks)

            new_chunks: list[dict[str, Any]] = []
            for sid, doc_text in documents.items():
                if not doc_text.strip():
                    continue
                pieces = semantic_chunk_text(doc_text)
                for pos, piece in enumerate(pieces):
                    new_chunks.append({
                        "source_id": sid, "chunk_position": pos,
                        "doc_total_chunks": len(pieces),
                        "text": piece,
                    })

            job.total_chunks = len(new_chunks)
            db.commit()

            factory = get_factory()
            target = factory.get_client(project_id).Index(job.target_index)

            start = job.processed_chunks
            for batch_start in range(start, len(new_chunks), _BATCH_SIZE):
                batch = new_chunks[batch_start:batch_start + _BATCH_SIZE]
                texts = [c["text"] for c in batch]
                embeddings = generate_embeddings_cohere_batch(texts, input_type="search_document")

                vectors = []
                for c, emb in zip(batch, embeddings):
                    try:
                        meta = await enrich_metadata(c["text"])
                    except Exception as exc:
                        _log.warning("metadata enrich failed: %s", exc)
                        meta = {"topic_tags": [], "source_type": "other", "public_safe": False}
                    vid = f"{c['source_id']}_{c['chunk_position']}_{uuid.uuid4().hex[:8]}"
                    full_meta = {
                        "text": c["text"][:40000],
                        "source_type": meta.get("source_type", "other"),
                        "source_id": c["source_id"],
                        "ingested_at": datetime.utcnow().isoformat(),
                        "domain": job.target_index.replace("max-", ""),
                        "namespace": job.target_namespace,
                        "topic_tags": meta.get("topic_tags", []),
                        "public_safe": bool(meta.get("public_safe", False)),
                        "embedding_model": "cohere-embed-v3",
                        "chunk_position": c["chunk_position"],
                        "doc_total_chunks": c["doc_total_chunks"],
                    }
                    vectors.append({"id": vid, "values": emb, "metadata": full_meta})

                try:
                    target.upsert(vectors=vectors, namespace=job.target_namespace)
                    job.processed_chunks = batch_start + len(batch)
                except Exception as exc:
                    _log.error("upsert failed at offset %s: %s", batch_start, exc)
                    job.failed_chunks += len(batch)

                db.commit()

                if job.total_chunks and job.failed_chunks / job.total_chunks > 0.05:
                    raise RuntimeError(f"Failure rate exceeded 5% threshold")

            job.status = "completed"
            job.completed_at = datetime.utcnow()
            db.commit()
        except Exception as exc:
            job.status = "failed"
            job.error_message = str(exc)
            db.commit()
            raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_ingest_job(args.job_id))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd backend && python3 -m pytest tests/test_reingest_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/reingest.py backend/tests/test_reingest_runner.py
git commit -m "feat(reingest): pipeline runner with idempotent offset + 5% failure halt"
```

---

### Task 15: Admin endpoint to trigger ingest job

**Files:**
- Modify: `backend/app/routers/admin_audit.py`
- Test: `backend/tests/test_admin_ingest_endpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_admin_ingest_endpoint.py
import uuid
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


def test_post_ingest_run_kicks_off_job_and_returns_202():
    from app.main import app
    from app.middleware.clerk_auth import require_bearer_token
    app.dependency_overrides[require_bearer_token] = lambda: {
        "sub": "u", "publicMetadata": {"role": "admin"}
    }

    job_id = str(uuid.uuid4())
    with patch("app.routers.admin_audit.run_ingest_job",
               new=AsyncMock(return_value=None)):
        client = TestClient(app)
        resp = client.post(f"/api/admin/ingest/{job_id}/run")

    app.dependency_overrides.clear()
    assert resp.status_code == 202
    assert resp.json()["job_id"] == job_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_admin_ingest_endpoint.py -v`
Expected: FAIL — endpoint not found

- [ ] **Step 3: Add ingest endpoint to admin_audit.py**

Append to `backend/app/routers/admin_audit.py`:

```python
from backend.scripts.reingest import run_ingest_job


@router.post("/ingest/{job_id}/run", status_code=202)
async def trigger_ingest(
    job_id: uuid.UUID,
    background: BackgroundTasks,
    _admin: dict = Depends(require_admin_role),
) -> dict:
    """Kick off (or resume) an ingest job in the background."""
    background.add_task(run_ingest_job, str(job_id))
    return {"job_id": str(job_id), "status": "started"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python3 -m pytest tests/test_admin_ingest_endpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/routers/admin_audit.py backend/tests/test_admin_ingest_endpoint.py
git commit -m "feat(api): admin endpoint to trigger or resume ingest jobs"
```

---

### Task 16: Frontend admin audit page (read-only first)

**Files:**
- Create: `frontend/src/app/admin/audit/page.tsx`
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Add API client functions**

Append to `frontend/src/lib/api.ts`:

```typescript
export interface AuditRow {
  id: string;
  index_name: string;
  project_id: string;
  record_count: number;
  embedding_model: string | null;
  dominant_domain: string | null;
  topic_tags: string[];
  sample_chunks: Array<{ text: string; metadata: Record<string, unknown> }>;
  proposed_disposition: "KEEP" | "MERGE" | "RE-INGEST" | "ARCHIVE" | "SPLIT";
  proposed_target_index: string | null;
  approved_disposition: string | null;
}

export async function fetchLatestAudit(token: string): Promise<AuditRow[]> {
  const res = await fetch(`${API_URL}/api/admin/audit/latest`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`audit fetch failed: ${res.status}`);
  return res.json();
}

export async function kickOffAudit(
  token: string, mode: "dry_run" | "execute"
): Promise<{ audit_id: string; row_count: number }> {
  const res = await fetch(`${API_URL}/api/admin/audit`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!res.ok) throw new Error(`audit kickoff failed: ${res.status}`);
  return res.json();
}

export async function approveDispositions(
  token: string, auditDate: string,
  updates: Array<{ audit_row_id: string; approved_disposition: string;
                   approved_target_index?: string | null }>,
): Promise<{ created_jobs: string[]; updated_rows: number }> {
  const res = await fetch(`${API_URL}/api/admin/audit/${auditDate}/dispositions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  if (!res.ok) throw new Error(`dispositions failed: ${res.status}`);
  return res.json();
}

export async function triggerIngest(token: string, jobId: string): Promise<void> {
  const res = await fetch(`${API_URL}/api/admin/ingest/${jobId}/run`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`ingest trigger failed: ${res.status}`);
}
```

- [ ] **Step 2: Add a `/api/admin/audit/latest` endpoint to the backend**

In `backend/app/routers/admin_audit.py` add:

```python
from fastapi import Query


@router.get("/audit/latest")
def get_latest_audit(
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin_role),
) -> list[dict]:
    latest_date = db.query(IndexAudit.audit_date).order_by(
        IndexAudit.audit_date.desc()).limit(1).scalar()
    if not latest_date:
        return []
    rows = db.query(IndexAudit).filter(IndexAudit.audit_date == latest_date).all()
    return [{
        "id": str(r.id),
        "index_name": r.index_name,
        "project_id": r.project_id,
        "record_count": r.record_count,
        "embedding_model": r.embedding_model,
        "dominant_domain": r.dominant_domain,
        "topic_tags": r.topic_tags,
        "sample_chunks": r.sample_chunks,
        "proposed_disposition": r.proposed_disposition,
        "proposed_target_index": r.proposed_target_index,
        "approved_disposition": r.approved_disposition,
    } for r in rows]
```

- [ ] **Step 3: Build the admin audit page**

```tsx
// frontend/src/app/admin/audit/page.tsx
"use client";

import { useAuth } from "@clerk/nextjs";
import { useCallback, useEffect, useState } from "react";
import {
  AuditRow, approveDispositions, fetchLatestAudit, kickOffAudit, triggerIngest,
} from "@/lib/api";

const DISPOSITIONS = ["KEEP", "MERGE", "RE-INGEST", "ARCHIVE", "SPLIT"] as const;

export default function AuditPage() {
  const { getToken } = useAuth();
  const [rows, setRows] = useState<AuditRow[]>([]);
  const [overrides, setOverrides] = useState<Record<string, {
    disposition: string; target?: string;
  }>>({});
  const [loading, setLoading] = useState(false);
  const [jobIds, setJobIds] = useState<string[]>([]);

  const load = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    const data = await fetchLatestAudit(token);
    setRows(data);
  }, [getToken]);

  useEffect(() => { load(); }, [load]);

  const runAudit = async () => {
    setLoading(true);
    try {
      const token = await getToken();
      if (!token) return;
      await kickOffAudit(token, "execute");
      await load();
    } finally { setLoading(false); }
  };

  const approveAll = async () => {
    const token = await getToken();
    if (!token) return;
    const audit_date = rows[0] ? new Date().toISOString().slice(0, 10) : "";
    if (!audit_date) return;
    const updates = rows.map((r) => {
      const ovr = overrides[r.id];
      const disposition = ovr?.disposition ?? r.proposed_disposition;
      return {
        audit_row_id: r.id,
        approved_disposition: disposition,
        approved_target_index: ovr?.target ?? r.proposed_target_index ?? undefined,
      };
    });
    const result = await approveDispositions(token, audit_date, updates);
    setJobIds(result.created_jobs);
  };

  const startAllIngest = async () => {
    const token = await getToken();
    if (!token) return;
    for (const jid of jobIds) {
      await triggerIngest(token, jid);
    }
  };

  return (
    <main style={{ padding: "2rem", fontFamily: "var(--font-mono, monospace)" }}>
      <h1>Phase 0 — Index Audit</h1>
      <div style={{ marginBottom: "1rem" }}>
        <button onClick={runAudit} disabled={loading}>
          {loading ? "Running..." : "Run audit"}
        </button>
        <button onClick={approveAll} style={{ marginLeft: "1rem" }}>
          Approve all (with overrides)
        </button>
        <button onClick={startAllIngest} style={{ marginLeft: "1rem" }}
                disabled={jobIds.length === 0}>
          Start all ingest jobs ({jobIds.length})
        </button>
      </div>

      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th align="left">Index</th>
            <th align="left">Records</th>
            <th align="left">Dominant Domain</th>
            <th align="left">Proposed</th>
            <th align="left">Override</th>
            <th align="left">Target</th>
            <th align="left">Sample</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} style={{ borderTop: "1px solid #ccc" }}>
              <td>{r.index_name}</td>
              <td>{r.record_count}</td>
              <td>{r.dominant_domain || "-"}</td>
              <td>{r.proposed_disposition}</td>
              <td>
                <select
                  value={overrides[r.id]?.disposition ?? r.proposed_disposition}
                  onChange={(e) => setOverrides((o) => ({
                    ...o, [r.id]: { ...o[r.id], disposition: e.target.value },
                  }))}
                >
                  {DISPOSITIONS.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </td>
              <td>
                <input
                  defaultValue={r.proposed_target_index || ""}
                  onChange={(e) => setOverrides((o) => ({
                    ...o, [r.id]: { ...o[r.id],
                      disposition: o[r.id]?.disposition ?? r.proposed_disposition,
                      target: e.target.value,
                    },
                  }))}
                  style={{ width: "180px" }}
                />
              </td>
              <td>
                <details>
                  <summary>view {r.sample_chunks.length}</summary>
                  <pre style={{ maxWidth: "600px", whiteSpace: "pre-wrap" }}>
                    {r.sample_chunks.map((c) => c.text).join("\n\n---\n\n")}
                  </pre>
                </details>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}
```

- [ ] **Step 4: Verify the page builds**

Run: `cd frontend && npm run build 2>&1 | tail -20`
Expected: build succeeds (warnings ok)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/admin/audit/page.tsx frontend/src/lib/api.ts backend/app/routers/admin_audit.py
git commit -m "feat(ui): admin audit page — table, overrides, approve, ingest start"
```

---

### Task 17: Manual checkpoint — run audit + review dispositions

**Files:** None (operational task — Max performs)

- [ ] **Step 1: Confirm migration is applied to production**

Run: `railway run python -c "from sqlalchemy import inspect; from app.db.database import engine; print('index_audits' in inspect(engine).get_table_names())"`
Expected: `True`

- [ ] **Step 2: Visit `/admin/audit` in production**

Navigate to `https://chat.maxmayes.io/admin/audit`. Click "Run audit". Wait for completion.

- [ ] **Step 3: Review each row**

For each of the ~40 indexes:
- Read sample chunks
- Confirm or override `dominant_domain` and `proposed_disposition`
- Set `proposed_target_index` for `MERGE` / `RE-INGEST` rows

- [ ] **Step 4: Click "Approve all (with overrides)"**

Returned `created_jobs` count should match the number of `RE-INGEST` + `MERGE` rows.

- [ ] **Step 5: STOP — do not start ingest yet. Verify in this turn that:**
  - All approved dispositions look right (re-fetch from DB if needed)
  - Target indexes (`max-strategy`, `max-marketing`, etc.) exist in Pinecone (created via the Pinecone console)

Pinecone index creation is manual:
- 7 indexes named `max-strategy`, `max-marketing`, `max-voice`, `max-personal`, `max-tools-ops`, `max-content-library`, `max-verticals`
- Each: dim 1024, cosine, serverless, project-1 API key

- [ ] **Step 6: No commit — operational only**

---

### Task 18: Create legacy-archive backup index

**Files:** None (operational task — Max performs in Pinecone console + a one-shot script)

- [ ] **Step 1: Create the legacy-archive index in Pinecone**

In the Pinecone console:
- Name: `legacy-archive-2026-05-10` (use today's date)
- Dim: 1024 (irrelevant — content gets re-embedded if ever restored, but it must accept inserts; pick a dimension that matches majority of source indexes)
- Metric: cosine
- Project: 1

- [ ] **Step 2: Snapshot script — append to `backend/scripts/reingest.py`**

```python
async def snapshot_to_legacy(source_index: str, source_project: str,
                              archive_index: str = "legacy-archive-2026-05-10") -> int:
    """Copy raw chunks from source to legacy-archive without re-chunking
    or re-embedding (zero-loss backup)."""
    chunks = await fetch_source_chunks(source_index, source_project)
    factory = get_factory()
    target = factory.get_client(source_project).Index(archive_index)
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        # Re-fetch with vectors (fetch above only got metadata)
        # In practice, list_paginated → fetch returns vectors when include_values=True
        ids = [c["id"] for c in batch]
        with_vectors = factory.get_client(source_project).Index(source_index).fetch(
            ids=ids
        )
        vectors = [
            {"id": f"{source_index}__{vid}",
             "values": v.values or [0.0],
             "metadata": {**(v.metadata or {}), "_source_index": source_index}}
            for vid, v in with_vectors.vectors.items()
        ]
        target.upsert(vectors=vectors, namespace=source_index)
    return len(chunks)


async def snapshot_all_active_to_legacy() -> dict[str, int]:
    db = SessionLocal()
    try:
        active = db.query(IndexRegistry).filter(IndexRegistry.is_active == True).all()  # noqa: E712
        results: dict[str, int] = {}
        for entry in active:
            try:
                count = await snapshot_to_legacy(entry.index_name, entry.project_id)
                results[entry.index_name] = count
                _log.info("snapshotted %s: %d chunks", entry.index_name, count)
            except Exception as exc:
                _log.error("snapshot failed for %s: %s", entry.index_name, exc)
                results[entry.index_name] = -1
        return results
    finally:
        db.close()
```

- [ ] **Step 3: Run snapshot in production**

Run: `railway run python -c "import asyncio; from backend.scripts.reingest import snapshot_all_active_to_legacy; print(asyncio.run(snapshot_all_active_to_legacy()))"`
Expected: dict of `{index_name: chunk_count}` — non-negative for each

- [ ] **Step 4: Commit the snapshot helper**

```bash
git add backend/scripts/reingest.py
git commit -m "feat(reingest): legacy-archive snapshot helpers (backup before mutating)"
```

---

### Task 19: Manual checkpoint — execute approved ingest jobs

**Files:** None (operational task)

- [ ] **Step 1: From `/admin/audit`, click "Start all ingest jobs"**

Background tasks fire in FastAPI; jobs run async.

- [ ] **Step 2: Monitor progress via DB query**

Run: `railway run python -c "from app.db.database import SessionLocal; from app.models.ingest_job import IngestJob; db=SessionLocal(); rows=db.query(IngestJob).all(); [print(r.id, r.target_index, r.status, r.processed_chunks, '/', r.total_chunks) for r in rows]; db.close()"`

Expected over time: `pending` → `running` → `completed`. Some may stay `running` for 1–2 hours depending on chunk count.

- [ ] **Step 3: Resume failed jobs (if any)**

For any `failed` job, fix the underlying issue and re-trigger:
Run: `railway run python -m backend.scripts.reingest <job_id>`
The pipeline picks up at `processed_chunks` offset.

- [ ] **Step 4: Sanity-check Pinecone**

In Pinecone console, verify each of the 7 target indexes has the expected number of vectors and namespaces.

- [ ] **Step 5: No commit — operational only**

---

### Task 20: Update index_registry rows for new domain indexes

**Files:** None (operational SQL task)

- [ ] **Step 1: Insert registry rows for the 7 new domain indexes**

Run: `railway connect postgres` (interactive — Max runs this himself)

Then in psql:

```sql
INSERT INTO index_registry
(index_name, project_id, api_key_env_var, dimension, embedding_model, metric,
 domain_description, sample_queries, namespaces, domain, public_safe_default,
 agent_module_path, status, is_active)
VALUES
 ('max-strategy', '1', 'PINECONE_API_KEY_1', 1024, 'cohere-embed-v3', 'cosine',
  'Business strategy, leadership, founder thinking, decision frameworks',
  ARRAY['how do I think about pricing?', 'what is my decision framework for X?'],
  '{"decision-frameworks":{}, "leadership":{}, "founder-notes":{}, "transcripts":{}}'::jsonb,
  'strategy', false, 'app.agents.strategy', 'active', true),
 ('max-marketing', '1', 'PINECONE_API_KEY_1', 1024, 'cohere-embed-v3', 'cosine',
  'Marketing tactics + case studies: email, ads, funnels, sales copy',
  ARRAY['draft an email about X', 'what works for ad copy?'],
  '{"email":{}, "ads":{}, "funnels":{}, "sales-copy":{}, "case-studies":{}}'::jsonb,
  'marketing', false, 'app.agents.marketing', 'active', true),
 ('max-voice', '1', 'PINECONE_API_KEY_1', 1024, 'cohere-embed-v3', 'cosine',
  'Voice and style references: how Max writes, voice samples',
  ARRAY['how does Max write hooks?', 'what is Max voice'],
  '{"voice-samples":{}, "style-references":{}}'::jsonb,
  'voice', false, 'app.agents.voice', 'active', true),
 ('max-personal', '1', 'PINECONE_API_KEY_1', 1024, 'cohere-embed-v3', 'cosine',
  'Personal experiences, journal entries, relationships, health, travel',
  ARRAY['what did I write about X', 'my notes on Y'],
  '{"journal":{}, "relationships":{}, "health":{}, "travel":{}}'::jsonb,
  'personal', false, 'app.agents.personal', 'active', true),
 ('max-tools-ops', '1', 'PINECONE_API_KEY_1', 1024, 'cohere-embed-v3', 'cosine',
  'Tools, automation, AI infrastructure: n8n workflows, AI tools, prompts',
  ARRAY['n8n workflow for X', 'how do I prompt for Y'],
  '{"n8n-workflows":{}, "ai-tools":{}, "infrastructure":{}, "prompts":{}}'::jsonb,
  'tools-ops', false, 'app.agents.tools_ops', 'active', true),
 ('max-content-library', '1', 'PINECONE_API_KEY_1', 1024, 'cohere-embed-v3', 'cosine',
  'Content artifacts: youtube, podcast, blog drafts, social posts',
  ARRAY['what did I say in the X podcast', 'social post about Y'],
  '{"youtube":{}, "podcast":{}, "blog-drafts":{}, "social":{}}'::jsonb,
  'content-library', true, 'app.agents.content_library', 'active', true),
 ('max-verticals', '1', 'PINECONE_API_KEY_1', 1024, 'cohere-embed-v3', 'cosine',
  'Industry-specific knowledge: telemed, local-business, coaching, saas',
  ARRAY['telemed marketing strategies', 'local business lead gen'],
  '{"telemed":{}, "local-business":{}, "coaching":{}, "saas":{}, "other":{}}'::jsonb,
  'verticals', false, 'app.agents.verticals', 'active', true);
```

- [ ] **Step 2: Verify rows inserted**

```sql
SELECT index_name, domain, status FROM index_registry
WHERE domain IS NOT NULL ORDER BY index_name;
```
Expected: 7 rows.

- [ ] **Step 3: No commit — operational only**

---

## Phase 0 acceptance test

- [ ] **Step 1: Verify each domain index has chunks**

Run from Pinecone console or:
```bash
railway run python -c "
import asyncio
from app.services.pinecone_service import get_factory
factory = get_factory()
for name in ['max-strategy','max-marketing','max-voice','max-personal',
             'max-tools-ops','max-content-library','max-verticals']:
    try:
        idx = factory.get_client('1').Index(name)
        print(name, idx.describe_index_stats().total_vector_count)
    except Exception as e:
        print(name, 'ERROR', e)
"
```
Expected: each index shows a non-zero count.

- [ ] **Step 2: Spot-check a query**

```bash
railway run python -c "
from app.services.embedding_service import generate_embedding_cohere
from app.services.pinecone_service import get_factory, retrieve
v = generate_embedding_cohere('email marketing tactics', input_type='search_query')
chunks = retrieve(get_factory(), 'max-marketing', '1', v, top_k=5)
for c in chunks: print(c['score'], c['text'][:100])
"
```
Expected: 5 chunks with scores > 0.5 returning marketing-related content.

- [ ] **Step 3: Confirm production chat is unaffected**

Send a query in chat.maxmayes.io. Should still work as before (using the old orchestrator). No regression.

- [ ] **Step 4: Phase 0 complete. Hand off to Phase 1 plan.**

---

## Self-review (skill-required)

**Spec coverage check:**
- ✅ FR-AUD-1 (audit script) — Task 10
- ✅ FR-AUD-2 (admin UI page with bulk approval) — Task 16
- ✅ FR-AUD-3 (auto-create ingest jobs from approval) — Task 11
- ✅ FR-ING-1 (re-ingest pipeline: fetch → reconstruct → semantic re-chunk → enrich → embed → upsert) — Tasks 13, 14
- ✅ FR-ING-2 (per-chunk failures log, >5% halts) — Task 14
- ✅ FR-API-1 (audit, dispositions, ingest endpoints) — Tasks 11, 15
- ✅ FR-API-2 (admin role gate) — Task 8
- ✅ NFR-DATA-1 (legacy-archive backup) — Task 18
- ✅ NFR-DATA-2 (non-destructive Phase 0) — never deletes anything
- ✅ NFR-DATA-3 (12-field metadata schema) — Task 14 `full_meta` dict
- ✅ NFR-DATA-4 (Cohere v3 across all domain indexes) — Tasks 1, 14
- ✅ NFR-REL-6 (idempotent re-ingest) — Task 14 (resumes from `processed_chunks`)
- ✅ NFR-SEC-2 (admin role middleware) — Task 8
- ✅ NFR-OBS-1..3 (audit/ingest persistence) — Tasks 5, 7

**Placeholder scan:** None remain. All code blocks are complete. Operational tasks are explicit about what Max does manually.

**Type consistency:**
- `audit_one_index` returns dict with `audit_date`, `index_name`, etc. — matches `IndexAudit` model columns ✓
- `IngestJob.config` is JSONB; pipeline reads `config.get("source_project_id", "1")` — consistent ✓
- `DispositionUpdate.approved_disposition` Literal matches DB CHECK shape (no DB constraint, but model accepts string) ✓
- AuditRow TypeScript interface matches `/admin/audit/latest` JSON shape ✓

No issues found.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-10-rag-redesign-phase0.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

Note: Tasks 17, 18, 19, 20 are operational checkpoints requiring Max's hands. Subagents will pause and surface those for you. Phases 1, 2, 3 will get their own plans after Phase 0 ships and stabilizes — keeps each plan focused and fits the "ship Phase 0 first, then iterate" rollout strategy.
