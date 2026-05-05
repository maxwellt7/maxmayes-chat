# Phase 2+3 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working end-to-end chat that accepts a user query, routes it to the correct Pinecone index, retrieves grounded context, verifies faithfulness, and streams a response in Max's voice — plus an admin dashboard to manage the Index Registry.

**Architecture:** Single `orchestrator.py` service runs all 5 pipeline steps (optimize → route → retrieve → verify → synthesize). A `PineconeClientFactory` lazily initializes one client per project. An `EmbeddingService` selects the correct model by dimension. The chat endpoint streams via SSE; the admin layer exposes CRUD + Pinecone auto-discovery.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Pinecone SDK v5, OpenAI SDK v1, Cohere SDK v5, Anthropic SDK, Next.js 14, TypeScript

---

## File Map

**Backend — new:**
- `backend/alembic.ini` — Alembic config pointing to `app/migrations`
- `backend/app/migrations/env.py` — migration env wired to app DB + models
- `backend/app/migrations/versions/001_initial_schema.py` — creates `index_registry` + `chat_messages`
- `backend/app/models/chat.py` — `ChatMessage` SQLAlchemy model
- `backend/app/services/pinecone_service.py` — `PineconeClientFactory` + `retrieve()`
- `backend/app/services/embedding_service.py` — `generate_embedding(query, dimension)`
- `backend/app/services/orchestrator.py` — 5-step pipeline, `run_pipeline()` async generator
- `backend/app/prompts/optimizer.txt` — Step A system prompt
- `backend/app/prompts/router.txt` — Step B system prompt
- `backend/app/prompts/verifier.txt` — Step D system prompt
- `backend/app/prompts/voice_synthesizer.txt` — Step E system prompt
- `backend/tests/__init__.py`
- `backend/tests/test_embedding_service.py`
- `backend/tests/test_pinecone_service.py`
- `backend/tests/test_orchestrator.py`

**Backend — modified:**
- `backend/requirements.txt` — add pinecone, openai, cohere, anthropic, pytest, pytest-asyncio
- `backend/Dockerfile` — run `alembic upgrade head` before uvicorn
- `backend/app/models/index_registry.py` — add unique constraint
- `backend/app/models/schemas.py` — add admin + chat schemas
- `backend/app/routers/admin.py` — replace placeholder with CRUD + discover
- `backend/app/routers/chat.py` — replace placeholder with SSE streaming endpoint
- `backend/app/main.py` — import new models so Base.metadata is complete

**Frontend — new:**
- `frontend/src/lib/sse.ts` — SSE stream reader for POST requests
- `frontend/src/app/admin/indexes/[id]/page.tsx` — index edit page
- `frontend/src/components/ChatWindow.tsx` — message list + scroll
- `frontend/src/components/MessageBubble.tsx` — user/assistant bubble
- `frontend/src/components/StreamingResponse.tsx` — SSE token consumer

**Frontend — modified:**
- `frontend/src/lib/api.ts` — add admin + streaming API methods
- `frontend/src/app/admin/page.tsx` — full rebuild with registry table + discover
- `frontend/src/app/chat/page.tsx` — full rebuild (convert to client component)
- `frontend/src/components/ChatInput.tsx` — add Cmd+Enter submit, Enter=newline
- `frontend/src/app/styles.css` — add chat + admin component styles

---

## Task 1: Backend Dependencies + Alembic Setup

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/Dockerfile`
- Create: `backend/alembic.ini`
- Create: `backend/app/migrations/env.py`
- Create: `backend/app/migrations/script.py.mako`

- [ ] **Step 1: Update requirements.txt**

Replace contents of `backend/requirements.txt`:

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic-settings>=2.5.2
sqlalchemy>=2.0.35
psycopg[binary]>=3.2.1
httpx>=0.27.2
python-jose[cryptography]>=3.3.0
alembic>=1.13.2
pinecone>=5.0.0
openai>=1.0.0
cohere>=5.0.0
anthropic>=0.25.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 2: Initialize Alembic**

Run from `backend/`:
```bash
cd backend && alembic init app/migrations
```

Expected: creates `alembic.ini` and `app/migrations/` directory with `env.py`, `script.py.mako`, `versions/`.

- [ ] **Step 3: Configure alembic.ini**

Replace the `sqlalchemy.url` line in `backend/alembic.ini` — set it to a placeholder since env.py will override it:

```ini
[alembic]
script_location = app/migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql://placeholder
```

- [ ] **Step 4: Configure app/migrations/env.py**

Replace the contents of `backend/app/migrations/env.py`:

```python
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.config import normalized_database_url
from app.db.database import Base
from app.models import index_registry, chat  # noqa: F401 — registers models with Base

config = context.config
config.set_main_option("sqlalchemy.url", normalized_database_url())

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 5: Update Dockerfile to run migrations on startup**

Replace the CMD line in `backend/Dockerfile`:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini .
COPY app ./app

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/Dockerfile backend/alembic.ini backend/app/migrations/
git commit -m "chore: add AI dependencies and configure Alembic migrations"
```

---

## Task 2: ChatMessage Model + DB Migration

**Files:**
- Create: `backend/app/models/chat.py`
- Create: `backend/app/migrations/versions/001_initial_schema.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Create ChatMessage model**

Create `backend/app/models/chat.py`:

```python
import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
```

- [ ] **Step 2: Update main.py to import all models**

Add model imports to `backend/app/main.py` so Base.metadata includes all tables:

```python
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.database import Base, engine
from app.models import chat, index_registry  # noqa: F401 — registers with Base.metadata
from app.models.schemas import HealthResponse
from app.routers import admin, chat as chat_router

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_allow_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router)
app.include_router(admin.router)


@app.get("/healthz", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")
```

- [ ] **Step 3: Generate Alembic migration**

Run from `backend/`:
```bash
cd backend && alembic revision --autogenerate -m "initial_schema"
```

Expected: creates `backend/app/migrations/versions/<hash>_initial_schema.py` with `create_table` operations for both `index_registry` and `chat_messages`.

- [ ] **Step 4: Apply migration locally (optional — runs automatically on deploy)**

```bash
cd backend && alembic upgrade head
```

Expected: `Running upgrade  -> <hash>, initial_schema`

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/chat.py backend/app/main.py backend/app/migrations/versions/
git commit -m "feat: add ChatMessage model and initial schema migration"
```

---

## Task 3: Pinecone Client Factory

**Files:**
- Create: `backend/app/services/pinecone_service.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_pinecone_service.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/__init__.py` (empty).

Create `backend/tests/test_pinecone_service.py`:

```python
from unittest.mock import MagicMock, patch

import pytest

from app.services.pinecone_service import PineconeClientFactory, retrieve


def test_factory_returns_client_for_project_1():
    with patch("app.services.pinecone_service.Pinecone") as MockPinecone:
        mock_client = MagicMock()
        MockPinecone.return_value = mock_client

        factory = PineconeClientFactory()
        client = factory.get_client("1")

        MockPinecone.assert_called_once_with(api_key=factory._key_for_project("1"))
        assert client is mock_client


def test_factory_caches_client():
    with patch("app.services.pinecone_service.Pinecone") as MockPinecone:
        factory = PineconeClientFactory()
        c1 = factory.get_client("1")
        c2 = factory.get_client("1")
        assert c1 is c2
        assert MockPinecone.call_count == 1


def test_factory_raises_for_unknown_project():
    factory = PineconeClientFactory()
    with pytest.raises(ValueError, match="Unknown project_id"):
        factory.get_client("99")


def test_retrieve_calls_correct_index():
    with patch("app.services.pinecone_service.Pinecone") as MockPinecone:
        mock_index = MagicMock()
        mock_index.query.return_value = MagicMock(
            matches=[
                MagicMock(metadata={"text": "chunk1"}, score=0.9),
                MagicMock(metadata={"text": "chunk2"}, score=0.8),
            ]
        )
        MockPinecone.return_value.Index.return_value = mock_index

        factory = PineconeClientFactory()
        results = retrieve(factory, "my-index", "1", [0.1, 0.2, 0.3], top_k=2)

        mock_index.query.assert_called_once_with(vector=[0.1, 0.2, 0.3], top_k=2, include_metadata=True)
        assert results == [{"text": "chunk1", "score": 0.9}, {"text": "chunk2", "score": 0.8}]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_pinecone_service.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `pinecone_service` doesn't exist yet.

- [ ] **Step 3: Create pinecone_service.py**

Create `backend/app/services/pinecone_service.py`:

```python
import os
from typing import Any

from pinecone import Pinecone


class PineconeClientFactory:
    _PROJECT_KEY_MAP = {
        "1": "PINECONE_API_KEY_1",
        "2": "PINECONE_API_KEY_2",
        "3": "PINECONE_API_KEY_3",
    }

    def __init__(self) -> None:
        self._cache: dict[str, Pinecone] = {}

    def _key_for_project(self, project_id: str) -> str:
        env_var = self._PROJECT_KEY_MAP.get(project_id)
        if not env_var:
            raise ValueError(f"Unknown project_id: {project_id}")
        return os.environ[env_var]

    def get_client(self, project_id: str) -> Pinecone:
        if project_id not in self._cache:
            self._cache[project_id] = Pinecone(api_key=self._key_for_project(project_id))
        return self._cache[project_id]


def retrieve(
    factory: PineconeClientFactory,
    index_name: str,
    project_id: str,
    vector: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    client = factory.get_client(project_id)
    index = client.Index(index_name)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [
        {"text": match.metadata.get("text", ""), "score": match.score}
        for match in results.matches
    ]


# Module-level singleton — shared across requests
_factory = PineconeClientFactory()


def get_factory() -> PineconeClientFactory:
    return _factory
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_pinecone_service.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/pinecone_service.py backend/tests/
git commit -m "feat: add Pinecone client factory and retrieval service"
```

---

## Task 4: Embedding Service

**Files:**
- Create: `backend/app/services/embedding_service.py`
- Create: `backend/tests/test_embedding_service.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_embedding_service.py`:

```python
from unittest.mock import MagicMock, patch

import pytest

from app.services.embedding_service import generate_embedding


def test_1536_dim_uses_openai_small():
    with patch("app.services.embedding_service.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )
        MockOpenAI.return_value = mock_client

        result = generate_embedding("test query", 1536)

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="test query"
        )
        assert len(result) == 1536


def test_2048_dim_uses_openai_large():
    with patch("app.services.embedding_service.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 2048)]
        )
        MockOpenAI.return_value = mock_client

        result = generate_embedding("test query", 2048)

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large", input="test query", dimensions=2048
        )
        assert len(result) == 2048


def test_1024_dim_uses_cohere():
    with patch("app.services.embedding_service.cohere") as mock_cohere_module:
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=[[0.1] * 1024])
        mock_cohere_module.Client.return_value = mock_client

        result = generate_embedding("test query", 1024)

        mock_client.embed.assert_called_once_with(
            texts=["test query"],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        assert len(result) == 1024


def test_unsupported_dimension_raises():
    with pytest.raises(ValueError, match="Unsupported embedding dimension"):
        generate_embedding("test query", 512)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_embedding_service.py -v
```

Expected: `ImportError` — `embedding_service` doesn't exist yet.

- [ ] **Step 3: Create embedding_service.py**

Create `backend/app/services/embedding_service.py`:

```python
import os

import cohere
from openai import OpenAI


def generate_embedding(query: str, dimension: int) -> list[float]:
    if dimension == 1536:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(model="text-embedding-3-small", input=query)
        return response.data[0].embedding

    if dimension == 2048:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(
            model="text-embedding-3-large", input=query, dimensions=2048
        )
        return response.data[0].embedding

    if dimension == 1024:
        co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
        response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        return response.embeddings[0]

    raise ValueError(f"Unsupported embedding dimension: {dimension}. Supported: 1024, 1536, 2048")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_embedding_service.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/embedding_service.py backend/tests/test_embedding_service.py
git commit -m "feat: add dimension-aware embedding service (Cohere 1024, OpenAI 1536/2048)"
```

---

## Task 5: System Prompts

**Files:**
- Create: `backend/app/prompts/optimizer.txt`
- Create: `backend/app/prompts/router.txt`
- Create: `backend/app/prompts/verifier.txt`
- Create: `backend/app/prompts/voice_synthesizer.txt`

- [ ] **Step 1: Create optimizer prompt**

Create `backend/app/prompts/optimizer.txt`:

```
You are a search query optimizer. Your only job is to reformulate the user's conversational query into a dense, semantically rich search query optimized for vector similarity search.

Rules:
- Strip filler words and conversational tone
- Expand abbreviations and acronyms  
- Fix spelling errors
- Make implicit context explicit where possible
- Output ONLY the optimized query string — no explanation, no punctuation wrapper, no quotes
- Do not answer the question
```

- [ ] **Step 2: Create router prompt**

Create `backend/app/prompts/router.txt`:

```
You are an index classifier. Given an optimized search query and a catalog of available Pinecone indexes, identify the single most relevant index.

Return valid JSON with exactly these fields:
{
  "index_name": "<exact index name from catalog>",
  "project_id": "<project_id from catalog>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence>"
}

If confidence is below 0.7, also include a "candidates" array with the next 1-2 best options:
{
  "candidates": [
    {"index_name": "...", "project_id": "..."}
  ]
}

Available indexes:
{catalog}

Query: {query}
```

- [ ] **Step 3: Create verifier prompt**

Create `backend/app/prompts/verifier.txt`:

```
You are an accuracy verifier. Given a user query and retrieved document chunks, assess whether the context contains enough information to answer the query faithfully.

Return valid JSON:
{
  "score": <0.0 to 1.0>,
  "recommendation": "proceed" | "proceed_with_caveat" | "insufficient_context",
  "reasoning": "<one sentence>"
}

Scoring guide:
- 0.8–1.0: Context directly answers the query → "proceed"
- 0.5–0.7: Context partially addresses the query, some gaps → "proceed_with_caveat"
- 0.0–0.4: Context does not support answering → "insufficient_context"

Query: {query}

Retrieved chunks:
{chunks}
```

- [ ] **Step 4: Create voice synthesizer prompt**

Create `backend/app/prompts/voice_synthesizer.txt`:

```
You are writing as Max Mayes. Use ONLY the information in the provided context to answer the question. Do not introduce knowledge from outside the context.

Max's voice profile (how he writes — match this exactly):
{voice_profile}

Context from Max's knowledge base:
{context}

{caveat_instruction}

Rules:
- Match Max's writing style, tone, and vocabulary precisely
- Every statement must be grounded in the provided context
- Write conversationally, as if Max is speaking directly to the reader
- Do not mention that you are using a knowledge base or context
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/prompts/
git commit -m "feat: add system prompts for all 5 pipeline steps"
```

---

## Task 6: Orchestrator Pipeline

**Files:**
- Create: `backend/app/services/orchestrator.py`
- Create: `backend/tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_orchestrator.py`:

```python
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.orchestrator import optimize_query, route_query, verify_accuracy


@pytest.mark.asyncio
async def test_optimize_query_returns_string():
    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="onboarding email sequence automation"))]
        )
        MockOpenAI.return_value = mock_client

        result = await optimize_query("hey whats the deal with that email onboarding thing")

        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_route_query_returns_index_and_project():
    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "index_name": "my-index",
                "project_id": "1",
                "confidence": 0.9,
                "reasoning": "Query matches domain"
            })))]
        )
        MockOpenAI.return_value = mock_client

        catalog = [{"index_name": "my-index", "project_id": "1", "domain_description": "test", "sample_queries": []}]
        result = await route_query("test query", catalog)

        assert result["index_name"] == "my-index"
        assert result["project_id"] == "1"
        assert result["confidence"] == 0.9


@pytest.mark.asyncio
async def test_verify_accuracy_proceed():
    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "score": 0.92,
                "recommendation": "proceed",
                "reasoning": "Context directly answers the query"
            })))]
        )
        MockOpenAI.return_value = mock_client

        result = await verify_accuracy("test query", [{"text": "relevant content", "score": 0.9}])

        assert result["recommendation"] == "proceed"
        assert result["score"] == 0.92


@pytest.mark.asyncio
async def test_verify_accuracy_insufficient():
    with patch("app.services.orchestrator.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "score": 0.2,
                "recommendation": "insufficient_context",
                "reasoning": "No relevant content found"
            })))]
        )
        MockOpenAI.return_value = mock_client

        result = await verify_accuracy("test query", [])

        assert result["recommendation"] == "insufficient_context"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_orchestrator.py -v
```

Expected: `ImportError` — `orchestrator` doesn't exist yet.

- [ ] **Step 3: Create orchestrator.py**

Create `backend/app/services/orchestrator.py`:

```python
import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.models.index_registry import IndexRegistry
from app.services.embedding_service import generate_embedding
from app.services.pinecone_service import get_factory, retrieve

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


async def optimize_query(raw_query: str) -> str:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": _load_prompt("optimizer.txt")},
            {"role": "user", "content": raw_query},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


async def route_query(
    optimized_query: str, catalog: list[dict[str, Any]]
) -> dict[str, Any]:
    catalog_text = "\n".join(
        f"- index_name: {e['index_name']} | project_id: {e['project_id']}\n"
        f"  description: {e['domain_description']}\n"
        f"  sample_queries: {', '.join(e.get('sample_queries', [])[:3])}"
        for e in catalog
    )
    prompt = _load_prompt("router.txt").replace("{catalog}", catalog_text).replace("{query}", optimized_query)

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


async def verify_accuracy(
    query: str, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    chunks_text = "\n\n".join(
        f"[{i+1}] (score: {c['score']:.2f})\n{c['text']}" for i, c in enumerate(chunks)
    )
    prompt = (
        _load_prompt("verifier.txt")
        .replace("{query}", query)
        .replace("{chunks}", chunks_text or "(no chunks retrieved)")
    )

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


async def _fetch_voice_profile(top_k: int = 5) -> str:
    factory = get_factory()
    # Voice index is project 1 by default — update if it's in a different project
    voice_index = os.environ.get("VOICE_INDEX_NAME", "max-copywriting-voice")
    voice_project = os.environ.get("VOICE_INDEX_PROJECT", "1")

    # Embed a generic query to fetch representative voice chunks
    try:
        vector = generate_embedding("writing style tone voice communication", 1536)
        chunks = retrieve(factory, voice_index, voice_project, vector, top_k=top_k)
        return "\n\n".join(c["text"] for c in chunks)
    except Exception:
        return ""


async def run_pipeline(
    raw_query: str, db: Session
) -> AsyncGenerator[str, None]:
    # Step A: optimize
    optimized = await optimize_query(raw_query)

    # Step B: route
    active_indexes = (
        db.query(IndexRegistry)
        .filter(IndexRegistry.is_active == True)  # noqa: E712
        .all()
    )
    catalog = [
        {
            "index_name": idx.index_name,
            "project_id": idx.project_id,
            "domain_description": idx.domain_description,
            "sample_queries": idx.sample_queries,
        }
        for idx in active_indexes
    ]

    if not catalog:
        yield "I don't have any indexes configured yet. Please add indexes in the admin dashboard."
        return

    route_result = await route_query(optimized, catalog)
    index_name = route_result["index_name"]
    project_id = route_result["project_id"]

    # Look up dimension + embedding model from registry
    registry_entry = (
        db.query(IndexRegistry)
        .filter(
            IndexRegistry.index_name == index_name,
            IndexRegistry.project_id == project_id,
        )
        .first()
    )
    if not registry_entry:
        yield "I couldn't find the target index in the registry. Please check the admin dashboard."
        return

    # Step C: retrieve
    vector = generate_embedding(optimized, registry_entry.dimension)
    factory = get_factory()
    chunks = retrieve(factory, index_name, project_id, vector, top_k=10)

    # Handle low-confidence multi-index fallback
    if route_result.get("confidence", 1.0) < 0.7 and route_result.get("candidates"):
        for candidate in route_result["candidates"][:2]:
            candidate_entry = (
                db.query(IndexRegistry)
                .filter(
                    IndexRegistry.index_name == candidate["index_name"],
                    IndexRegistry.project_id == candidate["project_id"],
                )
                .first()
            )
            if candidate_entry:
                extra_vector = generate_embedding(optimized, candidate_entry.dimension)
                extra_chunks = retrieve(
                    factory, candidate["index_name"], candidate["project_id"], extra_vector, top_k=5
                )
                chunks = sorted(chunks + extra_chunks, key=lambda c: c["score"], reverse=True)[:10]

    # Step D: verify
    verification = await verify_accuracy(raw_query, chunks)

    if verification["recommendation"] == "insufficient_context":
        yield "I don't have enough information in my knowledge base to answer that confidently."
        return

    # Step E: synthesize with voice
    context_text = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    )
    voice_profile = await _fetch_voice_profile()

    caveat = (
        "\nNote: Some parts of this answer have limited supporting context — indicate any uncertainty naturally."
        if verification["recommendation"] == "proceed_with_caveat"
        else ""
    )

    prompt_template = _load_prompt("voice_synthesizer.txt")
    system_prompt = (
        prompt_template
        .replace("{voice_profile}", voice_profile or "Write clearly and conversationally.")
        .replace("{context}", context_text)
        .replace("{caveat_instruction}", caveat)
    )

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": raw_query}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_orchestrator.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Add ANTHROPIC_API_KEY and VOICE_INDEX_PROJECT to config**

Update `backend/app/config.py` — add the new fields:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "chat-with-my-pinecone-backend"
    app_env: str = "development"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    cors_allow_origin: str = "http://localhost:3000"

    clerk_secret_key: str = ""

    pinecone_api_key_1: str = ""
    pinecone_api_key_2: str = ""
    pinecone_api_key_3: str = ""

    openai_api_key: str = ""
    cohere_api_key: str = ""
    anthropic_api_key: str = ""

    voice_index_name: str = "max-copywriting-voice"
    voice_index_project: str = "1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()


def normalized_database_url() -> str:
    raw = settings.database_url.strip()
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw
```

- [ ] **Step 6: Add ANTHROPIC_API_KEY to Railway**

```bash
cd backend && railway variables set ANTHROPIC_API_KEY="<max's anthropic key>" VOICE_INDEX_PROJECT="1"
```

> **Note for Max:** Replace `<max's anthropic key>` with your actual Anthropic API key from console.anthropic.com.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/orchestrator.py backend/tests/test_orchestrator.py backend/app/config.py
git commit -m "feat: add 5-step orchestration pipeline with SSE-ready generator"
```

---

## Task 7: Chat SSE Endpoint

**Files:**
- Modify: `backend/app/routers/chat.py`
- Modify: `backend/app/models/schemas.py`

- [ ] **Step 1: Update schemas.py**

Replace contents of `backend/app/models/schemas.py`:

```python
import uuid

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ChatResponse(BaseModel):
    request_id: str
    reply: str


class IndexRegistryCreate(BaseModel):
    index_name: str
    project_id: str
    api_key_env_var: str
    dimension: int
    embedding_model: str
    metric: str
    domain_description: str
    sample_queries: list[str]
    namespaces: dict = {}
    is_active: bool = True


class IndexRegistryUpdate(BaseModel):
    index_name: str | None = None
    project_id: str | None = None
    api_key_env_var: str | None = None
    dimension: int | None = None
    embedding_model: str | None = None
    metric: str | None = None
    domain_description: str | None = None
    sample_queries: list[str] | None = None
    namespaces: dict | None = None
    is_active: bool | None = None


class IndexRegistryResponse(BaseModel):
    id: str
    index_name: str
    project_id: str
    api_key_env_var: str
    dimension: int
    embedding_model: str
    metric: str
    domain_description: str
    sample_queries: list[str]
    namespaces: dict
    is_active: bool

    model_config = {"from_attributes": True}


class DiscoveredIndex(BaseModel):
    index_name: str
    project_id: str
    dimension: int
    metric: str
    already_in_registry: bool
```

- [ ] **Step 2: Replace chat.py with SSE endpoint**

Replace contents of `backend/app/routers/chat.py`:

```python
import json
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.middleware.clerk_auth import require_bearer_token
from app.models.chat import ChatMessage
from app.models.schemas import ChatRequest
from app.services.orchestrator import run_pipeline

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("")
async def chat_endpoint(
    payload: ChatRequest,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    request_id = str(uuid.uuid4())
    user_id = token  # Phase 1 placeholder — replace with Clerk JWT sub claim

    # Persist user message
    db.add(ChatMessage(
        session_id=payload.session_id,
        user_id=user_id,
        role="user",
        content=payload.message,
    ))
    db.commit()

    accumulated: list[str] = []

    async def event_stream():
        try:
            async for token_text in run_pipeline(payload.message, db):
                accumulated.append(token_text)
                data = json.dumps({"token": token_text, "request_id": request_id})
                yield f"data: {data}\n\n"

            # Persist assistant response
            full_reply = "".join(accumulated)
            db.add(ChatMessage(
                session_id=payload.session_id,
                user_id=user_id,
                role="assistant",
                content=full_reply,
            ))
            db.commit()

            yield "data: [DONE]\n\n"
        except Exception as exc:
            error_data = json.dumps({"error": str(exc), "request_id": request_id})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    return {
        "session_id": session_id,
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in messages
        ],
    }
```

- [ ] **Step 3: Verify the app starts without errors**

```bash
cd backend && pip install -r requirements.txt && python -c "from app.main import app; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add backend/app/routers/chat.py backend/app/models/schemas.py
git commit -m "feat: replace chat placeholder with SSE streaming endpoint"
```

---

## Task 8: Admin CRUD + Auto-Discovery

**Files:**
- Modify: `backend/app/routers/admin.py`

- [ ] **Step 1: Replace admin.py with full CRUD + discover**

Replace contents of `backend/app/routers/admin.py`:

```python
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pinecone import Pinecone
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.middleware.clerk_auth import require_bearer_token
from app.models.index_registry import IndexRegistry
from app.models.schemas import (
    DiscoveredIndex,
    IndexRegistryCreate,
    IndexRegistryResponse,
    IndexRegistryUpdate,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/indexes")
async def list_indexes(
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> dict:
    total = db.query(IndexRegistry).count()
    indexes = db.query(IndexRegistry).offset(skip).limit(limit).all()
    return {
        "indexes": [IndexRegistryResponse.model_validate(idx) for idx in indexes],
        "count": total,
    }


@router.post("/indexes", status_code=status.HTTP_201_CREATED)
async def create_index(
    payload: IndexRegistryCreate,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> IndexRegistryResponse:
    existing = (
        db.query(IndexRegistry)
        .filter(
            IndexRegistry.index_name == payload.index_name,
            IndexRegistry.project_id == payload.project_id,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Index '{payload.index_name}' in project '{payload.project_id}' already exists",
        )
    entry = IndexRegistry(**payload.model_dump())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return IndexRegistryResponse.model_validate(entry)


@router.get("/indexes/{index_id}")
async def get_index(
    index_id: str,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> IndexRegistryResponse:
    entry = db.query(IndexRegistry).filter(IndexRegistry.id == uuid.UUID(index_id)).first()
    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Index not found")
    return IndexRegistryResponse.model_validate(entry)


@router.patch("/indexes/{index_id}")
async def update_index(
    index_id: str,
    payload: IndexRegistryUpdate,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> IndexRegistryResponse:
    entry = db.query(IndexRegistry).filter(IndexRegistry.id == uuid.UUID(index_id)).first()
    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Index not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(entry, field, value)
    db.commit()
    db.refresh(entry)
    return IndexRegistryResponse.model_validate(entry)


@router.delete("/indexes/{index_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_index(
    index_id: str,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> None:
    entry = db.query(IndexRegistry).filter(IndexRegistry.id == uuid.UUID(index_id)).first()
    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Index not found")
    entry.is_active = False
    db.commit()


@router.post("/discover")
async def discover_indexes(
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    discovered: list[DiscoveredIndex] = []

    existing_keys = {
        (idx.index_name, idx.project_id)
        for idx in db.query(IndexRegistry.index_name, IndexRegistry.project_id).all()
    }

    for project_id in ["1", "2", "3"]:
        env_var = f"PINECONE_API_KEY_{project_id}"
        api_key = os.environ.get(env_var, "")
        if not api_key:
            continue

        try:
            pc = Pinecone(api_key=api_key)
            indexes = pc.list_indexes()
            for idx in indexes:
                name = idx.name
                spec = idx.dimension if hasattr(idx, "dimension") else None
                metric = idx.metric if hasattr(idx, "metric") else "cosine"

                # Fetch full spec if dimension not on list response
                if spec is None:
                    try:
                        desc = pc.describe_index(name)
                        spec = desc.dimension
                        metric = desc.metric
                    except Exception:
                        spec = 0

                discovered.append(
                    DiscoveredIndex(
                        index_name=name,
                        project_id=project_id,
                        dimension=spec or 0,
                        metric=metric,
                        already_in_registry=(name, project_id) in existing_keys,
                    )
                )
        except Exception as exc:
            # Skip projects with bad keys or network issues — don't block the others
            continue

    return {"discovered": [d.model_dump() for d in discovered], "count": len(discovered)}
```

- [ ] **Step 2: Run the full test suite to verify nothing broke**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: all existing tests PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/app/routers/admin.py
git commit -m "feat: add admin CRUD endpoints and Pinecone auto-discovery"
```

---

## Task 9: Deploy Backend

- [ ] **Step 1: Push to GitHub and redeploy Railway**

```bash
git push origin main
cd backend && railway up --detach
```

- [ ] **Step 2: Verify healthcheck**

```bash
curl https://chat-with-my-pinecone-backend-production.up.railway.app/healthz
```

Expected: `{"status":"ok"}`

- [ ] **Step 3: Verify discover endpoint works**

```bash
curl -s -X POST https://chat-with-my-pinecone-backend-production.up.railway.app/api/admin/discover \
  -H "Authorization: Bearer test-token" | python3 -m json.tool
```

Expected: JSON with `discovered` array listing your Pinecone indexes.

---

## Task 10: Frontend API Client + SSE Helper

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/lib/sse.ts`

- [ ] **Step 1: Update api.ts**

Replace contents of `frontend/src/lib/api.ts`:

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_URL) {
  console.warn("NEXT_PUBLIC_API_URL is not set.");
}

function apiUrl(path: string): string {
  if (!API_URL) throw new Error("Missing NEXT_PUBLIC_API_URL");
  return `${API_URL}${path}`;
}

// ---- Chat ----

export async function getChatHistory(token: string, sessionId: string) {
  const res = await fetch(apiUrl(`/api/chat/history/${sessionId}`), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Failed to fetch history: ${res.status}`);
  return res.json() as Promise<{ messages: { role: string; content: string; created_at: string }[] }>;
}

export async function startChatStream(
  token: string,
  message: string,
  sessionId: string
): Promise<Response> {
  const res = await fetch(apiUrl("/api/chat"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  if (!res.ok) throw new Error(`Chat request failed: ${res.status}`);
  return res;
}

// ---- Admin ----

export async function listIndexes(token: string) {
  const res = await fetch(apiUrl("/api/admin/indexes"), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Failed to list indexes: ${res.status}`);
  return res.json();
}

export async function getIndex(token: string, id: string) {
  const res = await fetch(apiUrl(`/api/admin/indexes/${id}`), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Failed to get index: ${res.status}`);
  return res.json();
}

export async function updateIndex(token: string, id: string, data: Record<string, unknown>) {
  const res = await fetch(apiUrl(`/api/admin/indexes/${id}`), {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Failed to update index: ${res.status}`);
  return res.json();
}

export async function createIndex(token: string, data: Record<string, unknown>) {
  const res = await fetch(apiUrl("/api/admin/indexes"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Failed to create index: ${res.status}`);
  return res.json();
}

export async function discoverIndexes(token: string) {
  const res = await fetch(apiUrl("/api/admin/discover"), {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`Discovery failed: ${res.status}`);
  return res.json() as Promise<{
    discovered: { index_name: string; project_id: string; dimension: number; metric: string; already_in_registry: boolean }[];
    count: number;
  }>;
}

export async function importIndex(
  token: string,
  discovered: { index_name: string; project_id: string; dimension: number; metric: string }
) {
  const dimensionModelMap: Record<number, string> = {
    1024: "embed-english-v3.0",
    1536: "text-embedding-3-small",
    2048: "text-embedding-3-large",
  };
  return createIndex(token, {
    index_name: discovered.index_name,
    project_id: discovered.project_id,
    api_key_env_var: `PINECONE_API_KEY_${discovered.project_id}`,
    dimension: discovered.dimension,
    embedding_model: dimensionModelMap[discovered.dimension] ?? "text-embedding-3-small",
    metric: discovered.metric,
    domain_description: "",
    sample_queries: [],
    is_active: false,
  });
}
```

- [ ] **Step 2: Create sse.ts**

Create `frontend/src/lib/sse.ts`:

```typescript
export type SSEToken = { token: string; request_id: string };
export type SSEError = { error: string; request_id: string };

export async function* readSSEStream(
  response: Response
): AsyncGenerator<SSEToken | SSEError> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6).trim();
      if (raw === "[DONE]") return;
      try {
        yield JSON.parse(raw) as SSEToken | SSEError;
      } catch {
        // malformed line — skip
      }
    }
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/lib/sse.ts
git commit -m "feat: add admin + streaming API client and SSE stream reader"
```

---

## Task 11: Chat Components

**Files:**
- Create: `frontend/src/components/ChatWindow.tsx`
- Create: `frontend/src/components/MessageBubble.tsx`
- Create: `frontend/src/components/StreamingResponse.tsx`
- Modify: `frontend/src/components/ChatInput.tsx`
- Modify: `frontend/src/app/styles.css`

- [ ] **Step 1: Create MessageBubble.tsx**

Create `frontend/src/components/MessageBubble.tsx`:

```tsx
"use client";

type Props = {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
};

export function MessageBubble({ role, content, isStreaming }: Props) {
  return (
    <div className={`message-bubble message-${role}`}>
      <div className="bubble-content">
        {content}
        {isStreaming && <span className="streaming-cursor" />}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create StreamingResponse.tsx**

Create `frontend/src/components/StreamingResponse.tsx`:

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { readSSEStream } from "@/lib/sse";
import { MessageBubble } from "./MessageBubble";

type Props = {
  response: Response;
  onComplete: (fullText: string) => void;
  onError: (error: string) => void;
};

export function StreamingResponse({ response, onComplete, onError }: Props) {
  const [text, setText] = useState("");
  const fullTextRef = useRef("");

  useEffect(() => {
    let cancelled = false;

    async function stream() {
      try {
        for await (const event of readSSEStream(response)) {
          if (cancelled) return;
          if ("error" in event) {
            onError(event.error);
            return;
          }
          fullTextRef.current += event.token;
          setText(fullTextRef.current);
        }
        if (!cancelled) onComplete(fullTextRef.current);
      } catch (err) {
        if (!cancelled) onError(String(err));
      }
    }

    stream();
    return () => { cancelled = true; };
  }, [response]);

  return <MessageBubble role="assistant" content={text} isStreaming />;
}
```

- [ ] **Step 3: Create ChatWindow.tsx**

Create `frontend/src/components/ChatWindow.tsx`:

```tsx
"use client";

import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import { StreamingResponse } from "./StreamingResponse";

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type Props = {
  messages: Message[];
  streamingResponse: Response | null;
  onStreamComplete: (text: string) => void;
  onStreamError: (error: string) => void;
};

export function ChatWindow({ messages, streamingResponse, onStreamComplete, onStreamError }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingResponse]);

  return (
    <div className="chat-window">
      {messages.map((m) => (
        <MessageBubble key={m.id} role={m.role} content={m.content} />
      ))}
      {streamingResponse && (
        <StreamingResponse
          response={streamingResponse}
          onComplete={onStreamComplete}
          onError={onStreamError}
        />
      )}
      <div ref={bottomRef} />
    </div>
  );
}
```

- [ ] **Step 4: Update ChatInput.tsx — Enter=newline, Cmd+Enter=submit**

Replace contents of `frontend/src/components/ChatInput.tsx`:

```tsx
"use client";

import { KeyboardEvent, useRef, useState } from "react";

type Props = {
  onSend: (message: string) => void;
  disabled?: boolean;
};

export function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      submit();
    }
    // plain Enter falls through — adds newline naturally
  };

  return (
    <div className="chat-input-container">
      <textarea
        ref={textareaRef}
        className="chat-textarea"
        value={value}
        rows={3}
        disabled={disabled}
        placeholder="Ask anything… (Cmd+Enter to send)"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <button
        className="chat-send-btn"
        onClick={submit}
        disabled={disabled || !value.trim()}
      >
        Send
      </button>
    </div>
  );
}
```

- [ ] **Step 5: Add chat + admin styles to styles.css**

Append to `frontend/src/app/styles.css`:

```css
/* ---- Chat ---- */
.chat-page {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 4rem);
  max-width: 800px;
  margin: 0 auto;
  padding: 1rem;
}

.chat-window {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 1rem 0;
}

.message-bubble {
  max-width: 80%;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

.message-user {
  align-self: flex-end;
  background: #1e3a5f;
  border: 1px solid #2b5282;
}

.message-assistant {
  align-self: flex-start;
  background: #131a2f;
  border: 1px solid #2b3658;
}

.streaming-cursor::after {
  content: "▋";
  animation: blink 0.8s step-end infinite;
  margin-left: 2px;
  color: #7c9ef5;
}

@keyframes blink {
  50% { opacity: 0; }
}

.chat-input-container {
  display: flex;
  gap: 0.5rem;
  align-items: flex-end;
  padding-top: 0.75rem;
  border-top: 1px solid #2b3658;
}

.chat-textarea {
  flex: 1;
  background: #131a2f;
  border: 1px solid #2b3658;
  border-radius: 8px;
  color: #e6eaf2;
  padding: 0.6rem 0.75rem;
  font-family: inherit;
  font-size: 0.95rem;
  resize: none;
  outline: none;
}

.chat-textarea:focus {
  border-color: #4a6fa5;
}

.chat-send-btn {
  background: #2b5282;
  color: #e6eaf2;
  border: none;
  border-radius: 8px;
  padding: 0.6rem 1.2rem;
  cursor: pointer;
  font-size: 0.9rem;
  white-space: nowrap;
}

.chat-send-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* ---- Admin ---- */
.admin-page {
  padding: 1.5rem 1rem;
  max-width: 1100px;
  margin: 0 auto;
}

.admin-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
}

.btn {
  background: #2b5282;
  color: #e6eaf2;
  border: none;
  border-radius: 6px;
  padding: 0.5rem 1rem;
  cursor: pointer;
  font-size: 0.875rem;
}

.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-sm { padding: 0.3rem 0.7rem; font-size: 0.8rem; }
.btn-danger { background: #7b2020; }
.btn-outline { background: transparent; border: 1px solid #2b3658; }

.registry-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.registry-table th,
.registry-table td {
  text-align: left;
  padding: 0.6rem 0.75rem;
  border-bottom: 1px solid #2b3658;
}

.registry-table th {
  color: #7c9ef5;
  font-weight: 500;
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.05em;
}

.badge {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
}

.badge-active { background: #1a3a2a; color: #4caf78; border: 1px solid #2a5a3a; }
.badge-inactive { background: #2a1a1a; color: #e05252; border: 1px solid #5a2a2a; }

.edit-form { display: flex; flex-direction: column; gap: 1.25rem; max-width: 720px; }
.form-field { display: flex; flex-direction: column; gap: 0.4rem; }
.form-label { font-size: 0.8rem; color: #7c9ef5; text-transform: uppercase; letter-spacing: 0.05em; }
.form-input, .form-textarea, .form-select {
  background: #131a2f;
  border: 1px solid #2b3658;
  border-radius: 6px;
  color: #e6eaf2;
  padding: 0.5rem 0.75rem;
  font-family: inherit;
  font-size: 0.9rem;
  outline: none;
}
.form-textarea { resize: vertical; min-height: 100px; }
.form-input:focus, .form-textarea:focus, .form-select:focus { border-color: #4a6fa5; }

.tag-list { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.4rem; }
.tag {
  background: #1e3a5f;
  border: 1px solid #2b5282;
  border-radius: 4px;
  padding: 0.2rem 0.5rem;
  font-size: 0.8rem;
  display: flex;
  align-items: center;
  gap: 0.3rem;
}
.tag-remove { cursor: pointer; color: #7c9ef5; background: none; border: none; padding: 0; }

.modal-backdrop {
  position: fixed; inset: 0; background: rgba(0,0,0,0.6);
  display: flex; align-items: center; justify-content: center; z-index: 50;
}
.modal {
  background: #131a2f; border: 1px solid #2b3658; border-radius: 10px;
  padding: 1.5rem; width: 90%; max-width: 700px; max-height: 80vh; overflow-y: auto;
}
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/ frontend/src/app/styles.css
git commit -m "feat: add chat components (ChatWindow, MessageBubble, StreamingResponse, ChatInput)"
```

---

## Task 12: Chat Page

**Files:**
- Modify: `frontend/src/app/chat/page.tsx`

- [ ] **Step 1: Replace chat page with full client component**

Replace contents of `frontend/src/app/chat/page.tsx`:

```tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { ChatInput } from "@/components/ChatInput";
import { ChatWindow, type Message } from "@/components/ChatWindow";
import { getChatHistory, startChatStream } from "@/lib/api";

const SESSION_KEY = "pinecone-chat-session";

function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return crypto.randomUUID();
  const existing = localStorage.getItem(SESSION_KEY);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(SESSION_KEY, id);
  return id;
}

export default function ChatPage() {
  const { getToken } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingResponse, setStreamingResponse] = useState<Response | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useRef(getOrCreateSessionId());

  useEffect(() => {
    async function loadHistory() {
      try {
        const token = await getToken();
        if (!token) return;
        const history = await getChatHistory(token, sessionId.current);
        setMessages(
          history.messages.map((m, i) => ({
            id: `hist-${i}`,
            role: m.role as "user" | "assistant",
            content: m.content,
          }))
        );
      } catch {
        // no history yet — start fresh
      }
    }
    loadHistory();
  }, [getToken]);

  const handleSend = useCallback(async (message: string) => {
    const token = await getToken();
    if (!token) return;

    setError(null);
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content: message },
    ]);
    setIsStreaming(true);

    try {
      const response = await startChatStream(token, message, sessionId.current);
      setStreamingResponse(response);
    } catch (err) {
      setIsStreaming(false);
      setError(String(err));
    }
  }, [getToken]);

  const handleStreamComplete = useCallback((text: string) => {
    setStreamingResponse(null);
    setIsStreaming(false);
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "assistant", content: text },
    ]);
  }, []);

  const handleStreamError = useCallback((err: string) => {
    setStreamingResponse(null);
    setIsStreaming(false);
    setError(err);
  }, []);

  return (
    <div className="chat-page">
      <ChatWindow
        messages={messages}
        streamingResponse={streamingResponse}
        onStreamComplete={handleStreamComplete}
        onStreamError={handleStreamError}
      />
      {error && (
        <p style={{ color: "#e05252", fontSize: "0.85rem", padding: "0.5rem 0" }}>
          {error}
        </p>
      )}
      <ChatInput onSend={handleSend} disabled={isStreaming} />
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/app/chat/page.tsx
git commit -m "feat: rebuild chat page with SSE streaming and session persistence"
```

---

## Task 13: Admin Index Table Page

**Files:**
- Modify: `frontend/src/app/admin/page.tsx`

- [ ] **Step 1: Replace admin page with registry table + discover**

Replace contents of `frontend/src/app/admin/page.tsx`:

```tsx
"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@clerk/nextjs";
import { discoverIndexes, importIndex, listIndexes, updateIndex } from "@/lib/api";

type IndexEntry = {
  id: string;
  index_name: string;
  project_id: string;
  dimension: number;
  embedding_model: string;
  is_active: boolean;
  domain_description: string;
};

type DiscoveredEntry = {
  index_name: string;
  project_id: string;
  dimension: number;
  metric: string;
  already_in_registry: boolean;
};

export default function AdminPage() {
  const { getToken } = useAuth();
  const [indexes, setIndexes] = useState<IndexEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [discovering, setDiscovering] = useState(false);
  const [discovered, setDiscovered] = useState<DiscoveredEntry[] | null>(null);
  const [importing, setImporting] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  const loadIndexes = useCallback(async () => {
    const token = await getToken();
    if (!token) return;
    try {
      const data = await listIndexes(token);
      setIndexes(data.indexes);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [getToken]);

  useEffect(() => { loadIndexes(); }, [loadIndexes]);

  const handleDiscover = async () => {
    const token = await getToken();
    if (!token) return;
    setDiscovering(true);
    setError(null);
    try {
      const data = await discoverIndexes(token);
      setDiscovered(data.discovered);
    } catch (err) {
      setError(String(err));
    } finally {
      setDiscovering(false);
    }
  };

  const handleImport = async (entry: DiscoveredEntry) => {
    const token = await getToken();
    if (!token) return;
    const key = `${entry.index_name}-${entry.project_id}`;
    setImporting((prev) => new Set(prev).add(key));
    try {
      await importIndex(token, entry);
      await loadIndexes();
    } catch (err) {
      setError(String(err));
    } finally {
      setImporting((prev) => { const s = new Set(prev); s.delete(key); return s; });
    }
  };

  const handleToggleActive = async (idx: IndexEntry) => {
    const token = await getToken();
    if (!token) return;
    try {
      await updateIndex(token, idx.id, { is_active: !idx.is_active });
      await loadIndexes();
    } catch (err) {
      setError(String(err));
    }
  };

  return (
    <div className="admin-page">
      <div className="admin-header">
        <h1>Index Registry</h1>
        <button className="btn" onClick={handleDiscover} disabled={discovering}>
          {discovering ? "Scanning Pinecone…" : "Discover from Pinecone"}
        </button>
      </div>

      {error && <p style={{ color: "#e05252", marginBottom: "1rem" }}>{error}</p>}

      {discovered && (
        <div className="card" style={{ marginBottom: "1.5rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <h3 style={{ margin: 0 }}>Discovered Indexes ({discovered.length})</h3>
            <button className="btn btn-sm btn-outline" onClick={() => setDiscovered(null)}>Dismiss</button>
          </div>
          <table className="registry-table">
            <thead>
              <tr><th>Index</th><th>Project</th><th>Dimension</th><th>Metric</th><th></th></tr>
            </thead>
            <tbody>
              {discovered.map((d) => {
                const key = `${d.index_name}-${d.project_id}`;
                return (
                  <tr key={key}>
                    <td>{d.index_name}</td>
                    <td>{d.project_id}</td>
                    <td>{d.dimension}</td>
                    <td>{d.metric}</td>
                    <td>
                      {d.already_in_registry ? (
                        <span style={{ color: "#888", fontSize: "0.8rem" }}>Already imported</span>
                      ) : (
                        <button
                          className="btn btn-sm"
                          disabled={importing.has(key)}
                          onClick={() => handleImport(d)}
                        >
                          {importing.has(key) ? "Importing…" : "Import"}
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {loading ? (
        <p>Loading registry…</p>
      ) : (
        <table className="registry-table">
          <thead>
            <tr><th>Index</th><th>Project</th><th>Dim</th><th>Model</th><th>Status</th><th>Description</th><th></th></tr>
          </thead>
          <tbody>
            {indexes.length === 0 && (
              <tr><td colSpan={7} style={{ color: "#888", padding: "2rem 0" }}>No indexes yet. Click "Discover from Pinecone" to get started.</td></tr>
            )}
            {indexes.map((idx) => (
              <tr key={idx.id}>
                <td>{idx.index_name}</td>
                <td>{idx.project_id}</td>
                <td>{idx.dimension}</td>
                <td style={{ fontSize: "0.8rem", color: "#aaa" }}>{idx.embedding_model}</td>
                <td>
                  <button
                    className={`badge ${idx.is_active ? "badge-active" : "badge-inactive"}`}
                    onClick={() => handleToggleActive(idx)}
                    style={{ cursor: "pointer", background: "none" }}
                  >
                    {idx.is_active ? "Active" : "Inactive"}
                  </button>
                </td>
                <td style={{ maxWidth: "240px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "#aaa", fontSize: "0.85rem" }}>
                  {idx.domain_description || <em style={{ color: "#555" }}>No description yet</em>}
                </td>
                <td>
                  <Link href={`/admin/indexes/${idx.id}`} className="btn btn-sm btn-outline">Edit</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/app/admin/page.tsx
git commit -m "feat: rebuild admin page with registry table and Pinecone discover flow"
```

---

## Task 14: Admin Index Edit Page

**Files:**
- Create: `frontend/src/app/admin/indexes/[id]/page.tsx`

- [ ] **Step 1: Create the edit page**

Create `frontend/src/app/admin/indexes/[id]/page.tsx`:

```tsx
"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@clerk/nextjs";
import { getIndex, updateIndex } from "@/lib/api";

type IndexEntry = {
  id: string;
  index_name: string;
  project_id: string;
  api_key_env_var: string;
  dimension: number;
  embedding_model: string;
  metric: string;
  domain_description: string;
  sample_queries: string[];
  is_active: boolean;
};

export default function EditIndexPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { getToken } = useAuth();

  const [entry, setEntry] = useState<IndexEntry | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newQuery, setNewQuery] = useState("");

  useEffect(() => {
    async function load() {
      const token = await getToken();
      if (!token) return;
      try {
        const data = await getIndex(token, id);
        setEntry(data);
      } catch (err) {
        setError(String(err));
      }
    }
    load();
  }, [id, getToken]);

  const handleSave = useCallback(async () => {
    if (!entry) return;
    const token = await getToken();
    if (!token) return;
    setSaving(true);
    setError(null);
    try {
      await updateIndex(token, entry.id, {
        domain_description: entry.domain_description,
        sample_queries: entry.sample_queries,
        is_active: entry.is_active,
        embedding_model: entry.embedding_model,
        metric: entry.metric,
      });
      router.push("/admin");
    } catch (err) {
      setError(String(err));
      setSaving(false);
    }
  }, [entry, getToken, router]);

  const addQuery = () => {
    const q = newQuery.trim();
    if (!q || !entry) return;
    setEntry({ ...entry, sample_queries: [...entry.sample_queries, q] });
    setNewQuery("");
  };

  const removeQuery = (i: number) => {
    if (!entry) return;
    setEntry({ ...entry, sample_queries: entry.sample_queries.filter((_, idx) => idx !== i) });
  };

  if (!entry) return <div className="admin-page"><p>{error ?? "Loading…"}</p></div>;

  return (
    <div className="admin-page">
      <div className="admin-header">
        <div>
          <h1>{entry.index_name}</h1>
          <p style={{ color: "#888", margin: "0.25rem 0 0", fontSize: "0.875rem" }}>
            Project {entry.project_id} · {entry.dimension}d · {entry.embedding_model}
          </p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button className="btn btn-outline" onClick={() => router.push("/admin")}>Cancel</button>
          <button className="btn" onClick={handleSave} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>

      {error && <p style={{ color: "#e05252", marginBottom: "1rem" }}>{error}</p>}

      <div className="edit-form">
        <div className="form-field">
          <label className="form-label">
            Domain Description
            <span style={{ color: "#888", fontWeight: 400, marginLeft: "0.5rem" }}>
              — What topics and content does this index contain? Be specific. The router LLM uses this to decide which index to query.
            </span>
          </label>
          <textarea
            className="form-textarea"
            style={{ minHeight: "140px" }}
            value={entry.domain_description}
            onChange={(e) => setEntry({ ...entry, domain_description: e.target.value })}
            placeholder="e.g. Contains Max's email copywriting frameworks, subject line formulas, and onboarding sequence templates."
          />
        </div>

        <div className="form-field">
          <label className="form-label">Sample Queries</label>
          <p style={{ color: "#888", fontSize: "0.825rem", margin: "0 0 0.5rem" }}>
            Representative questions that should route to this index. Add 3–5 examples.
          </p>
          <div className="tag-list">
            {entry.sample_queries.map((q, i) => (
              <span key={i} className="tag">
                {q}
                <button className="tag-remove" onClick={() => removeQuery(i)}>×</button>
              </span>
            ))}
          </div>
          <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem" }}>
            <input
              className="form-input"
              style={{ flex: 1 }}
              value={newQuery}
              onChange={(e) => setNewQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addQuery())}
              placeholder="Type a sample query and press Enter"
            />
            <button className="btn btn-sm" onClick={addQuery}>Add</button>
          </div>
        </div>

        <div className="form-field">
          <label className="form-label">Active</label>
          <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={entry.is_active}
              onChange={(e) => setEntry({ ...entry, is_active: e.target.checked })}
            />
            <span style={{ fontSize: "0.9rem" }}>Include this index in query routing</span>
          </label>
        </div>

        <div style={{ display: "flex", gap: "1rem" }}>
          <div className="form-field" style={{ flex: 1 }}>
            <label className="form-label">Embedding Model</label>
            <input className="form-input" value={entry.embedding_model}
              onChange={(e) => setEntry({ ...entry, embedding_model: e.target.value })} />
          </div>
          <div className="form-field" style={{ flex: 1 }}>
            <label className="form-label">Metric</label>
            <select className="form-select" value={entry.metric}
              onChange={(e) => setEntry({ ...entry, metric: e.target.value })}>
              <option value="cosine">cosine</option>
              <option value="dotproduct">dotproduct</option>
              <option value="euclidean">euclidean</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/app/admin/indexes/
git commit -m "feat: add admin index edit page with domain description and sample queries"
```

---

## Task 15: Final Deploy

- [ ] **Step 1: Push all changes**

```bash
git push origin main
```

- [ ] **Step 2: Redeploy backend**

```bash
cd backend && railway up --detach
```

- [ ] **Step 3: Redeploy frontend**

```bash
cd "/Users/maxmayes/Dropbox/01. Professional/02. AI Tools/Chat with My Pinecone" && vercel --prod --yes
```

- [ ] **Step 4: Smoke test the admin discover flow**

1. Open `https://maxmayes.io/admin`
2. Sign in with Clerk
3. Click "Discover from Pinecone" — should show your indexes
4. Import a few indexes
5. Click Edit on one — add a domain description and 3 sample queries, save

- [ ] **Step 5: Smoke test the chat**

1. Open `https://maxmayes.io/chat`
2. Send a message that matches one of your indexed domains
3. Verify response streams in and reflects Max's voice
4. Verify the message persists on page refresh

- [ ] **Step 6: Tag the release**

```bash
git tag v0.2.0 -m "Phase 2+3: Pinecone integration, registry, and chat pipeline"
git push origin v0.2.0
```
