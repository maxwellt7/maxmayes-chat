# RAG Redesign — Phase 1 Implementation Plan (orchestrator_v2 + Domain Agents)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the LangGraph orchestrator_v2 with 7 domain agents (each with iterative retrieval, up to 3 passes), mode-aware synthesis, persistent classification + citations + agent traces, and an eval harness — all behind a feature flag so production behavior doesn't change until Phase 2 flips the default.

**Architecture:** Adds `app.orchestrator_v2` (LangGraph), `app.agents.*` (7 modules), 3 new Postgres tables (`message_classifications`, `citations`, `agent_traces`) + `messages.cached_context`, new feature-flag-gated route, and a `backend/evals/` test harness with CI gate.

**Tech Stack:** LangGraph + Anthropic SDK (Opus 4.7 + Sonnet 4.6) + OpenAI SDK (gpt-4.1-mini) + Cohere SDK (embeddings + rerank) — all already wired in Phase 0.

**Source spec:** `docs/superpowers/specs/2026-05-10-rag-redesign-design.md`
**Architecture:** `docs/architecture-rag-redesign-2026-05-10.md`
**Prerequisite:** Phase 0 complete — 7 domain indexes populated + `index_registry` rows registered.

---

## Phase 1 deliverables

End state of this plan:
- LangGraph orchestrator_v2 running parallel domain agents with iterative refinement
- 7 domain agent modules, each its own Python file with a custom `Plan` node
- Mode-aware synthesis (personal/content/public) with auto-classifier
- Static MAX_STYLE_GUIDE in code — voice Pinecone leak eliminated
- Cross-session conversation memory via new `conversation-memory` Pinecone index
- `POST /api/chat?engine=v2` opt-in endpoint; existing `/api/chat` unchanged
- 3 new Postgres tables + `messages.cached_context` for re-stream support
- Eval harness with 30 frozen test cases gating CI

Production behavior is **unchanged** unless the client appends `?engine=v2`. Phase 2 will flip the default.

---

## Task 1: Postgres schema for Phase 1

**Files:**
- Create: `backend/app/models/message_classification.py`
- Create: `backend/app/models/citation.py`
- Create: `backend/app/models/agent_trace.py`
- Modify: `backend/app/models/__init__.py`
- Modify: `backend/app/models/chat.py` (add cached_context columns)
- Create: `backend/app/migrations/versions/003_phase1_orchestrator_v2.py`
- Test: `backend/tests/test_phase1_models.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_phase1_models.py
import importlib
import inspect


def test_classification_model_columns_present():
    from app.models.message_classification import MessageClassification
    source = inspect.getsource(MessageClassification.__module__.__loader__.load_module)
    # Lighter check via __dict__:
    expected = {
        "message_id", "detected_mode", "mode_confidence", "user_override",
        "intent", "topic_hints", "persona_target", "out_of_domain_score",
        "created_at",
    }
    src = inspect.getsource(MessageClassification)
    for name in expected:
        assert f"{name}:" in src, f"missing column {name}"


def test_citation_model_columns_present():
    from app.models.citation import Citation
    src = inspect.getsource(Citation)
    expected = {
        "id", "message_id", "position", "source_index", "source_namespace",
        "source_type", "source_id", "source_url", "chunk_text", "chunk_score",
        "ingested_at", "created_at",
    }
    for name in expected:
        assert f"{name}:" in src, f"missing column {name}"


def test_agent_trace_model_columns_present():
    from app.models.agent_trace import AgentTrace
    src = inspect.getsource(AgentTrace)
    expected = {
        "id", "message_id", "domain_name", "passes_used",
        "refinement_history", "self_assessment", "notes_for_synthesizer",
        "chunks_returned", "total_latency_ms", "created_at",
    }
    for name in expected:
        assert f"{name}:" in src, f"missing column {name}"
```

- [ ] **Step 2: Create models**

`backend/app/models/message_classification.py`:
```python
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, REAL, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class MessageClassification(Base):
    __tablename__ = "message_classifications"

    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    detected_mode: Mapped[str] = mapped_column(String(50), nullable=False)
    mode_confidence: Mapped[float] = mapped_column(REAL, nullable=False)
    user_override: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    intent: Mapped[str] = mapped_column(String(100), nullable=False)
    topic_hints: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list
    )
    persona_target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    out_of_domain_score: Mapped[float] = mapped_column(REAL, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
```

`backend/app/models/citation.py`:
```python
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, REAL, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Citation(Base):
    __tablename__ = "citations"
    __table_args__ = (
        UniqueConstraint("message_id", "position", name="citations_msg_pos_unique"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    source_index: Mapped[str] = mapped_column(String(255), nullable=False)
    source_namespace: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(100), nullable=False)
    source_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_score: Mapped[float] = mapped_column(REAL, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
```

`backend/app/models/agent_trace.py`:
```python
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class AgentTrace(Base):
    __tablename__ = "agent_traces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    domain_name: Mapped[str] = mapped_column(String(100), nullable=False)
    passes_used: Mapped[int] = mapped_column(Integer, nullable=False)
    refinement_history: Mapped[list] = mapped_column(JSONB, nullable=False)
    self_assessment: Mapped[dict] = mapped_column(JSONB, nullable=False)
    notes_for_synthesizer: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunks_returned: Mapped[int] = mapped_column(Integer, nullable=False)
    total_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
```

- [ ] **Step 3: Add `cached_context` columns to `messages` (in `app/models/chat.py`)**

Append to the existing `Message` (or equivalent) model the two columns:
```python
    cached_context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    cached_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
```

- [ ] **Step 4: Update `app/models/__init__.py` to import all new modules**

```python
from app.models import agent_trace as _agent_trace  # noqa: F401
from app.models import chat as _chat  # noqa: F401
from app.models import citation as _citation  # noqa: F401
from app.models import index_audit as _index_audit  # noqa: F401
from app.models import index_registry as _index_registry  # noqa: F401
from app.models import ingest_job as _ingest_job  # noqa: F401
from app.models import message_classification as _message_classification  # noqa: F401
from app.models import schemas as _schemas  # noqa: F401
```

- [ ] **Step 5: Alembic migration 003**

Hand-write at `backend/app/migrations/versions/003_phase1_orchestrator_v2.py` — covers all three tables + the `messages` column additions + partial index on `cached_until`. Use `revision = "003_phase1_orchestrator_v2"` and `down_revision = "002_phase0_audit_ingest"`.

- [ ] **Step 6: Run tests + commit**

```bash
cd backend && python3 -m pytest tests/test_phase1_models.py -v
git add backend/app/models/ backend/app/migrations/versions/003_*.py backend/tests/test_phase1_models.py
git commit -m "feat(db): Phase 1 schema — classifications, citations, agent traces, cached_context"
git push origin main
```

---

## Task 2: Static MAX_STYLE_GUIDE + mode config

**Files:**
- Create: `backend/app/orchestrator_v2/__init__.py`
- Create: `backend/app/orchestrator_v2/style.py`
- Create: `backend/app/orchestrator_v2/modes.py`
- Test: `backend/tests/test_modes_and_style.py`

- [ ] **Step 1: Write `style.py` with the static guide (no Pinecone)**

```python
# backend/app/orchestrator_v2/style.py
"""Static Max Mayes style guide used by the synthesizer.

This replaces the dynamic ``_fetch_voice_profile`` retrieval pattern that
silently injected topic content from the voice Pinecone into every answer.
Voice content is now retrievable as a normal domain agent only when the
query is genuinely about voice/style.
"""

MAX_STYLE_GUIDE = """
- Direct, opinionated, no hedging. Lead with the punch line, then explain.
- Short paragraphs. One-line sentences for emphasis.
- Speaks in second person to the reader ("you").
- Uses contractions. Occasional emphatic profanity ("fucking", "dumb").
- Refuses to bullshit. Honest about what he doesn't know.
- Concrete > abstract. Names specific tactics, frameworks, percentages, dollar amounts where relevant.
- Avoids: buzzwords, MBA-speak, hedge-words ("perhaps", "potentially", "it depends").
- Hooks: start with a strong claim, a counterintuitive observation, or "look, here's the thing".
- Closes: with a clear next move or a question that puts the ball back in the reader's court.
"""
```

- [ ] **Step 2: Write `modes.py` with per-mode config**

```python
# backend/app/orchestrator_v2/modes.py
"""Per-mode config: synthesizer model, refusal threshold, citation policy,
mode-specific prompt rules.

Adding a new mode = adding an entry to MODES + a rules block. No code
changes needed in synthesize.py.
"""
from typing import Literal, TypedDict

Mode = Literal["personal", "content_creation", "public"]


class ModeConfig(TypedDict):
    synthesizer_model: str
    refusal_threshold: float
    citations_required: bool
    public_safe_only: bool
    rules: str


_PERSONAL_RULES = """
Mode: personal — you're talking to Max himself.
- Write conversationally. Citations OFF by default; if you reference something specific,
  mention it naturally ("from that podcast with X..."), don't emit [1][2] tags.
- It's OK to push back, disagree, point out contradictions.
- If gaps were noted, say so plainly — Max wants honesty.
- If knowledge is thin, give your best partial take and flag what would make it stronger.
"""

_CONTENT_CREATION_RULES = """
Mode: content_creation — Max is drafting something he'll ship.
- Output IS the artifact (or strong draft). Match the implied format (email, ad, post, script).
- Voice precision matters most here — every sentence should sound like Max.
- Citations as [1][2] inline, required for any factual or example-based statement.
- Do NOT invent. If a specific claim is unsupported, write "[need source]" inline.
"""

_PUBLIC_RULES = """
Mode: public — someone other than Max is reading this.
- Citations [1][2] inline, ALWAYS. Every factual claim must trace to a chunk.
- Refuse cleanly if knowledge is insufficient. Do not extrapolate beyond what's cited.
- No personal asides. Speak about what Max has said/written/believes.
- Use only public_safe chunks (filtered upstream).
"""


MODES: dict[Mode, ModeConfig] = {
    "personal": {
        "synthesizer_model": "claude-sonnet-4-6",
        "refusal_threshold": 0.45,
        "citations_required": False,
        "public_safe_only": False,
        "rules": _PERSONAL_RULES,
    },
    "content_creation": {
        "synthesizer_model": "claude-opus-4-7",
        "refusal_threshold": 0.50,
        "citations_required": True,
        "public_safe_only": False,
        "rules": _CONTENT_CREATION_RULES,
    },
    "public": {
        "synthesizer_model": "claude-opus-4-7",
        "refusal_threshold": 0.65,
        "citations_required": True,
        "public_safe_only": True,
        "rules": _PUBLIC_RULES,
    },
}
```

- [ ] **Step 3: Test**

```python
# backend/tests/test_modes_and_style.py
from app.orchestrator_v2.modes import MODES, Mode
from app.orchestrator_v2.style import MAX_STYLE_GUIDE


def test_three_modes_defined():
    assert set(MODES.keys()) == {"personal", "content_creation", "public"}


def test_public_mode_filters_public_safe():
    assert MODES["public"]["public_safe_only"] is True
    assert MODES["personal"]["public_safe_only"] is False


def test_content_and_public_use_opus():
    assert MODES["content_creation"]["synthesizer_model"] == "claude-opus-4-7"
    assert MODES["public"]["synthesizer_model"] == "claude-opus-4-7"
    assert MODES["personal"]["synthesizer_model"] == "claude-sonnet-4-6"


def test_refusal_thresholds_monotonic():
    assert (
        MODES["personal"]["refusal_threshold"]
        < MODES["content_creation"]["refusal_threshold"]
        < MODES["public"]["refusal_threshold"]
    )


def test_style_guide_is_non_empty_string():
    assert isinstance(MAX_STYLE_GUIDE, str)
    assert len(MAX_STYLE_GUIDE.strip()) > 100
    assert "punch line" in MAX_STYLE_GUIDE
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/orchestrator_v2/ backend/tests/test_modes_and_style.py
git commit -m "feat(orchestrator_v2): static MAX_STYLE_GUIDE + per-mode config"
```

---

## Task 3: Classifier node

**Files:**
- Create: `backend/app/orchestrator_v2/nodes/__init__.py`
- Create: `backend/app/orchestrator_v2/nodes/classify.py`
- Create: `backend/app/prompts/classify.txt`
- Test: `backend/tests/test_classify_node.py`

- [ ] **Step 1: Write classify prompt**

`backend/app/prompts/classify.txt`:
```
You classify a user query for a multi-mode RAG system.

Return JSON:
{
  "mode": "personal" | "content_creation" | "public",
  "mode_confidence": 0.0-1.0,
  "intent": "factual_lookup" | "strategy_synthesis" | "drafting" | "recall_with_context" | "exploration",
  "topic_hints": ["...", ...],
  "out_of_domain_score": 0.0-1.0,
  "persona_target": null | "<short description>"
}

Mode detection rules:
- First-person directives ("draft me X", "write a Y") → content_creation
- Third-person reference ("a client asked", "for our audience") → public
- Self-referential ("what did I write about", "what's my take") → personal
- Ambiguous strategy questions → personal (Max talking to himself)

Set persona_target ONLY in content_creation mode and ONLY when the query implies an audience
(e.g., "for B2B founders", "for our podcast audience" → set; "draft me an email" alone → null).

out_of_domain_score = how unlikely the query relates to: business strategy, marketing,
personal life, AI/automation tools, Max's content library, or specific industry verticals
(higher = more out-of-domain). When in doubt, return < 0.2.

Conversation history (last few turns):
$history

Current query: $query
```

- [ ] **Step 2: Write classify.py**

```python
# backend/app/orchestrator_v2/nodes/classify.py
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Literal

from app.services.llm_service import call_openai_json

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "classify.txt"
_MODEL = "gpt-4.1-mini"


@dataclass
class ClassifierOutput:
    mode: Literal["personal", "content_creation", "public"]
    mode_confidence: float
    intent: str
    topic_hints: list[str]
    out_of_domain_score: float
    persona_target: str | None


async def classify_query(
    query: str, history: list[dict] | None = None
) -> ClassifierOutput:
    """Classify the query into mode/intent/topic_hints/persona_target.

    On parse failure or invalid mode, defaults to personal mode with low
    confidence so downstream pipeline can flag and use the override badge.
    """
    history_str = "\n".join(
        f"{m.get('role', '?')}: {m.get('content', '')[:200]}"
        for m in (history or [])[-6:]
    ) or "(no prior turns)"

    prompt = Template(_PROMPT_PATH.read_text()).safe_substitute(
        history=history_str,
        query=query,
    )

    try:
        raw = await call_openai_json(model=_MODEL, system="", user=prompt)
    except Exception:
        return ClassifierOutput(
            mode="personal", mode_confidence=0.0,
            intent="exploration", topic_hints=[],
            out_of_domain_score=0.0, persona_target=None,
        )

    mode = raw.get("mode")
    if mode not in ("personal", "content_creation", "public"):
        mode = "personal"
    return ClassifierOutput(
        mode=mode,
        mode_confidence=float(raw.get("mode_confidence", 0.0)),
        intent=str(raw.get("intent", "exploration")),
        topic_hints=list(raw.get("topic_hints", []))[:10],
        out_of_domain_score=float(raw.get("out_of_domain_score", 0.0)),
        persona_target=raw.get("persona_target") or None,
    )
```

- [ ] **Step 3: Test classify_node — mock LLM with several queries**

```python
# backend/tests/test_classify_node.py
from unittest.mock import AsyncMock, patch
import pytest

from app.orchestrator_v2.nodes.classify import classify_query


@pytest.mark.asyncio
async def test_classify_drafting_query():
    with patch(
        "app.orchestrator_v2.nodes.classify.call_openai_json",
        new=AsyncMock(return_value={
            "mode": "content_creation",
            "mode_confidence": 0.92,
            "intent": "drafting",
            "topic_hints": ["email", "Q3"],
            "out_of_domain_score": 0.05,
            "persona_target": "B2B founder",
        }),
    ):
        out = await classify_query("Draft me a cold email for Q3 telemed pitch")
    assert out.mode == "content_creation"
    assert out.persona_target == "B2B founder"


@pytest.mark.asyncio
async def test_classify_defaults_to_personal_on_invalid_mode():
    with patch(
        "app.orchestrator_v2.nodes.classify.call_openai_json",
        new=AsyncMock(return_value={"mode": "nonsense", "mode_confidence": 0.5}),
    ):
        out = await classify_query("anything")
    assert out.mode == "personal"


@pytest.mark.asyncio
async def test_classify_defaults_on_api_error():
    with patch(
        "app.orchestrator_v2.nodes.classify.call_openai_json",
        new=AsyncMock(side_effect=Exception("LLM down")),
    ):
        out = await classify_query("anything")
    assert out.mode == "personal"
    assert out.mode_confidence == 0.0
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/orchestrator_v2/nodes/ backend/app/prompts/classify.txt backend/tests/test_classify_node.py
git commit -m "feat(orchestrator_v2): classifier node — mode + intent + persona detection"
```

---

## Task 4: Optimize node

**Files:**
- Create: `backend/app/orchestrator_v2/nodes/optimize.py`
- Test: `backend/tests/test_optimize_node.py`

- [ ] **Step 1: Implement optimize.py — reuses existing `optimizer.txt` prompt**

```python
# backend/app/orchestrator_v2/nodes/optimize.py
from pathlib import Path

from app.services.llm_service import call_openai_text

_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "optimizer.txt"
)
_MODEL = "gpt-4.1-mini"


async def optimize_query(raw_query: str) -> str:
    """Reformulate raw query into dense semantic search string."""
    return await call_openai_text(
        model=_MODEL,
        system=_PROMPT_PATH.read_text(),
        user=raw_query,
        temperature=0,
    )
```

- [ ] **Step 2: Test**

```python
# backend/tests/test_optimize_node.py
from unittest.mock import AsyncMock, patch
import pytest

from app.orchestrator_v2.nodes.optimize import optimize_query


@pytest.mark.asyncio
async def test_optimize_returns_dense_string():
    with patch(
        "app.orchestrator_v2.nodes.optimize.call_openai_text",
        new=AsyncMock(return_value="optimized semantic dense query"),
    ):
        result = await optimize_query("hey whats the deal with email")
    assert result == "optimized semantic dense query"
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/orchestrator_v2/nodes/optimize.py backend/tests/test_optimize_node.py
git commit -m "feat(orchestrator_v2): optimize node — query reformulation"
```

---

## Task 5: Top Router node

**Files:**
- Create: `backend/app/orchestrator_v2/nodes/top_router.py`
- Create: `backend/app/prompts/top_router.txt`
- Test: `backend/tests/test_top_router.py`

- [ ] **Step 1: Write top_router prompt**

`backend/app/prompts/top_router.txt`:
```
You select 1-2 domain agents to consult for a user query.

Available domain agents:
$catalog

Return JSON:
{
  "selected_domains": [
    {"name": "<domain>", "confidence": 0.0-1.0, "reasoning": "<short>"}
  ],
  "out_of_domain": <true|false>
}

Rules:
- Return up to 2 domains, ordered by relevance.
- Set out_of_domain=true ONLY if NO domain plausibly contains relevant material
  (max confidence < 0.4 across all candidates).
- Topic hints: $topic_hints
- Optimized query: $query
```

- [ ] **Step 2: Implement top_router.py**

```python
# backend/app/orchestrator_v2/nodes/top_router.py
from dataclasses import dataclass
from pathlib import Path
from string import Template

from app.services.llm_service import call_openai_json

_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "top_router.txt"
)
_MODEL = "gpt-4.1-mini"


@dataclass
class RouterCandidate:
    name: str
    confidence: float
    reasoning: str


@dataclass
class TopRouterOutput:
    selected_domains: list[RouterCandidate]
    out_of_domain: bool


async def route_to_domains(
    optimized_query: str,
    topic_hints: list[str],
    domain_catalog: list[dict],
) -> TopRouterOutput:
    catalog_text = "\n".join(
        f"- {d['name']}: {d['description']}" for d in domain_catalog
    )
    prompt = Template(_PROMPT_PATH.read_text()).safe_substitute(
        catalog=catalog_text,
        topic_hints=", ".join(topic_hints) or "(none)",
        query=optimized_query,
    )

    raw = await call_openai_json(model=_MODEL, system="", user=prompt)

    candidates = []
    for d in (raw.get("selected_domains") or [])[:2]:
        if not isinstance(d, dict) or not d.get("name"):
            continue
        candidates.append(RouterCandidate(
            name=d["name"],
            confidence=float(d.get("confidence", 0.0)),
            reasoning=str(d.get("reasoning", "")),
        ))

    out_of_domain = bool(raw.get("out_of_domain", False))
    if candidates and max(c.confidence for c in candidates) < 0.4:
        out_of_domain = True

    return TopRouterOutput(selected_domains=candidates, out_of_domain=out_of_domain)
```

- [ ] **Step 3: Test (out_of_domain enforcement + normal routing)**

Write 4 tests covering: 2-domain selection, single-domain, out_of_domain auto-fire when max < 0.4, malformed candidates filtered.

- [ ] **Step 4: Commit**

---

## Task 6: Domain Agent base + protocol

**Files:**
- Create: `backend/app/agents/__init__.py`
- Create: `backend/app/agents/base.py`
- Create: `backend/app/prompts/agent_self_eval.txt`
- Create: `backend/app/prompts/agent_rewriter.txt`
- Test: `backend/tests/test_agent_base.py`

- [ ] **Step 1: Write base.py with the shared contract + sub-graph runner**

```python
# backend/app/agents/base.py
from __future__ import annotations

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

from app.services.embedding_service import generate_embedding_cohere
from app.services.llm_service import call_openai_json
from app.services.pinecone_service import get_factory, retrieve
from app.services.rerank_service import rerank as cohere_rerank

_log = logging.getLogger(__name__)

_SELF_EVAL_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "agent_self_eval.txt"
)
_REWRITER_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "agent_rewriter.txt"
)
_MAX_PASSES = 3
_FINAL_TOP_N = 12
_EVAL_MODEL = "gpt-4.1-mini"
_REWRITER_MODEL = "gpt-4.1-mini"


@dataclass
class RetrievalPlan:
    namespaces: list[str]
    k: int
    metadata_filter: dict | None = None


@dataclass
class DomainAgentInput:
    optimized_query: str
    original_query: str
    topic_hints: list[str]
    memory_context: str | None = None
    max_passes: int = _MAX_PASSES
    public_safe_only: bool = False


@dataclass
class DomainAgentOutput:
    chunks: list[dict[str, Any]] = field(default_factory=list)
    passes_used: int = 0
    refinement_history: list[dict] = field(default_factory=list)
    self_assessment: dict = field(default_factory=dict)
    notes_for_synthesizer: str | None = None


def _chunk_key(c: dict[str, Any]) -> str:
    return hashlib.sha1((c.get("text") or "").encode("utf-8")).hexdigest()


class DomainAgent(ABC):
    """Base domain agent. Subclasses implement only `plan` and define
    `name`, `description`, `pinecone_index`."""

    name: str
    description: str
    pinecone_index: str  # the Pinecone index this agent owns

    @abstractmethod
    def plan(self, inp: DomainAgentInput, pass_num: int) -> RetrievalPlan: ...

    async def run(self, inp: DomainAgentInput) -> DomainAgentOutput:
        chunks_acc: list[dict] = []
        history: list[dict] = []
        verdict = {"verdict": "partial", "confidence": 0.0, "gaps": []}
        current_query = inp.optimized_query
        passes = 0

        for pass_num in range(inp.max_passes):
            plan = self.plan(inp, pass_num)
            history.append({
                "pass": pass_num + 1,
                "query": current_query,
                "namespaces": plan.namespaces,
                "k": plan.k,
            })
            new = await self._retrieve(current_query, plan, inp.public_safe_only)
            # dedup against accumulator
            existing = {_chunk_key(c) for c in chunks_acc}
            for c in new:
                if _chunk_key(c) not in existing:
                    chunks_acc.append(c)
            passes = pass_num + 1

            verdict = await self._self_eval(inp, chunks_acc, pass_num + 1, inp.max_passes)
            if verdict["verdict"] in ("sufficient", "off-topic"):
                break
            if pass_num + 1 >= inp.max_passes:
                break

            current_query = await self._rewrite(inp, verdict)

        # within-domain rerank against original query
        ranked = await cohere_rerank(inp.original_query, chunks_acc, top_n=_FINAL_TOP_N)

        notes = None
        if verdict.get("gaps"):
            notes = f"{self.name} agent gaps: {'; '.join(verdict['gaps'][:3])}"

        return DomainAgentOutput(
            chunks=ranked,
            passes_used=passes,
            refinement_history=history,
            self_assessment=verdict,
            notes_for_synthesizer=notes,
        )

    async def _retrieve(
        self, query: str, plan: RetrievalPlan, public_safe_only: bool,
    ) -> list[dict]:
        """Embed query → Pinecone over plan.namespaces in parallel → flatten."""
        try:
            vector = await asyncio.to_thread(
                generate_embedding_cohere, query, "search_query",
            )
        except Exception as exc:
            _log.warning("embed failed for %s: %s", self.name, exc)
            return []

        async def _query_ns(ns: str) -> list[dict]:
            try:
                factory = get_factory()
                # NOTE: pinecone_service.retrieve doesn't currently accept a
                # metadata filter argument. T9 will extend it. Until then,
                # public_safe is filtered post-fetch.
                results = await asyncio.to_thread(
                    retrieve, factory, self.pinecone_index, "1", vector, plan.k, [ns],
                )
                if public_safe_only:
                    results = [r for r in results
                               if (r.get("metadata") or {}).get("public_safe")]
                # Tag source for citation later
                for r in results:
                    r["source_index"] = self.pinecone_index
                    r["source_namespace"] = ns
                return results
            except Exception as exc:
                _log.warning("retrieve ns=%s failed for %s: %s", ns, self.name, exc)
                return []

        nss = plan.namespaces or ["default"]
        chunk_lists = await asyncio.gather(*[_query_ns(ns) for ns in nss])
        merged = [c for lst in chunk_lists for c in lst]
        merged.sort(key=lambda c: c["score"], reverse=True)
        return merged[: plan.k]

    async def _self_eval(
        self, inp: DomainAgentInput, chunks: list[dict],
        pass_num: int, max_passes: int,
    ) -> dict:
        chunks_text = "\n\n".join(
            f"[{i+1}] (ns: {c.get('source_namespace','?')}, score: {c['score']:.2f})\n"
            f"{(c.get('text') or '')[:600]}"
            for i, c in enumerate(chunks[:10])
        ) or "(no chunks retrieved)"
        prompt = Template(_SELF_EVAL_PROMPT.read_text()).safe_substitute(
            domain_name=self.name,
            original_query=inp.original_query,
            optimized_query=inp.optimized_query,
            topic_hints=", ".join(inp.topic_hints) or "(none)",
            pass_number=pass_num,
            max_passes=max_passes,
            chunks=chunks_text,
        )
        try:
            return await call_openai_json(
                model=_EVAL_MODEL, system="", user=prompt, temperature=0,
            )
        except Exception:
            return {"verdict": "sufficient", "confidence": 0.5, "gaps": []}

    async def _rewrite(self, inp: DomainAgentInput, verdict: dict) -> str:
        prompt = Template(_REWRITER_PROMPT.read_text()).safe_substitute(
            original_query=inp.original_query,
            optimized_query=inp.optimized_query,
            domain_name=self.name,
            gaps="; ".join(verdict.get("gaps", [])) or "(none)",
        )
        try:
            raw = await call_openai_json(
                model=_REWRITER_MODEL, system="", user=prompt, temperature=0,
            )
            return str(raw.get("refine_query") or inp.optimized_query)
        except Exception:
            return inp.optimized_query
```

- [ ] **Step 2: Write `agent_self_eval.txt` and `agent_rewriter.txt`** (verbatim from spec §5.2 and §5.3)

- [ ] **Step 3: Tests** — mock all I/O. Verify the loop halts on `sufficient`, max-passes cap fires, dedup works.

- [ ] **Step 4: Commit**

---

## Task 7: Per-domain agent modules (7 agents)

Each agent is a thin wrapper. Same structure for all 7 — differs only in `name`, `description`, `pinecone_index`, and `plan` logic.

**Files (one task each, but grouped here for brevity):**
- `backend/app/agents/strategy.py`
- `backend/app/agents/marketing.py`
- `backend/app/agents/voice.py`
- `backend/app/agents/personal.py`
- `backend/app/agents/tools_ops.py`
- `backend/app/agents/content_library.py`
- `backend/app/agents/verticals.py`

Example `marketing.py` (use as template — write all 7 with their own namespace lists from spec §7.1):
```python
from app.agents.base import DomainAgent, DomainAgentInput, RetrievalPlan


class MarketingAgent(DomainAgent):
    name = "marketing"
    description = (
        "Marketing tactics + case studies: email, ads, funnels, sales copy, "
        "marketing case studies. Choose when query is about how to acquire, "
        "convert, or message to customers."
    )
    pinecone_index = "max-marketing"
    _NAMESPACES = ["email", "ads", "funnels", "sales-copy", "case-studies"]

    def plan(self, inp: DomainAgentInput, pass_num: int) -> RetrievalPlan:
        hints_lc = " ".join(inp.topic_hints).lower()
        # Prefer case-studies when a vertical is named
        if any(v in hints_lc for v in ["telemed", "local", "saas", "coaching"]):
            return RetrievalPlan(namespaces=["case-studies"] + self._NAMESPACES,
                                  k=15 if pass_num == 0 else 8)
        # Channel keywords prefer that channel
        for ch in ["email", "ads", "funnels", "sales-copy"]:
            if ch.replace("-", " ") in hints_lc or ch in hints_lc:
                return RetrievalPlan(namespaces=[ch] + self._NAMESPACES,
                                      k=12 if pass_num == 0 else 6)
        # Default: broad
        return RetrievalPlan(namespaces=self._NAMESPACES,
                              k=15 if pass_num == 0 else 8)
```

Each of the 7 agents should be its own task with a few tests verifying the planner picks the right namespaces given specific topic_hints.

Commit each agent independently so the diff is reviewable.

---

## Task 8: Domain agent registry + dispatcher

**Files:**
- Create: `backend/app/agents/registry.py`
- Test: `backend/tests/test_agent_registry.py`

```python
# backend/app/agents/registry.py
"""Maps domain name → DomainAgent class. Used by orchestrator's
run_domain_agents node to instantiate the right agents."""
from app.agents.base import DomainAgent
from app.agents.content_library import ContentLibraryAgent
from app.agents.marketing import MarketingAgent
from app.agents.personal import PersonalAgent
from app.agents.strategy import StrategyAgent
from app.agents.tools_ops import ToolsOpsAgent
from app.agents.verticals import VerticalsAgent
from app.agents.voice import VoiceAgent

_REGISTRY: dict[str, type[DomainAgent]] = {
    "strategy": StrategyAgent,
    "marketing": MarketingAgent,
    "voice": VoiceAgent,
    "personal": PersonalAgent,
    "tools-ops": ToolsOpsAgent,
    "content-library": ContentLibraryAgent,
    "verticals": VerticalsAgent,
}


def get_agent(name: str) -> DomainAgent:
    cls = _REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown domain agent: {name}")
    return cls()


def domain_catalog() -> list[dict]:
    """Catalog for the Top Router prompt."""
    return [
        {"name": agent_cls.name, "description": agent_cls.description}
        for agent_cls in _REGISTRY.values()
    ]
```

---

## Task 9: Conversation memory recall + persistence

**Files:**
- Create: `backend/app/orchestrator_v2/nodes/recall_memory.py`
- Create: `backend/app/orchestrator_v2/nodes/persist_memory.py`
- Test: `backend/tests/test_memory.py`

Memory index name: `conversation-memory`. Per-user namespace.

Recall: embed optimized query → Pinecone search filtered by user_id namespace → top-5 prior turn-summaries.

Persist: at end of pipeline, embed `f"Q: {raw_query}\nA: {summarized_response[:500]}"` and upsert.

---

## Task 10: Verifier node

**Files:**
- Create: `backend/app/orchestrator_v2/nodes/verify.py`
- Test: `backend/tests/test_verify_node.py`

Wrap existing `verifier.txt` prompt. Returns `{verdict, confidence, reasoning}` — synthesizer compares confidence vs mode's `refusal_threshold`.

---

## Task 11: Synthesizer node (streaming)

**Files:**
- Create: `backend/app/orchestrator_v2/nodes/synthesize.py`
- Create: `backend/app/prompts/synthesizer_v2.txt`
- Test: `backend/tests/test_synthesize_node.py`

Mode-aware:
- Picks model from `MODES[mode]["synthesizer_model"]`
- Picks rules block from `MODES[mode]["rules"]`
- Streams via `anthropic.AsyncAnthropic.messages.stream`
- Returns `AsyncIterator[str]` of token deltas

The synthesizer prompt template uses `$style_guide`, `$memory_context`, `$context`, `$gaps_summary`, `$query`, `$mode_rules`.

---

## Task 12: RRF merge + cross-domain rerank node

**Files:**
- Create: `backend/app/orchestrator_v2/nodes/fuse_and_rerank.py`
- Test: `backend/tests/test_fuse_and_rerank.py`

RRF k=60, then `cohere_rerank` against original query → top-10.

---

## Task 13: LangGraph orchestrator wiring

**Files:**
- Create: `backend/app/orchestrator_v2/graph.py`
- Test: `backend/tests/test_orchestrator_v2_graph.py`
- Modify: `backend/requirements.txt` (add `langgraph>=0.2,<0.3`)

Compose:
```
classify → optimize → (recall_memory ‖ top_router) → run_domain_agents (parallel) →
fuse_and_rerank → verify → synthesize → persist
```

Use LangGraph's parallel edges for the two `(recall, route)` and `(N domain agents)` gather points.

State spec: see architecture §6.3.

Output: `AsyncIterator[str]` of token deltas + final meta event.

---

## Task 14: API integration — feature-flag-gated v2 endpoint

**Files:**
- Modify: `backend/app/routers/chat.py`
- Test: `backend/tests/test_chat_v2_endpoint.py`

Add query param `engine: Literal["v1", "v2"] = "v1"` to existing `POST /api/chat`. When v2, call orchestrator_v2; otherwise existing pipeline.

Terminal SSE event `event: meta` with classification, citations, agent_traces (see spec §8.2).

---

## Task 15: Restream + sources endpoints

**Files:**
- Modify: `backend/app/routers/chat.py`
- Test: `backend/tests/test_restream_sources.py`

```
POST /api/chat/messages/{message_id}/restream  body: {mode}
GET  /api/chat/messages/{message_id}/sources
```

Restream reads `messages.cached_context`. Returns 410 if `cached_until` expired.

---

## Task 16: Frontend mode badge + citation chips + panels

**Files:**
- Create: `frontend/src/components/ModeBadge.tsx`
- Create: `frontend/src/components/CitationChip.tsx`
- Create: `frontend/src/components/SourcePanel.tsx`
- Create: `frontend/src/components/DebugPanel.tsx`
- Modify: `frontend/src/components/MessageBubble.tsx`
- Modify: `frontend/src/components/ChatWindow.tsx`
- Modify: `frontend/src/lib/api.ts` (parse meta event, restream call)

ModeBadge: pill showing `<mode> • <conf>`, amber when confidence < 0.7, click opens dropdown.

CitationChip: react-markdown plugin replaces `[N]` with chip components.

SourcePanel: collapsible, auto-expanded in public mode.

DebugPanel: collapsed by default, shows classifier output + per-agent traces.

---

## Task 17: Frontend wires `?engine=v2`

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/app/chat/page.tsx`

Pass `engine=v2` if URL has `?v2=true` or user has opted into v2 in localStorage. This is the Phase 1 opt-in mechanism.

---

## Task 18: Eval harness

**Files:**
- Create: `backend/evals/__init__.py`
- Create: `backend/evals/cases.py`
- Create: `backend/evals/runner.py`
- Create: `backend/evals/baseline.json`
- Create: `.github/workflows/evals.yml`

`cases.py` defines 30 test cases (see spec §11.3 — 10 regression, 8 mode-classification, 5 cross-domain, 4 out-of-domain, 3 voice consistency).

`runner.py`:
- For each case, calls `orchestrator_v2` against the live v2 pipeline (or a `test` env clone)
- Asserts: `expected_mode`, `expected_domains_hit`, `must_cite`, `must_not_say`, `should_say`
- An LLM judge (Claude Sonnet) scores the `judge_criteria` 1–5
- Aggregates pass-rate; writes JSON report

`baseline.json` stores the current pass-rate. CI fails any PR that drops it.

GitHub Actions workflow:
- Trigger: nightly + on PR
- Run: `python -m backend.evals.runner --report report.json`
- Compare to baseline; fail if regression

---

## Task 19: Smoke test against live Pinecone

**Files:** None — operational

Once everything above is shipped:
- Open `https://chat.maxmayes.io/chat?v2=true`
- Run the D2C-style failure query: "Help me come up with a good D2C marketing strategy"
- Verify: classifier picks correct mode, domains routed sensibly, out-of-domain refusal fires honestly OR retrieved chunks land + synthesis cites them
- Run a known-good query in your wheelhouse: "what do I think about cold email open rates?"
- Verify: response is grounded, cites chunks, mode badge shows correct mode

---

## Task 20: Phase 2 prep — eval baseline freeze

**Files:**
- Modify: `backend/evals/baseline.json`

Run the eval suite against v2, freeze the pass-rate as baseline. Phase 2 will flip default; baseline ensures the flip doesn't regress.

```bash
cd backend && python3 -m backend.evals.runner --write-baseline
git add backend/evals/baseline.json
git commit -m "evals: freeze Phase 1 baseline pass-rate"
```

---

## Phase 1 acceptance test

- [ ] All unit tests pass (`pytest tests/test_orchestrator_v2_* tests/test_classify_node*` etc.)
- [ ] Eval suite pass-rate ≥ 80% across all categories, 100% on regression cases
- [ ] D2C query returns a clean out-of-domain refusal OR a multi-domain response (not a hedged "local marketing" answer)
- [ ] Mode badge appears in UI when `?v2=true`
- [ ] Citation chips render and are clickable
- [ ] No regression in v1 behavior (`/api/chat` without engine flag works as before)

---

## Self-review

- All 39 FRs from architecture §2 are addressed (Tasks 1-17 map 1:1 to FR-CL/OPT/MEM/RT/DA/RRF/VER/SYN/PER/UI/API/FF prefixes).
- Eval suite (T18) covers NFR-QUAL-1..6.
- No orchestrator/agent code touches the `voice` Pinecone outside the `VoiceAgent` retrieve path (NFR-QUAL voice-leak fix).
- Feature flag (T14/T17) keeps v1 default until Phase 2.
- Type names consistent (`DomainAgentInput`, `DomainAgentOutput`, `ClassifierOutput`, `RouterCandidate`, `TopRouterOutput`).

---

## Execution Handoff

Same options as Phase 0:
1. Subagent-Driven (recommended) — one subagent per task, two-stage review
2. Inline Execution — same-session checkpoints

Phase 1 is ~20 tasks vs Phase 0's 20 — but Tasks 7 (agent modules) and 18 (eval harness) are themselves multi-step. Realistic effort: ~6–8 dev days, vs Phase 0's ~3.

Do NOT start Phase 1 until Phase 0 manual checkpoints (T17–T20 of `2026-05-10-rag-redesign-phase0.md`) are complete and 7 domain indexes are populated in Pinecone with metadata. Phase 1 depends on that.
