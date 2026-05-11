# RAG Redesign — Hierarchical Multi-Agent Pipeline with Mode-Aware Synthesis

**Status:** Draft, awaiting user approval
**Author:** Max Mayes (with Claude Code as scribe)
**Date:** 2026-05-10
**Project:** Chat with My Pinecone (`maxmayes-chat`)
**v2 deferrals tracked in:** `docs/superpowers/v2-backlog.md`

---

## 1. Problem

The current pipeline (single optimizer → single router → flat retrieve → verify → synthesize) cannot reliably surface relevant content from a 40-index, ~50K-record corpus. Three concrete symptoms:

1. **Voice profile leaks topic content into synthesis.** The synthesizer prompt receives 5 chunks pulled from the `max-copywriting-voice` Pinecone using the same hardcoded seed query (`"writing style tone voice communication"`) on every request. Those chunks happen to be heavily local-marketing-flavored. Combined with the rule "every statement must be grounded in the provided context," the model treats voice chunks as canonical content and frames every answer through local marketing — even for D2C, telemed, or other unrelated topics. Confirmed via inspection of `orchestrator.py:104` and `voice_synthesizer.txt`.

2. **Single-domain routing misses multi-domain queries.** The router picks one index per query. The fallback gate (`confidence < 0.7`) almost never fires for queries that have plausible single-domain matches even when the answer requires synthesizing across multiple domains. Result: shallow answers that look like they came from one corner of the corpus.

3. **No depth iteration.** Each query gets one Pinecone pass at `top_k=10`. If pass 1 misses the right chunks, the system has no recovery mechanism. The verifier can only refuse, not refine.

The corpus side compounds these issues: 40 indexes were created over time without consistent chunking, embedding model, metadata, or naming. Indexes are described inaccurately (the n8n workflow's prompts referenced indexes that didn't exist). Even a perfect orchestrator wouldn't recover from a corpus this disorganized.

## 2. Goals

**Primary**
- Replace single-router pipeline with hierarchical multi-agent orchestration that iterates retrieval per domain (up to 3 passes) and merges results across domains.
- Reorganize the corpus into 5–7 well-described domain indexes with consistent embedding model, chunking, and metadata.
- Eliminate the voice-profile leak by replacing dynamic voice retrieval with a static style guide. Voice content remains retrievable only when queries are genuinely about voice/style.
- Support three synthesis modes (personal / content_creation / public) auto-detected per query, each with its own prompt, refusal threshold, and citation policy.
- Add per-message classification, citation tracking, and agent observability to Postgres so Max can debug surprising responses.

**Secondary**
- Cross-session conversational memory via a `conversation-memory` Pinecone index with semantic recall.
- Inline `[N]` citation chips in the chat UI with expandable source panel and a debug panel showing classifier output, agent traces, and latency.
- Mode override badge per message; manual choice "stickies" for the rest of the session.
- Re-streamable responses when mode is switched manually (chunks cached for 5 min by message_id).

**Non-goals (v1)**
- Memory retention/decay policy (deferred to v2 — index grows unbounded for v1).
- Manual persona dropdown (classifier auto-detects only).
- Multi-user role-based access beyond Clerk's existing auth.
- Per-mode style-guide variants.

## 3. Success Criteria

**Quality (measured by eval suite, see §11)**
- 100% of regression cases pass: D2C-style query, telemed-affiliate query, and any "wrong-map" failure mode no longer produces local-marketing-framed answers.
- ≥85% of mode-classification cases pass: detected mode matches expected mode.
- ≥80% of cross-domain cases route to all expected domains.
- 100% of out-of-domain cases refuse cleanly per their mode (public mode strictest, personal mode softest).
- Voice consistency: same factual question asked in different modes produces same factual content with mode-appropriate register.

**Operational**
- Per-query latency: ≤2.6s to first synthesis token, ≤10s to completion (worst case, 3-pass agents). Current pipeline is ≤1.5s / ≤7s — accepting +1s for the quality lift.
- Per-query cost: ≤$0.020 (currently ~$0.006). 2.7× increase, dominated by Opus 4.7 in content_creation/public modes.
- Zero data loss during corpus migration (legacy archive retained 30 days in Pinecone, then exported to S3/R2 indefinitely).
- Eval pass-rate cannot drop below baseline on any PR (CI gate).

## 4. Architecture

### 4.1 Component Diagram

```
                         ┌───────────────────────────────┐
                         │  Vercel: Next.js chat UI      │
                         │  (chat.maxmayes.io)           │
                         │   - Mode badge per message    │
                         │   - Inline citation chips     │
                         │   - Source + debug panels     │
                         └──────────────┬────────────────┘
                                        │  POST /api/chat (SSE)
                                        ▼
              ┌──────────────────────────────────────────────────┐
              │  FastAPI on Railway                              │
              │  ─ Auth (Clerk JWT) ─ Sessions ─ SSE plumbing    │
              └──────────────┬───────────────────────────────────┘
                             ▼
              ┌──────────────────────────────────────────────────┐
              │  LangGraph Orchestrator (orchestrator_v2)        │
              │                                                  │
              │   1. Classify ─────► {mode, intent, hints}       │
              │   2. Optimize ─────► dense search query          │
              │   3. Recall Memory (parallel w/ #4)              │
              │   4. Top Router ───► picks 1-2 Domain Agents     │
              │                                                  │
              │   5. DOMAIN AGENTS (parallel)                    │
              │      Each is its own Python module + sub-graph:  │
              │        Plan → Retrieve → Self-Eval → (Refine)*   │
              │        → Finalize (within-domain rerank)         │
              │      Up to 3 passes per agent                    │
              │                                                  │
              │   6. Cross-domain RRF + Cohere Rerank            │
              │   7. Verify (sufficient context per mode?)       │
              │   8. Synthesize per mode (Sonnet or Opus)        │
              │      ─ tagged citations [N]                      │
              │      ─ static style guide (no voice leak)        │
              │      ─ STREAMING SSE                             │
              │   9. Persist (messages, citations, agent_traces, │
              │      conversation-memory index)                  │
              └──────────────┬───────────────────────────────────┘
                             ▼
                   ┌──────────────────────┐
                   │  Postgres (Railway)  │
                   │   ─ sessions         │
                   │   ─ messages         │
                   │   ─ message_classifications │
                   │   ─ citations        │
                   │   ─ agent_traces     │
                   │   ─ index_registry   │
                   │   ─ index_audits     │
                   │   ─ ingest_jobs      │
                   └──────────────────────┘
```

### 4.2 Pipeline data flow (canonical example)

User message: *"Help me think through how to position the telemed offer for the Q3 launch."*

| Step | Component | Latency budget | What happens |
|---|---|---|---|
| 1 | Classify | 150ms | gpt-4.1-mini reads message + last 4 turns. Returns `{mode: "content_creation", intent: "strategy_synthesis", topic_hints: ["telemed", "Q3 launch", "offer positioning"], persona_target: "B2B founder", out_of_domain_score: 0.05}` |
| 2 | Optimize | 200ms | gpt-4.1-mini → `"telemed offer positioning Q3 launch strategy positioning frameworks healthcare DTC"` |
| 3 | Recall Memory | 300ms (parallel w/ 4) | Embed query (Cohere v3), search `conversation-memory` index filtered by `user_id`, top-5. Summarize relevant prior turns into ~150 tokens. |
| 4 | Top Router | 250ms | gpt-4.1-mini reads catalog of 7 domain agents, returns `{selected_domains: [{name: "marketing", confidence: 0.92}, {name: "verticals", confidence: 0.78}], out_of_domain: false}` |
| 5 | Domain Agents (parallel) | ~1500ms | Each runs its sub-graph: Plan → Retrieve (Cohere embed → Pinecone) → Self-Eval → optionally Refine → Finalize (within-domain Cohere rerank, dedup, top-12). Cap 3 passes per agent. Detailed in §5. |
| 6 | Cross-domain RRF + Rerank | 400ms | Concat top-12 lists from each agent. RRF merge (k=60, rank-only). Cohere Rerank-v3.5 → top-10. |
| 7 | Verify | 250ms | gpt-4.1-mini judges if chunks support an answer. Returns `proceed`, `proceed_with_caveat`, or `insufficient_context`. Threshold per mode (see §6.3). |
| 8 | Synthesize | First token 600-1000ms; full 3-8s | Mode-aware prompt → Sonnet 4.6 (personal) or Opus 4.7 (content_creation, public). Streams via SSE. Emits `[N]` citation tags inline. |
| 9 | Persist | ~100ms async | Insert into `messages`, `message_classifications`, `citations`, `agent_traces`. Upsert summary chunk into `conversation-memory` index. |

**Total cold path:** ~2.0s before first synthesis token, ~5–10s for full response.

## 5. Domain Agents

The system has 7 domain agents in v1: `strategy`, `marketing`, `voice`, `personal`, `tools-ops`, `content-library`, `verticals` (verticals may collapse to namespaces under `marketing` if Phase-0 audit reveals low volume per vertical).

Each agent is **its own Python module** at `app.agents.<domain_name>` with a class implementing the `DomainAgent` protocol:

```python
class DomainAgentInput:
    optimized_query: str
    original_query: str
    topic_hints: list[str]
    memory_context: str | None
    max_passes: int = 3

class DomainAgentOutput:
    chunks: list[Chunk]
    passes_used: int
    refinement_history: list[str]
    self_assessment: dict
    notes_for_synthesizer: str | None
```

### 5.1 Sub-graph internals

```
Domain Agent: <name>
─────────────────────────────────────────────────────
   ┌──────────────────────────────────┐
   │  Plan                            │
   │  Per-agent custom logic:         │
   │  - which namespaces to hit       │
   │  - per-namespace k               │
   │  - metadata filters              │
   │  Returns: RetrievalPlan          │
   └──────────────┬───────────────────┘
                  ▼
   ┌──────────────────────────────────┐
   │  Retrieve (parallel namespaces)  │
   │  Embed (Cohere v3) → Pinecone    │
   └──────────────┬───────────────────┘
                  ▼
   ┌──────────────────────────────────┐
   │  Self-Evaluate                   │
   │  gpt-4.1-mini, JSON output       │
   │  {verdict, confidence, gaps}     │
   └──────────────┬───────────────────┘
                  ▼
        ┌──────────────────────┐
        │ verdict?             │
        └──┬──────────────┬────┘
           │              │
   sufficient        partial
   or off-topic     & passes < max
           │              │
           │              ▼
           │   ┌───────────────────────┐
           │   │  Rewriter             │
           │   │  Separate LLM call:   │
           │   │  takes gaps + query   │
           │   │  → focused refined    │
           │   │  query + ns targets   │
           │   └───────────┬───────────┘
           │               │
           │   ┌───────────▼───────────┐
           │   │  Refine plan          │
           │   │  - new query          │
           │   │  - tighter k (halved) │
           │   │  - target namespaces  │
           │   └───────────┬───────────┘
           │               │
           │               └─► loop back to Retrieve
           ▼
   ┌──────────────────────────────────┐
   │  Finalize                        │
   │  - Dedup chunks across passes    │
   │  - Within-domain Cohere rerank   │
   │  - Trim to top-12                │
   │  - Build self_assessment block   │
   │  - Build notes_for_synthesizer   │
   └──────────────┬───────────────────┘
                  ▼
       returns DomainAgentOutput
```

### 5.2 Self-Evaluation prompt (shared template)

```
Domain: {domain_name}
Original query: {original_query}
Optimized query: {optimized_query}
Topic hints: {topic_hints}
Pass: {pass_number} of {max_passes}
Chunks retrieved this pass:
[1] (ns: <ns>, score: 0.81) <text>
[2] (ns: <ns>, score: 0.79) <text>
...

Return JSON:
{
  "verdict": "sufficient" | "partial" | "off-topic",
  "confidence": 0.0-1.0,
  "gaps": ["short description of what's missing", ...]
}

Rules:
- "sufficient" = chunks could ground a good answer to the original query
- "partial" = relevant but key angle missing; gaps[] must be non-empty
- "off-topic" = wrong domain; this agent is a poor fit
- Be honest. Marking partial when chunks are fine wastes a pass.
```

### 5.3 Rewriter prompt (separate LLM call when verdict=partial)

Quality-over-cost decision logged in v2-backlog.md. A separate prompt produces tighter refined queries than asking one model to evaluate AND rewrite in one JSON output.

```
Original query: {original_query}
Optimized query that was searched: {optimized_query}
Domain: {domain_name}
Available namespaces: {namespaces}
Gaps identified by self-eval: {gaps}

Return JSON:
{
  "refine_query": "rephrased query targeting the gaps",
  "namespaces_to_target": ["..."] | null
}

Make the refined query maximally specific to the gaps.
Halve the breadth of the original — go deep, not wide.
```

### 5.4 Refine logic

When verdict is `partial` and `passes_used < max_passes`:
1. Use `refine_query` from rewriter (not original).
2. If `namespaces_to_target` is set, restrict to those namespaces.
3. Halve `k` from previous pass (focused, not broad — broad chunks already retrieved).
4. Carry forward chunks from prior passes (deduped by SHA1 of text).

After pass `max_passes` (3 in v1) the agent finalizes regardless of verdict — no infinite loops. If the final verdict is still `partial`, the unmet gaps are surfaced to the synthesizer in `notes_for_synthesizer`.

### 5.5 Two reranks (within-domain + cross-domain)

| Rerank | Where | Question answered |
|---|---|---|
| Within-domain | Each agent's Finalize step | "Of MY chunks across all passes, which are most relevant?" |
| Cross-domain | Orchestrator step 6 | "Across all domains' top picks, which are most relevant?" |

Both use Cohere Rerank-v3.5 against the **original query** (not the optimized or refined queries). RRF merge in step 6 is rank-only, so the cross-domain rerank is what restores quality-aware ranking after the merge.

### 5.6 Failure modes

| Failure | Behavior |
|---|---|
| Pinecone times out for one namespace | Skip that namespace, log, continue |
| All Pinecones for an agent fail | Return empty chunks + `notes_for_synthesizer="domain X retrieval failed"` |
| Self-eval LLM returns malformed JSON | Default to `verdict="sufficient"`, no refine |
| Pass 3 still says `partial` | Finalize anyway, surface `gaps` to synthesizer |
| Cohere rerank API outage | Graceful fallback to score-sort; log warning |
| Top Router picks an agent whose Pinecones are all empty | Agent returns empty + note; verifier picks it up |

## 6. Mode-Aware Synthesis

### 6.1 Mode detection

Classifier (step 1 in §4.2) returns:

```python
class ClassifierOutput:
    mode: Literal["personal", "content_creation", "public"]
    mode_confidence: float
    intent: Literal[
        "factual_lookup",
        "strategy_synthesis",
        "drafting",
        "recall_with_context",
        "exploration",
    ]
    topic_hints: list[str]
    out_of_domain_score: float
    persona_target: str | None
```

**Detection rules** (encoded in classifier prompt with examples):

| Signal | Mode |
|---|---|
| First-person directives ("draft me", "write a") | `content_creation` |
| Third-person reference ("a client asked", "for our audience") | `public` |
| Self-referential ("what did I", "what's my take on") | `personal` |
| Ambiguous strategy questions | `personal` (default — Max talking to himself) |
| Manual mode override from UI | Always wins; classifier ignored |

**Mode stickiness**: once the user manually selects a mode in the UI, that choice persists for the rest of the session. The classifier can still override on a subsequent message only if `mode_confidence > 0.85`.

### 6.2 Synthesizer prompt skeleton (shared across modes)

```
You are writing as Max Mayes.

[STYLE GUIDE — match cadence, tone, vocabulary; do NOT borrow topics or facts]
{static_max_style_guide}

[CONVERSATION CONTEXT — recent + recalled]
{memory_context}

[KNOWLEDGE — only source of factual content]
[1] (domain/namespace, source_type, date)
    {chunk_text}
[2] (domain/namespace, source_type, date)
    {chunk_text}
...

[GAPS NOTED BY DOMAIN AGENTS]
{gaps_summary}

[USER QUESTION]
{original_query}

[MODE-SPECIFIC RULES]
{mode_rules}
```

### 6.3 Mode-specific rules and refusal thresholds

| Mode | Model | Refusal threshold | Citations | Profanity allowed | Behavior on insufficient context |
|---|---|---|---|---|---|
| Personal | Sonnet 4.6 | 0.45 | Off by default; natural mentions OK | Yes | Partial answer + flag what's missing |
| Content creation | Opus 4.7 | 0.50 | `[N]` inline, required | Yes | Draft with `[need source]` markers for unsupported claims |
| Public | Opus 4.7 | 0.65 | `[N]` inline, always | Yes (still grounded — can't invent) | Refuse cleanly: "I don't have enough verified material to answer this for a public audience." Filter chunks by `metadata.public_safe == true`. |

### 6.4 Static style guide (kills the voice leak)

Lives in code, single source of truth:

```python
MAX_STYLE_GUIDE = """
- Direct, opinionated, no hedging. Lead with the punch line, then explain.
- Short paragraphs. One-line sentences for emphasis.
- Speaks in second person to the reader ("you").
- Uses contractions. Occasional emphatic profanity ("fucking", "dumb").
- Refuses to bullshit. Honest about what he doesn't know.
- Concrete > abstract. Names specific tactics, frameworks, percentages, dollar amounts where relevant.
- Avoids: buzzwords, MBA-speak, hedge-words ("perhaps", "potentially", "it depends").
- Hooks: starts with a strong claim, a counterintuitive observation, or "look, here's the thing".
- Closes: with a clear next move or a question that puts the ball back in the reader's court.
"""
```

The `max-voice` Pinecone is NOT deleted. It becomes a retrievable domain like any other — when a query is genuinely about Max's voice/style ("how do I write more like Max?"), the Top Router picks the `voice` agent and its chunks land in the KNOWLEDGE block legitimately, with citations. Voice content can no longer be silently injected as topic content because it's only retrieved when relevant.

### 6.5 Honesty primitives (in every mode)

1. **No invention.** Claims must come from KNOWLEDGE block or conversation context.
2. **Name the gap.** If question requires X and only Y is available, say so explicitly.
3. **Source attribution honesty.** Cite each chunk that supports a claim, don't generalize weak chunks into confident statements.

### 6.6 Classification-miss safety nets

- **Soft miss** (`mode_confidence` 0.5–0.7): Mode badge in UI is amber-tinted. User can one-click switch and the response re-streams in the new mode (chunks cached 5 min by message_id).
- **Hard miss** (`mode_confidence < 0.5`): Default to `personal` mode (lowest stakes), badge surfaces prominently.

## 7. Storage & Data Model

### 7.1 Pinecone schema (post-Phase-0)

All domain indexes use Cohere `embed-v3.0` (1024-dim, cosine).

| Index | Namespaces | Purpose |
|---|---|---|
| `max-strategy` | `decision-frameworks`, `leadership`, `founder-notes`, `transcripts` | Business strategy, leadership, founder thinking |
| `max-marketing` | `email`, `ads`, `funnels`, `sales-copy`, `case-studies` | Marketing tactics + case studies |
| `max-voice` | `voice-samples`, `style-references` | Retrievable voice/style content |
| `max-personal` | `journal`, `relationships`, `health`, `travel` | Personal experiences and context |
| `max-tools-ops` | `n8n-workflows`, `ai-tools`, `infrastructure`, `prompts` | Implementation, automation, AI tooling |
| `max-content-library` | `youtube`, `podcast`, `blog-drafts`, `social` | Content artifacts |
| `max-verticals` | `telemed`, `local-business`, `coaching`, `saas`, `other` | Industry-specific (audit may collapse to `max-marketing` namespaces) |
| `conversation-memory` | `{user_id}` per namespace | Cross-session semantic recall |
| `legacy-archive-{date}` | (read-only) | 30-day snapshot of original 40 indexes; exported to S3/R2 then deleted |

### 7.2 Standard chunk metadata (every vector)

```python
{
  "text": str,
  "source_type": str,             # "email" | "transcript" | "note" | "tweet" | "podcast" | ...
  "source_id": str | None,
  "source_url": str | None,
  "ingested_at": str,             # ISO timestamp
  "domain": str,                  # one of the 7 domain names
  "namespace": str,               # sub-category within domain
  "topic_tags": list[str],        # 3-5 LLM-derived tags
  "public_safe": bool,            # gates Public mode; default false
  "embedding_model": "cohere-embed-v3",
  "chunk_position": int,
  "doc_total_chunks": int,
}
```

### 7.3 Postgres schema (new tables)

```sql
CREATE TABLE message_classifications (
  message_id UUID PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
  detected_mode TEXT NOT NULL,
  mode_confidence REAL NOT NULL,
  user_override BOOLEAN NOT NULL DEFAULT false,
  intent TEXT NOT NULL,
  topic_hints JSONB NOT NULL DEFAULT '[]'::jsonb,
  persona_target TEXT,
  out_of_domain_score REAL NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE citations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  source_index TEXT NOT NULL,
  source_namespace TEXT NOT NULL,
  source_type TEXT NOT NULL,
  source_id TEXT,
  source_url TEXT,
  chunk_text TEXT NOT NULL,
  chunk_score REAL NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (message_id, position)
);
CREATE INDEX citations_message_id_idx ON citations(message_id);

CREATE TABLE agent_traces (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  domain_name TEXT NOT NULL,
  passes_used INTEGER NOT NULL,
  refinement_history JSONB NOT NULL,
  self_assessment JSONB NOT NULL,
  notes_for_synthesizer TEXT,
  chunks_returned INTEGER NOT NULL,
  total_latency_ms INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX agent_traces_message_id_idx ON agent_traces(message_id);

CREATE TABLE index_audits (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  audit_date DATE NOT NULL,
  index_name TEXT NOT NULL,
  project_id TEXT NOT NULL,
  record_count INTEGER NOT NULL,
  embedding_model TEXT,
  dominant_domain TEXT,
  topic_tags JSONB NOT NULL DEFAULT '[]'::jsonb,
  sample_chunks JSONB NOT NULL,
  proposed_disposition TEXT NOT NULL,
  proposed_target_index TEXT,
  approved_disposition TEXT,
  approved_at TIMESTAMPTZ,
  executed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX index_audits_unique ON index_audits(audit_date, index_name, project_id);

CREATE TABLE ingest_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_index TEXT NOT NULL,
  target_index TEXT NOT NULL,
  target_namespace TEXT NOT NULL,
  status TEXT NOT NULL,
  total_chunks INTEGER,
  processed_chunks INTEGER NOT NULL DEFAULT 0,
  failed_chunks INTEGER NOT NULL DEFAULT 0,
  error_message TEXT,
  config JSONB NOT NULL,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ingest_jobs_status_idx ON ingest_jobs(status);
```

Plus added columns on existing `index_registry`:
```sql
ALTER TABLE index_registry
  ADD COLUMN domain TEXT,
  ADD COLUMN public_safe_default BOOLEAN NOT NULL DEFAULT false,
  ADD COLUMN agent_module_path TEXT,
  ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
```

And a `cached_context` column on `messages` for re-stream support:
```sql
ALTER TABLE messages
  ADD COLUMN cached_context JSONB,           -- chunks + memory_context for 5-min re-stream
  ADD COLUMN cached_until TIMESTAMPTZ;
CREATE INDEX messages_cached_until_idx ON messages(cached_until)
  WHERE cached_until IS NOT NULL;
```

A periodic cleanup job nulls expired `cached_context`.

### 7.4 Re-ingest pipeline

CLI: `backend/scripts/reingest.py <ingest_job_id>`. Steps:

1. Fetch source chunks (paginate Pinecone).
2. Reconstruct documents (group by `source_id` if available).
3. Re-chunk semantically (LangChain `SemanticChunker` + Cohere embed-v3, ~512 token target).
4. Enrich metadata via LLM (gpt-4.1-mini): `topic_tags`, `source_type` (if missing), `public_safe` (sensitivity heuristic).
5. Embed with Cohere v3 (batch 96 chunks per call).
6. Upsert to target Pinecone index + namespace.
7. Update `ingest_jobs` row with stats.

Idempotent: re-running picks up at `processed_chunks` offset. Per-chunk failures log and continue; >5% failure rate halts the job.

## 8. UI + API Changes

### 8.1 Frontend changes

**Mode badge** on each assistant message:
- Pill next to "Max" showing `<mode> • <confidence>`
- Click → dropdown: Personal / Content / Public
- Confidence < 0.7 → amber-tinted
- Switch triggers re-stream (cached chunks); manual choice sticks for the session

**Inline citation chips**:
- `[N]` markers in response wrapped as interactive chips by react-markdown plugin
- Hover: tooltip with chunk preview + source label
- Click: scroll/expand source panel for that citation

**Source panel** (collapsed by default, auto-expanded in public mode):
- Lists all citations in order with chunk text, source label (`max-marketing/case-studies`), source URL if any
- "Open in admin" link → existing `/admin/indexes/{id}` route

**Debug panel** (collapsed by default):
- Classifier output (mode, intent, confidence, topic_hints)
- Per-domain agent: `passes_used`, `self_assessment`, `notes_for_synthesizer`
- Total latency breakdown

**Mobile**: mode badge collapses to icon; source panel becomes full-screen sheet; debug panel hidden by default.

### 8.2 API changes

**Existing**: `/api/chat`, `/api/chat/sessions`, `/api/admin/*` keep their shapes.

**`POST /api/chat`** — extended to emit a terminal SSE event:
```
event: meta
data: {
  "message_id": "uuid",
  "classification": { mode, mode_confidence, intent, topic_hints, persona_target },
  "citations": [...],
  "agent_traces": [...]
}
```

**`POST /api/chat/messages/{message_id}/restream`** (new):
- Body: `{mode}` — re-runs synthesis only, against cached chunks
- Marks `message_classifications.user_override = true`
- Returns SSE stream + meta event

**`GET /api/chat/messages/{message_id}/sources`** (new):
- On-demand fetch of citations + agent_traces (for revisiting old chats)

**`POST /api/admin/audit`** (new):
- Body: `{mode: "dry_run" | "execute"}`
- Async background task; writes `index_audits`

**`POST /api/admin/audit/{audit_id}/dispositions`** (new):
- Bulk approve dispositions; auto-creates `ingest_jobs` for `RE-INGEST`/`MERGE` rows

**`POST /api/admin/ingest/{job_id}/run`** (new):
- Triggers re-ingest job; SSE progress for admin UI

### 8.3 Admin page

Single combined `/admin/audit` page covering both audit and ingest. Sections:
1. **Audit table**: per-row dispositions, sample chunks (expandable), proposed disposition + reason, dropdown to override
2. **Bulk action**: "Approve all proposed" (per-row overrides take precedence)
3. **Ingest queue**: jobs created from approved dispositions; progress bars; error logs

### 8.4 Auth

Admin endpoints require `clerk_user.publicMetadata.role === "admin"` (new middleware). Public mode chunks gated by Pinecone metadata filter `public_safe == true`.

## 9. Costs

### 9.1 Phase 0 one-time cost (re-ingest ~50K records)

| Item | Cost |
|---|---|
| Cohere embed-v3 | ~$5–10 |
| Cohere semantic chunking embeddings (~2x base) | ~$10–20 |
| LLM metadata enrichment (gpt-4.1-mini) | ~$15–25 |
| Pinecone storage delta | net-negative (consolidation) |
| **Total** | **~$40–60 one-time** |

### 9.2 Per-query ongoing cost

| Stage | Current | New |
|---|---|---|
| Optimizer | $0.0001 | $0.0001 |
| Classifier | $0 | $0.0002 |
| Top router | $0.0002 | $0.0002 |
| Memory recall embed | $0 | $0.00001 |
| Domain agents (avg 1.4 passes × 2 agents, retrieve + self-eval + rewriter when needed) | $0 | ~$0.001 |
| Cohere reranks (within-domain × N + cross-domain) | $0.0005 | $0.0015 |
| Verifier | $0.0002 | $0.0002 |
| Synthesizer (Sonnet/Opus mix, weighted ~30% personal / 50% content / 20% public) | $0.005 | $0.012 |
| **Per-query total** | **~$0.006** | **~$0.016** |

2.7× increase. At any sane usage volume (<1000 queries/month), monthly burn stays under $20.

## 10. Migration Plan

Incremental, never breaks live chat.

| Phase | Build effort | Calendar | What ships |
|---|---|---|---|
| **0** | ~3 dev days | 1 week (Max approval + re-ingest runtime) | Audit script, audit table, admin UI page, re-ingest pipeline. Audit run + Max-approved dispositions executed. 7 domain indexes live alongside legacy 40. |
| **1** | ~5 dev days | 1.5 weeks | LangGraph orchestrator_v2 + 7 domain agents. New endpoint `/api/chat?engine=v2` (opt-in). Old default unchanged. Eval suite drafted. |
| **2** | ~2 dev days | 3 days | Default flips to v2 (`/api/chat?engine=v1` becomes 2-week safety fallback). UI gets mode badges, citations, source panels, debug panel. |
| **3** | ~0.5 dev days | 2 weeks observation + 1 day | Remove v1 fallback + old `orchestrator.py`. Export `legacy-archive-*` to S3/R2; delete from Pinecone. Drop deprecated `index_registry` columns. |
| **Total** | **~10–11 dev days** | **~4 weeks elapsed** | |

### 10.1 Rollback

| Phase | Rollback path |
|---|---|
| 0 | Re-ingest is non-destructive (legacy archive untouched until phase 3). Drop new domain indexes; revert audit migration. |
| 1 | v2 is opt-in via flag. Stop using `?v2=true`. ~0 minutes. |
| 2 | v1 fallback flag stays for 2 weeks. Flip default back. ~5 minutes. |
| 3 | Past this point, rollback requires git history. Why we wait 2 stable weeks. |

### 10.2 Observability during rollout

Three SQL-backed dashboards:
- Mode classification distribution (catch drift)
- Per-domain agent activation frequency (catch dead agents)
- Refusal rate per mode (catch over/under refusing)

## 11. Testing Strategy

### 11.1 Layer 1 — Unit tests (~50, all mocked)

Per node: classifier, optimizer, memory recall, top router, each domain agent's plan/retrieve/eval/refine/finalize, RRF merge, Cohere rerank, verifier, each mode synthesizer.

Bug-prevention specifics:
- Voice chunks don't appear in KNOWLEDGE block when query isn't about voice
- Public mode refuses when chunks lack `public_safe=true`
- Self-eval `sufficient` halts loop on pass 1
- Self-eval `partial` triggers refine until `max_passes`
- Failed Pinecone for one namespace doesn't crash agent
- RRF merge dedups identical chunk text
- Mode override flag overrides classifier output

### 11.2 Layer 2 — Integration tests (~15, real Pinecone, mocked LLMs)

Full pipeline against a fixed test corpus seeded into a `chat-with-my-pinecone-test` Railway env. Per test: known query → expected domains routed → expected chunks retrieved → expected mode-shaped response.

### 11.3 Layer 3 — Eval suite (~30 frozen test cases, real APIs, judged output)

Located at `backend/evals/`. Each test case:

```python
{
    "query": "Help me think through how to position telemed offer for Q3",
    "expected_mode": "content_creation",
    "expected_domains_hit": ["marketing", "verticals"],
    "must_cite": True,
    "must_not_say": ["local marketing", "ZIP code"],   # the leak we just diagnosed
    "should_say": ["positioning", "Q3"],
    "judge_criteria": "Response addresses telemed positioning specifically, doesn't default to local marketing examples, cites at least 2 sources",
}
```

Categories:
- **Regression** (10): D2C, telemed-affiliate, "wrong-map" failures we already know about
- **Mode classification** (8): one per mode × intent combo
- **Cross-domain** (5): queries needing 2+ domains
- **Out-of-domain refusal** (4): one per mode plus a "almost-but-not-quite-relevant" case
- **Voice consistency** (3): same factual question across modes

Drafted by Claude/me first, reviewed by Max. Run nightly in CI; PRs that drop pass-rate below baseline blocked.

## 12. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Re-ingest produces worse chunks than originals | Medium | Eval suite runs against pre/post indexes; revert from `legacy-archive-*` if quality drops |
| Classifier misroutes mode often | Medium | Mode override badge; user_override logged in DB; classifier prompt re-tunable without redeploy |
| LangGraph upgrade breaks something | Low-medium | Pin version; test upgrades in `chat-with-my-pinecone-test` first |
| Cohere API outage breaks reranking | Low | Graceful fallback already implemented; keep that pattern |
| Domain agents go infinite-loop | Low | Hard cap `max_passes=3`; logged in `agent_traces` |
| Per-query cost balloons | Low | `agent_traces` allows per-message cost calc; alert if monthly burn > 3× projected |

## 13. Open Questions

None at draft time. All Section 1–8 decisions captured. Items deferred to v2 are tracked in `docs/superpowers/v2-backlog.md`.

## 14. Appendix: Glossary

- **RRF (Reciprocal Rank Fusion)**: Merges multiple ranked lists by summing `1/(k + rank)` per item across lists. Rank-only; ignores raw scores. k=60 is the standard from Cormack et al. paper.
- **Cohere Rerank-v3.5**: Cross-encoder reranker. Takes a query + N candidate documents, returns relevance-scored ranking. Single biggest precision lever in production RAG.
- **Domain Agent**: A LangGraph sub-graph owning 1–3 Pinecone indexes/namespaces, capable of multi-pass iterative retrieval with self-evaluation.
- **Mode**: Synthesis behavior class. v1 has `personal`, `content_creation`, `public`. Each has its own prompt, refusal threshold, citation policy.
- **Public-safe**: A boolean per chunk gating whether public mode can retrieve it. Default false; flipped true after manual review or safe-by-default ingestion of obviously-public content.
- **Out-of-domain gate**: Pre-retrieval check that skips retrieval entirely when no domain agent has high enough confidence to plausibly answer.
