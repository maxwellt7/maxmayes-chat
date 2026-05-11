# Architecture — Chat with My Pinecone RAG Redesign

**Status:** Architecture draft, derived from approved spec
**Source spec:** `docs/superpowers/specs/2026-05-10-rag-redesign-design.md`
**Date:** 2026-05-10
**Author:** System Architect (Claude Code) on behalf of Max Mayes

---

## 1. System Overview

### 1.1 Purpose
Replace the current single-router RAG pipeline with a hierarchical multi-agent orchestrator backed by a reorganized 7-domain Pinecone corpus. Eliminate three diagnosed failure modes:

1. Voice-Pinecone leak that frames every answer through whatever happens to be top-5 in `max-copywriting-voice` for the seed query `"writing style tone voice communication"` (heavy local-marketing skew).
2. Single-domain routing that ignores multi-domain queries even when relevant material exists across indexes.
3. No iterative depth — pipeline gets one Pinecone pass per query and either answers or refuses; cannot refine.

### 1.2 Scope
In-scope (v1):
- LangGraph orchestrator (`orchestrator_v2`) running inside the existing FastAPI service
- 5–7 domain agents (Python modules), each with up to 3 retrieval passes
- Audit-driven corpus consolidation: 40 indexes → 7 domain indexes on Cohere `embed-v3.0`
- Mode-aware synthesis (personal / content_creation / public) with auto-detection
- Cross-session conversational memory via a `conversation-memory` Pinecone index
- Inline `[N]` citations, source panel, mode override badge, debug panel in the existing Vercel/Next.js UI
- Per-message classification, citations, and agent traces persisted to Postgres
- 4-phase migration with feature-flag rollback at each phase

Out-of-scope (v2 — see `docs/superpowers/v2-backlog.md`):
- Memory retention/decay policy
- Manual persona override UI
- Per-mode style-guide variants
- Re-stream chunk caching beyond Postgres TTL
- S3 export of legacy archive (action exists; tooling for it deferred)

### 1.3 Architectural drivers
The NFRs that most heavily constrain design choices, in priority order:

1. **Quality over cost** (explicit user direction). Every tradeoff favors retrieval/synthesis quality. Drives: per-agent custom modules, both reranks (within-domain + cross-domain), separate rewriter LLM call, Opus 4.7 on content/public modes.
2. **Zero data loss during migration**. Drives: legacy-archive Pinecone backup, feature-flag-based rollouts, no destructive Phase-0 actions.
3. **Voice/topic separation**. Drives: static `MAX_STYLE_GUIDE` in code (not Pinecone) plus retention of `max-voice` as a *retrievable* domain (not silently injected).
4. **Multi-purpose use case** — personal, content-creation, and public consumers from one backend. Drives: mode dimension as a first-class system attribute, not a UI option.
5. **Observability for debugging** — Max needs to know why the AI said what it said. Drives: `message_classifications`, `citations`, `agent_traces` tables, debug panel in UI.
6. **Latency budget tolerable, not aggressive**. <2.6s to first token, <10s full response. Drives: parallel domain agents (gather), parallel memory recall (with router), streaming SSE synthesis.

---

## 2. Functional Requirements

Extracted from spec. Identifier scheme: `FR-<section>-<index>`.

| ID | Requirement | Source |
|---|---|---|
| FR-CL-1 | Classify each query into a mode (`personal` / `content_creation` / `public`) plus intent, topic_hints, persona_target, out_of_domain_score | §6.1 |
| FR-CL-2 | When mode_confidence < 0.7 the UI must surface an amber-tinted mode badge for one-click override | §6.6 |
| FR-CL-3 | Manual mode override always wins over classifier output for that message | §6.1 |
| FR-CL-4 | Manual mode choice persists for the rest of the session unless classifier returns confidence > 0.85 with a different mode | §6.1, §7-A1 |
| FR-OPT-1 | Optimize raw query into a dense semantic search string via gpt-4.1-mini | §4.2 |
| FR-MEM-1 | Embed optimized query (Cohere v3) and search `conversation-memory` index filtered by `user_id` for top-5 prior turns | §4.2 |
| FR-MEM-2 | Persist a summary chunk per turn into `conversation-memory` index with `{user_id, session_id, mode, timestamp}` metadata | §4.2 step 9 |
| FR-RT-1 | Top router selects 1–2 domain agents from a 7-domain catalog, returning per-agent confidence | §4.2 |
| FR-RT-2 | When all router confidences < 0.4, raise `out_of_domain` and skip retrieval entirely | §6.5, derived from current orchestrator gate |
| FR-DA-1 | Each domain agent owns 1–3 Pinecone namespaces and exposes the standardized `DomainAgentInput`/`DomainAgentOutput` contract | §5 |
| FR-DA-2 | Each domain agent runs Plan → Retrieve → Self-Eval → (Refine via Rewriter)* → Finalize, capped at 3 passes | §5.1 |
| FR-DA-3 | Each domain agent's Plan node is a custom Python module per domain (not a shared config function) | §5 |
| FR-DA-4 | Self-Eval LLM (gpt-4.1-mini) returns `{verdict, confidence, gaps}`; Rewriter LLM returns `{refine_query, namespaces_to_target}` as a separate call | §5.2, §5.3 |
| FR-DA-5 | Refine pass uses rewriter output, halves k, optionally narrows namespaces, dedupes carry-forward chunks by SHA1(text) | §5.4 |
| FR-DA-6 | After max_passes, agent finalizes regardless of verdict; unmet gaps surface in `notes_for_synthesizer` | §5.4 |
| FR-DA-7 | Each agent runs Cohere Rerank-v3.5 (within-domain) against the original query at Finalize | §5.5 |
| FR-RRF-1 | Orchestrator merges per-agent top-12 lists with Reciprocal Rank Fusion (k=60), then runs final Cohere Rerank-v3.5 → top-10 | §4.2 step 6, §5.5 |
| FR-VER-1 | Verifier (gpt-4.1-mini) judges sufficiency; returns `proceed` / `proceed_with_caveat` / `insufficient_context` | §4.2 step 7 |
| FR-SYN-1 | Synthesizer prompt includes mode-specific rules block; static `MAX_STYLE_GUIDE` for voice; never the dynamic voice Pinecone | §6.2, §6.4 |
| FR-SYN-2 | Personal mode uses Sonnet 4.6; content_creation and public modes use Opus 4.7 | §6.3 |
| FR-SYN-3 | Public mode filters chunks by `metadata.public_safe == true` at the Pinecone query level | §6.3 |
| FR-SYN-4 | Public mode refusal threshold 0.65; content_creation 0.50; personal 0.45 | §6.3 |
| FR-SYN-5 | Synthesizer emits `[N]` citation tokens inline; required in content_creation and public modes | §6.3 |
| FR-SYN-6 | Streaming response via SSE; terminal `event: meta` carries classification + citations + agent_traces | §8.2 |
| FR-PER-1 | Persist per-message: classification, citations, agent_traces; insert summary chunk into conversation-memory index | §4.2 step 9, §7.3 |
| FR-PER-2 | Cache retrieved chunks + memory_context in `messages.cached_context` with 5-min TTL for re-stream support | §7.3 |
| FR-AUD-1 | Phase-0 audit script samples 8–10 chunks per index/namespace, LLM-classifies into domain bucket, proposes per-index disposition | §2.1 of spec, §7.3 schema |
| FR-AUD-2 | Admin UI page `/admin/audit` shows audit table with editable disposition dropdowns and "Approve all proposed" bulk action | §8.3 |
| FR-AUD-3 | Approving dispositions auto-creates `ingest_jobs` for `RE-INGEST` and `MERGE` rows | §8.2 |
| FR-ING-1 | Re-ingest pipeline: fetch source → reconstruct docs → semantic re-chunk (Cohere v3) → LLM metadata enrichment → embed → upsert; idempotent at chunk-offset granularity | §7.4 |
| FR-ING-2 | Per-chunk failures log and continue; >5% failure rate halts the job | §7.4 |
| FR-UI-1 | Mode badge per assistant message with click-to-switch dropdown | §8.1 |
| FR-UI-2 | Inline citation chips: hover tooltip with chunk preview, click to expand source panel | §8.1 |
| FR-UI-3 | Source panel: collapsed by default (expanded in public mode); auto-expand for low-confidence answers | §8.1 |
| FR-UI-4 | Debug panel: classifier output, per-agent passes_used + self_assessment + notes, latency breakdown | §8.1 |
| FR-UI-5 | Mode-switch triggers re-stream against cached chunks | §6.6 |
| FR-API-1 | New endpoints: `POST /api/chat/messages/{id}/restream`, `GET /api/chat/messages/{id}/sources`, `POST /api/admin/audit`, `POST /api/admin/audit/{id}/dispositions`, `POST /api/admin/ingest/{id}/run` | §8.2 |
| FR-API-2 | Admin endpoints require `clerk_user.publicMetadata.role === "admin"` | §8.4 |
| FR-FF-1 | Feature flag `engine=v2` on `POST /api/chat` enables the new orchestrator without changing default; default flips in Phase 2; v1 fallback retained for 2 weeks | §10 |

**Total: 39 FRs**

---

## 3. Non-Functional Requirements

| ID | Category | Requirement | Source |
|---|---|---|---|
| NFR-PERF-1 | Performance | First synthesis token within 2.6s of request (p95) | §3 success criteria |
| NFR-PERF-2 | Performance | Full response within 10s (p95) | §3 success criteria |
| NFR-PERF-3 | Performance | Domain agents must run in parallel (not sequential) when Top Router picks ≥2 | §4.2 step 5 |
| NFR-PERF-4 | Performance | Memory recall must run in parallel with Top Router | §4.2 step 3 |
| NFR-PERF-5 | Performance | Per-agent retrieval across namespaces must run in parallel | §5.1 |
| NFR-COST-1 | Cost | Per-query cost ≤ $0.020 (currently $0.006; new ~$0.016 projected) | §9.2 |
| NFR-COST-2 | Cost | Phase-0 one-time re-ingest cost ≤ $60 for ~50K records | §9.1 |
| NFR-QUAL-1 | Quality | 100% of regression eval cases pass (D2C, telemed-affiliate, "wrong-map" failures) | §11.3 |
| NFR-QUAL-2 | Quality | ≥85% of mode-classification eval cases pass | §11.3 |
| NFR-QUAL-3 | Quality | ≥80% of cross-domain eval cases route to all expected domains | §11.3 |
| NFR-QUAL-4 | Quality | 100% of out-of-domain cases refuse cleanly per mode threshold | §11.3 |
| NFR-QUAL-5 | Quality | Voice consistency: same factual question across modes returns same factual content with mode-appropriate register | §11.3 |
| NFR-QUAL-6 | Quality | Eval pass-rate cannot drop below baseline on any PR (CI gate) | §3, §11.3 |
| NFR-REL-1 | Reliability | Failure of one Pinecone namespace within an agent must not crash the agent | §5.6 |
| NFR-REL-2 | Reliability | Failure of one entire domain agent must not crash the orchestrator | §5.6 |
| NFR-REL-3 | Reliability | Cohere rerank failure must gracefully fall back to score-sort (within-domain) or RRF order (cross-domain) | §5.6 |
| NFR-REL-4 | Reliability | Domain agents must enforce hard `max_passes=3` cap (no infinite loops) | §5.4, §5.6 |
| NFR-REL-5 | Reliability | Self-eval JSON parse failure must default to `verdict="sufficient"` (no halt) | §5.6 |
| NFR-REL-6 | Reliability | Re-ingest jobs must be idempotent (resumable at processed_chunks offset) | §7.4 |
| NFR-DATA-1 | Data Integrity | Zero data loss during corpus migration: `legacy-archive-{date}` Pinecone retained 30 days, then exported to S3/R2 indefinitely | §3, §10 |
| NFR-DATA-2 | Data Integrity | Phase 0 is non-destructive: original 40 indexes untouched until Phase 3 cleanup, after 2-week stable Phase-2 observation | §10.1 |
| NFR-DATA-3 | Data Integrity | All chunks across all indexes use the standardized metadata schema (12 fields) post-Phase-0 | §7.2 |
| NFR-DATA-4 | Data Integrity | All domain indexes use Cohere `embed-v3.0` (1024-dim, cosine) so cross-domain RRF math is well-defined | §7.1 |
| NFR-SEC-1 | Security | Auth via Clerk JWT (existing); no change to user-facing surface | §8.4 |
| NFR-SEC-2 | Security | Admin endpoints gated by `publicMetadata.role === "admin"` middleware (new) | §8.4 |
| NFR-SEC-3 | Security | Public mode chunks must be filtered server-side by `metadata.public_safe == true` (cannot be bypassed by client) | §6.3, §8.4 |
| NFR-SEC-4 | Security | Conversation memory namespaces partitioned by `user_id`; retrieval must filter by user_id (no cross-user leakage) | §7.1 derived |
| NFR-SEC-5 | Security | API keys (Cohere, OpenAI, Anthropic, Pinecone) stored in Railway env vars, never logged or returned in responses | Existing platform behavior, reaffirmed |
| NFR-MAINT-1 | Maintainability | Each domain agent is its own Python module so adding/modifying a domain requires no changes to other agents or core orchestrator | §5 |
| NFR-MAINT-2 | Maintainability | Adding a new domain agent requires: new module, registry row, planner config, router catalog blurb. No code changes elsewhere. | §5 conclusion |
| NFR-MAINT-3 | Maintainability | Mode behavior (prompts, thresholds, model choice) is configurable in one location per mode | §6.3 |
| NFR-MAINT-4 | Maintainability | Static `MAX_STYLE_GUIDE` lives in code as a single source of truth, not split across prompts or Pinecone chunks | §6.4 |
| NFR-MAINT-5 | Maintainability | Eval suite drafted by Claude/Max review pattern; PRs blocked on regression in eval pass-rate (CI gate) | §11.3 |
| NFR-OBS-1 | Observability | Persist per-message classification (mode, intent, confidence, override flag) | §7.3 |
| NFR-OBS-2 | Observability | Persist per-message citations (position, source, score) for trust + audit | §7.3 |
| NFR-OBS-3 | Observability | Persist per-agent traces (passes_used, refinement_history, self_assessment, latency) for debugging | §7.3 |
| NFR-OBS-4 | Observability | Three rollout dashboards: mode classification distribution, per-domain activation frequency, refusal rate per mode | §10.2 |
| NFR-OBS-5 | Observability | Debug panel in UI surfaces classifier output + agent traces + latency for any message | §8.1 |
| NFR-AVAIL-1 | Availability | Service stays available during all 4 migration phases (no big-bang cutover) | §10 |
| NFR-AVAIL-2 | Availability | Each phase rollback procedure documented and ≤ 5 minutes elapsed time (Phase 1 / 2) | §10.1 |
| NFR-AVAIL-3 | Availability | Existing Railway/Vercel deployment topology preserved (no platform migration) | §4 component diagram |
| NFR-CMP-1 | Compatibility | Backward compatibility with existing chat session UI in sidebar (read-only display of v1-era sessions during transition) | §10 Phase 2 |
| NFR-CMP-2 | Compatibility | New endpoints additive; no breaking changes to existing `/api/chat` shape (only an additional terminal SSE event) | §8.2 |

**Total: 41 NFRs across 9 categories**

---

## 4. Architectural Drivers (NFRs That Shape Design)

The dominant drivers, in order of design influence:

1. **NFR-QUAL-* (six requirements)** — eval pass-rates dictate every component-level decision. Design implications: hierarchical agent pattern (depth via iteration), two reranks, mode-aware synthesis, separate rewriter LLM, Opus on quality-critical modes, MAX_STYLE_GUIDE separation.
2. **NFR-DATA-1/2/3/4** — zero-loss migration + standardized metadata + unified embedding model. Design implications: Phase-0 audit-then-execute pattern, legacy-archive-{date} backup, Cohere v3 across all domain indexes, semantic chunker.
3. **NFR-AVAIL-1/2** — service stays live throughout. Design implications: feature-flag gated rollout, additive (not replacement) endpoints, dual-orchestrator window in Phase 1.
4. **NFR-MAINT-1/2** — domain agents must be additive, not coupled. Design implications: per-agent Python module pattern, standardized contract, registry table with `agent_module_path`.
5. **NFR-OBS-1..5** — Max needs to debug surprises. Design implications: three new tables (classifications, citations, traces), debug panel in UI, dashboards.
6. **NFR-PERF-1..5** — latency budget shapes parallelism. Design implications: gather() across domain agents, gather() across namespaces, parallel memory-recall + router, SSE streaming for synthesis.
7. **NFR-SEC-2/3/4** — admin gate, public-safe filter, user_id memory partitioning. Design implications: new admin middleware, server-side Pinecone metadata filter, user_id namespace per `conversation-memory` user.

---

## 5. Architecture Pattern

### 5.1 Selected pattern: **Modular Monolith with Orchestrated Agent Sub-graphs**

The system fits Level 2 complexity (one team, one deployable, multiple bounded modules with clear interfaces). Pattern decomposition:

- **Application architecture**: Modular monolith. Single FastAPI process on Railway. Modules: `app.api`, `app.orchestrator_v2`, `app.agents.*`, `app.services.*` (Pinecone, Cohere, embedding), `app.db`, `app.middleware`.
- **Orchestration architecture**: LangGraph state machine inside the orchestrator module. Each node is a pure function over typed state. Sub-graphs (one per domain agent) compose into the parent graph. Streaming output from the synthesizer node is bridged into FastAPI's SSE response.
- **Data architecture**: CRUD against Postgres for control-plane tables (sessions, messages, classifications, citations, traces, audits, ingest_jobs); vector retrieval against Pinecone for the data plane.
- **Integration architecture**: REST + SSE for client-server. Internal LLM/vector calls are direct HTTPS to Cohere/OpenAI/Anthropic/Pinecone via their official Python SDKs. No message queue (overkill at this scale; persist-then-respond pattern handles all async needs via cached_context TTL).

### 5.2 Why not microservices
Each domain agent is a *logical* service, not a deployment unit. Splitting into separate processes would add deployment complexity, intra-service network latency (currently zero — they're function calls in the same process), and shared-state coordination (the cached chunks and conversation memory are easier as one process). Monolith retains the option to split later if a single domain agent grows enough to warrant its own SLO.

### 5.3 Why LangGraph (not raw Python)
Three properties LangGraph provides cheaply that hand-rolled would require:
- **Typed state** with reducers (essential for parallel domain-agent gathers that merge results).
- **Streaming** that composes through nested sub-graphs (synthesis token-stream needs to surface from inside the graph to the SSE response).
- **Replay and observability** via state checkpointing — useful for debugging the multi-pass agent loop.

Alternative considered: hand-rolled `asyncio.gather` + manual state. Quality-equivalent but ~3× more code for the parallel/streaming/checkpoint plumbing. LangGraph adds one dependency; ROI is positive.

### 5.4 Why FastAPI stays
The current FastAPI service handles auth, sessions, SSE plumbing, admin CRUD. None of that benefits from being moved. The orchestrator is *additive* — a new module called from the existing `/api/chat` endpoint. Replacing FastAPI would be unrelated scope.

---

## 6. Component Design

### 6.1 Component map

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer: Edge / Client                                                   │
│  ─────────────────────────────────────                                  │
│  Vercel (Next.js 14 App Router)                                         │
│  ├─ ChatPage                                                            │
│  │  ├─ ModeBadge (new)                                                  │
│  │  ├─ MessageBubble (extended: citation chips, source panel,           │
│  │  │   debug panel)                                                    │
│  │  └─ ChatInput (Enter sends, IME-safe)                                │
│  ├─ AdminAuditPage (new)                                                │
│  └─ Existing AdminIndexes pages                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │ HTTPS, SSE
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer: API                                                             │
│  ─────────────────────────────────────                                  │
│  FastAPI on Railway                                                     │
│  ├─ app.api.chat                                                        │
│  │  ├─ POST /api/chat                       (extended)                  │
│  │  ├─ POST /api/chat/messages/{id}/restream (new)                      │
│  │  ├─ GET  /api/chat/sessions               (existing)                 │
│  │  └─ GET  /api/chat/messages/{id}/sources  (new)                      │
│  ├─ app.api.admin                                                       │
│  │  ├─ POST /api/admin/audit                 (new)                      │
│  │  ├─ POST /api/admin/audit/{id}/dispositions (new)                    │
│  │  ├─ POST /api/admin/ingest/{id}/run       (new)                      │
│  │  └─ Existing /api/admin/indexes/*                                    │
│  └─ app.middleware                                                      │
│     ├─ ClerkAuthMiddleware (existing)                                   │
│     └─ AdminRoleMiddleware (new — checks publicMetadata.role)           │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer: Orchestration                                                   │
│  ─────────────────────────────────────                                  │
│  app.orchestrator_v2 (LangGraph)                                        │
│  ├─ nodes/classify.py        (gpt-4.1-mini)                             │
│  ├─ nodes/optimize.py        (gpt-4.1-mini)                             │
│  ├─ nodes/recall_memory.py   (Cohere embed + Pinecone)                  │
│  ├─ nodes/top_router.py      (gpt-4.1-mini)                             │
│  ├─ nodes/run_domain_agents.py (gather across selected agents)          │
│  ├─ nodes/rrf_merge.py       (pure function)                            │
│  ├─ nodes/cross_rerank.py    (Cohere Rerank-v3.5)                       │
│  ├─ nodes/verify.py          (gpt-4.1-mini)                             │
│  ├─ nodes/synthesize.py      (Sonnet 4.6 or Opus 4.7, streaming)        │
│  ├─ nodes/persist.py         (Postgres + memory upsert)                 │
│  └─ graph.py                 (LangGraph composition + state spec)       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer: Domain Agents (one Python module per domain)                    │
│  ─────────────────────────────────────                                  │
│  app.agents.base                                                        │
│  ├─ DomainAgent protocol (Input/Output dataclasses)                     │
│  ├─ Shared sub-graph composition helper                                 │
│  ├─ Shared self_eval_node (uses shared prompt template)                 │
│  ├─ Shared rewriter_node                                                │
│  └─ Shared finalize_node (within-domain Cohere rerank)                  │
│  app.agents.strategy        (custom plan logic)                         │
│  app.agents.marketing       (custom plan logic)                         │
│  app.agents.voice           (custom plan logic — hand-tuned for style)  │
│  app.agents.personal        (custom plan logic)                         │
│  app.agents.tools_ops       (custom plan logic)                         │
│  app.agents.content_library (custom plan logic)                         │
│  app.agents.verticals       (custom plan logic)                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer: Services (external API adapters, mostly existing)               │
│  ─────────────────────────────────────                                  │
│  app.services                                                           │
│  ├─ pinecone_service.py    (existing; multi-project key factory)        │
│  ├─ embedding_service.py   (extended: Cohere v3 alongside OpenAI)       │
│  ├─ rerank_service.py      (new: Cohere Rerank-v3.5 wrapper)            │
│  ├─ llm_service.py         (new: thin wrappers for OpenAI/Anthropic     │
│  │                          with shared error handling + retry)         │
│  └─ chunking_service.py    (new: SemanticChunker wrapper for ingest)    │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer: Storage                                                         │
│  ─────────────────────────────────────                                  │
│  Postgres (Railway):                                                    │
│    sessions, messages, message_classifications (new), citations (new),  │
│    agent_traces (new), index_registry (extended), index_audits (new),   │
│    ingest_jobs (new)                                                    │
│  Pinecone:                                                              │
│    7 domain indexes (Cohere v3 / 1024-dim / cosine)                     │
│    + conversation-memory (per-user namespaces)                          │
│    + legacy-archive-{date} (read-only snapshot)                         │
│  S3/R2 (new, Phase-3):                                                  │
│    Cold archive of legacy-archive-{date} as compressed JSONL            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Component responsibilities

**Edge / Client (Next.js)** — already exists, four additions:
- `ModeBadge` component, rendered next to "Max" on each assistant message; controls re-stream via `POST /api/chat/messages/{id}/restream`.
- `MessageBubble` extension to parse `[N]` markers (during react-markdown render) into interactive `CitationChip` components.
- `SourcePanel` and `DebugPanel` lazy-loaded panels reading from the message's `meta` SSE event payload (or `GET /api/chat/messages/{id}/sources` on revisit).
- `AdminAuditPage` with audit table + ingest queue.

**API (FastAPI)** — extends existing routers:
- Adds `restream`, `sources`, `audit`, `dispositions`, `ingest/run` endpoints.
- New `AdminRoleMiddleware` checks Clerk `publicMetadata.role`.
- SSE assembly: orchestrator_v2 returns an async generator of events; FastAPI streams them.

**Orchestrator (LangGraph)** — the new heart of the system.
- Each node is a typed pure function `(state) → state_update`.
- Parallel nodes use LangGraph's native parallel-edge syntax (no manual asyncio.gather in node bodies).
- The synthesize node's output is a streaming async iterator; LangGraph passes this through unchanged so the FastAPI endpoint can yield tokens as they arrive.
- State is checkpointed at each node boundary (in-memory; not persisted across requests).

**Domain Agents** — encapsulated retrieval intelligence.
- `app.agents.base.DomainAgent` is the shared contract + sub-graph composition helper.
- Each concrete agent overrides only the `plan` method and provides a domain description constant for the Top Router catalog.
- Self-eval, rewriter, retrieve, finalize are shared — domain customization is *just* in planning (which namespaces, what k, what filters, how to interpret topic_hints).

**Services** — external API adapters with shared error handling.
- All LLM and rerank calls go through `app.services.llm_service` / `rerank_service` (single retry policy, single timeout config, single observability hook).
- `embedding_service` extended to support both Cohere v3 (new default) and OpenAI (legacy compatibility for the migration window).

**Storage** — see §7 (data model) for full schema.

### 6.3 Component interfaces

**Domain Agent contract (FR-DA-1):**
```python
@dataclass
class DomainAgentInput:
    optimized_query: str
    original_query: str
    topic_hints: list[str]
    memory_context: str | None
    max_passes: int = 3
    rerank_against_query: str | None = None  # defaults to original_query

@dataclass
class DomainAgentOutput:
    chunks: list[Chunk]                 # ordered, deduped, post-rerank
    passes_used: int
    refinement_history: list[dict]      # [{pass_num, query, namespaces, k}]
    self_assessment: dict               # {verdict, confidence, gaps}
    notes_for_synthesizer: str | None

class DomainAgent(Protocol):
    name: str                           # e.g., "marketing"
    description: str                    # for Top Router catalog

    def plan(self, input: DomainAgentInput, pass_num: int) -> RetrievalPlan: ...
    async def run(self, input: DomainAgentInput) -> DomainAgentOutput: ...
```

**Orchestrator state (LangGraph):**
```python
class OrchestratorState(TypedDict, total=False):
    # Inputs
    raw_query: str
    session_id: UUID
    user_id: str
    history: list[Message]
    mode_override: str | None

    # Stage outputs
    classification: ClassifierOutput
    optimized_query: str
    memory_context: str
    selected_domains: list[RouterCandidate]
    domain_outputs: list[DomainAgentOutput]    # parallel-merged
    fused_chunks: list[Chunk]                  # post-RRF
    final_chunks: list[Chunk]                  # post-cross-rerank
    verification: VerifierOutput

    # Output
    response_stream: AsyncIterator[str]        # passed through to SSE
    message_id: UUID
```

**REST endpoints (FR-API-1):** see §8 (API specifications).

---

## 7. Data Model

### 7.1 Postgres schema

New tables (full DDL in spec §7.3 — summary here):

| Table | Cardinality | Retention | Purpose |
|---|---|---|---|
| `message_classifications` | 1:1 with `messages` | Forever | Per-message mode + intent + override flag |
| `citations` | 1:N with `messages` | Forever | Inline `[N]` source chunks per response |
| `agent_traces` | 1:N with `messages` | Forever | Per-domain-agent execution record (passes, gaps, latency) |
| `index_audits` | N per audit run | Forever (audit history) | Phase-0 audit results + dispositions |
| `ingest_jobs` | N per audit run | 90 days post-completion | Long-running re-ingest job state |

Extensions to existing tables:
- `index_registry`: `+domain TEXT`, `+public_safe_default BOOLEAN`, `+agent_module_path TEXT`, `+status TEXT`
- `messages`: `+cached_context JSONB`, `+cached_until TIMESTAMPTZ` (5-min TTL for re-stream)

Index strategy:
- `citations(message_id)` — UI rendering
- `agent_traces(message_id)` — UI debug panel
- `messages(cached_until) WHERE cached_until IS NOT NULL` — partial index for cleanup job
- `index_audits(audit_date, index_name, project_id) UNIQUE` — prevent duplicate audit rows
- `ingest_jobs(status)` — admin queue view

### 7.2 Pinecone schema

| Index | Dim | Metric | Embedding Model | Namespaces | Status |
|---|---|---|---|---|---|
| `max-strategy` | 1024 | cosine | cohere-embed-v3 | `decision-frameworks`, `leadership`, `founder-notes`, `transcripts` | New (v1) |
| `max-marketing` | 1024 | cosine | cohere-embed-v3 | `email`, `ads`, `funnels`, `sales-copy`, `case-studies` | New (v1) |
| `max-voice` | 1024 | cosine | cohere-embed-v3 | `voice-samples`, `style-references` | New (v1) |
| `max-personal` | 1024 | cosine | cohere-embed-v3 | `journal`, `relationships`, `health`, `travel` | New (v1) |
| `max-tools-ops` | 1024 | cosine | cohere-embed-v3 | `n8n-workflows`, `ai-tools`, `infrastructure`, `prompts` | New (v1) |
| `max-content-library` | 1024 | cosine | cohere-embed-v3 | `youtube`, `podcast`, `blog-drafts`, `social` | New (v1) |
| `max-verticals` | 1024 | cosine | cohere-embed-v3 | `telemed`, `local-business`, `coaching`, `saas`, `other` | New (v1, may collapse) |
| `conversation-memory` | 1024 | cosine | cohere-embed-v3 | `{user_id}` per user | New (v1) |
| `legacy-archive-{date}` | mixed | mixed | mixed | mirror of source | Cold (Phase 0–3) |
| Original 40 indexes | mixed | mixed | mixed | as-is | Live during Phase 0–2; archived Phase 3 |

Standardized chunk metadata schema (12 fields, NFR-DATA-3): see spec §7.2.

### 7.3 Embedding strategy
**One embedding model across all domain indexes (NFR-DATA-4):** Cohere `embed-v3.0`, 1024-dim, cosine. Single model means cross-domain RRF is mathematically clean (rank positions are comparable; raw scores aren't, but RRF doesn't use raw scores). Same vendor as the reranker simplifies ops.

The `max-voice` index intentionally uses the same embedding model as the topic indexes — voice is *content* now, retrievable like any other content, not a magic prompt-injection layer.

### 7.4 Re-ingest data flow
Source-of-truth chain for re-ingestion (FR-ING-1):
1. **Source**: paginated read from existing Pinecone index (whatever embedding it uses).
2. **Document reconstruction**: group chunks by `source_id` if metadata supports it; otherwise treat as standalone chunks.
3. **Semantic re-chunk**: LangChain `SemanticChunker` over reconstructed text, using Cohere embeddings to find sentence-pair similarity troughs.
4. **Metadata enrichment**: gpt-4.1-mini one-shot per chunk → `{topic_tags, source_type (if missing), public_safe (sensitivity heuristic)}`.
5. **Embed**: Cohere v3 in 96-chunk batches.
6. **Upsert**: into `(target_index, target_namespace)`.
7. **Idempotency**: `ingest_jobs.processed_chunks` advances after each successful upsert; rerunning resumes at offset (NFR-REL-6).

---

## 8. API Specifications

### 8.1 New endpoints

**`POST /api/chat`** (extended)
- Existing request shape unchanged
- SSE response now ends with a terminal `event: meta` payload:
```json
{
  "message_id": "uuid",
  "classification": {
    "mode": "content_creation",
    "mode_confidence": 0.91,
    "intent": "drafting",
    "topic_hints": ["telemed", "Q3 launch"],
    "persona_target": "B2B founder"
  },
  "citations": [
    {
      "position": 1,
      "source_index": "max-marketing",
      "source_namespace": "case-studies",
      "source_type": "transcript",
      "source_url": null,
      "chunk_text": "...",
      "chunk_score": 0.87
    }
  ],
  "agent_traces": [
    {
      "domain_name": "marketing",
      "passes_used": 2,
      "self_assessment": {"verdict": "sufficient", "confidence": 0.88},
      "notes_for_synthesizer": null
    }
  ]
}
```

**`POST /api/chat/messages/{message_id}/restream`** (new)
- Body: `{"mode": "personal" | "content_creation" | "public"}`
- Effect: re-runs synthesis-only against `messages.cached_context` (must not be expired); marks `message_classifications.user_override = true`
- Returns: SSE stream + meta event (same shape as `/api/chat`)
- Errors: `404` if message not found, `410 Gone` if cached_context expired, `400` if invalid mode

**`GET /api/chat/messages/{message_id}/sources`** (new)
- Returns: full citation list + agent_traces from DB (used when revisiting old chats)
- Response: `{citations: [...], agent_traces: [...], classification: {...}}`

**`POST /api/admin/audit`** (new, admin-only)
- Body: `{"mode": "dry_run" | "execute"}`
- Async: creates a background task; immediately returns `{audit_id, status_url}`
- Status endpoint via existing pattern

**`POST /api/admin/audit/{audit_id}/dispositions`** (new, admin-only)
- Body: `[{audit_row_id, approved_disposition, approved_target_index?}]`
- Effect: updates `index_audits` rows, auto-creates `ingest_jobs` for `RE-INGEST` and `MERGE` rows
- Returns: list of created job IDs

**`POST /api/admin/ingest/{job_id}/run`** (new, admin-only)
- Triggers (or resumes) a re-ingest job
- Returns: SSE stream of progress events `{processed, total, failed}`

### 8.2 Versioning strategy
Backward compatibility via additive changes only (NFR-CMP-2). Existing endpoints retain shape; `event: meta` is a new SSE event the old client ignores. No `/v2` URL prefix needed.

Feature-flag gating during Phase 1:
- `POST /api/chat?engine=v2` → orchestrator_v2
- `POST /api/chat` (no flag) → existing orchestrator
- Phase 2 flips defaults; `?engine=v1` becomes the legacy fallback for 2 weeks.

### 8.3 Auth model
- All `/api/chat/*` endpoints: existing Clerk JWT bearer (NFR-SEC-1).
- All `/api/admin/*` endpoints: Clerk JWT + new `AdminRoleMiddleware` (NFR-SEC-2) checking `publicMetadata.role === "admin"`. Returns 403 on mismatch.

---

## 9. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Frontend framework | Next.js 14 App Router | Already in use; no migration cost; React 18 server components fit the streaming SSE pattern well |
| Frontend deploy | Vercel | Already in use; custom domain `chat.maxmayes.io` configured |
| Backend framework | FastAPI 0.115+ | Already in use; native async fits the SSE + parallel-LLM patterns; no replacement justified |
| Backend deploy | Railway (Docker) | Already in use; Postgres + service in same project; deployment topology preserved (NFR-AVAIL-3) |
| Orchestration framework | LangGraph (latest pinned in Phase 1) | Typed state, native parallel edges, sub-graph composition, streaming through nested graphs — properties that hand-rolled asyncio would require ~3× more code to replicate. Pin version to control upgrade risk (NFR-REL-? mitigated). |
| Vector database | Pinecone (existing tier) | Already in use; multi-project API keys already configured; namespace-per-user pattern works at small N |
| Embedding model | Cohere `embed-v3.0` (1024-dim) | Same vendor as reranker (simpler ops); cheap; consistent across all domain indexes (NFR-DATA-4); matches dim of legacy max-copywriting-voice index (some skip-reingest cases) |
| Reranker | Cohere Rerank-v3.5 | Same vendor as embeddings; best-in-class precision lift on heterogeneous corpora; graceful degradation already implemented (NFR-REL-3) |
| Classifier / Optimizer / Router / Verifier / Self-Eval / Rewriter | gpt-4.1-mini (OpenAI) | Cheap, low-latency, JSON-mode reliable. Total spend across all small-LLM nodes is ~$0.002/query — quality of these classification steps doesn't justify Sonnet-tier cost. |
| Synthesizer (personal mode) | Anthropic Sonnet 4.6 | Snappy first-token (~500ms), strong voice consistency, sufficient for personal mode where output tolerates lighter polish |
| Synthesizer (content_creation, public) | Anthropic Opus 4.7 | Highest output quality where it matters most (drafted artifacts, public-facing). +5x cost vs Sonnet acceptable per quality-over-cost direction (NFR-COST-1 still met because Opus is only ~70% of mode mix) |
| Re-chunker | LangChain `SemanticChunker` + Cohere embeddings | Better than fixed-token splitters on heterogeneous content; finds natural breakpoints; same Cohere vendor as the embedding/rerank stack |
| Auth | Clerk (existing) | Already integrated; JWT + role metadata supports admin gate (NFR-SEC-2) without re-platforming |
| Database | Postgres 16 (Railway managed) | Already in use; JSONB columns handle flexible per-message metadata cleanly; partial index pattern supports the `messages.cached_until` cleanup |
| Object storage | Cloudflare R2 or AWS S3 (Phase 3) | Cheap indefinite storage for `legacy-archive-{date}` exports; vendor decision deferred to Phase 3 (R2 if cost-priority, S3 if integration-priority) |
| Test framework | pytest + pytest-asyncio | Already in use; supports async generator testing for streaming nodes |
| Eval framework | Custom harness in `backend/evals/` | Anthropic-style judge prompts; lightweight; runs in CI via GitHub Actions |
| Observability | SQL queries + simple Grafana panels (or Postgres-backed dashboard) | All required signals already persisted to `agent_traces` + `message_classifications`; no APM platform needed at single-user scale (NFR-OBS-4) |

---

## 10. NFR Mapping (Coverage Matrix)

| NFR | Architectural Decision(s) | Component(s) |
|---|---|---|
| NFR-PERF-1 (first token ≤2.6s) | Parallel domain agents + parallel namespace retrieval + parallel memory-recall/router; SSE streaming from synthesizer | Orchestrator (parallel edges), Domain Agents (asyncio.gather over namespaces), Synthesize node |
| NFR-PERF-2 (full ≤10s) | max_passes=3 hard cap; Opus token-budget capped at 2048; SSE streams as tokens arrive | Domain Agent finalize, Synthesize node |
| NFR-PERF-3..5 (parallelism) | LangGraph parallel edges for gather; per-agent asyncio.gather for namespace queries | Orchestrator graph, Domain Agent retrieve node |
| NFR-COST-1 (≤$0.020/query) | Small-LLM (gpt-4.1-mini) for all classification/eval/router steps; Sonnet only for personal mode; Opus only for content/public | Classify, Optimize, TopRouter, Verify, SelfEval, Rewriter, Synthesize |
| NFR-COST-2 (≤$60 re-ingest) | Cohere v3 chosen for cheap embeddings; semantic chunker batched; metadata enrichment uses gpt-4.1-mini | Re-ingest pipeline |
| NFR-QUAL-1..6 | Hierarchical agent pattern + iterative refinement + two reranks + mode-aware synthesis + static voice guide + eval suite as CI gate | Domain Agents, Synthesize, Eval suite (CI) |
| NFR-REL-1 (one ns failure) | Per-namespace try/except inside Domain Agent retrieve node; logs and continues | Domain Agent retrieve |
| NFR-REL-2 (one agent failure) | Per-agent try/except in `run_domain_agents` node; agent contributes empty output + note | Orchestrator run_domain_agents node |
| NFR-REL-3 (rerank failure) | rerank_service wrapper falls back to score-sort (within-domain) or RRF order (cross-domain) | rerank_service |
| NFR-REL-4 (max_passes cap) | Hard `max_passes=3` constant in DomainAgent.run; finalize forced after | Domain Agent base |
| NFR-REL-5 (self-eval JSON parse) | Try/except around JSON parse; default to verdict=sufficient | Shared self_eval_node |
| NFR-REL-6 (re-ingest idempotent) | `ingest_jobs.processed_chunks` advances per successful upsert; resume from offset | Re-ingest pipeline, ingest_jobs table |
| NFR-DATA-1 (zero loss) | `legacy-archive-{date}` index created in Phase 0 before any modification; deleted only after Phase-3 export to S3/R2 | Phase-0 audit, Phase-3 cleanup |
| NFR-DATA-2 (non-destructive Phase 0) | Phase-0 outputs only INSERT into new tables/indexes; original 40 indexes untouched until Phase 3 | Audit pipeline, ingest pipeline |
| NFR-DATA-3 (standard metadata schema) | Re-ingest pipeline applies the 12-field schema at upsert time; metadata enrichment LLM fills missing fields | Re-ingest pipeline step 4 |
| NFR-DATA-4 (Cohere v3 across domains) | embedding_service defaults to Cohere v3; re-ingest pipeline uses it | embedding_service, re-ingest |
| NFR-SEC-1 (Clerk auth) | Existing ClerkAuthMiddleware unchanged | app.middleware.clerk_auth |
| NFR-SEC-2 (admin gate) | New AdminRoleMiddleware checks `clerk_user.publicMetadata.role` | app.middleware.admin_role |
| NFR-SEC-3 (public-safe filter) | Pinecone metadata filter `public_safe == true` applied server-side in Public mode retrieve calls | Domain Agent retrieve (mode-aware) |
| NFR-SEC-4 (memory partitioning) | `conversation-memory` index uses `{user_id}` as namespace; queries always include user_id filter | recall_memory node, persist node |
| NFR-SEC-5 (no key leakage) | API keys read from settings, never returned in responses; existing logging policy retained | All service modules |
| NFR-MAINT-1 (per-agent isolation) | Each agent in own module; depends only on `agents.base` + `services.*` | app.agents.* layout |
| NFR-MAINT-2 (additive new domain) | New domain = new module + index_registry row + planner config + router catalog blurb. Zero changes elsewhere. | DomainAgent protocol, index_registry.agent_module_path |
| NFR-MAINT-3 (mode config locality) | Each mode's prompt + threshold + model is a single dict in `app.orchestrator_v2.modes` | Synthesize node config |
| NFR-MAINT-4 (style-guide single source) | `MAX_STYLE_GUIDE` constant in code; not in DB, not in Pinecone | Synthesize node |
| NFR-MAINT-5 (CI eval gate) | GitHub Actions workflow runs `backend/evals/` nightly + on PR; baseline pass-rate stored in repo | CI config |
| NFR-OBS-1..3 (persistence) | Three new tables; persist node writes them after every message | Postgres, persist node |
| NFR-OBS-4 (dashboards) | SQL queries against new tables; rendered in simple admin dashboard or Grafana | Admin UI / Grafana |
| NFR-OBS-5 (debug panel) | Frontend reads classification + agent_traces from terminal SSE meta event or `/sources` endpoint | MessageBubble.DebugPanel |
| NFR-AVAIL-1 (always available) | 4-phase rollout, feature-flag gating, dual-orchestrator window in Phase 1 | Migration plan |
| NFR-AVAIL-2 (≤5 min rollback) | Phase 1: stop using `?v2=true`. Phase 2: flip default flag. | Migration plan |
| NFR-AVAIL-3 (deployment topology) | No change to Vercel + Railway + Pinecone topology; only intra-app changes | Component diagram unchanged at platform level |
| NFR-CMP-1 (legacy session display) | Old chat sessions remain readable in sidebar; only new sessions get v2 features | Frontend MessageBubble degrades gracefully when meta event absent |
| NFR-CMP-2 (no breaking API changes) | All new endpoints additive; meta event additive on existing endpoint | API design |

**Coverage: 41/41 NFRs mapped.**

---

## 11. Trade-off Analysis

### 11.1 Quality vs cost
**Decision:** Quality wins every time. Per-agent custom modules (vs shared config), both reranks, separate rewriter LLM call, Opus 4.7 on content/public modes.

**Cost impact:** Per-query cost goes from $0.006 to $0.016 (2.7×).

**Why acceptable:** Single-user system. At any sane volume (<1000 queries/month), monthly cost stays under $20. Cost is an order of magnitude below the value of better answers.

**Reversibility:** All quality knobs (rerank on/off, model choice per mode, max_passes) are config. If cost ever becomes a constraint, can dial back without architecture changes.

### 11.2 Monolith vs microservices for domain agents
**Decision:** Monolith. Each domain agent is a Python module, not a service.

**Why:** Single team, single deploy, no inter-service network cost, shared cache and conversation memory live in one process. Microservices add complexity (deployment, observability, network) with no benefit at this scale.

**When to revisit:** If a single domain agent ever needs its own SLO (e.g., the `voice` agent becomes a public API that other apps consume), split it then. Module boundaries are designed to make that split mechanical.

### 11.3 LangGraph vs hand-rolled orchestration
**Decision:** LangGraph.

**Trade:** Adds a dependency that could break on upgrades or have its own bugs.

**Why accepted:** Typed state, parallel edges, sub-graph composition, and streaming through nested graphs would take ~3× more code to hand-roll well. ROI positive even accounting for upgrade risk.

**Mitigation:** Pin version in requirements.txt; test upgrades in `chat-with-my-pinecone-test` env first.

### 11.4 Voice as static guide vs retrievable domain (kept both)
**Decision:** Static guide for synthesis style; retrievable domain for genuine voice queries. Belt and suspenders.

**Why:** The static guide kills the leak (no more accidental topic injection from voice chunks). Keeping `max-voice` as a domain agent preserves the value when "how does Max write hooks?" is the actual question.

**Cost:** Slight prompt-engineering complexity — synthesizer must cite voice chunks when they're in the KNOWLEDGE block, but rely only on STYLE GUIDE when they're not.

### 11.5 Single embedding model across all domains
**Decision:** Cohere v3 across all 7 domain indexes (and conversation-memory).

**Trade:** Loses per-domain optimization (e.g., a code-tuned embedding for `tools-ops`).

**Why accepted:** Cross-domain RRF requires comparable rank positions. Heterogeneous embedding models break the merge math. Cohere v3 is strong across diverse text types — gain from mixed embeddings would be marginal vs the cost of breaking RRF.

**When to revisit:** If a specific domain's retrieval quality is provably hurt by Cohere v3 (eval-measured), consider that domain getting its own model + a separate non-RRF merge path. v2 problem.

### 11.6 Memory unbounded in v1
**Decision:** Conversation memory grows forever in v1.

**Trade:** Cost grows linearly with usage; recall quality may degrade as the index gets large.

**Why accepted:** At single-user scale, even months of daily use doesn't reach concerning sizes. Adding retention policy now would be premature optimization.

**v2 deferral:** Decay heuristic, summarization, eviction (already in v2-backlog.md).

### 11.7 No mode override at the API level (UI only)
**Decision:** Mode override is a UI affordance; the API auto-classifies on every request unless `mode` is explicitly passed.

**Trade:** A misclassified mode requires a UI round-trip (re-stream) rather than a one-shot fix.

**Why accepted:** Re-stream is fast (chunks cached). Forcing the classifier to be the default keeps API simple.

---

## 12. Deployment Architecture

### 12.1 Deployment topology (preserved from current)
```
            ┌───────────────────┐
            │  GitHub           │
            │  maxwellt7/       │
            │  maxmayes-chat    │
            └─────────┬─────────┘
                      │ git push origin main
        ┌─────────────┴──────────────┐
        ▼                            ▼
┌──────────────┐            ┌────────────────────┐
│  Vercel      │            │  Railway            │
│  Auto-deploy │            │  Auto-deploy        │
│  Next.js     │            │  Docker FastAPI     │
│  → chat.     │            │  → chat-with-my-    │
│    maxmayes  │            │    pinecone-backend │
│    .io       │            │    .up.railway.app  │
└──────────────┘            │                     │
                            │  Postgres (managed) │
                            └────────────────────┘
```

No platform changes. New components live inside the existing FastAPI service.

### 12.2 New environment variables (Railway)
Already present (verified): `COHERE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `PINECONE_API_KEY_*`, `CLERK_SECRET_KEY`, `DATABASE_URL`, `CORS_ALLOW_ORIGIN`.

To add:
- (Phase 3) `ARCHIVE_S3_BUCKET` / `ARCHIVE_R2_*` — for legacy archive export
- (Optional) `LANGGRAPH_CHECKPOINT_BACKEND=memory` — confirmed default

### 12.3 Migration phasing (recap)

| Phase | Calendar | Production state | Rollback |
|---|---|---|---|
| 0 | Week 1 | Old pipeline live; new domain indexes populated alongside | Drop new indexes, revert audit migration |
| 1 | Weeks 2–3 | Old pipeline still default; v2 opt-in via `?engine=v2` | Stop using flag |
| 2 | Week 4 | v2 default; v1 fallback via `?engine=v1` for 2 weeks | Flip default flag back (~5 min) |
| 3 | Week 5+ | v1 removed; legacy archive exported | Past this point requires git-history restore |

### 12.4 CI / CD
- GitHub Actions: existing build/test pipeline preserved
- Add: nightly eval-suite run on `backend/evals/`
- Add: PR check that runs eval-suite against the PR branch and blocks merge if pass-rate < baseline (baseline stored in `backend/evals/baseline.json`)

---

## 13. Future Considerations

Beyond the v2 backlog (which captures known deferrals), three areas worth thinking about:

1. **Multi-user scaling.** Current design assumes one user (Max). Conversation-memory partitioning by user_id is in place; admin role gate is in place. But: per-user usage quotas, per-user public_safe defaults, and rate limiting are not specified. If/when this opens to clients, those become must-haves.

2. **Tool use beyond Pinecone.** The Domain Agent contract is rigid about Pinecone retrieval. Adding agents that call external tools (web search, calculator, calendar lookup) requires extending the contract or introducing a parallel "ToolAgent" abstraction. v3 territory.

3. **Online learning from user feedback.** The system has all the data (citations, mode overrides, debug traces) needed to learn — e.g., automatically tune Top Router prompts when override rates climb on certain query types. Not in scope, but the data foundation is there for it.

---

## 14. Open Items

None at architecture-document time. All design decisions captured. The implementation plan (next step via `superpowers:writing-plans`) will turn this into executable tasks.
