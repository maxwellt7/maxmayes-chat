# Master Plan — maxmayes-chat → Production Product

Status: **v2** — integrates five specialist audits
Author: Chief Architect
Date: 2026-08-08

Companion documents:
- [`DEFECT-REGISTER.md`](./DEFECT-REGISTER.md) — 50 evidence-backed defects with file/line references, severity, and owning phase.
- [`architecture-rag-redesign-2026-05-10.md`](./architecture-rag-redesign-2026-05-10.md) — the prior v2 RAG design. **This plan builds on it, and does not restart it.**

---

## 0. Verified current state

Everything in this section was confirmed by direct inspection, not assumed.

### What works
- FastAPI backend + Next.js 14 frontend, deployed (Railway + Vercel).
- 5-stage RAG pipeline: optimize → route → parallel retrieve → RRF → Cohere rerank → verify → Claude synthesis.
- 80/80 backend tests pass. Frontend typechecks clean.
- Sensible data model foundations already exist: `index_registry` (with `domain`, `public_safe_default`, `status` columns), `chat_messages`, `index_audits`, `ingest_jobs`.
- A thoughtful ingest pipeline: semantic chunking, LLM metadata enrichment, Cohere v3 embeddings.

### What is broken (blocking)

| # | Defect | Evidence | Impact |
|---|---|---|---|
| B1 | **Total auth bypass.** Clerk JWT signature is never verified; the payload is base64-decoded and trusted. `CLERK_SECRET_KEY` is declared and unused. | `backend/app/middleware/clerk_auth.py:32-44`, `config.py:18` | Anyone can forge `{"sub":"x","publicMetadata":{"role":"admin"}}` and read the entire knowledge base + control admin. |
| B2 | **No privacy enforcement.** `public_safe` is computed per-chunk at ingest and never read at query time. | `prompts/metadata_enrich.txt` vs. `orchestrator.py` (no filter anywhere) | Personal journals, health, finances, relationships are retrievable by any caller. |
| B3 | **Conversation memory does not exist.** `run_pipeline(raw_query, db)` receives only the latest message. | `routers/chat.py:39`, `orchestrator.py:291` | Every follow-up ("what did you mean?", "expand on that") is answered as a cold first message. This is the single largest cause of "answers aren't tight". |
| B4 | **Self-destructing index registry.** Any transient Pinecone error permanently sets `is_active = False` and commits. | `orchestrator.py:279-288` | One network blip silently removes a knowledge domain from the product, forever, with no alert. |
| B5 | **Raw exceptions streamed to the client.** | `routers/chat.py:55-57` | Leaks internals; also means a failed answer is never persisted. |
| B6 | **No citations.** Chunks are numbered `[1]..[n]` in the prompt but no source data is ever returned. | `orchestrator.py:373-375` | Users cannot verify anything; we cannot debug bad answers. |
| B7 | **Project was not under version control.** Git root resolved to `/Users/maxmayes`; zero tracked files. | `git rev-parse --show-toplevel` | No history, no review, no parallel work. **Fixed** — dedicated repo initialized at commit `dc858b4`. |
| B8 | **No CI, no integration tests, no eval harness, unpinned dependencies.** | absent `.github/`, `requirements.txt` uses `>=` | Non-reproducible builds; no way to prove a retrieval change improved anything. |

### What is architecturally weak (non-blocking but must change)

- **Routing is a single LLM guess over prose descriptions.** No lexical/hybrid fallback, no query-type awareness, no learning from outcomes. The `0.4` out-of-domain threshold is arbitrary and unvalidated.
- **Pure dense retrieval.** Exact-match questions (names, numbers, dates, product names) are exactly where dense-only retrieval fails, and exactly what "specific questions" means.
- **4 sequential LLM round-trips before the first token** (optimize → route → verify → synthesize) plus a per-request Pinecone fetch for the voice profile using a hardcoded seed query. Time-to-first-token is dominated by avoidable serialization.
- **Fixed `top_k` (10,5,5) and `top_n=8`** regardless of query complexity.
- **No caching** of embeddings, routing decisions, or the voice profile. **No rate limiting** → unbounded cost exposure the moment this is public.

---

## 1. Product definition

Before architecture, the thing being built:

**A public-facing AI persona of Max Mayes**, grounded in his knowledge base, that:
- answers strangers' questions accurately and in his voice, with citations;
- never leaks private material;
- can be spoken to out loud;
- captures and qualifies leads into GoHighLevel;
- eventually answers Instagram DMs;
- gets measurably better every week from real usage.

**Three audiences, three trust tiers.** This is the organizing principle of the whole system:

| Persona | Who | Corpus access | Capabilities |
|---|---|---|---|
| `public` | Anonymous / free signup | Public-safe corpus only | Chat, voice, lead capture |
| `member` | Authenticated customer (future paid tier) | Public-safe + member-only corpus | + history, saved threads |
| `owner` | Max, and only Max | Everything | + admin, private corpus, ingest, evals |

Every retrieval, every prompt, and every API response is scoped by persona. Persona is derived **server-side from a verified identity**, never from a request parameter.

---

## 2. Target architecture

```
                        ┌──────────────────────────────────────┐
   Browser / Mobile ───►│  Next.js 14 (Vercel)                 │
   Voice (WebRTC)       │  Clerk-authenticated                 │
   Instagram (webhook)  └──────────────┬───────────────────────┘
                                       │ verified Bearer JWT
                        ┌──────────────▼───────────────────────┐
                        │  FastAPI (Railway)                    │
                        │  ┌─────────────────────────────────┐ │
                        │  │ AuthN/AuthZ  → persona resolver │ │
                        │  ├─────────────────────────────────┤ │
                        │  │ Rate limit / quota / cost guard │ │
                        │  ├─────────────────────────────────┤ │
                        │  │ ANSWER ENGINE                    │ │
                        │  │  1 understand (1 LLM call:       │ │
                        │  │    rewrite + classify + route)   │ │
                        │  │  2 retrieve  (hybrid: dense+BM25,│ │
                        │  │    persona-filtered, fan-out)    │ │
                        │  │  3 fuse + rerank                 │ │
                        │  │  4 redact (PII/secret scrub)     │ │
                        │  │  5 synthesize (stream + cite)    │ │
                        │  │  6 guard (output privacy check)  │ │
                        │  ├─────────────────────────────────┤ │
                        │  │ Learning loop / telemetry        │ │
                        │  └─────────────────────────────────┘ │
                        └───┬─────────┬──────────┬─────────────┘
                            │         │          │
                  ┌─────────▼──┐ ┌────▼─────┐ ┌──▼──────────────┐
                  │ Postgres   │ │ Pinecone │ │ External        │
                  │ users,     │ │ public / │ │ GHL, Instagram, │
                  │ threads,   │ │ member / │ │ ElevenLabs,     │
                  │ telemetry, │ │ private  │ │ OpenAI/Anthropic│
                  │ evals,leads│ │ tiers    │ │ /Cohere         │
                  └────────────┘ └──────────┘ └─────────────────┘
```

### 2.1 Privacy: four layers, defense in depth

The current design's fatal assumption is that an LLM-assigned `public_safe` boolean is a sufficient control. It is not — it is one probabilistic signal. Privacy will be enforced as an **access boundary**, not a filter.

- **L1 — Corpus partitioning (primary control).** Content is physically separated into `public` / `member` / `private` Pinecone namespaces. The public answer engine is only ever handed registry entries whose `visibility='public'`. Private vectors are not reachable by a public request even if every other layer fails, because the code path never receives their coordinates.
- **L2 — Query-time metadata filter.** Pinecone `filter` clause pinned to the caller's persona, injected server-side. Belt and braces for anything mis-partitioned.
- **L3 — Post-retrieval redaction.** Deterministic scrubbing of emails, phone numbers, street addresses, card/account numbers, API keys, and a configurable named-entity denylist (third-party names, client names) before text ever reaches the synthesizer.
- **L4 — Output guard.** A fast check on the generated answer before it reaches a public user, plus prompt-injection hardening in the synthesizer system prompt. Fails closed.

Plus an **admin "leak audit" tool**: run a corpus of adversarial probes against the public persona and report anything that surfaces private material. This runs in CI.

**Migration reality:** every existing chunk must be classified into a visibility tier before public launch. LLM classification proposes; Max approves in bulk via the admin UI; **default is `private`** for anything unreviewed. Nothing becomes public by omission.

### 2.2 Answer quality: what actually changes

| Problem | Fix |
|---|---|
| No conversation context | Pass last N turns into a **conversation-aware rewriter** that resolves pronouns/ellipsis into a standalone query. |
| 4 serial LLM calls | Collapse rewrite + query-type classification + routing into **one** structured call. Run the verifier **concurrently** with synthesis rather than before it. |
| Dense-only retrieval misses exact matches | **Hybrid retrieval**: dense + lexical (Postgres FTS / BM25 over a chunk mirror), fused with RRF. This is the single biggest lever for "specific questions". |
| LLM router guesses wrong | With a consolidated ~7-index corpus, **fan out to all eligible indexes** and let the cross-encoder rerank decide. Routing becomes an optimization, not a correctness dependency. Keep the LLM router only to prune when the catalog is large. |
| Fixed `top_k` | Budget scales with query type: lookup → narrow+precise; synthesis → wide; comparison → multi-subquery. |
| No citations | Return structured source events over SSE; render inline footnotes with document title, date, and type. |
| Voice profile fetched per request | Compile once into a stored, versioned voice-profile record; refresh on a schedule, not per request. |
| No way to know if changes help | **Eval harness** (§2.3) gates every retrieval change. |

### 2.3 The eval harness (non-negotiable, built early)

Nothing about "best logic possible to answer specific questions tightly" is verifiable without measurement. Before any retrieval rewrite lands:

- A golden set of ~100 questions across domains, each with expected source documents and a reference answer.
- Automated metrics: **retrieval recall@k**, **citation precision** (did cited chunks actually support the claim), **groundedness** (LLM judge), **refusal correctness** (does it refuse what it should, and only that), **privacy leakage rate**, **p50/p95 time-to-first-token**, **cost per query**.
- Runs in CI on every PR touching the answer engine. A regression on groundedness or leakage blocks merge.

### 2.4 Relationship to the existing v2 RAG design

A detailed v2 design already exists (May 2026): LangGraph orchestrator, seven domain agents over consolidated `max-*` indexes, mode-aware synthesis (`personal` / `content_creation` / `public`), a `conversation-memory` index with per-user namespaces, a static `MAX_STYLE_GUIDE` replacing the voice leak, citations, `agent_traces`, and an `?engine=v2` feature flag. It is well reasoned and largely unimplemented.

**We adopt it, with one structural correction.**

In that design, `public` is a **mode** — chosen by a classifier, overridable in the UI. For an internal tool that is fine. For a public product it is a critical flaw: a classifier deciding whether a stranger sees Max's journal is a coin flip with an unacceptable downside.

The correction: **persona is resolved server-side from verified identity and hard-constrains mode.** A `public` caller is pinned to public-safe retrieval and cannot be moved out of it by a classifier, a UI toggle, or a prompt. Mode classification survives as a *quality* mechanism selecting tone and model within the persona's permitted envelope — never as an access-control mechanism.

We also defer the full seven-agent LangGraph sub-graph machinery (plan → retrieve → self-eval → refine × 3) until the eval harness can prove it earns its latency. The measured wins — conversation history, hybrid retrieval, provenance-rich context, citations — come first and are cheaper.

### 2.5 Voice architecture

Researched and decided: **speech-to-text → our existing RAG → text-to-speech**, orchestrated by LiveKit Agents with `run_pipeline` wired in as a custom `llm_node`. Deepgram for streaming STT with eager end-of-turn so retrieval starts before the user finishes speaking; ElevenLabs Flash with a professional voice clone of Max for TTS, streamed at sentence boundaries.

**Rejected: OpenAI Realtime speech-to-speech.** Three independent disqualifiers — it cannot run Claude (our synthesis quality and persona live there), the tool-call round-trip to reach our retrieval makes latency *worse* not better, and it would paraphrase our carefully synthesized answer in a preset voice we cannot clone Max into.

**The honest latency position:** a genuine retrieve-rerank-synthesize turn has a floor around 900ms–1.2s. That is "thoughtful", not "instant". This is the correct trade — the differentiator is grounded, in-voice answers, not reflexes. The failure mode to engineer against is *silence*, not latency: a pre-synthesized in-voice acknowledgement ("let me pull that up…") covers the gap and is cancelled if retrieval returns fast. Conversational turns ("say that again") bypass retrieval entirely via a fast path.

The pipeline latency fixes this requires — prefetched voice profile, merged understand-stage, non-blocking verifier, native async clients — are the *same* fixes Phase 3 makes for the text product. Voice is not a detour; it is a forcing function.

### 2.6 Integration constraints that shape the schedule

**Instagram is the schedule risk. GoHighLevel is not.**

GoHighLevel: API v2 with a Private Integration Token (v1 is deprecated and can no longer mint keys). Callable within a day. Three traps to design around — upserting `tags` **overwrites all existing tags** (use the dedicated add/remove tag endpoints), custom field IDs are per-location (resolve by `key` at runtime), and **there is no sandbox**, so integration tests run against production with live workflow triggers and must use a dedicated test sub-account with outbound email/SMS disabled.

Instagram: a hard 4–8 week gate that is paperwork, not engineering. Business Verification must complete *before* App Review; App Review now runs ~20 days per cycle and rejection is all-or-nothing. Three constraints shape the product itself:
- **You cannot initiate a conversation.** The user must DM first.
- **24-hour reply window.** The `HUMAN_AGENT` tag extends it to 7 days but is for humans — automated sends with it are rejected and misuse risks API access. The other message tags were removed in April 2026.
- **A human escalation path is mandatory** and is checked at App Review, and automated experiences must respond within **30 seconds** — a real latency budget for an LLM pipeline.

Given that, we **launch Instagram through ManyChat or Chatfuel** (Meta Business Partners running on the official APIs, live in hours) and treat direct integration as a later migration once volume justifies owning the Meta relationship. Business Verification paperwork starts in Phase 1 regardless, in parallel, so the option stays open.

### 2.7 Getting smarter

- **Capture**: every query, rewrite, route decision, retrieved chunk ID, citation, latency, cost, and user feedback (thumbs, "that's wrong", corrections) into a telemetry table.
- **Mine**: weekly automated report of unanswered/refused questions, low-groundedness answers, and queries whose top chunks scored poorly → a **knowledge gap list** for Max.
- **Close the loop**: Max answers gap questions in the admin UI → those become curated Q&A pairs in a high-priority index, immediately improving future answers.
- **Tune**: track which index actually produced cited chunks per query type, and feed that back into routing priors.
- **Promote**: highly-rated answers become cached/canonical responses (semantic cache) — faster and cheaper.

---

## 3. The team

I am staffing ten specialist agents. Each owns a workstream, a branch, and an acceptance gate. I integrate, review, and arbitrate.

| # | Agent | Charter | Primary artifacts |
|---|---|---|---|
| A1 | **Security & Identity** | Clerk JWT verification, persona resolution, RBAC, rate limiting, secrets hygiene | `app/security/*`, `users` table, auth tests |
| A2 | **Privacy & Data Governance** | 4-layer privacy model, visibility classification + migration, redaction, leak-audit tooling | `app/privacy/*`, classification pipeline, adversarial probe suite |
| A3 | **Retrieval / Answer Engine** | Conversation memory, hybrid retrieval, routing, citations, latency | `app/engine/*` |
| A4 | **Evaluation & Learning** | Eval harness, golden set, telemetry schema, gap mining, feedback loop | `app/evals/*`, `evals/` datasets |
| A5 | **Platform & Reliability** | Timeouts, retries, error taxonomy, structured logging, tracing, CI/CD, migrations, pinned deps | `.github/workflows/*`, `app/core/*` |
| A6 | **Product Frontend** | Customer-facing UI, onboarding, chat UX, citations, mobile, a11y | `frontend/src/app/*` |
| A7 | **Admin & Analytics** | Admin dashboard: questions, gaps, leads, costs, evals, privacy audit | `frontend/src/app/admin/*`, analytics endpoints |
| A8 | **Voice** | Real-time speech in/out, barge-in, Max's cloned voice | `app/voice/*`, WebRTC client |
| A9 | **Integrations** | GoHighLevel lead sync, Instagram DM agent, webhook security | `app/integrations/*` |
| A10 | **QA / Red Team** | Adversarial testing of every gate: auth bypass, prompt injection, privacy leakage, load | `tests/security/*`, load scripts |

**Working agreement:** one branch per workstream off `main`; every branch must pass typecheck + tests + evals + a red-team pass from A10 before I merge; no workstream may weaken a gate owned by another.

---

## 4. Phased delivery

Each phase has an explicit **gate** — objective criteria I verify before the next phase starts. Phases 3–7 parallelize across agents; Phases 0–2 are strictly sequential because everything depends on them.

### Phase 0 — Foundation ✅ (complete)
Dedicated git repo, secrets-safe `.gitignore`, local dev environment, green baseline (80/80 tests, clean typecheck).

### Phase 1 — Make it safe *(A1, A5, A10)*
Nothing else may ship before this.
1. Real Clerk JWT verification (RS256 via JWKS, `exp`/`nbf`/`azp` validation, cached keys).
2. `users` table; roles stored **in our database**, not read from token claims. Owner seeded by explicit allowlist.
3. Authorization on every endpoint; admin routes require owner role. Tenant isolation verified on chat history.
4. Error taxonomy — no raw exceptions to clients; structured logs with request IDs.
5. Rate limiting + per-user cost quotas.
6. Fix B4 (circuit-breaker instead of permanent deactivation) and B5 (persist partial answers, surface clean errors).
7. Pinned dependencies; CI running lint + typecheck + tests on every push.

**Gate:** A10 red-team attempts forged-token admin access, cross-user history reads, and unauthenticated chat — all must fail. CI green.

### Phase 2 — Make it private *(A2, A10)*
1. Visibility tiering of the corpus; migration + bulk review UI; default-private.
2. Persona-scoped retrieval (L1 + L2).
3. Redaction layer (L3) and output guard (L4).
4. Adversarial probe suite in CI.

**Gate:** 200-probe adversarial suite yields **zero** private-material leaks against the `public` persona. Max signs off on the visibility classification.

### Phase 3 — Make it good *(A3, A4)*
1. Eval harness + golden set **first**.
2. Conversation-aware rewriting; merged understand-stage; concurrent verification.
3. Hybrid dense+lexical retrieval; adaptive budgets; fan-out+rerank.
4. Citations end to end.
5. Voice profile compiled and cached.

**Gate:** measurable improvement vs. Phase-2 baseline on the golden set — recall@10, citation precision, groundedness all up; p95 time-to-first-token down. No privacy regression.

### Phase 4 — Make it usable *(A6)*
Customer-facing interface: landing → sign-up → guided first question → chat with citations, streaming, error states, mobile, accessibility. Clerk onboarding flows.

**Gate:** Lighthouse + a11y thresholds; real end-to-end session on a phone; no unhandled error states.

### Phase 5 — Make it observable *(A7)*
Admin dashboard: live question feed, knowledge gaps, user analytics, cost/latency, eval history, privacy audit, lead pipeline.

**Gate:** Max can answer "what are people asking, what am I missing, what is it costing me" in under 30 seconds.

### Phase 6 — Make it convert *(A9)*
Lead capture and qualification; GoHighLevel contact/opportunity sync; conversation-driven lead scoring; retry-safe outbound queue.

**Gate:** a test lead flows from chat → GHL contact with tags, notes, and pipeline stage; failures retry without duplicating.

### Phase 7 — Make it speak *(A8)*
LiveKit Agents orchestration; Deepgram streaming STT with eager end-of-turn; `run_pipeline` as the `llm_node`; ElevenLabs Flash TTS on Max's professional voice clone; cached in-voice acknowledgement to cover retrieval latency; barge-in; fast path for conversational turns.

**Gate:** p95 turn latency within the agreed budget on real hardware over a real network; graceful degradation to text; **identical privacy guarantees as text** (same persona resolution, same filters — voice is a transport, not a bypass).

### Phase 8 — Make it reach *(A9)*
Instagram DMs via ManyChat/Chatfuel webhook into our answer engine, with persona pinned to `public`. Mandatory human escalation path and owner kill switch. `is_echo` filtering (the classic infinite-loop bug). Sub-30-second response budget enforced.

**Meta Business Verification starts in Phase 1**, in parallel — it gates App Review, which gates any future direct integration, and it is the longest pole in the schedule.

**Gate:** DMs answered within policy and inside the 30s budget; human handoff works; kill switch works; no echo loops under load.

### Phase 9 — Make it live *(all)*
Production hardening, staging environment, runbook, monitoring/alerting, backups, load test, launch.

---

## 5. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Private knowledge leaks to a stranger | **Existential** | Four-layer model with physical partitioning as primary; default-private; adversarial suite in CI; staged launch |
| Meta App Review rejected or slow | High | Start immediately in parallel; design Instagram as an isolated, last, optional module; middleware fallback |
| Cost blow-up from public traffic | High | Rate limits, per-user quotas, semantic caching, cheap-model routing, hard budget alarms |
| Voice quality/latency disappoints | Medium | Prove latency budget with a spike before committing; text always available |
| Retrieval rewrite regresses answers | Medium | Eval harness built before the rewrite; every change gated on it |
| Persona/voice cloning consent | Medium | It is Max's own voice; retain written consent record per vendor requirements |
| Solo-owner bus factor on ops | Medium | Runbook, IaC-style config, documented recovery |

---

## 6. Decisions I need from Max

These block or materially shape work. Defaults are what I will assume if I hear nothing.

1. **Who can use it?** Public/anonymous, or signup-gated? *(Default: signup-gated free tier; anonymous later.)*
2. **Paid tiers now or later?** *(Default: later — build the boundary, don't build billing yet.)*
3. **Privacy review capacity.** Corpus-wide tiering needs your sign-off. Bulk-approve by index, or review samples? *(Default: LLM proposes per index, you approve at index+topic granularity, default-private.)*
4. **Credentials for local verification.** I cannot verify retrieval, privacy, or latency end-to-end without Pinecone/OpenAI/Anthropic/Cohere/Clerk keys and a Postgres URL in a local `.env`. *(Blocking for Phase 3 validation.)*
5. **Instagram account + Meta developer access**, and whether direct API or middleware (ManyChat et al.) is acceptable.
6. **GoHighLevel** location ID + integration token, and the target pipeline/stages.
7. **Voice** — do you have an existing ElevenLabs voice clone, or do we create one?

---

## 7. Definition of done

The product is production-ready when:
- No unauthenticated or forged-token path reaches any data. *(A10 verified)*
- The adversarial privacy suite passes with zero leaks. *(CI enforced)*
- Golden-set groundedness and citation precision exceed agreed thresholds. *(CI enforced)*
- p95 time-to-first-token is within budget under load.
- A real user can sign up, ask, hear, and convert — on a phone.
- Max can see what people ask, what's missing, and what it costs.
- Errors alert someone; there is a runbook; there are backups.
