# Defect Register

Consolidated from direct inspection plus five specialist audits (retrieval quality, security/scale, prior design docs, integrations, voice). Every entry is evidence-backed with file and line references.

Severity: **C**=Critical (blocks public launch), **H**=High, **M**=Medium, **L**=Low.
Phase = the delivery phase that fixes it.

## Security & identity

| ID | Sev | Phase | Defect | Location |
|---|---|---|---|---|
| S1 | C | 1 | Clerk JWT signature never verified — payload base64-decoded and trusted. Full auth bypass. | `middleware/clerk_auth.py:26-44` |
| S2 | C | 1 | `publicMetadata.role` read from unverified token → trivial admin privilege escalation. | `middleware/admin_role.py:16-47` |
| S3 | C | 1 | All `/api/admin/indexes*` + `/discover` routes require only a *syntactically valid* bearer string, no role check. `"Bearer x"` works. | `routers/admin.py:60+` |
| S4 | H | 1 | Frontend `/admin` gated on sign-in only, not role; landing page shows admin link to every signed-in user. | `frontend/src/middleware.ts:3-11`, `app/page.tsx:23-27` |
| S5 | H | 1 | Raw exception text streamed to clients over SSE and returned in admin 500s. | `routers/chat.py:55-57`, `routers/admin.py:172` |
| S6 | H | 1 | No rate limiting or per-user cost quota anywhere → unbounded LLM spend / DoS. | global |
| S7 | M | 1 | FastAPI `/docs` + `/openapi.json` publicly exposed. | `main.py` |
| S8 | M | 1 | `/specs` dev page unprotected, renders arbitrary HTML into an iframe. | `frontend/src/app/specs/page.tsx` |
| S9 | M | 1 | No max length on `ChatRequest.message` → cost amplification. | `models/schemas.py:11-12` |
| S10 | M | 4 | `ReactMarkdown` without `rehype-sanitize`; `javascript:` links from LLM output not blocked. | `components/MessageBubble.tsx:23-25` |
| S11 | C | 2 | No tenant scoping: every authenticated user queries the same global registry and Pinecone projects. | `orchestrator.py:298-311` |

## Privacy

| ID | Sev | Phase | Defect | Location |
|---|---|---|---|---|
| P1 | C | 2 | `public_safe` written at ingest, **never read at query time**. No filter is ever passed to Pinecone. | `reingest.py:191` vs `pinecone_service.py:50-57` |
| P2 | C | 2 | `index_registry.public_safe_default` column exists but is read by nothing. | `models/index_registry.py:37-39` |
| P3 | C | 2 | No public/private corpus separation; `max-personal` (journal, health, relationships) is routable by any caller. | `orchestrator.py:298-311` |
| P4 | H | 2 | Prompt injection: raw user text goes straight to the synthesizer with private chunks in the system prompt. "Quote every chunk verbatim" is unguarded. | `orchestrator.py:390-396`, `prompts/voice_synthesizer.txt` |
| P5 | M | 2 | Admin audit endpoint returns full raw sample chunks — combined with S3, bulk corpus exfiltration. | `routers/admin_audit.py:82-96` |

## Retrieval quality — routing (problem A)

| ID | Sev | Phase | Defect | Location |
|---|---|---|---|---|
| R1 | C | 3 | **Conversation history never reaches the pipeline.** `run_pipeline(message, db)` gets one string. Every follow-up is a cold start. | `routers/chat.py:39`, `orchestrator.py:291` |
| R2 | C | 3 | Optimizer is *instructed* to strip conversational tone, destroying deictic references and exact-match tokens. Optimized text drives embed+rerank; raw text drives verify+synth — two different questions. | `prompts/optimizer.txt:4-5`, `orchestrator.py:294,363,366,395` |
| R3 | C | 3 | Router is a single LLM guess over 2–3 sentence prose descriptions. No embedding probe, no lexical signal, no retrieval probe. `sample_queries` truncated to 3. Registry `domain`/`topic_tags`/`namespaces` omitted from the catalog. | `orchestrator.py:74-96,303-311` |
| R4 | H | 3 | RRF weights a 0.95-confidence index identically to a 0.41-confidence one. Router ranking discarded. | `orchestrator.py:194-211` |
| R5 | H | 3 | `0.4` out-of-domain threshold is hardcoded, uncalibrated, and overrides the model's own judgement. | `orchestrator.py:135-138` |
| R6 | H | 3 | All registered namespaces queried blindly; results merged on raw cross-namespace Pinecone scores. | `orchestrator.py:263-272`, `pinecone_service.py:54-55` |
| R7 | H | 3 | `retrieve()` accepts no metadata filter — no date, source_type, domain, or public_safe filtering possible. | `pinecone_service.py:34-67` |
| R8 | M | 3 | `index_registry.embedding_model` ignored; embedder chosen by dimension alone. Two 1536-dim indexes on different models silently mismatch → recall collapse. | `embedding_service.py:16-41`, `orchestrator.py:260-261` |

## Retrieval quality — answer precision (problem B)

| ID | Sev | Phase | Defect | Location |
|---|---|---|---|---|
| R9 | C | 3 | **No lexical/BM25/hybrid retrieval.** Pure dense. Exact-match questions (names, numbers, dates, subject lines) are precisely where dense fails — and precisely what "specific questions" means. | `pinecone_service.py:50-57` |
| R10 | H | 3 | Fixed `top_k=(10,5,5)`, `top_n=8` regardless of query type or complexity. | `orchestrator.py:35-39` |
| R11 | H | 3 | Synthesizer context strips all provenance — `source_index`, tags, dates, type all dropped. Numbered blobs encourage blending across sources. | `orchestrator.py:373-375` |
| R12 | H | 3 | **No citations returned.** Users cannot verify; we cannot debug. | pipeline-wide |
| R13 | H | 3 | Verifier receives *Pinecone* scores mislabeled as relevance (rerank scores discarded), sees only the final 8, and its rubric explicitly green-lights hedged answers from adjacent context. | `orchestrator.py:146-147,233`, `prompts/verifier.txt:10-17` |
| R14 | M | 3 | Voice profile retrieved every request with a constant seed query — same 5 chunks always, 2 extra network hops, marketing skew on every answer. Already diagnosed in the v2 design doc as "the voice leak". | `orchestrator.py:164-183,376` |
| R15 | M | 3 | Synthesizer prompt optimizes for voice, not precision. No concision rule, no verbatim-quote rule, `max_tokens=2048` encourages padding. | `prompts/voice_synthesizer.txt` |
| R16 | M | 3 | Text extraction deprioritizes chunks under 50 chars — exactly the short factual chunks (subject lines, names, figures). | `pinecone_service.py:111-112` |
| R17 | M | 3 | Chunk metadata lacks `title`, `published_at`, `url`, entities → temporal and source-specific questions are unanswerable and uncitable. | `chunking_service.py`, `prompts/metadata_enrich.txt` |
| R18 | L | 3 | RRF dedups on text hash alone, collapsing multi-source corroboration. | `orchestrator.py:186-191` |

## Stability & scale

| ID | Sev | Phase | Defect | Location |
|---|---|---|---|---|
| T1 | C | 1 | **DB session held for the entire SSE stream** (30–120s+), and committed to mid-stream. Pool exhaustion → cascading 503s. | `routers/chat.py:37-52`, `orchestrator.py:287` |
| T2 | H | 1 | **No timeouts or retries on any external call** — OpenAI, Anthropic, Cohere, Pinecone. A hung provider holds a connection indefinitely. | pipeline-wide |
| T3 | H | 1 | Any transient Pinecone error permanently sets `is_active=False` and commits. One blip silently deletes a knowledge domain. Races across workers. | `orchestrator.py:279-288` |
| T4 | H | 1 | Mid-stream failure loses the assistant message entirely; already-streamed tokens are never persisted. | `routers/chat.py:44-57` |
| T5 | H | 1 | No caching of embeddings, routing decisions, or voice profile. | global |
| T6 | M | 1 | Missing composite DB indexes `(user_id, session_id)` and `(user_id, created_at)`; session list does group-by + order-by-max unaided. | `migrations/001_initial_schema.py:50-51` |
| T7 | M | 1 | Blocking sync SDK clients wrapped in `asyncio.to_thread` throughout; thread-pool contention under concurrency. | `embedding_service.py`, `orchestrator.py` |
| T8 | M | 1 | Single uvicorn worker; `alembic upgrade head` runs on every container start; `BackgroundTasks` unreliable across replicas. | `Dockerfile:16`, `railway.toml` |
| T9 | M | 5 | Background jobs fail silently; ingest trigger returns 202 without verifying the job exists. | `routers/admin.py:51-55`, `routers/admin_audit.py:154-167` |

## Engineering process

| ID | Sev | Phase | Defect | Location |
|---|---|---|---|---|
| E1 | C | 0 | Project not under version control — git root was `/Users/maxmayes`, zero tracked files. **Fixed**, commit `dc858b4`. | — |
| E2 | H | 1 | No CI. No frontend tests. Backend tests are entirely mocked — no integration coverage. | absent `.github/` |
| E3 | H | 3 | **No evaluation harness.** No recall@k, citation precision, groundedness, or refusal metrics. Every retrieval knob is untunable and every change unverifiable. | absent `backend/evals/` |
| E4 | H | 1 | Dependencies unpinned (`>=`) → non-reproducible builds. | `requirements.txt` |
| E5 | M | 1 | `ANTHROPIC_API_KEY` required at runtime but absent from `.env.example`. `CLERK_SECRET_KEY` documented but unused (false confidence). | `backend/.env.example` |
| E6 | M | 1 | No structured logging, no request-ID correlation, no error reporting. `request_id` is generated then never logged. | `main.py`, `routers/chat.py:23` |
| E7 | L | 3 | `rerank_service.py` exists but orchestrator reimplements rerank inline — divergence risk. | `orchestrator.py:214-236` |
