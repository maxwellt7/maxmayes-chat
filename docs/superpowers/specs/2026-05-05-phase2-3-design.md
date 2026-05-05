# Phase 2+3 Design: Pinecone Integration, Registry & Chat Pipeline

**Date:** 2026-05-05
**Approach:** Option B — Lean core, ship fast
**Scope:** Collapsed Phase 2 (Pinecone integration + registry) and Phase 3 (orchestration pipeline) into a single build

---

## 1. Goals

- Working end-to-end chat: user submits a query, gets a streamed response grounded in Pinecone data, written in Max's voice
- Admin dashboard to manage the Index Registry, seeded via auto-discovery from all 3 Pinecone projects
- No hallucination: accuracy verifier short-circuits before generation if context is insufficient

---

## 2. Architecture Overview

Two parallel tracks that connect at the chat endpoint:

**Track 1 — Registry & Admin (configuration layer)**
Auto-discovery reads all indexes from the 3 Pinecone projects and seeds the registry with technical metadata. Max writes domain descriptions and sample queries per index via the admin UI. This data drives the router.

**Track 2 — Chat Pipeline (execution layer)**
A single `orchestrator.py` service runs all 5 pipeline steps in sequence. It reads from the registry to determine which index to hit and which embedding model to use. The chat endpoint streams the response back via SSE.

**Voice profile:** Queried from the dedicated `max-copywriting-voice` Pinecone index at Step E. Top 5 chunks injected into the generation prompt as voice context.

---

## 3. New Dependencies

Add to `backend/requirements.txt`:
- `pinecone` — Pinecone Python SDK
- `openai` — embeddings (1536/2048 dim) and generation
- `cohere` — embeddings (1024 dim)

---

## 4. Backend: Chat Pipeline

Single service: `app/services/orchestrator.py`

### Step A — Query Optimizer
- Model: `gpt-4.1-mini`
- Input: raw user message
- Output: dense, search-optimized query string
- Strips conversational filler, expands context, corrects spelling

### Step B — Query Router
- Model: `gpt-4.1-mini`
- Input: optimized query + active registry catalog (index name, domain description, sample queries)
- Output: `{ index_name, project_id, confidence }` as JSON
- If confidence is low: returns top 2-3 candidates, retrieval runs in parallel across them, results merged before Step D

### Step C — Retrieval
1. Look up target index in registry → get `dimension`, `embedding_model`, `api_key_env_var`, `metric`
2. Generate query embedding at correct dimension using correct model
3. Initialize Pinecone client for the target project (lazy-init, cached by `project_id`)
4. Query with `top_k=10`
5. Return matched chunks + similarity scores

**Dimension → embedding model mapping:**
| Dimension | Model | Provider |
|---|---|---|
| 1024 | `embed-english-v3.0` | Cohere |
| 1536 | `text-embedding-3-small` | OpenAI |
| 2048 | `text-embedding-3-large` (truncated) | OpenAI |

### Step D — Accuracy Verifier
- Model: `gpt-4.1-mini`
- Input: retrieved chunks + original user query
- Output: `{ score: 0.0–1.0, recommendation: "proceed" | "proceed_with_caveat" | "insufficient_context" }`
- `insufficient_context` → short-circuit, return transparent fallback message (no generation)
- `proceed_with_caveat` → generation includes uncertainty disclaimer

### Step E — Voice Synthesizer
- Model: `claude-sonnet-4-6`
- Fetch top 5 chunks from `max-copywriting-voice` Pinecone index as voice context
- Combine: voice context + retrieved chunks + original query
- Strict grounding: LLM instructed to use only retrieved context, no external knowledge
- Stream response tokens back via SSE

### Chat Endpoint
`POST /api/chat` → `StreamingResponse` yielding SSE events
- Accepts `{ message: string, session_id: string }`
- Runs orchestrator, streams Step E output token by token
- Persists user message + assistant response to `chat_messages` table

---

## 5. Backend: Admin Layer

### Auto-Discovery
`POST /api/admin/discover`
- Calls Pinecone list-indexes API across all 3 projects (using `PINECONE_API_KEY_1/2/3`)
- Reads technical metadata per index (dimension, metric) from Pinecone
- Returns indexes not yet in the registry as draft candidates
- Client clicks "Import" to seed selected indexes; description + sample queries left blank

### Registry CRUD
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/admin/indexes` | List all (paginated) |
| `POST` | `/api/admin/indexes` | Create manually |
| `GET` | `/api/admin/indexes/{id}` | Fetch single entry |
| `PATCH` | `/api/admin/indexes/{id}` | Update fields |
| `DELETE` | `/api/admin/indexes/{id}` | Soft delete (sets `is_active=false`) |

### DB Migration
Alembic migration creates `index_registry` and `chat_messages` tables. Migration runs automatically at app startup via `alembic upgrade head` in the Dockerfile entrypoint.

**`chat_messages` table schema:**
| Column | Type |
|---|---|
| `id` | UUID (PK) |
| `session_id` | VARCHAR |
| `user_id` | VARCHAR (from Clerk JWT) |
| `role` | VARCHAR (`user` / `assistant`) |
| `content` | TEXT |
| `created_at` | TIMESTAMP |

---

## 6. Frontend: Admin UI

### `/admin` — Index Registry Table
- Table columns: index name, project, dimension, embedding model, active toggle, last updated, Edit button
- "Discover from Pinecone" button → calls `/api/admin/discover`, shows modal with discovered indexes, "Import Selected" action
- Active/inactive toggle calls `PATCH` endpoint inline

### `/admin/indexes/[id]` — Edit Page
Full-screen edit form per index:
- `domain_description` — large textarea (this is the most important field for routing accuracy)
- `sample_queries` — tag-style input (add/remove individual query strings)
- Technical fields: index name, project ID, API key env var, dimension, embedding model, metric (editable but pre-filled)
- `is_active` toggle
- Save and Cancel buttons
- Navigates back to `/admin` on save

---

## 7. Frontend: Chat UI

### `/chat` — Full Rebuild
Replace current placeholder with a real chat interface.

**Layout:** Full-height page, scrollable message history, fixed input bar at bottom.

**Message display:**
- User messages: right-aligned bubbles
- Assistant messages: left-aligned bubbles with streaming typewriter effect
- Loading indicator while Steps A–D run before first token (~1–3s)

**Input behavior:**
- `Enter` → new line
- `Cmd+Enter` or Send button → submit
- Input disabled while response is streaming

**Message persistence:**
- Stored in `chat_messages` table via backend
- Loaded on page mount (survives refresh)
- Keyed by `session_id` (stored in localStorage)

**Error states:**
- `insufficient_context` → friendly inline message in the chat, not an error state
- Network error → toast notification, message preserved in input field

**New components:**
- `ChatWindow.tsx` — message list + auto-scroll
- `MessageBubble.tsx` — user vs assistant rendering, handles streaming state
- `StreamingResponse.tsx` — SSE consumer, feeds tokens to MessageBubble
- `ChatInput.tsx` — replaces current placeholder (Enter=newline, Cmd+Enter=submit)

---

## 8. File Changes Summary

**Backend — new files:**
- `app/services/orchestrator.py` — 5-step pipeline
- `app/services/pinecone_service.py` — client factory + retrieval
- `app/services/embedding_service.py` — dimension-aware embedding
- `app/migrations/` — Alembic migration for index_registry + chat_messages

**Backend — modified files:**
- `app/routers/admin.py` — replace placeholder with full CRUD + discover
- `app/routers/chat.py` — replace placeholder with SSE streaming endpoint
- `app/models/schemas.py` — add request/response schemas for new endpoints
- `requirements.txt` — add pinecone, openai, cohere
- `Dockerfile` — add `alembic upgrade head` to entrypoint

**Frontend — new files:**
- `src/app/admin/indexes/[id]/page.tsx` — index edit page
- `src/components/ChatWindow.tsx`
- `src/components/MessageBubble.tsx`
- `src/components/StreamingResponse.tsx`
- `src/lib/sse.ts` — SSE client helper

**Frontend — modified files:**
- `src/app/chat/page.tsx` — full rebuild
- `src/app/admin/page.tsx` — full rebuild with registry table + discover
- `src/components/ChatInput.tsx` — update input behavior
- `src/lib/api.ts` — add admin + streaming API methods

---

## 9. Out of Scope (deferred to Phase 4)

- Performance profiling and SLO enforcement
- OpenTelemetry tracing
- Rate limiting and abuse detection
- Multi-index parallel fallback UI indicators
- Conversation management (rename, delete sessions)
