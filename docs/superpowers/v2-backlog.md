# v2 Backlog — Deferred from RAG Redesign Spec

Living document. Items here are explicitly out of scope for the current redesign but tracked so they don't get lost.

## Deferred from Section 3 (Pipeline Data Flow)
- **Memory retention policy.** The `conversation-memory` index grows unbounded in v1. v2 needs:
  - Decay heuristic (e.g., halve weight on chunks > 90 days old at recall time)
  - Periodic summarization job (collapse N old turns into a single summary chunk)
  - Per-user storage caps with eviction (oldest first or lowest-recall-frequency first)
  - User-facing "forget this" controls

## Quality-over-cost decisions logged from Section 4
These were upgraded vs the original "shared function / one LLM call" recommendation. Logging them so we can revisit if cost ever becomes a constraint:

- Per-agent Plan nodes are custom Python modules, not a shared config function. Cost: more code to maintain. Benefit: domain-specific retrieval intelligence (e.g., the `voice` agent can do something fundamentally different from the `verticals` agent).
- Within-domain Cohere rerank PLUS cross-domain Cohere rerank. Doubles Cohere call volume per query. Worth it for top-K precision.
- Refinement-query rewriter is a separate LLM call from self-evaluation. Adds ~150ms + ~$0.0001 per refine. Benefit: a focused rewriter prompt produces better refined queries than asking one model to evaluate AND rewrite in one JSON output.

## Deferred from Section 5 (Synthesis)
- Manual persona override dropdown for `content_creation` mode. Classifier auto-detects persona_target in v1; if it gets it wrong, user has to rephrase. v2: per-message dropdown.
- Per-mode style-guide variants. v1 ships one MAX_STYLE_GUIDE used across all modes (with mode rules layering on top). v2: separate guides per mode if voice drift becomes an issue (e.g., public mode wants slightly more measured cadence).
- Re-streaming optimization: cache the retrieved chunks so a mode-switch re-stream skips retrieval. v1 keeps it simple by caching for 5 min in Postgres; v2 could move to Redis.

## Decisions logged from Section 6 (Storage)
- Backup index `legacy-archive-{date}` retained in Pinecone for 30 days, then exported to S3/R2 as compressed JSONL and the Pinecone index is deleted. Need to wire the export job in v1 implementation.
- "Show debug" toggle on each assistant message in the UI surfaces `agent_traces` + `message_classifications` per response. Collapsed by default. v2 could expand this into a full conversation-replay tool.

(More items will be appended as the spec progresses.)
