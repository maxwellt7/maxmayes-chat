import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import log_and_convert
from app.db.database import get_db, get_session_factory, session_scope
from app.models.chat import ChatMessage
from app.models.schemas import ChatRequest
from app.security.dependencies import get_current_principal
from app.security.principal import Principal
from app.services import rate_limit, spend_guard
from app.services.cost import estimate_chat_request_usd
from app.services.orchestrator import run_pipeline

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _client_ip(request: Request) -> str:
    """The caller's address, as well as it can be known.

    Behind Railway's proxy `request.client.host` is the proxy, so
    `X-Forwarded-For` is consulted when configured. The leftmost entry is the
    client as reported by the proxy chain — and is caller-controlled, so per-IP
    limiting is a speed bump rather than a boundary. Per-account limiting and the
    global spend ceiling are the controls that actually bound cost.
    """
    if settings.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


@router.post("")
async def chat_endpoint(
    payload: ChatRequest,
    request: Request,
    principal: Principal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    request_id = str(uuid.uuid4())
    user_id = principal.user_id

    # --- everything below runs on the request-scoped session, before streaming ---

    # Order matters. Rate limits are checked before anything is written or
    # reserved, so a flood costs one indexed row lock rather than a pipeline run.
    rate_limit.enforce(
        db, f"acct:{user_id}", rate_limit.account_policy()
    )
    rate_limit.enforce(
        db, f"ip:{rate_limit.hash_ip(_client_ip(request))}", rate_limit.ip_policy()
    )

    # Reserve the estimated cost against the global daily ceiling *before* any
    # provider call. Raises and refuses if the day's budget is spent.
    estimated_usd = estimate_chat_request_usd(payload.message)
    spend_guard.reserve(
        db,
        estimated_usd,
        user_id=user_id,
        request_id=request_id,
        kind="chat",
    )

    db.add(ChatMessage(
        session_id=payload.session_id,
        user_id=user_id,
        role="user",
        content=payload.message,
    ))
    db.commit()

    # Hand the pooled connection back before a single token is generated.
    #
    # FastAPI holds a `yield` dependency open until the response is finished, and
    # for a StreamingResponse that is the whole stream — up to the ~120s timeout
    # budget. With a pool of 10 plus 20 overflow, thirty concurrent chats would
    # take every connection and the thirty-first request, including unrelated
    # ones, would block for `pool_timeout` and fail. `get_current_principal`
    # shares this same session via FastAPI's dependency cache, so closing it here
    # releases the auth path's connection too.
    #
    # `db` must not be used again after this point. `Session.close()` is
    # idempotent and the session would silently check out a *new* connection on
    # next use, which would quietly reintroduce the bug. Post-stream work uses
    # `session_scope()` instead.
    db.close()

    accumulated: list[str] = []

    def _produced_an_answer() -> bool:
        # Whitespace-only output is not an answer: a pipeline that emitted one
        # empty chunk before failing has produced nothing worth keeping and
        # nothing worth charging for.
        return bool("".join(accumulated).strip())

    def _persist_reply(session: Session) -> None:
        session.add(ChatMessage(
            session_id=payload.session_id,
            user_id=user_id,
            role="assistant",
            content="".join(accumulated),
        ))
        session.commit()

    def _settle_unfinished() -> None:
        """Persist and reconcile a turn that ended without reaching either the
        success or the error path.

        In practice that means the client hung up: Starlette throws
        `GeneratorExit`/`CancelledError` into the generator at the `yield`, and
        neither derives from `Exception`, so the handler below never sees it.
        Previously that lost the turn *and* leaked the reservation, so a client
        that disconnected repeatedly would ratchet the day's ceiling closed
        against everyone. Same rule as the error path: tokens the user saw are
        kept and paid for, a turn that produced nothing is refunded.
        """
        try:
            with session_scope() as session:
                if _produced_an_answer():
                    _persist_reply(session)
                else:
                    spend_guard.release(session, estimated_usd)
        except Exception:
            _log.error(
                "chat_stream_settle_failed request_id=%s", request_id, exc_info=True
            )

    async def event_stream() -> AsyncIterator[str]:
        settled = False
        try:
            async for token_text in run_pipeline(
                payload.message, get_session_factory()
            ):
                accumulated.append(token_text)
                data = json.dumps({"token": token_text, "request_id": request_id})
                yield f"data: {data}\n\n"

            with session_scope() as session:
                _persist_reply(session)
            settled = True
            yield "data: [DONE]\n\n"
        except Exception as exc:
            settled = True
            # The client learns a code and a request ID. Everything that could
            # identify an index, a namespace, a provider or a query stays in the
            # log, keyed by that ID.
            client = log_and_convert(
                exc, request_id=request_id, context="chat_stream"
            )

            if _produced_an_answer():
                # Tokens already reached the user, so they were really paid for —
                # refunding here would understate spend, which is the unsafe
                # direction. Keep what was generated rather than dropping the
                # partial answer on the floor.
                try:
                    with session_scope() as session:
                        _persist_reply(session)
                except Exception:
                    _log.error(
                        "partial_reply_persist_failed request_id=%s",
                        request_id,
                        exc_info=True,
                    )
            else:
                # Nothing was produced, so the reservation is returned. Otherwise
                # a failing provider would burn the day's budget on empty answers.
                with session_scope() as session:
                    spend_guard.release(session, estimated_usd)

            yield f"data: {json.dumps(client.payload(request_id))}\n\n"
        finally:
            if not settled:
                _settle_unfinished()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions")
async def list_sessions(
    principal: Principal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    user_id = principal.user_id
    rows = (
        db.query(
            ChatMessage.session_id,
            func.min(ChatMessage.created_at).label("created_at"),
            func.max(ChatMessage.created_at).label("last_message_at"),
            func.count(ChatMessage.id).label("message_count"),
        )
        .filter(ChatMessage.user_id == user_id)
        .group_by(ChatMessage.session_id)
        .order_by(func.max(ChatMessage.created_at).desc())
        .limit(50)
        .all()
    )
    previews = {}
    for row in rows:
        first = (
            db.query(ChatMessage.content)
            .filter(
                ChatMessage.session_id == row.session_id,
                ChatMessage.user_id == user_id,
                ChatMessage.role == "user",
            )
            .order_by(ChatMessage.created_at)
            .first()
        )
        previews[row.session_id] = (first[0][:80] if first else "") if first else ""
    return {
        "sessions": [
            {
                "session_id": row.session_id,
                "preview": previews.get(row.session_id, ""),
                "message_count": row.message_count,
                "created_at": row.created_at.isoformat(),
                "last_message_at": row.last_message_at.isoformat(),
            }
            for row in rows
        ]
    }


@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    principal: Principal = Depends(get_current_principal),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    # `session_id` comes from the caller, so the `user_id` predicate is the only
    # thing standing between one user and another's transcript. It is not
    # optional and must not be relaxed into a post-filter.
    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == principal.user_id,
        )
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
