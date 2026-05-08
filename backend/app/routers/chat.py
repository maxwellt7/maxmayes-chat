import json
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.middleware.clerk_auth import extract_user_id, require_bearer_token
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
    user_id = extract_user_id(token)

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


@router.get("/sessions")
async def list_sessions(
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    from sqlalchemy import func
    user_id = extract_user_id(token)
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
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    user_id = extract_user_id(token)
    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == user_id,
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
