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
