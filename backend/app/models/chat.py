import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.db.types import uuid_type


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        # The queries that actually run are "this user's messages in this
        # session, in order" and "this user's sessions, most recent first".
        # Neither is served by the single-column indexes below.
        Index("ix_chat_messages_user_session", "user_id", "session_id", "created_at"),
        Index("ix_chat_messages_user_created", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        uuid_type(), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
