import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base

# Roles are stored here rather than read from Clerk token claims. A Clerk
# session token does not carry `publicMetadata` unless the instance is
# explicitly configured to add it, and metadata a user can influence must never
# be an authorization input. This table is the single source of truth.
ROLE_OWNER = "owner"
ROLE_MEMBER = "member"
VALID_ROLES = (ROLE_OWNER, ROLE_MEMBER)


class UserAccount(Base):
    __tablename__ = "user_accounts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    clerk_user_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ROLE_MEMBER, server_default=ROLE_MEMBER
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
