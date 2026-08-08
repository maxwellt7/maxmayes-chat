import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.db.types import json_type, uuid_type


class IndexAudit(Base):
    __tablename__ = "index_audits"
    __table_args__ = (
        UniqueConstraint(
            "audit_date", "index_name", "project_id", name="index_audits_unique"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        uuid_type(), primary_key=True, default=uuid.uuid4
    )
    audit_date: Mapped[date] = mapped_column(Date, nullable=False)
    index_name: Mapped[str] = mapped_column(String(255), nullable=False)
    project_id: Mapped[str] = mapped_column(String(50), nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    dominant_domain: Mapped[str | None] = mapped_column(String(100), nullable=True)
    topic_tags: Mapped[list] = mapped_column(json_type(), nullable=False, default=list)
    sample_chunks: Mapped[list] = mapped_column(json_type(), nullable=False)
    proposed_disposition: Mapped[str] = mapped_column(String(50), nullable=False)
    proposed_target_index: Mapped[str | None] = mapped_column(String(255), nullable=True)
    approved_disposition: Mapped[str | None] = mapped_column(String(50), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    executed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
