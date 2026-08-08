import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.db.types import json_type, uuid_type


class IngestJob(Base):
    __tablename__ = "ingest_jobs"
    # Named explicitly to match what migration 002 created and what is deployed.
    # Left to SQLAlchemy's default naming this becomes `ix_ingest_jobs_status`,
    # and autogenerate then proposes dropping the real index and adding an
    # identical one under a new name on every run.
    __table_args__ = (Index("ingest_jobs_status_idx", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(
        uuid_type(), primary_key=True, default=uuid.uuid4
    )
    source_index: Mapped[str] = mapped_column(String(255), nullable=False)
    target_index: Mapped[str] = mapped_column(String(255), nullable=False)
    target_namespace: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processed_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
