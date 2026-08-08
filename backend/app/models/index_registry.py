import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.db.types import json_type, text_array_type, uuid_type


class IndexRegistry(Base):
    __tablename__ = "index_registry"

    id: Mapped[uuid.UUID] = mapped_column(
        uuid_type(), primary_key=True, default=uuid.uuid4
    )
    index_name: Mapped[str] = mapped_column(String(255), nullable=False)
    project_id: Mapped[str] = mapped_column(String(50), nullable=False)
    api_key_env_var: Mapped[str] = mapped_column(String(255), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    metric: Mapped[str] = mapped_column(String(50), nullable=False)
    domain_description: Mapped[str] = mapped_column(Text, nullable=False)
    sample_queries: Mapped[list[str]] = mapped_column(
        text_array_type(), nullable=False
    )
    namespaces: Mapped[dict[str, Any]] = mapped_column(
        json_type(), nullable=False, default=dict
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    domain: Mapped[str | None] = mapped_column(String(100), nullable=True)
    public_safe_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    agent_module_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")
