import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pinecone import Pinecone
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.middleware.clerk_auth import require_bearer_token
from app.models.index_registry import IndexRegistry
from app.models.schemas import (
    DiscoveredIndex,
    IndexRegistryCreate,
    IndexRegistryResponse,
    IndexRegistryUpdate,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/indexes")
async def list_indexes(
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> dict:
    total = db.query(IndexRegistry).count()
    indexes = db.query(IndexRegistry).offset(skip).limit(limit).all()
    return {
        "indexes": [IndexRegistryResponse.model_validate(idx) for idx in indexes],
        "count": total,
    }


@router.post("/indexes", status_code=status.HTTP_201_CREATED)
async def create_index(
    payload: IndexRegistryCreate,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> IndexRegistryResponse:
    existing = (
        db.query(IndexRegistry)
        .filter(
            IndexRegistry.index_name == payload.index_name,
            IndexRegistry.project_id == payload.project_id,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Index '{payload.index_name}' in project '{payload.project_id}' already exists",
        )
    entry = IndexRegistry(**payload.model_dump())
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return IndexRegistryResponse.model_validate(entry)


@router.get("/indexes/{index_id}")
async def get_index(
    index_id: str,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> IndexRegistryResponse:
    try:
        entry_uuid = uuid.UUID(index_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid index_id")
    entry = db.query(IndexRegistry).filter(IndexRegistry.id == entry_uuid).first()
    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Index not found")
    return IndexRegistryResponse.model_validate(entry)


@router.patch("/indexes/{index_id}")
async def update_index(
    index_id: str,
    payload: IndexRegistryUpdate,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> IndexRegistryResponse:
    try:
        entry_uuid = uuid.UUID(index_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid index_id")
    entry = db.query(IndexRegistry).filter(IndexRegistry.id == entry_uuid).first()
    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Index not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(entry, field, value)
    db.commit()
    db.refresh(entry)
    return IndexRegistryResponse.model_validate(entry)


@router.delete("/indexes/{index_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_index(
    index_id: str,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> None:
    try:
        entry_uuid = uuid.UUID(index_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid index_id")
    entry = db.query(IndexRegistry).filter(IndexRegistry.id == entry_uuid).first()
    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Index not found")
    entry.is_active = False
    db.commit()


@router.post("/discover")
async def discover_indexes(
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    discovered: list[DiscoveredIndex] = []

    existing_keys = {
        (idx.index_name, idx.project_id)
        for idx in db.query(IndexRegistry.index_name, IndexRegistry.project_id).all()
    }

    project_keys = {
        "1": settings.pinecone_api_key_1,
        "2": settings.pinecone_api_key_2,
        "3": settings.pinecone_api_key_3,
    }

    for project_id, api_key in project_keys.items():
        if not api_key:
            continue

        try:
            pc = Pinecone(api_key=api_key)
            indexes = pc.list_indexes()
            for idx in indexes:
                name = idx.name
                dimension = getattr(idx, "dimension", None)
                metric = getattr(idx, "metric", "cosine")

                # Fetch full spec if dimension wasn't on the list response
                if dimension is None:
                    try:
                        desc = pc.describe_index(name)
                        dimension = desc.dimension
                        metric = desc.metric
                    except Exception:
                        dimension = 0

                discovered.append(
                    DiscoveredIndex(
                        index_name=name,
                        project_id=project_id,
                        dimension=dimension or 0,
                        metric=metric,
                        already_in_registry=(name, project_id) in existing_keys,
                    )
                )
        except Exception:
            # Skip projects whose Pinecone calls fail; don't block the others.
            continue

    return {"discovered": [d.model_dump() for d in discovered], "count": len(discovered)}
