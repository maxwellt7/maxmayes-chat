import asyncio
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import SessionLocal, get_db
from app.middleware.clerk_auth import require_bearer_token
from app.models.index_registry import IndexRegistry
from app.models.schemas import (
    DiscoveredIndex,
    IndexRegistryCreate,
    IndexRegistryResponse,
    IndexRegistryUpdate,
)
from app.services.auto_describe import generate_index_description
from app.services.pinecone_service import get_factory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _parse_index_id(index_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(index_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid index_id",
        )


async def _auto_describe_and_save(index_id: uuid.UUID) -> None:
    """Background task: regenerate description + sample queries for an index."""
    db = SessionLocal()
    try:
        entry = db.query(IndexRegistry).filter(IndexRegistry.id == index_id).first()
        if not entry:
            return
        try:
            result = await generate_index_description(
                entry.index_name, entry.project_id, entry.dimension
            )
            entry.domain_description = result["domain_description"]
            entry.sample_queries = result["sample_queries"]
            db.commit()
        except Exception as exc:
            logger.warning(
                "auto-describe failed for %s (%s): %s",
                entry.index_name, index_id, exc,
            )
    finally:
        db.close()


@router.get("/indexes")
async def list_indexes(
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    include_inactive: bool = False,
) -> dict:
    base_query = db.query(IndexRegistry)
    if not include_inactive:
        base_query = base_query.filter(IndexRegistry.is_active == True)  # noqa: E712
    total = base_query.count()
    indexes = base_query.offset(skip).limit(limit).all()
    return {
        "indexes": [IndexRegistryResponse.model_validate(idx) for idx in indexes],
        "count": total,
    }


@router.post("/indexes", status_code=status.HTTP_201_CREATED)
async def create_index(
    payload: IndexRegistryCreate,
    background_tasks: BackgroundTasks,
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

    # Auto-describe in the background if the user didn't provide one
    if not entry.domain_description.strip():
        background_tasks.add_task(_auto_describe_and_save, entry.id)

    return IndexRegistryResponse.model_validate(entry)


@router.get("/indexes/{index_id}")
async def get_index(
    index_id: str,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> IndexRegistryResponse:
    entry = db.query(IndexRegistry).filter(IndexRegistry.id == _parse_index_id(index_id)).first()
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
    entry = db.query(IndexRegistry).filter(IndexRegistry.id == _parse_index_id(index_id)).first()
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
    entry = db.query(IndexRegistry).filter(IndexRegistry.id == _parse_index_id(index_id)).first()
    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Index not found")
    entry.is_active = False
    db.commit()


@router.post("/indexes/{index_id}/auto-describe")
async def auto_describe_index(
    index_id: str,
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> IndexRegistryResponse:
    entry = db.query(IndexRegistry).filter(IndexRegistry.id == _parse_index_id(index_id)).first()
    if not entry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Index not found")

    try:
        result = await generate_index_description(
            entry.index_name, entry.project_id, entry.dimension
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Auto-describe failed: {exc}",
        )

    entry.domain_description = result["domain_description"]
    entry.sample_queries = result["sample_queries"]
    db.commit()
    db.refresh(entry)
    return IndexRegistryResponse.model_validate(entry)


@router.post("/indexes/auto-describe-all")
async def auto_describe_all(
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
    only_empty: bool = True,
) -> dict:
    query = db.query(IndexRegistry)
    if only_empty:
        query = query.filter(IndexRegistry.domain_description == "")
    entries = query.all()

    succeeded: list[str] = []
    failed: list[dict] = []
    for entry in entries:
        try:
            result = await generate_index_description(
                entry.index_name, entry.project_id, entry.dimension
            )
            entry.domain_description = result["domain_description"]
            entry.sample_queries = result["sample_queries"]
            db.commit()
            succeeded.append(entry.index_name)
        except Exception as exc:
            db.rollback()
            failed.append({"index_name": entry.index_name, "error": str(exc)})
            logger.warning("auto-describe-all failed for %s: %s", entry.index_name, exc)

    return {
        "succeeded": succeeded,
        "failed": failed,
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
    }


@router.post("/indexes/health-check")
async def health_check_all_indexes(
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    """Ping each active index in Pinecone. Deactivate any that 404."""
    active = db.query(IndexRegistry).filter(IndexRegistry.is_active == True).all()  # noqa: E712
    factory = get_factory()
    healthy: list[str] = []
    deactivated: list[dict] = []

    for entry in active:
        try:
            client = factory.get_client(entry.project_id)
            # Just hit describe_index — no need to embed/query
            await asyncio.to_thread(client.describe_index, entry.index_name)
            healthy.append(entry.index_name)
        except Exception as exc:
            entry.is_active = False
            db.commit()
            deactivated.append({"index_name": entry.index_name, "reason": str(exc)[:200]})
            logger.warning("health-check deactivated %s: %s", entry.index_name, exc)

    return {
        "healthy": healthy,
        "healthy_count": len(healthy),
        "deactivated": deactivated,
        "deactivated_count": len(deactivated),
    }


@router.post("/discover")
async def discover_indexes(
    token: str = Depends(require_bearer_token),
    db: Session = Depends(get_db),
) -> dict:
    discovered: list[DiscoveredIndex] = []
    partial_failures: list[str] = []

    existing_keys = {
        (idx.index_name, idx.project_id)
        for idx in db.query(IndexRegistry.index_name, IndexRegistry.project_id).all()
    }

    factory = get_factory()

    for project_id in ("1", "2", "3"):
        try:
            pc = factory.get_client(project_id)
        except ValueError:
            # Project not configured — skip silently
            continue

        try:
            indexes = await asyncio.to_thread(pc.list_indexes)
            for idx in indexes:
                name = idx.name
                dimension = getattr(idx, "dimension", None)
                metric = getattr(idx, "metric", "cosine")

                if dimension is None:
                    try:
                        desc = await asyncio.to_thread(pc.describe_index, name)
                        dimension = desc.dimension
                        metric = desc.metric
                    except Exception as inner_exc:
                        logger.warning(
                            "describe_index failed for %s in project %s: %s",
                            name, project_id, inner_exc,
                        )
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
        except Exception as exc:
            failure_msg = f"project {project_id}: {type(exc).__name__}: {exc}"
            logger.warning("discover failed: %s", failure_msg)
            partial_failures.append(failure_msg)
            continue

    return {
        "discovered": [d.model_dump() for d in discovered],
        "count": len(discovered),
        "partial_failures": partial_failures,
    }
