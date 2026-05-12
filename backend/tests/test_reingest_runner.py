"""Tests for app.scripts.reingest.run_ingest_job."""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_job():
    """Build a fake IngestJob with attribute assignment."""
    job = MagicMock()
    job.id = uuid.uuid4()
    job.source_index = "src-index"
    job.target_index = "max-marketing"
    job.target_namespace = "email"
    job.status = "pending"
    job.processed_chunks = 0
    job.failed_chunks = 0
    job.total_chunks = None
    job.started_at = None
    job.completed_at = None
    job.config = {"source_project_id": "1"}
    return job


@pytest.mark.asyncio
async def test_run_ingest_job_marks_completed_on_success():
    from app.scripts.reingest import run_ingest_job

    job = _make_job()
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = job

    fake_chunks = [
        {
            "id": f"c{i}",
            "text": "x" * 200,
            "metadata": {"source_id": "doc1", "chunk_position": i},
        }
        for i in range(3)
    ]

    with patch("app.scripts.reingest.SessionLocal", return_value=fake_db), patch(
        "app.scripts.reingest.IngestJob"
    ), patch(
        "app.scripts.reingest.fetch_source_chunks",
        new=AsyncMock(return_value=fake_chunks),
    ), patch(
        "app.scripts.reingest.semantic_chunk_text",
        return_value=["combined chunk text " * 10],
    ), patch(
        "app.scripts.reingest.enrich_metadata",
        new=AsyncMock(return_value={
            "topic_tags": ["a"],
            "source_type": "email",
            "public_safe": True,
        }),
    ), patch(
        "app.scripts.reingest.generate_embeddings_cohere_batch",
        return_value=[[0.1] * 1024],
    ), patch("app.scripts.reingest.get_factory") as mock_factory:
        mock_index = MagicMock()
        mock_factory.return_value.get_client.return_value.Index.return_value = mock_index

        await run_ingest_job(str(job.id))

    assert job.status == "completed"
    assert job.processed_chunks == 1  # one chunk after semantic_chunk_text collapsed to 1
    assert mock_index.upsert.called


@pytest.mark.asyncio
async def test_run_ingest_job_resumes_from_processed_offset():
    """If job already has processed_chunks > 0, the runner should skip that
    prefix of work on next invocation."""
    from app.scripts.reingest import run_ingest_job

    job = _make_job()
    job.processed_chunks = 100  # pretend we already did 100 chunks
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = job

    # Source has 3 source-chunks → re-chunked to 3 → already-processed
    fake_chunks = [
        {"id": f"c{i}", "text": "x" * 200,
         "metadata": {"source_id": f"d{i}", "chunk_position": 0}}
        for i in range(3)
    ]

    with patch("app.scripts.reingest.SessionLocal", return_value=fake_db), patch(
        "app.scripts.reingest.IngestJob"
    ), patch(
        "app.scripts.reingest.fetch_source_chunks",
        new=AsyncMock(return_value=fake_chunks),
    ), patch(
        "app.scripts.reingest.semantic_chunk_text",
        return_value=["x" * 200],
    ), patch(
        "app.scripts.reingest.enrich_metadata",
        new=AsyncMock(return_value={
            "topic_tags": [],
            "source_type": "other",
            "public_safe": False,
        }),
    ), patch(
        "app.scripts.reingest.generate_embeddings_cohere_batch",
        return_value=[[0.1] * 1024],
    ), patch("app.scripts.reingest.get_factory") as mock_factory:
        mock_index = MagicMock()
        mock_factory.return_value.get_client.return_value.Index.return_value = mock_index

        await run_ingest_job(str(job.id))

    # processed_chunks (100) > total_chunks (3), so no batches run
    assert not mock_index.upsert.called
    assert job.status == "completed"


@pytest.mark.asyncio
async def test_run_ingest_job_short_circuits_when_already_completed():
    from app.scripts.reingest import run_ingest_job

    job = _make_job()
    job.status = "completed"
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = job

    with patch("app.scripts.reingest.SessionLocal", return_value=fake_db), patch(
        "app.scripts.reingest.IngestJob"
    ):
        await run_ingest_job(str(job.id))

    # Should not touch Pinecone or commit anything beyond the no-op
    assert job.status == "completed"


@pytest.mark.asyncio
async def test_run_ingest_job_marks_failed_on_exception():
    from app.scripts.reingest import run_ingest_job

    job = _make_job()
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = job

    with patch("app.scripts.reingest.SessionLocal", return_value=fake_db), patch(
        "app.scripts.reingest.IngestJob"
    ), patch(
        "app.scripts.reingest.fetch_source_chunks",
        new=AsyncMock(side_effect=RuntimeError("pinecone down")),
    ):
        with pytest.raises(RuntimeError, match="pinecone down"):
            await run_ingest_job(str(job.id))

    assert job.status == "failed"
    assert "pinecone down" in (job.error_message or "")


@pytest.mark.asyncio
async def test_run_ingest_job_missing_job_raises():
    from app.scripts.reingest import run_ingest_job

    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = None

    with patch("app.scripts.reingest.SessionLocal", return_value=fake_db), patch(
        "app.scripts.reingest.IngestJob"
    ):
        with pytest.raises(ValueError, match="not found"):
            await run_ingest_job(str(uuid.uuid4()))
