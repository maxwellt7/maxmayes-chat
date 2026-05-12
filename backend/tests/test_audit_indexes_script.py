"""Tests for app.scripts.audit_indexes.audit_one_index."""
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from app.scripts.audit_indexes import audit_one_index


@pytest.mark.asyncio
async def test_audit_one_index_returns_row_with_classification():
    fake_stats = {
        "total_vector_count": 1500,
        "dimension": 1024,
        "namespaces": {"default": {"vector_count": 1500}},
    }
    fake_sample = [
        {"text": "marketing email tactics ad copy " * 20, "metadata": {}}
    ] * 8

    with patch(
        "app.scripts.audit_indexes._fetch_stats",
        new=AsyncMock(return_value=fake_stats),
    ), patch(
        "app.scripts.audit_indexes._sample_chunks",
        new=AsyncMock(return_value=fake_sample),
    ), patch(
        "app.scripts.audit_indexes.call_openai_json",
        new=AsyncMock(return_value={
            "dominant_domain": "marketing",
            "topic_tags": ["email", "ads"],
            "proposed_disposition": "KEEP",
            "proposed_target_index": "max-marketing",
            "reasoning": "homogeneous marketing content",
        }),
    ):
        row = await audit_one_index(
            index_name="copywriting-resources",
            project_id="1",
            audit_date=date(2026, 5, 11),
        )

    assert row["index_name"] == "copywriting-resources"
    assert row["project_id"] == "1"
    assert row["record_count"] == 1500
    assert row["embedding_model"] == "dim=1024"
    assert row["dominant_domain"] == "marketing"
    assert row["proposed_disposition"] == "KEEP"
    assert row["proposed_target_index"] == "max-marketing"
    assert len(row["sample_chunks"]) == 8


@pytest.mark.asyncio
async def test_audit_one_index_defaults_disposition_to_keep_on_missing_field():
    fake_stats = {
        "total_vector_count": 100,
        "dimension": 1024,
        "namespaces": {},
    }

    with patch(
        "app.scripts.audit_indexes._fetch_stats",
        new=AsyncMock(return_value=fake_stats),
    ), patch(
        "app.scripts.audit_indexes._sample_chunks",
        new=AsyncMock(return_value=[]),
    ), patch(
        "app.scripts.audit_indexes.call_openai_json",
        new=AsyncMock(return_value={
            "dominant_domain": "mixed",
            "topic_tags": [],
            # proposed_disposition deliberately omitted to test default
        }),
    ):
        row = await audit_one_index(
            index_name="empty-test",
            project_id="1",
            audit_date=date.today(),
        )

    assert row["proposed_disposition"] == "KEEP"
    assert row["proposed_target_index"] is None
