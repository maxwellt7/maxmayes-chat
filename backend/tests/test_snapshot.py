"""Tests for the legacy-archive snapshot helpers."""
from unittest.mock import MagicMock, patch

import pytest

from app.scripts.reingest import snapshot_to_legacy


@pytest.mark.asyncio
async def test_snapshot_to_legacy_paginates_and_upserts_with_namespaces():
    fake_source = MagicMock()
    fake_source.list_paginated.side_effect = [
        MagicMock(
            vectors=[MagicMock(id="a"), MagicMock(id="b")],
            pagination=MagicMock(next="c1"),
        ),
        MagicMock(vectors=[MagicMock(id="c")], pagination=None),
    ]
    fake_source.fetch.side_effect = [
        MagicMock(
            vectors={
                "a": MagicMock(values=[0.1], metadata={"text": "alpha"}),
                "b": MagicMock(values=[0.2], metadata={"text": "beta"}),
            }
        ),
        MagicMock(
            vectors={"c": MagicMock(values=[0.3], metadata={"text": "gamma"})}
        ),
    ]

    fake_archive = MagicMock()

    factory = MagicMock()

    def _index_factory(name):
        return fake_source if name == "src" else fake_archive

    factory.get_client.return_value.Index.side_effect = _index_factory

    with patch("app.scripts.reingest.get_factory", return_value=factory):
        count = await snapshot_to_legacy("src", "1", "legacy-archive-test")

    assert count == 3
    # archive.upsert was called with namespace=source_index
    upsert_calls = fake_archive.upsert.call_args_list
    assert len(upsert_calls) == 2
    for call in upsert_calls:
        assert call.kwargs["namespace"] == "src"


@pytest.mark.asyncio
async def test_snapshot_to_legacy_handles_empty_source():
    fake_source = MagicMock()
    fake_source.list_paginated.return_value = MagicMock(
        vectors=[], pagination=None
    )
    fake_archive = MagicMock()
    factory = MagicMock()
    factory.get_client.return_value.Index.side_effect = (
        lambda name: fake_source if name == "src" else fake_archive
    )

    with patch("app.scripts.reingest.get_factory", return_value=factory):
        count = await snapshot_to_legacy("src", "1", "legacy-archive-test")

    assert count == 0
    fake_archive.upsert.assert_not_called()
