"""Tests for RecallService (search, scoring, assembly)."""

from __future__ import annotations

import pytest

from rems.config import REMSConfig
from rems.models.event import Event
from rems.models.metabolism import Shadow
from rems.services.recall_service import RecallService
from rems.storage.database import Database
from rems.storage.repository import EventRepository, RoleRepository
from rems.storage.vector_store import VectorStore


@pytest.fixture()
def recall_service(config: REMSConfig, db: Database, tmp_dir):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    vector_store = VectorStore(config)
    return RecallService(config, event_repo, role_repo, vector_store), event_repo, vector_store


class TestRecallBlock:
    def test_empty_store_returns_empty(self, recall_service):
        svc, _, _ = recall_service
        block = svc.build_recall_block("hello")
        assert block.items == []
        assert block.total_length == 0

    def test_recall_finds_indexed_event(self, recall_service):
        svc, event_repo, vector_store = recall_service

        event = Event(content_raw="张三去了北京参加会议")
        event_repo.save(event)
        vector_store.add_event(event.event_id, event.content_raw, {"is_abstract": False})

        block = svc.build_recall_block("北京的会议")
        assert len(block.items) >= 1
        assert block.items[0].event_id == event.event_id

    def test_context_package(self, recall_service):
        svc, _, _ = recall_service
        shadow = Shadow(content="previous context")
        pkg = svc.build_context_package("new input", shadow)
        assert pkg.current_input == "new input"
        assert pkg.shadow.content == "previous context"


class TestScoring:
    def test_time_decay(self):
        from datetime import datetime
        score = RecallService._time_decay(datetime.now())
        assert 0.99 < score <= 1.0

    def test_time_decay_old(self):
        from datetime import datetime, timedelta
        old = datetime.now() - timedelta(days=60)
        score = RecallService._time_decay(old, half_life_days=30)
        assert score < 0.3

    def test_rrf_merge_uses_both_streams(self, recall_service):
        svc, _, _ = recall_service
        a = Event(content_raw="A")
        b = Event(content_raw="B")
        c = Event(content_raw="C")

        merged = svc._rrf_merge(
            {
                a.event_id: (a, 0.9),
                b.event_id: (b, 0.8),
            },
            {
                b.event_id: (b, 0.7, 10.0),
                c.event_id: (c, 0.6, 1.0),
            },
            [],
        )

        ids = [event.event_id for event, _ in merged]
        assert b.event_id == ids[0]
        assert set(ids) == {a.event_id, b.event_id, c.event_id}
