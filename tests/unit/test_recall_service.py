"""Tests for RecallService (tri-band search, scoring, assembly)."""

from __future__ import annotations

import pytest

from rems.config import REMSConfig
from rems.embedding.tri_band import TriBandEncoder
from rems.models.event import Event
from rems.models.metabolism import Shadow
from rems.services.recall_service import RecallService
from rems.storage.database import Database
from rems.storage.repository import EventRepository, RoleRepository
from rems.storage.vector_store import VectorStore


@pytest.fixture()
def recall_service(config: REMSConfig, db: Database):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    vector_store = VectorStore(config)
    tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    vector_store.set_tri_band(tri_band)
    svc = RecallService(
        config, event_repo, role_repo, vector_store, tri_band=tri_band,
    )
    return svc, event_repo, vector_store


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
        vector_store.upsert_event_vectors(event)

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

    def test_single_stream_score_orders_by_distance(self, recall_service):
        svc, _, _ = recall_service
        a = Event(content_raw="A")
        b = Event(content_raw="B")
        stream = {
            a.event_id: (a, 0.2, 0.5),
            b.event_id: (b, 0.8, 0.5),
        }
        ranked = svc._single_stream_score(stream, [])
        assert ranked[0][0].event_id == a.event_id
