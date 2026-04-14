"""Tests for EventService (seal, persist, index)."""

from __future__ import annotations

import json

import pytest

from rems.config import REMSConfig
from rems.models.event import Event, EventRoleEntry, Importance
from rems.services.event_service import EventService
from rems.skills.role_extraction import RoleExtractionSkill
from rems.skills.summary_generation import SummaryGenerationSkill
from rems.storage.database import Database
from rems.storage.repository import EventRepository
from rems.storage.vector_store import VectorStore

from .conftest import FakeLLM


@pytest.fixture()
def event_service(config: REMSConfig, db: Database, fake_llm: FakeLLM, tmp_dir):
    event_repo = EventRepository(db)
    vector_store = VectorStore(config)
    summary_skill = SummaryGenerationSkill(fake_llm, config)
    role_skill = RoleExtractionSkill(fake_llm, config)
    return EventService(config, fake_llm, event_repo, vector_store, summary_skill, role_skill)
    


class TestSealEvent:
    def test_seal_creates_event(self, event_service: EventService, fake_llm: FakeLLM):
        fake_llm.push_response({"summary": "核心事实", "char_count": 4})
        fake_llm.push_response("装饰文本")

        event = event_service.seal_event(
            "今天天气晴朗，适合散步。",
            skip_roles=True,
        )

        assert isinstance(event, Event)
        assert event.event_id.startswith("EVT-")
        assert event.content_raw == "今天天气晴朗，适合散步。"
        assert event.status.value == "active"

    def test_seal_persists(self, event_service: EventService, fake_llm: FakeLLM):
        fake_llm.push_response({"summary": "fact", "char_count": 4})
        fake_llm.push_response("decor")

        event = event_service.seal_event("persisted text", skip_roles=True)
        loaded = event_service.get_event(event.event_id)

        assert loaded is not None
        assert loaded.content_raw == "persisted text"

    def test_seal_with_roles(self, event_service: EventService, fake_llm: FakeLLM):
        fake_llm.push_response({"summary": "s", "char_count": 1})
        fake_llm.push_response("d")

        roles = [EventRoleEntry(role_id="ROL-test", importance=Importance.A)]
        event = event_service.seal_event("some event", role_entries=roles)
        assert len(event.role_list) == 1
        assert event.role_list[0].role_id == "ROL-test"
