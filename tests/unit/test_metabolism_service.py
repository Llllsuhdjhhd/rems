"""Tests for MetabolismService (shadow, unclosed events, triggers)."""

from __future__ import annotations


import pytest

from rems.config import REMSConfig
from rems.services.event_service import EventService
from rems.services.metabolism_service import MetabolismService
from rems.skills.boundary_detection import BoundaryDetectionSkill
from rems.skills.role_extraction import RoleExtractionSkill
from rems.skills.summary_generation import SummaryGenerationSkill
from rems.storage.database import Database
from rems.storage.repository import EventRepository, MetabolismRepository
from rems.storage.vector_store import VectorStore

from .conftest import FakeLLM


@pytest.fixture()
def metabolism_service(config: REMSConfig, db: Database, fake_llm: FakeLLM, tmp_dir):
    event_repo = EventRepository(db)
    meta_repo = MetabolismRepository(db)
    vector_store = VectorStore(config)
    summary_skill = SummaryGenerationSkill(fake_llm, config)
    role_skill = RoleExtractionSkill(fake_llm, config)
    event_service = EventService(config, fake_llm, event_repo, vector_store, summary_skill, role_skill)
    boundary_skill = BoundaryDetectionSkill(fake_llm, config)
    return MetabolismService(config, meta_repo, boundary_skill, event_service)


class TestProcessInput:
    def test_completed_event_sealed(self, metabolism_service: MetabolismService, fake_llm: FakeLLM):
        # boundary detection returns one completed event
        fake_llm.push_response({
            "completed_events": [{"content": "A thing happened.", "continuation_of": None}],
            "remaining_shadow": "",
            "new_unclosed": [],
        })
        # summary for the sealed event
        fake_llm.push_response({"summary": "thing happened", "char_count": 14})
        # role extraction
        fake_llm.push_response({"roles": [], "depronom_text": "A thing happened."})
        # decoration
        fake_llm.push_response("warm colours")

        events = metabolism_service.process_input("A thing happened.")
        assert len(events) == 1
        assert events[0].content_raw == "A thing happened."

    def test_no_completed_returns_empty(self, metabolism_service: MetabolismService, fake_llm: FakeLLM):
        fake_llm.push_response({
            "completed_events": [],
            "remaining_shadow": "partial text",
            "new_unclosed": [],
        })

        events = metabolism_service.process_input("partial text")
        assert events == []

    def test_force_save(self, metabolism_service: MetabolismService, fake_llm: FakeLLM):
        # summary
        fake_llm.push_response({"summary": "forced", "char_count": 6})
        # role extraction
        fake_llm.push_response({"roles": [], "depronom_text": "forced content"})
        # decoration
        fake_llm.push_response("decor")

        events = metabolism_service.process_input("forced content", force_save=True)
        assert len(events) >= 1
