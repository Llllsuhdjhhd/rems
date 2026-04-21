"""Tests for AbstractionService (cluster detection, abstract event creation)."""

from __future__ import annotations


import pytest

from rems.config import REMSConfig
from rems.models.event import Event
from rems.services.abstraction_service import AbstractionService
from rems.skills.inductive_evolution import InductiveEvolutionSkill
from rems.skills.summary_generation import SummaryGenerationSkill
from rems.storage.database import Database
from rems.storage.repository import EventRepository
from rems.storage.vector_store import VectorStore

from .conftest import FakeLLM


@pytest.fixture()
def abstraction_env(config: REMSConfig, db: Database, fake_llm: FakeLLM, tmp_dir):
    config.recall_cluster_threshold = 3
    event_repo = EventRepository(db)
    vector_store = VectorStore(config)
    evolution_skill = InductiveEvolutionSkill(fake_llm, config)
    summary_skill = SummaryGenerationSkill(fake_llm, config)
    svc = AbstractionService(config, event_repo, vector_store, evolution_skill, summary_skill)
    return svc, event_repo, vector_store, fake_llm


class TestCheckAndAbstract:
    def test_below_threshold_returns_none(self, abstraction_env):
        svc, event_repo, vector_store, _ = abstraction_env

        e = Event(content_raw="lonely event")
        event_repo.save(e)
        vector_store.add_event(e.event_id, e.content_raw, {"is_abstract": False})

        result = svc.check_and_abstract(e)
        assert result is None

    def test_above_threshold_creates_abstract(self, abstraction_env):
        svc, event_repo, vector_store, fake_llm = abstraction_env

        events = []
        for i in range(5):
            e = Event(
                content_raw=f"张三在第{i+1}次会议中讨论了项目进度",
                summaries={"L1": f"第{i+1}次会议讨论进度"},
            )
            event_repo.save(e)
            vector_store.add_event(e.event_id, e.content_raw, {"is_abstract": False})
            events.append(e)

        # evolution response
        fake_llm.push_response({
            "content_raw": "张三多次参与项目进度会议",
            "insight": "张三对项目进度高度关注",
            "decoration": "会议室的灯光总是温暖的",
            "roles": [{"role_id": "张三", "importance": "S", "l3_decision": "积极推进", "emotion_trend": {"vedana": {}, "klesha": {}}}],
        })
        # summary for the abstract event
        fake_llm.push_response({"summary": "多次会议讨论", "char_count": 6})

        result = svc.check_and_abstract(events[0])
        if result is not None:
            assert result.is_abstract is True
            assert result.insight is not None
            assert result.source_events is not None
