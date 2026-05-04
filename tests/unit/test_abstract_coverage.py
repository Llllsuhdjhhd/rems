"""abstract_coverage 累加与回忆 RRF 降权。"""

from __future__ import annotations

import math

import pytest

from rems.config import REMSConfig
from rems.models.event import Event
from rems.services.abstraction_service import AbstractionService
from rems.services.recall_service import RecallService
from rems.skills.inductive_evolution import InductiveEvolutionSkill
from rems.skills.summary_generation import SummaryGenerationSkill
from rems.storage.database import Database
from rems.storage.repository import AbstractedSubsetRepository, EventRepository, RecallLogRepository
from rems.storage.vector_store import VectorStore

from ..conftest import FakeLLM


@pytest.fixture()
def abstraction_env(config: REMSConfig, db: Database, fake_llm: FakeLLM):
    config.abstract_subset_min_size = 3
    config.abstract_subset_min_support = 3
    config.abstract_narrative_coherence_enabled = False

    event_repo = EventRepository(db)
    vector_store = VectorStore(config)
    recall_log_repo = RecallLogRepository(db)
    fired_repo = AbstractedSubsetRepository(db)
    evolution_skill = InductiveEvolutionSkill(fake_llm, config)
    summary_skill = SummaryGenerationSkill(fake_llm, config)

    svc = AbstractionService(
        config,
        event_repo,
        vector_store,
        evolution_skill,
        summary_skill,
        recall_log_repo,
        fired_repo,
    )
    return svc, event_repo, vector_store, recall_log_repo, fake_llm


def test_recall_coverage_weight_identity_when_zero():
    e = Event(content_raw="x")
    assert RecallService._recall_coverage_weight(e, 2.0) == 1.0


def test_recall_coverage_weight_exponential():
    e = Event(content_raw="x", abstract_coverage=1.0)
    w = RecallService._recall_coverage_weight(e, 2.0)
    assert math.isclose(w, math.exp(-2.0))


def test_repo_apply_increment_and_cap(config: REMSConfig, db):
    from rems.storage.repository import EventRepository

    repo = EventRepository(db)
    a = Event(content_raw="a")
    repo.save(a)
    repo.apply_abstract_coverage_increment({a.event_id}, 1.5, cap=3.0)
    again = repo.get(a.event_id)
    assert again is not None
    assert math.isclose(again.abstract_coverage, 1.5)
    repo.apply_abstract_coverage_increment({a.event_id}, 5.0, cap=3.0)
    capped = repo.get(a.event_id)
    assert capped is not None
    assert math.isclose(capped.abstract_coverage, 3.0)


def test_synthesis_bumps_member_coverage(abstraction_env):
    """抽象合成后，source 基本事件的 abstract_coverage 递增。"""
    svc, event_repo, vector_store, recall_log_repo, fake_llm = abstraction_env

    events = [
        Event(content_raw=f"事件 {i}", summaries={"L1": f"L{i}"})
        for i in range(3)
    ]
    for e in events:
        event_repo.save(e)
        vector_store.add_event(e.event_id, e.content_raw, {"is_abstract": False})

    common = [events[0].event_id, events[1].event_id, events[2].event_id]
    for k in range(3):
        recall_log_repo.append(recall_id=f"RCL-{k}", event_ids=common)

    fake_llm.push_response({"content_raw": "抽象", "insight": ""})
    for _ in range(5):
        fake_llm.push_response({"summary": "s", "char_count": 10})

    svc.mine_and_synthesize()

    for eid in common:
        e = event_repo.get(eid)
        assert e is not None
        assert e.abstract_coverage > 0
