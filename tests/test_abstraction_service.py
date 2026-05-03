"""Tests for AbstractionService.

白皮书 §3.2：抽象事件**唯一**触发路径 = recall_log 上的频繁子集挖掘
（``mine_and_synthesize``）。旧测试还在调用早期"向量聚类"接口
``check_and_abstract``，已经在架构升级时移除——本文件按新接口重写。
"""

from __future__ import annotations

import pytest

from rems.config import REMSConfig
from rems.models.event import Event
from rems.services.abstraction_service import AbstractionService
from rems.skills.inductive_evolution import InductiveEvolutionSkill
from rems.skills.summary_generation import SummaryGenerationSkill
from rems.storage.database import Database
from rems.storage.repository import (
    AbstractedSubsetRepository,
    EventRepository,
    RecallLogRepository,
)
from rems.storage.vector_store import VectorStore

from .conftest import FakeLLM


@pytest.fixture()
def abstraction_env(config: REMSConfig, db: Database, fake_llm: FakeLLM, tmp_dir):
    # 让小数据集就能触发挖掘：子集大小最小 3，被回忆 3 次。
    config.abstract_subset_min_size = 3
    config.abstract_subset_min_support = 3

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


def _save_event(event_repo: EventRepository, vector_store: VectorStore, content: str) -> Event:
    e = Event(content_raw=content, summaries={"L1": content[:30]})
    event_repo.save(e)
    vector_store.add_event(e.event_id, e.content_raw, {"is_abstract": False})
    return e


class TestMineAndSynthesize:
    def test_below_support_returns_empty(self, abstraction_env):
        svc, event_repo, vector_store, recall_log_repo, _ = abstraction_env

        events = [_save_event(event_repo, vector_store, f"事件 {i}") for i in range(3)]
        # 仅一次回忆 → support=1 < 3，挖不出来
        recall_log_repo.append(
            recall_id="RCL-1",
            event_ids=[e.event_id for e in events],
        )

        result = svc.mine_and_synthesize()
        assert result == []

    def test_meets_support_creates_abstract(self, abstraction_env):
        svc, event_repo, vector_store, recall_log_repo, fake_llm = abstraction_env

        events = [
            _save_event(event_repo, vector_store, f"张三在第{i+1}次会议讨论项目进度")
            for i in range(4)
        ]
        # 同一组三事件被反复回忆 3 次 → 频繁子集 {e0, e1, e2}, support=3
        common = [events[0].event_id, events[1].event_id, events[2].event_id]
        for k in range(3):
            recall_log_repo.append(
                recall_id=f"RCL-{k}",
                event_ids=common + [events[3].event_id] if k == 0 else common,
            )

        # InductiveEvolutionSkill.synthesize 期望的字段（白皮书 §3.1）。
        fake_llm.push_response({
            "content_raw": "张三多次参与项目进度会议",
            "insight": "张三对项目进度高度关注",
        })
        # SummaryGenerationSkill.generate 默认会循环若干层；提前压几个响应做兜底。
        for _ in range(5):
            fake_llm.push_response({"summary": "多次会议讨论", "char_count": 6})

        created = svc.mine_and_synthesize()
        assert len(created) == 1
        ae = created[0]
        assert ae.is_abstract is True
        assert set(ae.source_events) >= set(common)

    def test_idempotent_when_subset_already_fired(self, abstraction_env):
        svc, event_repo, vector_store, recall_log_repo, fake_llm = abstraction_env

        events = [_save_event(event_repo, vector_store, f"e{i}") for i in range(3)]
        common = [e.event_id for e in events]
        for k in range(3):
            recall_log_repo.append(recall_id=f"RCL-{k}", event_ids=common)

        fake_llm.push_response({"content_raw": "abstract", "insight": ""})
        for _ in range(5):
            fake_llm.push_response({"summary": "s", "char_count": 1})

        first = svc.mine_and_synthesize()
        assert len(first) == 1

        # recall_log 已被同一抽象 id 替换，且 fired_repo 标记过 → 第二次空跑。
        second = svc.mine_and_synthesize()
        assert second == []
