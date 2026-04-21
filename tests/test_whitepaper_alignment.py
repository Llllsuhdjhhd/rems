from __future__ import annotations

from rems.models.event import CompressionBudget
from rems.services.event_service import EventService
from rems.skills.boundary_detection import BoundaryDetectionSkill
from rems.skills.role_extraction import RoleExtractionSkill
from rems.skills.summary_generation import SummaryGenerationSkill
from rems.storage.repository import EventRepository


class _DummyJSONLLM:
    def __init__(self, payload: dict):
        self.payload = payload
        self.last_messages: list[dict[str, str]] | None = None

    def complete_json(
        self,
        task_type: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> dict:
        self.last_messages = messages
        return self.payload


class _NoopVectorStore:
    def add_event(self, event_id: str, text: str, metadata: dict) -> None:
        return None


def test_boundary_detect_uses_decoded_remaining_shadow(config):
    llm = _DummyJSONLLM(
        {
            "completed_events": [
                {
                    "content_raw_indices": [1],
                    "summaries": {"L1": "甲"},
                    "continuation_of": None,
                }
            ],
            # 这里故意返回冲突原文，验证实现优先使用 indices 解码
            "remaining_shadow": "SHOULD_NOT_USE",
            "remaining_shadow_indices": [2],
            "new_unclosed_indices": [],
        }
    )
    skill = BoundaryDetectionSkill(llm, config)

    result = skill.detect("甲。", "乙。")

    assert result.completed_events[0].content_raw == "甲。"
    assert result.remaining_shadow == "乙。"


def test_boundary_prompt_includes_budget_table(config):
    llm = _DummyJSONLLM(
        {
            "completed_events": [],
            "remaining_shadow_indices": [],
            "new_unclosed_indices": [],
        }
    )
    skill = BoundaryDetectionSkill(llm, config)
    budget = CompressionBudget(
        raw_len=100,
        total_budget=30,
        summary_level_budgets={"L1": 18, "L2": 10},
        snapshot_level_budgets={},
        wp_budget_per_role=5,
        decoration_budget=3,
    )

    skill.detect("旧残影。", "新输入。", budget=budget)
    user_prompt = llm.last_messages[1]["content"]

    assert "字数预算表" in user_prompt
    assert "L1: 18 字以内" in user_prompt


def test_role_extraction_prompt_includes_snapshot_budget(config):
    llm = _DummyJSONLLM({"roles": []})
    skill = RoleExtractionSkill(llm, config)
    budget = CompressionBudget(
        raw_len=120,
        total_budget=36,
        summary_level_budgets={},
        snapshot_level_budgets={"L3": 24, "L2": 14, "L1": 9},
        wp_budget_per_role=7,
        decoration_budget=3,
    )

    skill.extract("张三和李四讨论项目进展。", budget=budget)
    user_prompt = llm.last_messages[1]["content"]

    assert "角色快照预算表" in user_prompt
    assert "L3: 24 字以内" in user_prompt


def test_event_service_suspicious_fallback_creates_temp_role(config, db, fake_llm):
    event_repo = EventRepository(db)
    vector_store = _NoopVectorStore()
    summary_skill = SummaryGenerationSkill(fake_llm, config)
    role_skill = RoleExtractionSkill(fake_llm, config)
    service = EventService(config, fake_llm, event_repo, vector_store, summary_skill, role_skill)

    # 角色抽取返回空，触发 is_suspicious 兜底分支。
    fake_llm.push_response({"roles": []})
    event = service.seal_event(
        "这条输入很模糊但达到强制落库条件。",
        is_suspicious=True,
        pre_summaries={"L1": "模糊输入被封存"},
    )

    assert len(event.role_list) == 1
    assert event.role_list[0].role_id.startswith("临时记录角色_")
