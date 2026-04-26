from __future__ import annotations

from rems.config import REMSConfig
from rems.models.event import Event, EventRoleEntry, Importance, RoleSnapshot
from rems.skills.inductive_evolution import InductiveEvolutionSkill


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


def test_uses_basic_content_raw_and_role_context():
    config = REMSConfig()
    llm = _DummyJSONLLM(
        {
            "content_raw": "张三多次围绕项目进度协调风险、里程碑与测试计划。",
            "insight": "SHOULD_BE_IGNORED_WHEN_DISABLED",
        }
    )
    skill = InductiveEvolutionSkill(llm, config)
    event = Event(
        content_raw="张三在会议室讨论项目进度：梳理风险清单。",
        summaries={"L1": "摘要不应作为抽象输入"},
        role_list=[
            EventRoleEntry(
                role_id="ROL-zhangsan",
                importance=Importance.A,
                role_snapshot=RoleSnapshot(l1_mention="张三梳理项目风险"),
            )
        ],
    )

    abstract = skill.synthesize([event])
    assert llm.last_messages is not None
    user_prompt = llm.last_messages[1]["content"]

    assert "content_raw:" in user_prompt
    assert "张三在会议室讨论项目进度：梳理风险清单。" in user_prompt
    assert "摘要不应作为抽象输入" not in user_prompt
    assert "ROL-zhangsan" in user_prompt
    assert abstract.insight is None


def test_abstract_insight_switch():
    config = REMSConfig(enable_abstract_insight=True)
    llm = _DummyJSONLLM(
        {
            "content_raw": "张三多次围绕项目进度协调风险、里程碑与测试计划。",
            "insight": "张三倾向通过连续会议推进不确定事项。",
        }
    )
    skill = InductiveEvolutionSkill(llm, config)
    event = Event(content_raw="张三在会议室讨论项目进度。")

    abstract = skill.synthesize([event])
    assert llm.last_messages is not None
    user_prompt = llm.last_messages[1]["content"]

    assert "`insight`：开启" in user_prompt
    assert abstract.insight == "张三倾向通过连续会议推进不确定事项。"
