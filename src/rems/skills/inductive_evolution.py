from __future__ import annotations

import logging

# 归纳演化：多条基本/低阶事件 → 合成一条抽象事件（``is_abstract=True``）。
# 抽象事件不登记任何角色（白皮书 §3.2），但合成输入始终使用叶子基本事件的 content_raw，
# 并附带角色线索，防止压缩时把主体关系抹掉。

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import EVOLUTION_SYSTEM, EVOLUTION_USER
from ..models.event import Event, EventStatus, generate_event_id

logger = logging.getLogger(__name__)


class InductiveEvolutionSkill:
    """Synthesise an abstract event from a cluster of basic/lower-order events.

    由多条基本/低阶事件归纳生成抽象事件（``is_abstract=True``，并填充 ``source_events``、
    ``content_raw``、可选 ``insight`` / ``decoration``）。抽象事件 **无角色、无角色快照、无情感模型**；
    summaries 由外层 ``SummaryGenerationSkill`` 基于 ``content_raw`` 递归生成（白皮书 §3.1–§3.3）。
    """

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def synthesize(
        self,
        events: list[Event],
        abstraction_level: int = 1,
        *,
        evidence_events: list[Event] | None = None,
    ) -> Event:
        """Create an abstract event.

        ``events`` remains the logical source set written to ``source_events``. ``evidence_events``
        is the flattened leaf-basic evidence used in the prompt, so higher-order abstractions
        still compress from basic ``content_raw`` rather than from prior abstract text.
        """
        evidence = evidence_events or events
        event_contents = self._build_content_raw_text(evidence)
        target_content_len = self._average_content_len(evidence)
        insight_enabled = self._config.enable_abstract_insight
        insight_instruction = (
            "- `insight`：开启。请提炼跨事件的认知/规律，强调角色行为模式或关系变化；不要复述事实本身。"
            if insight_enabled
            else "- `insight`：关闭。不要输出 `insight` 字段。"
        )
        json_schema = (
            '{\n'
            '  "content_raw": "压缩后的抽象事件主文本",\n'
            '  "insight": "跨事件提炼出的认知/规律",\n'
            '  "decoration": "主观装饰（可选；无则省略或写 null）"\n'
            '}'
            if insight_enabled
            else '{\n'
            '  "content_raw": "压缩后的抽象事件主文本",\n'
            '  "decoration": "主观装饰（可选；无则省略或写 null）"\n'
            '}'
        )

        user_msg = EVOLUTION_USER.format(
            count=len(evidence),
            event_contents=event_contents,
            target_content_len=target_content_len,
            insight_instruction=insight_instruction,
            json_schema=json_schema,
        )

        data = self._llm.complete_json(
            "abstraction",
            [
                {"role": "system", "content": EVOLUTION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )

        return Event(
            event_id=generate_event_id(),
            content_raw=data.get("content_raw", ""),
            is_abstract=True,
            status=EventStatus.ACTIVE,
            decoration=data.get("decoration"),
            insight=data.get("insight") if insight_enabled else None,
            abstraction_level=abstraction_level,
            source_events=[e.event_id for e in events],
            role_list=[],
        )

    # ------------------------------------------------------------------

    def _build_content_raw_text(self, events: list[Event]) -> str:
        lines: list[str] = []
        for e in events:
            role_context = self._build_role_context(e)
            lines.append(
                f"### 基本事件 {e.event_id} (创建于 {e.create_time.isoformat()})\n"
                f"content_raw:\n{e.content_raw}\n"
                f"角色线索:\n{role_context}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _average_content_len(events: list[Event]) -> int:
        if not events:
            return 0
        return max(1, int(sum(len(e.content_raw) for e in events) / len(events)))

    @staticmethod
    def _build_role_context(event: Event) -> str:
        if not event.role_list:
            return "无显式角色线索"
        lines: list[str] = []
        for entry in event.role_list:
            snap = entry.role_snapshot
            fragments = [
                snap.l1_mention,
                snap.l2_interaction,
                snap.l3_decision,
            ]
            summary = " / ".join(f for f in fragments if f)
            importance = entry.importance.value if hasattr(entry.importance, "value") else str(entry.importance)
            lines.append(f"- {entry.role_id}（{importance}）：{summary or '无快照'}")
        return "\n".join(lines)
