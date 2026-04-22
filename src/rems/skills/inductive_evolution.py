from __future__ import annotations

import logging
import random

# 归纳演化：多条基本/低阶事件 → 合成一条抽象事件（``is_abstract=True``）。
# 抽象事件不登记任何角色（白皮书 §3.2），所以 prompt 只要求 content_raw + insight（+ 可选 decoration）。
# ``hallucination_anchor_prob``：按概率强制使用子事件 L1（或原文）作为证据输入，抑制
# "上一轮推理当新事实"的幻觉闭环（白皮书 §3.3）。

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import EVOLUTION_SYSTEM, EVOLUTION_USER
from ..models.event import Event, EventStatus, generate_event_id

logger = logging.getLogger(__name__)


class InductiveEvolutionSkill:
    """Synthesise an abstract event from a cluster of basic/lower-order events.

    由多条基本/低阶事件归纳生成抽象事件（``is_abstract=True``，并填充 ``source_events``、
    ``insight``、``content_raw``、可选 ``decoration``）。抽象事件 **无角色、无角色快照、无情感模型**；
    summaries 由外层 ``SummaryGenerationSkill`` 基于 ``content_raw`` 递归生成（白皮书 §3.1–§3.3）。
    """

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def synthesize(self, events: list[Event], abstraction_level: int = 1) -> Event:
        use_anchor = random.random() < self._config.hallucination_anchor_prob
        summaries_text = self._build_summaries_text(events, anchor=use_anchor)

        if use_anchor:
            logger.debug(
                "Hallucination control: anchoring to raw L1 summaries for %d events", len(events),
            )

        user_msg = EVOLUTION_USER.format(
            count=len(events),
            event_summaries=summaries_text,
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
            insight=data.get("insight"),
            abstraction_level=abstraction_level,
            source_events=[e.event_id for e in events],
            role_list=[],
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _build_summaries_text(events: list[Event], *, anchor: bool) -> str:
        lines: list[str] = []
        for e in events:
            if anchor:
                # Force-anchor: prefer L1 (most faithful compression of raw fact); 白皮书 §3.3 锚定事实层。
                text = e.summaries.get("L1", e.content_raw)
                label = "L1（锚定事实层）"
            else:
                # Normal: use mid-level summary — 常规路径，用中间档摘要平衡信息量与抽象度。
                text = e.summaries.get(e.mid_summary_key, e.content_raw)
                label = e.mid_summary_key
            lines.append(
                f"### 事件 {e.event_id} (创建于 {e.create_time.isoformat()}) [{label}]\n{text}"
            )
        return "\n\n".join(lines)
