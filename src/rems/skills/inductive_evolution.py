from __future__ import annotations

import logging
import random

# 归纳演化：由多条基本事件合成抽象事件（is_abstract=True，insight/content_raw 等由 LLM 给出）。
# hallucination_anchor_prob 控制是否强制用子事件 L1 锚定，抑制幻觉闭环（白皮书 3.3）。

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import EVOLUTION_SYSTEM, EVOLUTION_USER
from ..models.event import (
    EmotionalModel,
    Event,
    EventRoleEntry,
    EventStatus,
    Importance,
    Klesha,
    RoleSnapshot,
    Vedana,
    generate_event_id,
)

logger = logging.getLogger(__name__)


def _emotion_subfields(raw: Any, field_names: set[str]) -> dict[str, float]:
    """Keep only known keys and coerce to float; skip values the LLM returned as labels."""
    if not isinstance(raw, dict):
        if raw:
            logger.warning("inductive_evolution: skip non-dict emotion field %r", raw)
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        if k not in field_names:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            logger.warning("inductive_evolution: skip non-numeric emotion field %s=%r", k, v)
    return out


class InductiveEvolutionSkill:
    """Synthesises an abstract event from a cluster of basic/lower-order events.

    Hallucination Control (white paper section 3.3):
        With probability ``config.hallucination_anchor_prob`` the synthesiser
        is forced to use each sub-event's *raw L1 summary* (or content_raw)
        instead of the mid-level synthesised text.  This prevents the model
        from recursively abstracting its own prior inferences as if they were
        ground-truth facts, preserving a path back to objective evidence.

    由多条基本/低阶事件归纳生成抽象事件（``is_abstract=True``，并填充 ``source_events``、
    ``insight`` 等）。防幻觉：以 ``hallucination_anchor_prob`` 概率强制各子事件只提供 L1 或原文
    作为证据输入，避免模型把上一轮抽象结论当作「新事实」再次递归抽象，从而在链路上保留回到客观 L0/L1 的锚点（白皮书 3.3）。
    """

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def synthesize(self, events: list[Event], abstraction_level: int = 1) -> Event:
        use_anchor = random.random() < self._config.hallucination_anchor_prob
        summaries_text = self._build_summaries_text(events, anchor=use_anchor)

        if use_anchor:
            logger.debug(
                "Hallucination control: anchoring to raw L1 summaries for %d events", len(events)
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

        role_entries: list[EventRoleEntry] = []
        for rd in data.get("roles", []):
            trend = rd.get("emotion_trend", {}) or {}
            vedana_d = _emotion_subfields(trend.get("vedana") or {}, set(Vedana.model_fields))
            klesha_d = _emotion_subfields(trend.get("klesha") or {}, set(Klesha.model_fields))
            role_entries.append(EventRoleEntry(
                role_id=rd.get("role_id", ""),
                importance=Importance(rd["importance"]) if rd.get("importance") in Importance.__members__ else Importance.C,
                role_snapshot=RoleSnapshot(l3_decision=rd.get("l3_decision")),
                emotional_model=EmotionalModel(
                    vedana=Vedana(**vedana_d),
                    klesha=Klesha(**klesha_d),
                ),
            ))

        return Event(
            event_id=generate_event_id(),
            content_raw=data.get("content_raw", ""),
            is_abstract=True,
            status=EventStatus.ACTIVE,
            decoration=data.get("decoration"),
            insight=data.get("insight"),
            abstraction_level=abstraction_level,
            source_events=[e.event_id for e in events],
            role_list=role_entries,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _build_summaries_text(events: list[Event], *, anchor: bool) -> str:
        lines: list[str] = []
        for e in events:
            if anchor:
                # Force-anchor: prefer L1 (most faithful compression of raw fact); 白皮书 3.3 锚定事实层。
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
