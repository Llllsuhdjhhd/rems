from __future__ import annotations

import logging
from typing import Optional

# 边界检测技能：残影 + 当前输入 + 未完成库 → 已闭环片段、剩余残影、新未完成条目。
# Prompt 内嵌白皮书 1.1.7 防碎片化聚合原则（同一段落内琐碎动作合并为一条基本事件）。

from pydantic import BaseModel, Field

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import BOUNDARY_SYSTEM, BOUNDARY_USER, build_user_mode_block
from ..models.metabolism import UnclosedEvent

logger = logging.getLogger(__name__)


class CompletedFragment(BaseModel):
    content_raw: str
    summaries: dict[str, str] = Field(default_factory=dict)
    continuation_of: Optional[str] = None


class NewUnclosed(BaseModel):
    content: str
    logical_gaps: Optional[str] = None


class BoundaryResult(BaseModel):
    completed_events: list[CompletedFragment] = Field(default_factory=list)
    remaining_shadow: str = ""
    new_unclosed: list[NewUnclosed] = Field(default_factory=list)


class BoundaryDetectionSkill:
    """LLM-driven boundary detector producing structured JSON for metabolism.

    输入当前残影文本、本轮用户输入与未完成事件摘要，调用 ``boundary_detection`` 模型输出
    ``completed_events`` / ``remaining_shadow`` / ``new_unclosed``，供 ``MetabolismService``
    封存或挂起（白皮书 4.1、1.1.7 防碎片化条款体现在系统/用户 prompt 中）。
    """

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def detect(
        self,
        shadow_content: str,
        current_input: str,
        unclosed_events: list[UnclosedEvent] | None = None,
        char_budget: int | None = None,
    ) -> BoundaryResult:
        unclosed_summary = "无" if not unclosed_events else "\n".join(
            f"- ID={ue.id}, 片段={ue.merged_content[:80]}…, 缺={ue.logical_gaps or '未知'}"
            for ue in (unclosed_events or [])
        )

        budget_hint = f"\n【字数预算】请尽量将每个事件的 L1 摘要控制在 {char_budget} 字以内。\n" if char_budget else ""

        user_msg = BOUNDARY_USER.format(
            shadow=shadow_content or "（空）",
            unclosed_summary=unclosed_summary,
            current_input=current_input,
            budget_hint=budget_hint,
        )

        # 白皮书 2.2：system prompt 首部注入模式块，让边界检测也感知代词归属约束。
        system_msg = build_user_mode_block(self._config) + BOUNDARY_SYSTEM

        data = self._llm.complete_json(
            "boundary_detection",
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        completed = []
        for item in data.get("completed_events", []):
            extracted_raw = item.get("content_raw", "").strip()
            if not extracted_raw:
                continue
            
            completed.append(CompletedFragment(
                content_raw=extracted_raw,
                summaries=item.get("summaries", {}),
                continuation_of=item.get("continuation_of"),
            ))

        new_unc = []
        for item in data.get("new_unclosed", []):
            if isinstance(item, dict):
                content = item.get("content", "").strip()
                if content:
                    new_unc.append(NewUnclosed(
                        content=content,
                        logical_gaps=item.get("logical_gaps"),
                    ))
            elif isinstance(item, (str, bytes)):
                content = str(item).strip()
                if content:
                    new_unc.append(NewUnclosed(
                        content=content,
                        logical_gaps=None,
                    ))

        return BoundaryResult(
            completed_events=completed,
            remaining_shadow=data.get("remaining_shadow", ""),
            new_unclosed=new_unc,
        )
