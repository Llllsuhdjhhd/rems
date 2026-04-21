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
from ..utils.text import segment_sentences, format_indexed_text, decode_indices

logger = logging.getLogger(__name__)


class CompletedFragment(BaseModel):
    content_raw: str
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
    ) -> BoundaryResult:
        unclosed_summary = "无" if not unclosed_events else "\n".join(
            f"- ID={ue.id}, 片段={ue.merged_content[:80]}…, 缺={ue.logical_gaps or '未知'}"
            for ue in (unclosed_events or [])
        )

        # 1. 对整体输入（残影 + 当前）进行分句编码
        full_raw = (shadow_content + "\n" + current_input).strip()
        sentences = segment_sentences(full_raw)
        indexed_input = format_indexed_text(sentences)

        # 2. 构造 prompt：纯事件切分，不涉及摘要/角色等衍生字段
        user_msg = BOUNDARY_USER.format(
            shadow=shadow_content or "（空）",
            unclosed_summary=unclosed_summary,
            indexed_input=indexed_input,
        )

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
            indices = item.get("content_raw_indices", [])
            # 解码: 根据序号还原 content_raw
            extracted_raw = decode_indices(sentences, indices)
            if not extracted_raw:
                continue

            completed.append(CompletedFragment(
                content_raw=extracted_raw,
                continuation_of=item.get("continuation_of"),
            ))

        # 解码剩余残影（白皮书 4.2：优先遵循序号编码协议）
        shadow_indices = data.get("remaining_shadow_indices", [])
        remaining_shadow = decode_indices(sentences, shadow_indices)
        if not remaining_shadow:
            # 兼容历史输出：若模型仍返回原文字段，则回退使用。
            remaining_shadow = data.get("remaining_shadow", "")

        new_unc = []
        for item in data.get("new_unclosed_indices", []):
            # new_unclosed 现在也是序号列表或单个序号
            if isinstance(item, list):
                content = decode_indices(sentences, item)
            else:
                content = decode_indices(sentences, [item])
            
            if content:
                new_unc.append(NewUnclosed(
                    content=content,
                    logical_gaps=None, # 序号模式暂不强制要求 gap 描述
                ))

        return BoundaryResult(
            completed_events=completed,
            remaining_shadow=remaining_shadow,
            new_unclosed=new_unc,
        )
