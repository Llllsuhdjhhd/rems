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
    content: str
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

        user_msg = BOUNDARY_USER.format(
            shadow=shadow_content or "（空）",
            unclosed_summary=unclosed_summary,
            current_input=current_input,
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

        # 直接使用原文截断以替代 LLM 输出全量事件内容
        combined_text = shadow_content
        if combined_text and current_input:
            combined_text += "\n" + current_input
        elif current_input:
            combined_text = current_input
            
        completed = []
        last_idx = 0
        
        import re

        for item in data.get("completed_events", []):
            end_snip = item.get("end_snippet", "").strip()
            if not end_snip:
                continue
                
            # 改进正则：处理转义字符，并允许字符间存在任意空白/换行
            # 先对 snippet 做预处理，压缩连续空白
            clean_snip = re.sub(r"\s+", "", end_snip)
            if not clean_snip:
                continue
                
            # 为每个字符建立容错模式
            pattern_parts = []
            for c in clean_snip:
                pattern_parts.append(re.escape(c))
            pattern_str = r"\s*".join(pattern_parts)
            
            match = re.search(pattern_str, combined_text[last_idx:], re.DOTALL)
            
            if match:
                cut_idx = last_idx + match.end()
                content_chunk = combined_text[last_idx:cut_idx].strip()
                # 检查截取内容是否过短（通常不应发生）
                if len(content_chunk) > 5:
                    completed.append(CompletedFragment(
                        content=content_chunk,
                        continuation_of=item.get("continuation_of"),
                    ))
                    last_idx = cut_idx
            else:
                logger.warning("Failed to locate end_snippet in text: %s", end_snip)
                # 匹配失败时不应吞掉全文，而是尝试跳过
                continue

        new_unc = [
            NewUnclosed(
                content=item.get("content", ""),
                logical_gaps=item.get("logical_gaps"),
            )
            for item in data.get("new_unclosed", [])
            if item.get("content")
        ]

        return BoundaryResult(
            completed_events=completed,
            remaining_shadow=data.get("remaining_shadow", ""),
            new_unclosed=new_unc,
        )
