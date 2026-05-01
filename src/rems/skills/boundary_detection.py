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
    new_unclosed: list[NewUnclosed] = Field(default_factory=list)


class BoundaryDetectionSkill:
    """LLM-driven boundary detector producing structured JSON for metabolism.

    输入当前残影文本、本轮用户输入与未完成事件摘要，调用 ``boundary_detection`` 模型输出
    ``completed_events`` / ``remaining_shadow`` / ``new_unclosed``，供 ``MetabolismService``
    封存或挂起（白皮书 4.1、1.1.7 防碎片化条款体现在系统/用户 prompt 中）。
    """

    @staticmethod
    def _decode_new_unclosed_list(raw: list, sentences: list[str]) -> list[NewUnclosed]:
        """Map ``new_unclosed_indices`` JSON to ``NewUnclosed`` list.

        - **扁平数字列表** ``[1,2,3]``：视为**同一条**未完成叙事内连续句子（只落库一行）。
        - **嵌套列表** ``[[1,2],[8,9]]``：多线程时多条未完成，每组一行。
        若对扁平表逐项解码，会误将 ``[1,2,3]`` 拆成 3 条只含一句的未完成（DB 里「一条叙事多行」）
        —— 这是此前异常膨胀的主要原因。
        """
        if not raw:
            return []
        is_flat_indices = all(
            isinstance(x, (int, float)) and not isinstance(x, bool)
            for x in raw
        )
        if is_flat_indices:
            idxs = [int(x) for x in raw]
            content = decode_indices(sentences, idxs)
            if not content:
                return []
            return [NewUnclosed(content=content, logical_gaps=None)]

        new_unc: list[NewUnclosed] = []
        for item in raw:
            if isinstance(item, list):
                content = decode_indices(sentences, item)
            else:
                content = decode_indices(sentences, [int(item)])
            if content:
                new_unc.append(NewUnclosed(content=content, logical_gaps=None))
        return new_unc

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

        # 1. 对整体输入（残影 + 当前）进行分句编码；分别记录两段范围以便 prompt 显式标记。
        # 旧实现拼接后再分句，导致模型无法分辨"shadow 内容"与"当前输入"，是 P1-9 修复点。
        shadow_sentences = segment_sentences(shadow_content) if shadow_content else []
        current_sentences = segment_sentences(current_input) if current_input else []
        sentences = shadow_sentences + current_sentences
        shadow_count = len(shadow_sentences)
        # 区间提示：让模型知道 [1..shadow_count] 是已存在残影，剩余是新输入。
        if shadow_count > 0 and current_sentences:
            range_hint = (
                f"（【1..{shadow_count}】=既有残影；"
                f"【{shadow_count + 1}..{len(sentences)}】=本轮新输入）"
            )
        elif shadow_count > 0:
            range_hint = f"（【1..{shadow_count}】=既有残影；本轮无新输入）"
        elif current_sentences:
            range_hint = f"（【1..{len(sentences)}】=本轮新输入；无既有残影）"
        else:
            range_hint = "（无任何输入）"

        indexed_input = format_indexed_text(sentences)

        # 2. 构造 prompt：纯事件切分，不涉及摘要/角色等衍生字段
        user_msg = BOUNDARY_USER.format(
            shadow=shadow_content or "（空）",
            unclosed_summary=unclosed_summary,
            indexed_input=indexed_input,
            range_hint=range_hint,
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

        new_unc = self._decode_new_unclosed_list(
            data.get("new_unclosed_indices", []),
            sentences,
        )

        return BoundaryResult(
            completed_events=completed,
            new_unclosed=new_unc,
        )
