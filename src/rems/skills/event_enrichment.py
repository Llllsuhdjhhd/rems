from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# 事件充实（Event Enrichment）技能：
#   生成「L1..Ln 递归摘要」；角色列表由 RoleExtractionSkill 单独抽取。
#   替代原来分离的 SummaryGenerationSkill + RoleExtractionSkill 组合（仅基本事件流使用）。
# 对照《REMS 记忆系统规范解析》1.1.3、2.1、2.3。

from pydantic import BaseModel, Field

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import ENRICHMENT_USER, build_enrichment_system_message
from ..skills.role_extraction import ExtractedRole

if TYPE_CHECKING:
    from ..models.event import CompressionBudget
    from ..models.role import Role
    from .role_extraction import RoleExtractionSkill

logger = logging.getLogger(__name__)


class EnrichmentResult(BaseModel):
    """Unified output of a single event enrichment LLM call.

    ``summaries``       — L1..Ln dict, fuse-stop rule honored by the model.
    ``summary_lengths`` — len(text) per level, computed locally for downstream audits.
    ``actual_max_level``— number of generated levels; mirrors ``SummaryResult.actual_max_level`` semantics.
    ``roles``           — extracted role list (snapshots + emotion), same shape as ``RoleExtractionResult.roles``.
    """

    summaries: dict[str, str] = Field(default_factory=dict)
    summary_lengths: dict[str, int] = Field(default_factory=dict)
    actual_max_level: int = 0
    roles: list[ExtractedRole] = Field(default_factory=list)


class EventEnrichmentSkill:
    """Single-call skill producing summaries + roles for a sealed basic event.

    把原来两次 LLM 调用（summary_generation + role_extraction）合并为一次，
    输入：``content_raw`` + 可选 ``known_roles`` + ``budget``；
    输出：``summaries`` / ``summary_lengths`` / ``actual_max_level`` / ``roles``。
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: REMSConfig,
        role_fallback: "RoleExtractionSkill | None" = None,
    ):
        self._llm = llm
        self._config = config
        self._role_fallback = role_fallback

    def enrich(
        self,
        content_raw: str,
        *,
        known_roles: "list[Role] | None" = None,
        budget: "CompressionBudget | None" = None,
    ) -> EnrichmentResult:
        summary_budget_text = "按系统默认要求"
        if budget:
            summary_budget_text = "\n".join(
                f"- {lvl}: {b} 字以内" for lvl, b in budget.summary_level_budgets.items()
            )

        fuse_min = self._config.summary_fuse_min_chars
        user_msg = ENRICHMENT_USER.format(
            content_raw=content_raw,
            summary_budget_table=summary_budget_text,
            fuse_min_chars=fuse_min,
        )
        system_msg = build_enrichment_system_message(self._config, fuse_min_chars=fuse_min)

        data = self._llm.complete_json(
            "event_enrichment",
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        # ---- Summaries --------------------------------------------------
        raw_summaries = data.get("summaries") or {}
        summaries: dict[str, str] = {}
        if isinstance(raw_summaries, dict):
            for k, v in raw_summaries.items():
                if not isinstance(k, str) or not k.startswith("L") or not k[1:].isdigit():
                    continue
                if not isinstance(v, str):
                    continue
                text = v.strip()
                if not text:
                    continue
                summaries[k] = text
        summary_lengths = {k: len(v) for k, v in summaries.items()}
        actual_max_level = len(summaries)

        roles = []
        if self._role_fallback:
            try:
                extraction = self._role_fallback.extract(content_raw, known_roles=known_roles)
                roles = list(extraction.roles)
            except Exception as e:
                logger.warning("Fallback role extraction failed: %s", e)

        return EnrichmentResult(
            summaries=summaries,
            summary_lengths=summary_lengths,
            actual_max_level=actual_max_level,
            roles=roles,
        )
