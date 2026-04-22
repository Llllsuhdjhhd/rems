from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# 事件充实（Event Enrichment）技能：
#   一次 LLM 调用同时产出「L1..Ln 递归摘要」+「角色列表（含快照与 Vedana/Klesha）」。
#   替代原来分离的 SummaryGenerationSkill + RoleExtractionSkill 组合（仅基本事件流使用）。
#   角色级语义卡片（insight）刷新由下游按「主要角色」过滤后另起调用，见 RoleService._refresh_semantic_card。
# 对照《REMS 记忆系统规范解析》1.1.3、2.1、2.3。

from pydantic import BaseModel, Field

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import ENRICHMENT_USER, build_enrichment_system_message
from ..models.event import EmotionalModel, Klesha, RoleSnapshot, Vedana
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
        *,
        role_fallback: "RoleExtractionSkill | None" = None,
    ):
        self._llm = llm
        self._config = config
        # 合规模型仍漏掉 roles 时，用一次专用 role_extraction 调用补全（与主调用摘要结果合并）。
        self._role_fallback = role_fallback

    def enrich(
        self,
        content_raw: str,
        *,
        known_roles: "list[Role] | None" = None,
        budget: "CompressionBudget | None" = None,
    ) -> EnrichmentResult:
        known_desc = "无已知角色" if not known_roles else "\n".join(
            f"- {r.role_id}: {r.name} ({r.entity_type}), 别名={r.aliases}"
            for r in (known_roles or [])
        )

        summary_budget_text = "按系统默认要求"
        snapshot_budget_text = "按系统默认要求"
        if budget:
            summary_budget_text = "\n".join(
                f"- {lvl}: {b} 字以内" for lvl, b in budget.summary_level_budgets.items()
            )
            snapshot_budget_text = "\n".join(
                f"- {lvl}: {b} 字以内" for lvl, b in budget.snapshot_level_budgets.items()
            )

        fuse_min = self._config.summary_fuse_min_chars
        user_msg = ENRICHMENT_USER.format(
            known_roles=known_desc,
            content_raw=content_raw,
            summary_budget_table=summary_budget_text,
            snapshot_budget_table=snapshot_budget_text,
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

        # ---- Roles ------------------------------------------------------
        extracted: list[ExtractedRole] = []
        role_rows = data.get("roles", []) or data.get("characters", [])
        for rd in role_rows if isinstance(role_rows, list) else []:
            if not isinstance(rd, dict):
                continue
            snap = rd.get("snapshot") or {}
            emo = rd.get("emotion") or {}
            vedana_d = emo.get("vedana") or {}
            klesha_d = emo.get("klesha") or {}

            v_init: dict = {}
            if isinstance(vedana_d, dict):
                for k, v in vedana_d.items():
                    if k in Vedana.model_fields:
                        try:
                            v_init[k] = float(v)
                        except (TypeError, ValueError):
                            continue

            k_init: dict = {}
            if isinstance(klesha_d, dict):
                for k, v in klesha_d.items():
                    if k in Klesha.model_fields:
                        try:
                            k_init[k] = float(v)
                        except (TypeError, ValueError):
                            continue

            extracted.append(ExtractedRole(
                role_id=rd.get("role_id"),
                name=rd.get("name", ""),
                entity_type=rd.get("entity_type", "person"),
                importance=rd.get("importance", "C"),
                snapshot=RoleSnapshot(
                    l1_mention=snap.get("l1_mention") if isinstance(snap, dict) else None,
                    l2_interaction=snap.get("l2_interaction") if isinstance(snap, dict) else None,
                    l3_decision=snap.get("l3_decision") if isinstance(snap, dict) else None,
                ),
                emotional_model=EmotionalModel(
                    vedana=Vedana(**v_init),
                    klesha=Klesha(**k_init),
                ),
            ))

        if not extracted and self._role_fallback and content_raw.strip():
            logger.info("EventEnrichment: empty roles; running role_extraction fallback")
            fr = self._role_fallback.extract(
                content_raw, known_roles=known_roles, budget=budget,
            )
            extracted = list(fr.roles)

        return EnrichmentResult(
            summaries=summaries,
            summary_lengths=summary_lengths,
            actual_max_level=actual_max_level,
            roles=extracted,
        )
