from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# 事件充实（Event Enrichment）技能（白皮书 1.1.3、2.1、2.3）：
#   单次 LLM 调用同时产出「L1..Ln 递归摘要 + 角色列表（含 8 维情绪）」。
#   替代原来分离的 SummaryGenerationSkill + RoleExtractionSkill 组合（仅基本事件流使用）。
#   - full_mode=True ：summaries + roles 一次拿全；
#   - full_mode=False：仅 summaries（外层若已传入 role_entries，可省一次 roles 解析）。

from pydantic import BaseModel, Field

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import (
    ENRICHMENT_FULL_USER,
    ENRICHMENT_SUMMARY_ONLY_USER,
    build_enrichment_system_message,
)
from ..models.event import BasicEmotionVector, EmotionalModel, RoleSnapshot
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
    """Single-call skill producing summaries (+ optional roles) for a sealed basic event.

    把原来两次 LLM 调用（summary_generation + role_extraction）合并为一次：
        - 调用方在 ``enrich(...)`` 中给 ``skip_roles=True``：只解析 summaries（用于 pipeline
          已经在 pre-recall 阶段抽过角色的场景，避免二次提取）；
        - 否则同时解析 summaries + roles，单次 LLM 往返完成事件充实。
        - 仅当解析失败 + 显式提供 ``role_fallback`` 时，才会作为兜底再调一次 RoleExtractionSkill。
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
        skip_roles: bool = False,
    ) -> EnrichmentResult:
        summary_budget_text = "按系统默认要求"
        if budget:
            summary_budget_text = "\n".join(
                f"- {lvl}: {b} 字以内" for lvl, b in budget.summary_level_budgets.items()
            )

        fuse_min = self._config.summary_fuse_min_chars

        if skip_roles:
            user_msg = ENRICHMENT_SUMMARY_ONLY_USER.format(
                content_raw=content_raw,
                summary_budget_table=summary_budget_text,
                fuse_min_chars=fuse_min,
            )
        else:
            known_desc = "无已知角色" if not known_roles else "\n".join(
                f"- {r.role_id}: {r.name} ({r.entity_type}), 别名={r.aliases}"
                for r in known_roles
            )
            snapshot_budgets_text = "按系统默认要求"
            if budget:
                snapshot_budgets_text = "\n".join(
                    f"- {lvl}: {b} 字以内" for lvl, b in budget.snapshot_level_budgets.items()
                )
            user_msg = ENRICHMENT_FULL_USER.format(
                content_raw=content_raw,
                known_roles=known_desc,
                summary_budget_table=summary_budget_text,
                snapshot_budgets=snapshot_budgets_text,
                fuse_min_chars=fuse_min,
            )

        system_msg = build_enrichment_system_message(
            self._config, fuse_min_chars=fuse_min, full_mode=not skip_roles,
        )

        data = self._llm.complete_json(
            "event_enrichment",
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        summaries = self._parse_summaries(data.get("summaries"))
        summary_lengths = {k: len(v) for k, v in summaries.items()}
        actual_max_level = len(summaries)

        roles: list[ExtractedRole] = []
        if not skip_roles:
            roles = self._parse_roles(data.get("roles"))
            # 兜底：解析失败但配置了 fallback 时，单独再调一次（保留旧行为以增强健壮性）。
            if not roles and self._role_fallback is not None:
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

    @staticmethod
    def _parse_summaries(raw: object) -> dict[str, str]:
        out: dict[str, str] = {}
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            if not isinstance(k, str) or not k.startswith("L") or not k[1:].isdigit():
                continue
            if not isinstance(v, str):
                continue
            text = v.strip()
            if not text:
                continue
            out[k] = text
        return out

    @staticmethod
    def _parse_roles(raw: object) -> list[ExtractedRole]:
        if not isinstance(raw, list):
            return []
        out: list[ExtractedRole] = []
        for rd in raw:
            if not isinstance(rd, dict):
                continue
            snap = rd.get("snapshot") or {}
            emo = rd.get("emotion") or {}
            emotion_init: dict[str, float] = {}
            if isinstance(emo, dict):
                for k, v in emo.items():
                    if k in BasicEmotionVector.model_fields:
                        try:
                            emotion_init[k] = float(v)
                        except (TypeError, ValueError):
                            emotion_init[k] = 0.0
            emotion = BasicEmotionVector(**emotion_init)
            out.append(ExtractedRole(
                role_id=rd.get("role_id"),
                name=rd.get("name", ""),
                entity_type=rd.get("entity_type", "person"),
                importance=rd.get("importance", "C"),
                snapshot=RoleSnapshot(
                    l1_mention=snap.get("l1_mention") if isinstance(snap, dict) else None,
                    l2_interaction=snap.get("l2_interaction") if isinstance(snap, dict) else None,
                    l3_decision=snap.get("l3_decision") if isinstance(snap, dict) else None,
                ),
                emotional_model=EmotionalModel.from_emotion(emotion),
            ))
        return out
