from __future__ import annotations

import logging
from typing import Optional

# 角色抽取：从 L0 得到 role_list 候选、快照层级与 Vedana/Klesha；支持已知角色表去代词化（白皮书 2.1）。

from pydantic import BaseModel, Field

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import ROLE_EXTRACTION_SYSTEM, ROLE_EXTRACTION_USER, build_user_mode_block
from ..models.event import (
    EmotionalModel,
    EventRoleEntry,
    Importance,
    Klesha,
    RoleSnapshot,
    Vedana,
)
from ..models.role import Role

logger = logging.getLogger(__name__)


class ExtractedRole(BaseModel):
    role_id: Optional[str] = None
    name: str = ""
    entity_type: str = "person"
    importance: str = "C"
    snapshot: RoleSnapshot = Field(default_factory=RoleSnapshot)
    emotional_model: EmotionalModel = Field(default_factory=EmotionalModel)


class RoleExtractionResult(BaseModel):
    roles: list[ExtractedRole] = Field(default_factory=list)


class RoleExtractionSkill:
    """Extracts structured roles (snapshots + Vedana/Klesha) from raw event text via LLM JSON.

    将 ``content_raw`` 与可选 ``known_roles`` 描述一并送入 ``role_extraction`` 模型，解析为
    ``ExtractedRole`` 列表；``to_event_role_entry`` 再转为持久化用的 ``EventRoleEntry``（白皮书 2.1，去代词化在 prompt 层强调）。
    """

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def extract(self, content_raw: str, known_roles: list[Role] | None = None, snapshot_budget: int | None = None) -> RoleExtractionResult:
        known_desc = "无已知角色" if not known_roles else "\n".join(
            f"- {r.role_id}: {r.name} ({r.entity_type}), 别名={r.aliases}"
            for r in (known_roles or [])
        )

        # 白皮书 1.2：前置预算约束注入
        budget_hint = ""
        if snapshot_budget is not None:
            budget_hint = f"\n【字数预算】每个角色的 L2 互动白描请控制在 {snapshot_budget} 字以内。\n"

        user_msg = ROLE_EXTRACTION_USER.format(
            known_roles=known_desc,
            content_raw=content_raw,
            budget_hint=budget_hint,
        )

        # 白皮书 2.2：在系统提示词首部注入单人/多人模式块，指导模型做代词消解。
        system_msg = build_user_mode_block(self._config) + ROLE_EXTRACTION_SYSTEM

        data = self._llm.complete_json(
            "role_extraction",
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        extracted: list[ExtractedRole] = []
        for rd in data.get("roles", []):
            snap = rd.get("snapshot", {})
            emo = rd.get("emotion", {})
            vedana_d = emo.get("vedana", {})
            klesha_d = emo.get("klesha", {})

            extracted.append(ExtractedRole(
                role_id=rd.get("role_id"),
                name=rd.get("name", ""),
                entity_type=rd.get("entity_type", "person"),
                importance=rd.get("importance", "C"),
                snapshot=RoleSnapshot(
                    l1_mention=snap.get("l1_mention"),
                    l2_interaction=snap.get("l2_interaction"),
                    l3_decision=snap.get("l3_decision"),
                ),
                emotional_model=EmotionalModel(
                    vedana=Vedana(**{k: float(v) for k, v in vedana_d.items() if k in Vedana.model_fields}),
                    klesha=Klesha(**{k: float(v) for k, v in klesha_d.items() if k in Klesha.model_fields}),
                ),
            ))

        return RoleExtractionResult(
            roles=extracted,
        )

    @staticmethod
    def to_event_role_entry(er: ExtractedRole, assigned_role_id: str) -> EventRoleEntry:
        return EventRoleEntry(
            role_id=assigned_role_id,
            importance=Importance(er.importance) if er.importance in Importance.__members__ else Importance.C,
            role_snapshot=er.snapshot,
            emotional_model=er.emotional_model,
        )
