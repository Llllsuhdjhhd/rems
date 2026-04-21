from __future__ import annotations

import logging
import math
from typing import Optional

# 基本事件封存：L0 → 递归摘要 → 角色抽取 → decoration → 持久化与向量索引。
# 对照《REMS 记忆系统规范解析》1.1（事件字段）、1.1.6（decoration）。

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import DECORATION_SYSTEM, DECORATION_USER
from ..models.event import CompressionBudget, Event, EventRoleEntry, EventStatus
from ..skills.event_enrichment import EnrichmentResult, EventEnrichmentSkill
from ..skills.role_extraction import ExtractedRole, RoleExtractionSkill
from ..storage.repository import EventRepository
from ..storage.vector_store import VectorStore
from .emotion_service import EMAEvolver

logger = logging.getLogger(__name__)


class EventService:
    """Orchestrates the full lifecycle of a Basic Event:

    content_raw -> summaries -> role extraction -> decoration -> persist.

    基本事件全生命周期编排：从 ``content_raw`` 生成多级摘要，抽取角色与情感快照，
    生成非事实 ``decoration``，将完整 ``Event`` 持久化并写入向量索引（供回忆检索）。
    """

    def __init__(
        self,
        config: REMSConfig,
        llm: LLMProvider,
        event_repo: EventRepository,
        vector_store: VectorStore,
        enrichment_skill: EventEnrichmentSkill,
        role_service: RoleService | None = None,
        emotion_evolver: EMAEvolver | None = None,
    ):
        self._config = config
        self._llm = llm
        self._event_repo = event_repo
        self._vector = vector_store
        # 一次 LLM 调用同时产出摘要 + 角色（替代原 summary_skill + role_skill 二次调用）。
        self._enrichment_skill = enrichment_skill
        self._role_service = role_service
        # 可选：若传入 EMAEvolver，则在封存前做情感动态演化并计算 activation_energy（白皮书 2.5）。
        self._emotion_evolver = emotion_evolver

    # ------------------------------------------------------------------
    def seal_event(
        self,
        content_raw: str,
        *,
        role_entries: list[EventRoleEntry] | None = None,
        skip_roles: bool = False,
        known_roles: list | None = None,
        is_suspicious: bool = False,
        input_id: str | None = None,
    ) -> Event:
        """Create, enrich, persist and index a new basic event.

        创建、丰富字段、持久化并索引一条新的基本事件（``is_abstract`` 默认为 False）。
        可选传入已构造好的 ``role_entries`` 或 ``skip_roles`` 跳过角色抽取；
        ``known_roles`` 供 Enrichment 技能做去代词化对齐。超长 ``content_raw`` 会在 ``len_msg`` 处截断。

        实现上：一次 ``EventEnrichmentSkill.enrich`` 调用同时产出多级摘要与角色列表，
        取代原来分两次调用摘要与角色技能的做法，减少 LLM 往返与上下文重复。
        """
        length_cap = self._config.len_msg
        if len(content_raw) > length_cap:
            logger.warning(
                "content_raw (%d chars) exceeds len_msg (%d), truncating",
                len(content_raw), length_cap,
            )
            content_raw = content_raw[:length_cap]

        # 白皮书 1.2：前置预算计算 —— 在调用任何 LLM 之前确定性地算出各衍生数据项的字符预算
        budget = self._compute_budget(len(content_raw))
        logger.debug(
            "Compression budget: total=%d, summary=%d, snap/role=%d, wp/role=%d, deco=%d",
            budget.total_budget, budget.summary_budget,
            budget.snapshot_budget_per_role, budget.wp_budget_per_role, budget.decoration_budget,
        )

        # 只有在既没有预置 role_entries、又未显式 skip_roles 时，才需要跑 enrichment 抽角色。
        # 但摘要永远需要，因此总是调用 enrichment 产出 summaries（角色部分按需使用）。
        need_roles = role_entries is None and not skip_roles
        enrichment: EnrichmentResult = self._enrichment_skill.enrich(
            content_raw,
            known_roles=known_roles,
            budget=budget,
        )

        if need_roles:
            roles = list(enrichment.roles)

            if not roles and is_suspicious:
                import secrets
                from ..models.event import Importance

                roles.append(ExtractedRole(
                    name=f"临时记录角色_{secrets.token_hex(2)}",
                    entity_type="unknown",
                    importance=Importance.D.value,
                ))

            id_mapping: dict[str, str] = {}
            if self._role_service:
                id_mapping = self._role_service.resolve_and_register(
                    roles, is_suspicious=is_suspicious,
                )

            role_entries = []
            for er in roles:
                lookup_key = er.role_id or er.name
                assigned_id = id_mapping.get(lookup_key, lookup_key)
                role_entries.append(RoleExtractionSkill.to_event_role_entry(er, assigned_id))

        # 3. 生成主观装饰 (Decoration) - 受开关管控
        decoration = ""
        if self._config.enable_decoration:
            decoration = self._generate_decoration(content_raw, char_budget=budget.decoration_budget)
        else:
            logging.debug("Decoration skipped per config.")

        # 白皮书 1.2：计算实际压缩率 sum_len / raw_len
        compression_ratio = self._compute_compression_ratio(
            content_raw, enrichment, role_entries or [], decoration,
        )

        event = Event(
            content_raw=content_raw,
            summaries=enrichment.summaries,
            summary_lengths=enrichment.summary_lengths,
            actual_max_level=enrichment.actual_max_level,
            role_list=role_entries or [],
            status=EventStatus.ACTIVE,
            decoration=decoration,
            input_id=input_id,
            compression_ratio=compression_ratio,
        )

        # EMA 动态演化 + activation_energy 计算（白皮书 2.5）。
        # 必须在 save 之前，让 ORM 落盘时带上"历史心境调和后的情感"和激活能量。
        if self._emotion_evolver is not None:
            try:
                self._emotion_evolver.evolve_event(event)
            except Exception:
                logger.debug("EMA evolution skipped due to error", exc_info=True)

        self._event_repo.save(event)
        self._index_event(event)

        logger.info(
            "Sealed event %s (%d chars, %d roles, ratio=%.4f)",
            event.event_id, event.event_length, len(event.role_list), compression_ratio,
        )
        return event

    # ------------------------------------------------------------------
    def get_event(self, event_id: str) -> Optional[Event]:
        return self._event_repo.get(event_id)

    def list_basic_events(self, *, is_abstracted: bool | None = None) -> list[Event]:
        return self._event_repo.list_all(is_abstract=False, is_abstracted=is_abstracted)

    def mark_abstracted(self, event_id: str) -> None:
        self._event_repo.update_status(event_id, is_abstracted=True)

    # ------------------------------------------------------------------
    def _generate_decoration(self, content_raw: str, *, char_budget: int | None = None) -> str:
        budget_hint = f"【字数预算】请将装饰描述控制在 {char_budget} 字以内。\n" if char_budget else ""
        try:
            return self._llm.complete(
                "summary",
                [
                    {"role": "system", "content": DECORATION_SYSTEM},
                    {"role": "user", "content": DECORATION_USER.format(
                        content_raw=content_raw,
                        budget_hint=budget_hint,
                    )},
                ],
            ).strip()
        except Exception:
            logger.debug("Decoration generation failed, skipping", exc_info=True)
            return ""

    def _index_event(self, event: Event) -> None:
        # 检索主键优先 L1（保真压缩），无则退回原文（与白皮书 1.1.3 一致）。
        index_text = event.summaries.get("L1", event.content_raw)
        metadata: dict = {
            "is_abstract": event.is_abstract,
            "status": event.status.value,
            "event_length": event.event_length,
        }
        if event.role_list:
            metadata["role_ids"] = ",".join(r.role_id for r in event.role_list)
        self._vector.add_event(event.event_id, index_text, metadata)

    # ------------------------------------------------------------------
    # Compression budget & ratio (白皮书 1.2)
    # ------------------------------------------------------------------

    def _compute_budget(self, raw_len: int, role_count_estimate: int = 2) -> CompressionBudget:
        """Pre-compute character budgets for all derived data components.

        根据已知的 ``raw_len`` 和配置的目标压缩率，计算出各衍生数据项的字符预算。
        针对短文本引入宽恕机制（平滑补偿）：文字越短，允许保留的比例越高。
        支持多层级指数衰减预算。
        """
        cfg = self._config
        
        # 白皮书 1.2.2：短文本宽恕机制 (平滑补偿公式)
        relaxation_threshold = 500.0
        effective_ratio = cfg.compression_target_ratio + (1.0 - cfg.compression_target_ratio) * math.exp(-raw_len / relaxation_threshold)
        
        total = int(raw_len * effective_ratio * cfg.compression_budget_multiplier)
        total = max(total, 60)

        role_n = max(role_count_estimate, 1)
        
        # 1. 摘要层级预算 (L1 -> L10 指数衰减)
        summary_total_b = int(total * cfg.budget_ratio_summary)
        summary_l1 = max(summary_total_b, cfg.summary_fuse_min_chars)
        summary_budgets = {"L1": summary_l1}
        for i in range(2, 11):
            prev_b = summary_budgets[f"L{i-1}"]
            # 按 summary_decay_factor 递减，保底为 summary_fuse_min_chars
            new_b = max(int(prev_b * cfg.summary_decay_factor), cfg.summary_fuse_min_chars)
            summary_budgets[f"L{i}"] = new_b

        # 2. 角色快照层级预算 (L3 -> L2 -> L1 指数衰减)
        # 注意：L3 是最详细的，L1 是最精简的。
        snapshot_total_per_role = int(total * cfg.budget_ratio_snapshot / role_n)
        snapshot_l3 = max(snapshot_total_per_role, 30) # 保底 L3 稍长
        snapshot_budgets = {"L3": snapshot_l3}
        # 逆向衰减：L2 = L3 * factor, L1 = L2 * factor
        snapshot_l2 = max(int(snapshot_l3 * cfg.snapshot_decay_factor), 15)
        snapshot_l1 = max(int(snapshot_l2 * cfg.snapshot_decay_factor), 10)
        snapshot_budgets["L2"] = snapshot_l2
        snapshot_budgets["L1"] = snapshot_l1

        wp_b = int(total * cfg.budget_ratio_wp / role_n)
        deco_b = int(total * cfg.budget_ratio_decoration)

        return CompressionBudget(
            raw_len=raw_len,
            total_budget=total,
            summary_level_budgets=summary_budgets,
            snapshot_level_budgets=snapshot_budgets,
            wp_budget_per_role=max(wp_b, 10),
            decoration_budget=max(deco_b, 10),
            role_count_estimate=role_n,
        )

    @staticmethod
    def _compute_compression_ratio(
        content_raw: str,
        enrichment: "EnrichmentResult | SummaryResult",
        role_entries: list[EventRoleEntry],
        decoration: str | None,
    ) -> float:
        """Calculate actual sum_len / raw_len after all derived data is generated.

        封存后审计用：实际的衍生数据总量与原始数据的比值。
        配合 L1-L10 体系，审计平衡点取 L5（中阶压缩层级）；若未到 L5 则退到实际最高级。
        仅依赖 ``enrichment.summaries``，所以 EventEnrichment 与 SummaryGeneration 的返回
        结构都可传入（只需具备 ``summaries: dict[str, str]`` 字段）。
        """
        raw_len = len(content_raw)
        if raw_len == 0:
            return 0.0

        summaries = enrichment.summaries
        audit_level = "L5"
        if audit_level not in summaries:
            if summaries:
                levels = sorted(summaries.keys(), key=lambda k: int(k[1:]))
                audit_level = levels[-1]
            else:
                audit_level = None

        summary_len = len(summaries.get(audit_level, "")) if audit_level else 0

        # 角色快照（主角取 L2 标准级，配角只有 L1）
        snapshot_len = 0
        for re in role_entries:
            snap = re.role_snapshot
            snap_text = snap.l2_interaction or snap.l1_mention or ""
            snapshot_len += len(snap_text)

        # 装饰
        deco_len = len(decoration) if decoration else 0

        sum_len = summary_len + snapshot_len + deco_len
        return sum_len / raw_len
