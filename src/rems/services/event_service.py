from __future__ import annotations

import logging
from typing import Optional

# 基本事件封存：L0 → 递归摘要 → 角色抽取 → decoration → 持久化与向量索引。
# 对照《REMS 记忆系统规范解析》1.1（事件字段）、1.1.6（decoration）。

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import DECORATION_SYSTEM, DECORATION_USER
from ..models.event import CompressionBudget, Event, EventRoleEntry, EventStatus
from ..skills.role_extraction import RoleExtractionSkill
from ..skills.summary_generation import SummaryGenerationSkill
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
        summary_skill: SummaryGenerationSkill,
        role_skill: RoleExtractionSkill,
        role_service: RoleService | None = None,
        emotion_evolver: EMAEvolver | None = None,
    ):
        self._config = config
        self._llm = llm
        self._event_repo = event_repo
        self._vector = vector_store
        self._summary_skill = summary_skill
        self._role_skill = role_skill
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
        pre_summaries: dict[str, str] | None = None,
    ) -> Event:
        """Create, enrich, persist and index a new basic event.

        创建、丰富字段、持久化并索引一条新的基本事件（``is_abstract`` 默认为 False）。
        可选传入已构造好的 ``role_entries`` 或 ``skip_roles`` 跳过角色抽取；
        ``known_roles`` 供角色技能做去代词化对齐。超长 ``content_raw`` 会在 ``len_msg`` 处截断。
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

        if pre_summaries:
            from ..skills.summary_generation import SummaryResult
            # 合并摘要架构：直接使用上游传入的 summaries
            summary_result = SummaryResult(
                summaries=pre_summaries,
                summary_lengths={k: len(v) for k, v in pre_summaries.items()},
                actual_max_level=len(pre_summaries),
            )
        else:
            # 兼容旧逻辑/应急后置降级：使用独立摘要技能
            summary_result = self._summary_skill.generate(content_raw, char_budget=budget.summary_budget)

        if role_entries is None and not skip_roles:
            extraction = self._role_skill.extract(
                content_raw,
                known_roles=known_roles,
                snapshot_budget=budget.snapshot_budget_per_role,
            )
            
            if not extraction.roles and is_suspicious:
                import secrets
                from ..models.event import Importance
                
                mock_er = ExtractedRole(
                    name=f"临时记录角色_{secrets.token_hex(2)}",
                    entity_type="unknown",
                    importance=Importance.D
                )
                extraction.roles.append(mock_er)
            
            # Resolve extracted names to real persistent IDs
            id_mapping = {}
            if self._role_service:
                id_mapping = self._role_service.resolve_and_register(extraction.roles, is_suspicious=is_suspicious)
            
            role_entries = []
            for er in extraction.roles:
                lookup_key = er.role_id or er.name
                assigned_id = id_mapping.get(lookup_key, lookup_key)
                role_entries.append(RoleExtractionSkill.to_event_role_entry(er, assigned_id))

        decoration = self._generate_decoration(content_raw, char_budget=budget.decoration_budget)

        # 白皮书 1.2：计算实际压缩率 sum_len / raw_len
        compression_ratio = self._compute_compression_ratio(
            content_raw, summary_result, role_entries or [], decoration,
        )

        event = Event(
            content_raw=content_raw,
            summaries=summary_result.summaries,
            summary_lengths=summary_result.summary_lengths,
            actual_max_level=summary_result.actual_max_level,
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

        根据已知的 ``raw_len`` 和配置的目标压缩率、缩放因子、分项比例，确定性地计算出
        各衍生数据项的字符预算，用于注入 LLM 提示词（白皮书 1.2.2）。
        """
        cfg = self._config
        total = int(raw_len * cfg.compression_target_ratio * cfg.compression_budget_multiplier)
        total = max(total, 20)  # 最低保底

        role_n = max(role_count_estimate, 1)
        summary_b = int(total * cfg.budget_ratio_summary)
        snapshot_b = int(total * cfg.budget_ratio_snapshot / role_n)
        wp_b = int(total * cfg.budget_ratio_wp / role_n)
        deco_b = int(total * cfg.budget_ratio_decoration)

        return CompressionBudget(
            raw_len=raw_len,
            total_budget=total,
            summary_budget=max(summary_b, cfg.summary_fuse_min_chars),
            snapshot_budget_per_role=max(snapshot_b, 10),
            wp_budget_per_role=max(wp_b, 10),
            decoration_budget=max(deco_b, 10),
            role_count_estimate=role_n,
        )

    @staticmethod
    def _compute_compression_ratio(
        content_raw: str,
        summary_result: "SummaryResult",
        role_entries: list[EventRoleEntry],
        decoration: str | None,
    ) -> float:
        """Calculate actual sum_len / raw_len after all derived data is generated.

        封存后审计用：实际的衍生数据总量与原始数据的比值，记录在 Event 上供追踪（白皮书 1.2.1）。
        """
        raw_len = len(content_raw)
        if raw_len == 0:
            return 0.0

        # 默认级摘要（取中位级别）
        mid_key = None
        if summary_result.summaries:
            levels = sorted(summary_result.summaries.keys(), key=lambda k: int(k[1:]))
            mid_key = levels[len(levels) // 2]
        summary_len = len(summary_result.summaries.get(mid_key, "")) if mid_key else 0

        # 角色快照（默认级 l2_interaction）+ 白描条目（同一文本）
        snapshot_len = 0
        for re in role_entries:
            snap_text = re.role_snapshot.l2_interaction or re.role_snapshot.l1_mention or ""
            snapshot_len += len(snap_text)

        # 装饰
        deco_len = len(decoration) if decoration else 0

        sum_len = summary_len + snapshot_len + deco_len
        return sum_len / raw_len
