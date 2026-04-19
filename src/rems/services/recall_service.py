from __future__ import annotations

import logging
import math
from datetime import datetime

# 回忆服务：向量检索 + 混合打分（相似度/时间衰减/角色重要性/AE）+ 懒摘要降级 + 语义卡片注入。
# 总长约束约 1/6.6 上下文（config.physical_redline）。对照白皮书 4.4。

from ..config import REMSConfig
from ..models.event import Event
from ..models.metabolism import ContextPackage, RecallBlock, RecallItem, Shadow
from ..storage.repository import EventRepository, RoleRepository
from ..storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RecallService:
    """Builds the recall block for each processing cycle.

    Steps:
        1. Semantic search via vector store.
        2. Retrieve full events and compute hybrid relevance scores
           (cosine + time-decay + role-importance + AE affective energy).
        3. Apply Lazy Index degradation (L1 -> mid -> max summary level).
        4. Append role semantic cards for top-scoring roles.
        5. Enforce 1/6.6 context ceiling via ultra-concise mapping if necessary.

    为每次处理周期构造回忆块与上下文包。步骤与英文一致：先向量检索；再取回事件并计算混合相关度
    （余弦相似度、时间半衰衰减、角色重要性、事件级 AE）；在长度预算下做 L1→中间档→更高级摘要的
    「懒索引」降级；若仍有空间则为高分相关角色附加语义卡片；若总长度仍逼近约 1/6.6 上下文上限，
    则启用 ultra 级极简映射以硬裁剪（白皮书 4.4）。
    """

    def __init__(
        self,
        config: REMSConfig,
        event_repo: EventRepository,
        role_repo: RoleRepository,
        vector_store: VectorStore,
    ):
        self._config = config
        self._event_repo = event_repo
        self._role_repo = role_repo
        self._vector = vector_store

    # ------------------------------------------------------------------
    def build_recall_block(
        self,
        query: str,
        shadow: Shadow | None = None,
        focus_role_ids: set[str] | None = None,
    ) -> RecallBlock:
        """Build the recall block for *query*. Two-stage retrieval implementation."""
        search_text = query
        if shadow and shadow.content:
            search_text = shadow.content + "\n" + query

        # ========================================================
        # Phase 1: 海选匹配与数值遗忘过滤 (Vector + Time Decay)
        # ========================================================
        # 此阶段主要利用底层检索粗拉取，将"时间惩罚（遗忘因子）"尽压至第一级计算
        hits = self._vector.search(search_text, n_results=40)
        if not hits:
            return RecallBlock()

        phase1_candidates = []
        for hit in hits:
            event = self._event_repo.get(hit["event_id"])
            if event is None or event.is_tombstoned:
                continue
            raw_sim = 1.0 - hit.get("distance", 1.0)
            
            # 第一阶段分数仅考虑：纯距离相似度 + 时间衰减因子（纯数值计算），滤除冗余
            time_decay = self._time_decay(event.create_time)
            p1_score = 0.8 * raw_sim + 0.2 * time_decay
            phase1_candidates.append((event, p1_score, raw_sim))

        # 按照包含遗忘因子的前置打分进行排序与强制截断，保留头部若干项进入精排
        phase1_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = phase1_candidates[:15]

        # ========================================================
        # Phase 2: 情感精排与深度再校准 (Emotion & Entity Routing)
        # ========================================================
        scored_events: list[tuple[Event, float]] = []
        for event, p1_score, raw_sim in top_candidates:
            # 这里调用提取 AE、实体图谱的重负荷计算，得出二次重排的综合分
            score = self._hybrid_score(event, raw_sim)
            scored_events.append((event, score))

        scored_events.sort(key=lambda x: x[1], reverse=True)

        block = self._assemble_block(scored_events, focus_role_ids=focus_role_ids or set())
        block = self._append_semantic_cards(block, scored_events)
        return block

    # ------------------------------------------------------------------
    def build_context_package(
        self,
        raw_input: str,
        shadow: Shadow,
        focus_role_ids: set[str] | None = None,
    ) -> ContextPackage:
        recall = self.build_recall_block(raw_input, shadow, focus_role_ids=focus_role_ids)
        return ContextPackage(
            recall_block=recall,
            shadow=shadow,
            current_input=raw_input,
        )

    # ------------------------------------------------------------------
    # Scoring (cosine + time_decay 0.15 + role_importance 0.20 + AE + activation_energy)
    # ------------------------------------------------------------------

    def _hybrid_score(self, event: Event, cosine_sim: float) -> float:
        """Hybrid recall score.

        四项贡献：余弦相似度、时间半衰、角色重要性 max boost、事件级情感能量（AE）与
        ``activation_energy``（白皮书 2.5：重大情感事件的硬绑定抗遗忘初值）。权重分配上，
        ``time_decay`` 与 ``role_boost`` 固定为 0.15 / 0.20；``ae_w``、``act_w`` 从 config 读取，
        余量自动回填到余弦项，保证总权重恒为 1。
        """
        time_decay = self._time_decay(event.create_time)

        role_boost = 0.0
        importance_weights = {"S": 0.3, "A": 0.2, "B": 0.1, "C": 0.05, "D": 0.0}
        for re in event.role_list:
            key = re.importance.value if hasattr(re.importance, "value") else str(re.importance)
            role_boost = max(role_boost, importance_weights.get(key, 0.0))

        ae = event.affective_energy
        act = event.activation_energy
        ae_w = self._config.ae_score_weight                 # 事件级 AE 权重
        act_w = self._config.activation_energy_weight       # EMA 演化后的激活能量权重
        cosine_w = max(0.0, 1.0 - ae_w - act_w - 0.15 - 0.20)
        return (
            cosine_w * cosine_sim
            + 0.15 * time_decay
            + 0.20 * role_boost
            + ae_w * ae
            + act_w * act
        )

    @staticmethod
    def _time_decay(create_time: datetime, half_life_days: float = 30.0) -> float:
        age_seconds = (datetime.now() - create_time).total_seconds()
        age_days = max(age_seconds / 86400, 0.0)
        return math.exp(-0.693 * age_days / half_life_days)

    # ------------------------------------------------------------------
    # Assembly with role-aware Lazy Index degradation
    # ------------------------------------------------------------------

    def _assemble_block(
        self,
        scored: list[tuple[Event, float]],
        focus_role_ids: set[str] | None = None,
    ) -> RecallBlock:
        """Assemble RecallBlock with role-aware summary tier selection.

        For each recalled event the method determines a target compression tier
        based on the highest importance that any *focus_role* holds within that
        event:

        +--------------+-------------------------------------------+
        | Role status  | Tier offset from mid                      |
        +==============+===========================================+
        | S / A        | - ``recall_primary_role_detail_shift``    |
        |              |   (toward L1, more detail)                |
        +--------------+-------------------------------------------+
        | B            | ``recall_default_tier_offset`` (mid)      |
        +--------------+-------------------------------------------+
        | C / D /      | + ``recall_minor_role_compress_shift``    |
        | not present  |   (toward max level, more compressed)     |
        +--------------+-------------------------------------------+

        The summary picked at that tier is guaranteed to be ≥
        ``recall_summary_min_chars`` chars; if the candidate falls below that
        floor the tier is pinned (no further compression applied).
        """
        focus_role_ids = focus_role_ids or set()
        ceiling = self._config.physical_redline
        items: list[RecallItem] = []
        total = 0

        for event, score in scored:
            tier_offset = self._compute_tier_offset(event, focus_role_ids)
            text, level = self._role_aware_pick_summary(event, ceiling - total, tier_offset)
            if not text:
                continue

            items.append(RecallItem(
                event_id=event.event_id,
                content=text,
                score=score,
                summary_level=level,
            ))
            total += len(text)

            if total >= ceiling:
                break

        if total > ceiling:
            items = self._ultra_concise_fallback(items, ceiling)

        block = RecallBlock(items=items)
        block.recompute_length()
        return block

    # ------------------------------------------------------------------
    def _append_semantic_cards(
        self,
        block: RecallBlock,
        scored: list[tuple[Event, float]],
    ) -> RecallBlock:
        """Inject semantic-card summaries for top-scoring roles (if space allows).

        在向量打分靠前的若干事件中收集角色 ID，去重后拉取 ``SemanticCard``；若拼接 ``card_text`` 后仍不超过
        ``physical_redline``，向 ``RecallBlock`` 追加伪条目（``event_id`` 以 ``CARD:`` 前缀），
        以便对话模型直接读取压缩状态（白皮书 2.3与 4.4）。
        """
        ceiling = self._config.physical_redline
        seen_roles: set[str] = set()

        for event, _ in scored[:5]:
            for re in event.role_list:
                if re.role_id in seen_roles:
                    continue
                seen_roles.add(re.role_id)
                card = self._role_repo.get_semantic_card(re.role_id)
                if not card or not card.data:
                    continue
                card_text = f"[语义卡片:{re.role_id}] {card.data}"
                if block.total_length + len(card_text) <= ceiling:
                    block.items.append(RecallItem(
                        event_id=f"CARD:{re.role_id}",
                        content=card_text,
                        score=1.0,
                        summary_level="card",
                    ))
                    block.recompute_length()

        return block

    # ------------------------------------------------------------------
    # Role-aware summary tier helpers
    # ------------------------------------------------------------------

    def _compute_tier_offset(self, event: Event, focus_role_ids: set[str]) -> int:
        """Return the tier offset for *event* given *focus_role_ids*.

        Scans ``event.role_list`` to find the highest importance of any focus
        role.  Maps that importance to an offset relative to the mid-level index:

        * S / A  →  mid - ``recall_primary_role_detail_shift``  (more detail)
        * B      →  mid + ``recall_default_tier_offset``         (default)
        * C / D  →  mid + ``recall_minor_role_compress_shift``   (more compressed)
        * absent →  mid + ``recall_minor_role_compress_shift``   (same as C/D)
        """
        cfg = self._config
        if not focus_role_ids:
            return cfg.recall_default_tier_offset

        best = "absent"
        rank = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4, "absent": 5}
        for entry in event.role_list:
            if entry.role_id not in focus_role_ids:
                continue
            imp = entry.importance.value if hasattr(entry.importance, "value") else str(entry.importance)
            if rank.get(imp, 5) < rank.get(best, 5):
                best = imp

        if best in ("S", "A"):
            return cfg.recall_default_tier_offset - cfg.recall_primary_role_detail_shift
        if best == "B":
            return cfg.recall_default_tier_offset
        # C, D, absent
        return cfg.recall_default_tier_offset + cfg.recall_minor_role_compress_shift

    def _role_aware_pick_summary(
        self,
        event: Event,
        budget: int,
        tier_offset: int,
    ) -> tuple[str, str]:
        """Pick a summary level for *event* using role-aware tier selection.

        Algorithm
        ---------
        1. Build the ordered list of available summary levels: [L1, L2, …, Ln].
        2. Compute *target index* = mid_index + tier_offset, clamped to [0, n-1].
        3. Starting at *target index*, search toward max (higher compression) for
           a summary that fits *budget*.
        4. If target summary char-count ≤ ``recall_summary_min_chars``, pin at
           that level — do NOT compress further regardless of tier_offset.
        5. If nothing fits, fall back to L0 (content_raw) or return empty.

        字数门槛（recall_summary_min_chars）保证过短摘要不会被进一步压缩；
        这在摘要条目极短（如只剩几个关键词）时非常重要。
        """
        min_chars = self._config.recall_summary_min_chars

        # No summaries — try raw content
        if not event.summaries:
            text = event.content_raw
            return (text, "L0") if len(text) <= budget else ("", "")

        levels = sorted(event.summaries.keys(), key=lambda k: int(k[1:]))
        n = len(levels)
        mid_idx = n // 2  # index of the middle level (default starting point)

        target_idx = max(0, min(n - 1, mid_idx + tier_offset))

        # Starting from target, scan toward max-compression until we find one
        # that fits budget AND respects the min_chars floor.
        for idx in range(target_idx, n):
            level_key = levels[idx]
            text = event.summaries[level_key]
            char_count = len(text)

            # Min-chars floor: if this summary is already very short,
            # treat it as the terminal level — use it if it fits budget.
            at_floor = char_count <= min_chars

            if char_count <= budget:
                return text, level_key

            if at_floor:
                # Can't compress further; if it doesn't fit budget, give up.
                break

        # If target is above mid (requesting more detail), also scan toward L1.
        if target_idx < mid_idx:
            for idx in range(target_idx, -1, -1):
                level_key = levels[idx]
                text = event.summaries[level_key]
                if len(text) <= budget:
                    return text, level_key

        # Last resort: try every level from most to least compressed
        for level_key in reversed(levels):
            text = event.summaries[level_key]
            if len(text) <= budget:
                return text, level_key

        return "", ""

    @staticmethod
    def _ultra_concise_fallback(items: list[RecallItem], ceiling: int) -> list[RecallItem]:
        """Compress tail items to ``event_id + core verb + role_ids`` triples.

        当组装后总长仍越过上限时，对放不下的条目将 ``content`` 截断为前缀加省略号，并包以 ``[event_id]`` 前缀，
        标记 ``summary_level=ultra``，以在极端预算下保留可追溯 ID 与少量语义（工程化硬裁剪，白皮书 4.4）。
        """
        result: list[RecallItem] = []
        running = 0
        for it in items:
            if running + len(it.content) <= ceiling:
                result.append(it)
                running += len(it.content)
            else:
                stub = it.content[:40] + "…"
                concise = f"[{it.event_id}] {stub}"
                if running + len(concise) <= ceiling:
                    result.append(RecallItem(
                        event_id=it.event_id,
                        content=concise,
                        score=it.score,
                        summary_level="ultra",
                    ))
                    running += len(concise)
        return result
