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
        """Build the recall block for *query*. Multi-stage retrieval implementation (白皮书 4.4)."""
        search_text = query
        if shadow and shadow.content:
            search_text = shadow.content + "\n" + query

        # ========================================================
        # Phase 1: Expansion & Numerically Scored Search (Vector + Time)
        # ========================================================
        # Increased hits via recall_expansion_factor (approx 2x redline) 
        # to ensure enough variety for dynamic compression.
        expansion_hits = 60  # Heuristic for ~2x redline; could be dynamic if needed.
        hits = self._vector.search(search_text, n_results=expansion_hits)
        if not hits:
            return RecallBlock()

        phase1_candidates = []
        for hit in hits:
            event = self._event_repo.get(hit["event_id"])
            if event is None or event.is_tombstoned:
                continue
            raw_sim = 1.0 - hit.get("distance", 1.0)
            
            time_decay = self._time_decay(event.create_time)
            p1_score = 0.8 * raw_sim + 0.2 * time_decay
            phase1_candidates.append((event, p1_score, raw_sim))

        # Sort by P1 score and keep a generous set for fine-ranking
        phase1_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = phase1_candidates[:30] # Double the previous set for scaling

        # ========================================================
        # Phase 2: Hybrid Scoring & Emotional Filter (1.2x Redline)
        # ========================================================
        scored_events: list[tuple[Event, float]] = []
        for event, p1_score, raw_sim in top_candidates:
            score = self._hybrid_score(event, raw_sim)
            scored_events.append((event, score))

        scored_events.sort(key=lambda x: x[1], reverse=True)

        # Apply intermediate filter target (1.2/6.6)
        redline = self._config.physical_redline
        intermediate_ceiling = int(redline * self._config.recall_intermediate_filter_factor)
        
        filtered_events: list[tuple[Event, float]] = []
        current_len = 0
        for event, score in scored_events:
            tier_offset = self._compute_tier_offset(event, focus_role_ids or set())
            text, _ = self._role_aware_pick_summary(event, 9999, tier_offset)
            if not text:
                continue
            
            # Integrity: Stop BEFORE the item that would exceed the intermediate ceiling
            if current_len + len(text) > intermediate_ceiling and filtered_events:
                break
                
            filtered_events.append((event, score))
            current_len += len(text)

        block = self._assemble_block(filtered_events, focus_role_ids=focus_role_ids or set())
        block = self._append_semantic_cards(block, filtered_events)
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
        """Assemble RecallBlock with Sliding Window Dynamic Compression (白皮书 4.4 演化).
        
        Logic:
        1. Split scored events into Head (Top 2/3) and Tail (Bottom 1/3).
        2. Iteratively compress the Tail group by increasing summary tier offsets.
        3. If Tail is fully compressed, shift items from Head into Tail and repeat.
        """
        focus_role_ids = focus_role_ids or set()
        ceiling = self._config.physical_redline
        head_ratio = self._config.recall_head_ratio
        
        # Initial categorization
        n = len(scored)
        head_count = int(n * head_ratio)
        
        # We store the base tier offset and current 'extra' compression for each item
        items_data = []
        for i, (event, score) in enumerate(scored):
            base_offset = self._compute_tier_offset(event, focus_role_ids)
            items_data.append({
                "event": event,
                "score": score,
                "base_offset": base_offset,
                "extra_compression": 0,
                "is_head": i < head_count
            })

        max_iterations = 20 # Safety break
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            # 1. Trial assembly
            current_items = []
            total_len = 0
            for data in items_data:
                offset = data["base_offset"] + data["extra_compression"]
                text, level = self._role_aware_pick_summary(data["event"], 9999, offset)
                if not text: continue
                
                # Integrity check for final assemble: stop before crossing redline
                if total_len + len(text) > ceiling and current_items:
                    # Keep ``total_len`` as the actual assembled length.
                    # Over-counting here would trigger unnecessary extra compression.
                    break

                current_items.append(RecallItem(
                    event_id=data["event"].event_id,
                    content=text,
                    score=data["score"],
                    summary_level=level
                ))
                total_len += len(text)
            
            # 2. Check if we fit
            if total_len <= ceiling or not items_data:
                break
                
            # 3. Progressive Compression
            # Target the Tail group (including those shifted from Head)
            tail_indices = [i for i, d in enumerate(items_data) if not d["is_head"]]
            
            can_compress_tail = False
            if tail_indices:
                # 判断「还能不能再压一档」：比较当前档与再加一档后选出的摘要。
                # 若选出的 level_key 发生改变且文本确实更短，才允许 extra_compression++；
                # 否则说明已经到了该事件的最高压缩档（``_role_aware_pick_summary`` 的 clamp），
                # 继续递增没有意义，避免空转循环耗尽 ``max_iterations``。
                for idx in tail_indices:
                    event = items_data[idx]["event"]
                    current_level = items_data[idx]["base_offset"] + items_data[idx]["extra_compression"]
                    cur_text, cur_key = self._role_aware_pick_summary(event, 9999, current_level)
                    next_text, next_key = self._role_aware_pick_summary(event, 9999, current_level + 1)
                    if next_key and next_key != cur_key and len(next_text) < len(cur_text):
                        items_data[idx]["extra_compression"] += 1
                        can_compress_tail = True
            
            # 4. Shifting Logic
            if not can_compress_tail:
                # If tail is maxed out, shift one item from Head to Tail
                head_indices = [i for i, d in enumerate(items_data) if d["is_head"]]
                if head_indices:
                    last_head_idx = head_indices[-1]
                    logger.info("Shifting item %s from Head to Tail for compression", items_data[last_head_idx]["event"].event_id)
                    items_data[last_head_idx]["is_head"] = False
                    # On shift, we might want to immediately apply one level of compression
                    items_data[last_head_idx]["extra_compression"] += 1
                else:
                    # Everything is already in Tail and maxed out. 
                    # Use ultra-concise fallback and exit.
                    current_items = self._ultra_concise_fallback(current_items, ceiling)
                    break

        block = RecallBlock(items=current_items)
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
