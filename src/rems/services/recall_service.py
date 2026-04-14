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
    def build_recall_block(self, query: str, shadow: Shadow | None = None) -> RecallBlock:
        # 查询词拼接残影，使「未封存上下文」参与语义命中（白皮书 4.4）。
        search_text = query
        if shadow and shadow.content:
            search_text = shadow.content + "\n" + query

        hits = self._vector.search(search_text, n_results=20)
        if not hits:
            return RecallBlock()

        scored_events: list[tuple[Event, float]] = []
        for hit in hits:
            event = self._event_repo.get(hit["event_id"])
            if event is None or event.is_tombstoned:
                continue
            raw_sim = 1.0 - hit.get("distance", 1.0)
            score = self._hybrid_score(event, raw_sim)
            scored_events.append((event, score))

        scored_events.sort(key=lambda x: x[1], reverse=True)

        block = self._assemble_block(scored_events)

        # Append semantic cards for top roles (within ceiling)
        block = self._append_semantic_cards(block, scored_events)

        return block

    # ------------------------------------------------------------------
    def build_context_package(self, raw_input: str, shadow: Shadow) -> ContextPackage:
        recall = self.build_recall_block(raw_input, shadow)
        return ContextPackage(
            recall_block=recall,
            shadow=shadow,
            current_input=raw_input,
        )

    # ------------------------------------------------------------------
    # Scoring (cosine 0.5 + time_decay 0.15 + role_importance 0.2 + AE 0.15)
    # ------------------------------------------------------------------

    def _hybrid_score(self, event: Event, cosine_sim: float) -> float:
        time_decay = self._time_decay(event.create_time)

        role_boost = 0.0
        importance_weights = {"S": 0.3, "A": 0.2, "B": 0.1, "C": 0.05, "D": 0.0}
        for re in event.role_list:
            key = re.importance.value if hasattr(re.importance, "value") else str(re.importance)
            role_boost = max(role_boost, importance_weights.get(key, 0.0))

        ae = event.affective_energy
        ae_w = self._config.ae_score_weight           # default 0.15
        cosine_w = 1.0 - ae_w - 0.15 - 0.20          # remainder to cosine ≈ 0.50
        return cosine_w * cosine_sim + 0.15 * time_decay + 0.20 * role_boost + ae_w * ae

    @staticmethod
    def _time_decay(create_time: datetime, half_life_days: float = 30.0) -> float:
        age_seconds = (datetime.now() - create_time).total_seconds()
        age_days = max(age_seconds / 86400, 0.0)
        return math.exp(-0.693 * age_days / half_life_days)

    # ------------------------------------------------------------------
    # Assembly with Lazy Index degradation
    # ------------------------------------------------------------------

    def _assemble_block(self, scored: list[tuple[Event, float]]) -> RecallBlock:
        ceiling = self._config.physical_redline  # 10/66 ≈ 1/6.6
        items: list[RecallItem] = []
        total = 0

        for event, score in scored:
            text, level = self._pick_summary(event, ceiling - total)
            if not text:
                continue

            item = RecallItem(
                event_id=event.event_id,
                content=text,
                score=score,
                summary_level=level,
            )
            total += len(text)
            items.append(item)

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
    def _pick_summary(self, event: Event, budget: int) -> tuple[str, str]:
        """Choose the best summary level that fits within *budget*.

        在无摘要时尝试整段 ``content_raw``（标记 L0）；否则按 L1、L2… 顺序寻找首个长度不超过 *budget* 的级别；
        若均过长则尝试最短一级；仍超出则返回空串对，交由上层跳过或触发 ultra 回退（懒索引降级，白皮书 4.4）。
        """
        if not event.summaries:
            text = event.content_raw
            return (text, "L0") if len(text) <= budget else ("", "")

        levels = sorted(event.summaries.keys(), key=lambda k: int(k[1:]))

        for level_key in levels:
            text = event.summaries[level_key]
            if len(text) <= budget:
                return text, level_key

        shortest = event.summaries[levels[-1]]
        if len(shortest) <= budget:
            return shortest, levels[-1]

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
