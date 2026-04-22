from __future__ import annotations

import logging

# 抽象事件：向量近邻达阈值则归纳合成 is_abstract=True，并标记子事件 is_abstracted。
# 对照《REMS 记忆系统规范解析》3.2（召回聚集/演化扫描）、3.3（锚定概率在技能层）。

from ..config import REMSConfig
from ..models.event import Event
from ..skills.inductive_evolution import InductiveEvolutionSkill
from ..skills.summary_generation import SummaryGenerationSkill
from ..storage.repository import EventRepository
from ..storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class AbstractionService:
    """Detects when a recall cluster exceeds threshold and synthesises an abstract event.

    以某条锚点事件的向量表示检索近邻；``check_and_abstract`` 在同时满足
    ``abstraction_vector_min_total_events`` 与 L1 总长度 > ``len_msg * abstraction_vector_l1_len_msg_min_ratio`` 时
    调用 ``InductiveEvolutionSkill`` 合成 ``is_abstract=True`` 的新事件。
    ``_reconsolidate`` 触发的 ``abstract_event_cluster`` 不经过本函数（白皮书 3.2、4.4）。
    """

    def __init__(
        self,
        config: REMSConfig,
        event_repo: EventRepository,
        vector_store: VectorStore,
        evolution_skill: InductiveEvolutionSkill,
        summary_skill: SummaryGenerationSkill,
    ):
        self._config = config
        self._event_repo = event_repo
        self._vector = vector_store
        self._evolution = evolution_skill
        self._summary = summary_skill

    # ------------------------------------------------------------------
    def check_and_abstract(self, anchor_event: Event) -> Event | None:
        """Vector-neighbour cluster must reach both size and L1-total vs ``len_msg`` (see config).

        近邻数（不含锚点）从 ``abstraction_vector_min_total_events-1`` 起向上扩展，直到
        簇内 L1 总长度**严格大于** ``len_msg * abstraction_vector_l1_len_msg_min_ratio``；仍不足则再增加近邻
        直至满足或耗尽检索结果。近邻需 ``not is_abstract`` 且 ``not is_abstracted``。
        """
        cfg = self._config
        min_total = max(2, cfg.abstraction_vector_min_total_events)
        min_related = min_total - 1
        n_search = max(64, min_total * 3)
        ratio = cfg.abstraction_vector_l1_len_msg_min_ratio

        index_text = anchor_event.summaries.get("L1", anchor_event.content_raw)
        hits = self._vector.search(index_text, n_results=n_search)

        candidates: list[Event] = []
        seen: set[str] = {anchor_event.event_id}
        for h in hits:
            eid = h.get("event_id")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            evt = self._event_repo.get(eid)
            if not evt or evt.is_abstract or evt.is_abstracted:
                continue
            candidates.append(evt)

        if len(candidates) < min_related:
            return None

        for k in range(min_related, len(candidates) + 1):
            rel = candidates[:k]
            cluster = [anchor_event] + rel
            l1sum = self._sum_l1_text_len(cluster)
            if ratio > 0.0 and l1sum <= cfg.len_msg * ratio:
                continue
            if ratio <= 0.0 and l1sum <= 0:
                continue
            return self.abstract_event_cluster(cluster)
        return None

    @staticmethod
    def _sum_l1_text_len(events: list[Event]) -> int:
        return sum(
            len((e.summaries or {}).get("L1") or e.content_raw or "")
            for e in events
        )

    def abstract_event_cluster(self, cluster: list[Event]) -> Event | None:
        """Synthesize an abstract event from an explicit list of events.
        
        It calculates abstraction level, synthesizes content, generates summaries,
        saves to repo, and marks sources as abstracted. (白皮书 3.1 & 4.4).
        """
        if not cluster:
            return None
            
        max_level = max((e.abstraction_level or 0) for e in cluster) + 1

        logger.info(
            "Abstracting explicit cluster of %d events (level=%d)",
            len(cluster), max_level,
        )

        abstract_event = self._evolution.synthesize(cluster, abstraction_level=max_level)

        sr = self._summary.generate(abstract_event.content_raw)
        abstract_event.summaries = sr.summaries
        abstract_event.summary_lengths = sr.summary_lengths
        abstract_event.actual_max_level = sr.actual_max_level

        self._event_repo.save(abstract_event)
        self._index_abstract(abstract_event)

        for evt in cluster:
            if not evt.is_abstract:
                self._event_repo.update_status(evt.event_id, is_abstracted=True)

        return abstract_event

    # ------------------------------------------------------------------
    def run_background_evolution(self) -> list[Event]:
        """Scan un-abstracted basic events and cluster when threshold met.

        列出尚未被吸收的 basic events，逐个作为锚点调用 ``check_and_abstract``；若生成抽象事件，
        将其 ``source_events`` 记入 ``processed_ids`` 以避免同一批子事件重复参与后续锚点扫描（白皮书 3.2 演化驱动触发）。
        """
        candidates = self._event_repo.list_all(is_abstract=False, is_abstracted=False)
        if len(candidates) < self._config.abstraction_vector_min_total_events:
            return []

        created: list[Event] = []
        processed_ids: set[str] = set()

        for evt in candidates:
            if evt.event_id in processed_ids:
                continue
            result = self.check_and_abstract(evt)
            if result:
                created.append(result)
                for sid in result.source_events or []:
                    processed_ids.add(sid)

        return created

    # ------------------------------------------------------------------
    def _index_abstract(self, event: Event) -> None:
        text = event.summaries.get("L1", event.content_raw)
        self._vector.add_event(
            event.event_id,
            text,
            {
                "is_abstract": True,
                "status": event.status.value,
                "abstraction_level": event.abstraction_level or 1,
            },
        )
