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

    以某条锚点事件的向量表示检索近邻事件；当非抽象基本事件数量达到 ``recall_cluster_threshold`` 时，
    调用 ``InductiveEvolutionSkill`` 合成 ``is_abstract=True`` 的新事件，写入摘要与向量索引，
    并将簇内来源事件标记 ``is_abstracted``。``run_background_evolution`` 则扫描未吸收的基本事件批量尝试（白皮书 3.2）。
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
        """If the anchor event's recall cluster >= threshold, generate an abstract event.

        以 *anchor_event* 的 L1（无则原文）为查询向量，检索近邻事件 ID；在排除自身、过滤抽象/缺失后，
        若相关基本事件数量仍不低于 ``recall_cluster_threshold``，则将锚点与它们组成簇，
        计算 ``abstraction_level = max(簇内已有 abstraction_level)+1``，调用演化技能合成抽象事件，
        再跑摘要链、落库、写入向量索引，并把簇内基本事件标为 ``is_abstracted``。否则返回 ``None``。
        """
        index_text = anchor_event.summaries.get("L1", anchor_event.content_raw)
        hits = self._vector.search(index_text, n_results=self._config.recall_cluster_threshold + 5)

        related_ids = [
            h["event_id"]
            for h in hits
            if h["event_id"] != anchor_event.event_id
        ]

        if len(related_ids) < self._config.recall_cluster_threshold:
            return None

        related_events: list[Event] = []
        for eid in related_ids[:self._config.recall_cluster_threshold + 3]:
            evt = self._event_repo.get(eid)
            if evt and not evt.is_abstract:
                related_events.append(evt)

        if len(related_events) < self._config.recall_cluster_threshold:
            return None

        cluster = [anchor_event] + related_events
        max_level = max((e.abstraction_level or 0) for e in cluster) + 1

        logger.info(
            "Abstracting cluster of %d events (anchor=%s, level=%d)",
            len(cluster), anchor_event.event_id, max_level,
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
        if len(candidates) < self._config.recall_cluster_threshold:
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
