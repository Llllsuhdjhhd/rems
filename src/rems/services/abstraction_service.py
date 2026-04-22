from __future__ import annotations

import logging

# 抽象事件：唯一触发路径 = 回忆块 event_id 集合中的「极大频繁子集」挖掘（白皮书 §3.2）。
# - 子集大小 >= abstract_subset_min_size（默认 6，可配置）
# - 支持度（被多少条回忆块整体覆盖） >= abstract_subset_min_support（默认 5）
# 达到阈值的子集合成抽象事件；之后用抽象事件 id 在 recall_log 中替换该子集，保持统一命名空间。

from ..config import REMSConfig
from ..models.event import Event
from ..skills.inductive_evolution import InductiveEvolutionSkill
from ..skills.summary_generation import SummaryGenerationSkill
from ..storage.repository import (
    AbstractedSubsetRepository,
    EventRepository,
    RecallLogRepository,
)
from ..storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class AbstractionService:
    """Synthesize abstract events from maximal frequent subsets of recall logs (白皮书 §3.2).

    抽象事件特点（白皮书 §3.1/§3.2）：
        - ``role_list=[]``：抽象事件不登记角色，不写入任何角色白描，也不触发语义卡片；
        - ``summaries``：与基本事件同规则（``SummaryGenerationSkill`` L1…Ln 递归 + 熔断）；
        - ``insight``：由 LLM 基于 ``content_raw`` 生成规律/见解，不由各级 summaries 派生。

    触发是「唯一路径」——不再经向量近邻锚点或回忆压力区重组；由 ``mine_and_synthesize``
    扫描 ``recall_log`` 全量历史，找支持度 ≥ ``abstract_subset_min_support`` 且大小
    ≥ ``abstract_subset_min_size`` 的极大子集，逐个合成并登记替换。
    """

    def __init__(
        self,
        config: REMSConfig,
        event_repo: EventRepository,
        vector_store: VectorStore,
        evolution_skill: InductiveEvolutionSkill,
        summary_skill: SummaryGenerationSkill,
        recall_log_repo: RecallLogRepository,
        abstracted_subset_repo: AbstractedSubsetRepository,
    ):
        self._config = config
        self._event_repo = event_repo
        self._vector = vector_store
        self._evolution = evolution_skill
        self._summary = summary_skill
        self._recall_log_repo = recall_log_repo
        self._fired_repo = abstracted_subset_repo

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def mine_and_synthesize(self) -> list[Event]:
        """Scan ``recall_log``; synthesize one abstract event per maximal frequent subset.

        每次回忆结束后调用一次即可。返回新合成的抽象事件列表（可能为空）。
        """
        cfg = self._config
        min_size = max(2, cfg.abstract_subset_min_size)
        min_support = max(2, cfg.abstract_subset_min_support)

        rows = self._recall_log_repo.list_all()
        transactions = [frozenset(ids) for _, ids in rows if len(ids) >= min_size]
        if len(transactions) < min_support:
            return []

        maximal = _find_maximal_frequent_subsets(transactions, min_size, min_support)
        if not maximal:
            return []

        created: list[Event] = []
        for subset, support in maximal:
            if self._fired_repo.is_fired(subset):
                continue
            evt = self._synthesize_from_subset(subset, support)
            if evt is not None:
                created.append(evt)
                self._fired_repo.mark_fired(subset, evt.event_id)
                # 用抽象事件 id 替换 recall_log 中的该子集，保持统一命名空间。
                replaced = self._recall_log_repo.replace_subset(set(subset), evt.event_id)
                logger.info(
                    "Abstract %s created from %d events (support=%d); rewrote %d recall_log rows",
                    evt.event_id, len(subset), support, replaced,
                )
        return created

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesize_from_subset(self, subset: frozenset[str], support: int) -> Event | None:
        events: list[Event] = []
        for eid in subset:
            e = self._event_repo.get(eid)
            if e is None:
                logger.debug("abstract mining: skip missing event %s", eid)
                continue
            if e.is_tombstoned:
                continue
            events.append(e)
        if len(events) < self._config.abstract_subset_min_size:
            logger.debug(
                "abstract mining: subset shrunk below min_size after DB filtering (%d < %d)",
                len(events), self._config.abstract_subset_min_size,
            )
            return None

        max_level = max((e.abstraction_level or 0) for e in events) + 1
        abstract_event = self._evolution.synthesize(events, abstraction_level=max_level)

        # 抽象事件的摘要与基本事件同规则（白皮书 §1.1.3）。
        sr = self._summary.generate(abstract_event.content_raw)
        abstract_event.summaries = sr.summaries
        abstract_event.summary_lengths = sr.summary_lengths
        abstract_event.actual_max_level = sr.actual_max_level

        self._event_repo.save(abstract_event)
        self._index_abstract(abstract_event)

        for evt in events:
            if not evt.is_abstract:
                self._event_repo.update_status(evt.event_id, is_abstracted=True)

        return abstract_event

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


# =====================================================================
# Maximal frequent subset mining (pragmatic, closure-style)
# =====================================================================


def _find_maximal_frequent_subsets(
    transactions: list[frozenset[str]],
    min_size: int,
    min_support: int,
) -> list[tuple[frozenset[str], int]]:
    """Return ``[(subset, support)]`` for every maximal frequent itemset.

    Pragmatic closure enumeration:
        1. Seed candidates with pairwise intersections of size >= min_size.
        2. Iteratively intersect candidates with each transaction to discover
           smaller closures (also ``>= min_size``).
        3. For each discovered closure C compute support = |{T : C ⊆ T}|;
           keep C if support >= min_support.
        4. Filter maximal: drop any C ⊊ C' still in the frequent set.

    The algorithm is **closure-complete** for the common case and O(n²·|avg|)
    per iteration; adequate for the scales we expect (<~1k recalls).
    """
    n = len(transactions)
    if n < min_support:
        return []

    seen: set[frozenset[str]] = set()
    queue: list[frozenset[str]] = []

    # Seed with pairwise intersections.
    for i in range(n):
        for j in range(i + 1, n):
            inter = transactions[i] & transactions[j]
            if len(inter) >= min_size and inter not in seen:
                seen.add(inter)
                queue.append(inter)

    if not queue:
        return []

    frequent: dict[frozenset[str], int] = {}

    # BFS closure expansion.
    idx = 0
    while idx < len(queue):
        C = queue[idx]
        idx += 1

        support = sum(1 for T in transactions if C.issubset(T))
        if support >= min_support and len(C) >= min_size:
            frequent[C] = support

        if len(C) <= min_size:
            continue
        # Further shrinking only makes sense if support already meets the floor.
        if support < min_support:
            continue
        for T in transactions:
            if C.issubset(T):
                continue
            sub = C & T
            if len(sub) >= min_size and sub not in seen:
                seen.add(sub)
                queue.append(sub)

    if not frequent:
        return []

    # Maximal filter: drop any subset that has a strict superset in ``frequent``.
    items = sorted(frequent.keys(), key=lambda s: -len(s))
    maximal: list[frozenset[str]] = []
    for C in items:
        if any(C < M for M in maximal):
            continue
        maximal.append(C)
    return [(C, frequent[C]) for C in maximal]
