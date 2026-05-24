"""Fine-grained recall tracing: dual-stream hits, RRF merge, tiered vector where, reinforcement."""

from __future__ import annotations

from datetime import datetime, timedelta
import pytest

from rems.config import REMSConfig
from rems.models.event import Event, EventRoleEntry, EmotionalModel, Importance, RoleSnapshot
from rems.models.metabolism import Shadow
from rems.models.role import Role, WhitePaintingEntry
from rems.services.recall_service import RecallService
from rems.storage.repository import EventRepository, RoleRepository
from rems.storage.vector_store import VectorStore


def _hit_preview(h: dict) -> dict:
    d = {
        k: h.get(k)
        for k in ("event_id", "wp_id", "distance")
        if k in h and h.get(k) is not None
    }
    doc = h.get("document") or ""
    d["document_preview"] = doc[:160]
    return d


def install_recall_trace(svc: RecallService) -> tuple[dict, callable]:
    trace: dict = {"stream_a": [], "stream_b": [], "rrf": [], "vector_searches": []}
    orig_a = svc._get_stream_a_hits
    orig_b = svc._get_stream_b_hits
    orig_rrf = svc._rrf_merge
    vector = svc._vector
    orig_search = vector.search

    def wrapped_search(*args, **kwargs):
        out = orig_search(*args, **kwargs)
        trace["vector_searches"].append({
            "kwargs": {k: v for k, v in kwargs.items()},
            "n_results": len(out),
        })
        return out

    def wrap_a(query: str, focus_ids: set[str]):
        hits = orig_a(query, focus_ids)
        trace["stream_a"].append({
            "query_prefix": query[:160],
            "n_focus_roles": len(focus_ids),
            "hits": [_hit_preview(h) for h in hits[:20]],
        })
        return hits

    def wrap_b(q: str):
        hits = orig_b(q)
        trace["stream_b"].append({
            "query_prefix": q[:220],
            "hits": [_hit_preview(h) for h in hits[:20]],
        })
        return hits

    def wrap_rrf(stream_a, stream_b, focus_entries):
        merged = orig_rrf(stream_a, stream_b, focus_entries)
        trace["rrf"].append({
            "merged_top": [(e.event_id, round(float(s), 6)) for e, s in merged[:12]],
        })
        return merged

    vector.search = wrapped_search  # type: ignore[method-assign]
    svc._get_stream_a_hits = wrap_a  # type: ignore[method-assign]
    svc._get_stream_b_hits = wrap_b  # type: ignore[method-assign]
    svc._rrf_merge = wrap_rrf  # type: ignore[method-assign]

    def uninstall():
        vector.search = orig_search  # type: ignore[method-assign]
        svc._get_stream_a_hits = orig_a  # type: ignore[method-assign]
        svc._get_stream_b_hits = orig_b  # type: ignore[method-assign]
        svc._rrf_merge = orig_rrf  # type: ignore[method-assign]

    return trace, uninstall


def _where_has_role_anchor(where: object) -> bool:
    """True if Chroma where clause filters by role_<id> (``$or`` for multi-role, or single key)."""
    if isinstance(where, dict):
        if any(str(k).startswith("role_") for k in where):
            return True
        return any(_where_has_role_anchor(v) for v in where.values())
    if isinstance(where, list):
        return any(_where_has_role_anchor(v) for v in where)
    return False


def _seed_event_with_wp(
    *,
    event_repo: EventRepository,
    role_repo: RoleRepository,
    vector_store: VectorStore,
    content: str,
    index_text: str,
    role_id: str,
    wp_summary: str,
    create_time: datetime | None = None,
    forgetting_factor: float = 1.0,
) -> Event:
    create_time = create_time or datetime.now()
    entry = EventRoleEntry(
        role_id=role_id,
        importance=Importance.B,
        role_snapshot=RoleSnapshot(l1_mention="mention", l2_interaction=wp_summary),
        emotional_model=EmotionalModel(),
    )
    ev = Event(
        content_raw=content,
        summaries={"L1": content[:80] + " L1", "L2": content[:50] + " L2", "L3": content[:30] + " L3"},
        role_list=[entry],
        create_time=create_time,
    )
    event_repo.save(ev)
    meta = {
        "is_abstract": False,
        "create_time": create_time.timestamp(),
        f"role_{role_id}": True,
        "role_ids": [role_id],
    }
    vector_store.add_event(ev.event_id, index_text, meta)
    now = datetime.now()
    wp = WhitePaintingEntry(
        event_id=ev.event_id,
        role_summary=wp_summary,
        forgetting_factor=forgetting_factor,
        memory_weight=0.5,
        create_time=create_time,
        last_accessed_time=now,
    )
    role_repo.add_white_painting_entry(role_id, wp)
    vector_store.add_white_painting(role_id, ev.event_id, wp_summary, {"create_time": create_time.timestamp()})
    return ev


@pytest.fixture()
def trace_recall_service(config: REMSConfig, db, tmp_dir):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    vector_store = VectorStore(config)
    role = Role(role_id="ROL-trace-1", name="TracerOne")
    role_repo.save(role)
    base = datetime.now() - timedelta(hours=2)
    e_shared = _seed_event_with_wp(
        event_repo=event_repo,
        role_repo=role_repo,
        vector_store=vector_store,
        content="TracerOne 在北京项目讨论数据库优化方案",
        index_text="TracerOne 北京 项目 数据库 优化",
        role_id=role.role_id,
        wp_summary="TracerOne 参与技术会议讨论延迟",
        create_time=base,
    )
    e_noise = _seed_event_with_wp(
        event_repo=event_repo,
        role_repo=role_repo,
        vector_store=vector_store,
        content="另一角色在海南度假",
        index_text="海南 度假 海滩",
        role_id=role.role_id,
        wp_summary="无关休闲",
        create_time=base + timedelta(hours=1),
    )
    svc = RecallService(config, event_repo, role_repo, vector_store)
    return svc, event_repo, role_repo, vector_store, role.role_id, e_shared, e_noise


class TestRecallTraceHooks:
    def test_trace_records_stream_a_b_and_rrf(self, trace_recall_service):
        svc, _, _, _, rid, e_shared, e_noise = trace_recall_service
        trace, uninstall = install_recall_trace(svc)
        try:
            focus = {rid}
            ents = [
                EventRoleEntry(
                    role_id=rid,
                    importance=Importance.A,
                    role_snapshot=RoleSnapshot(l2_interaction="讨论延迟与索引"),
                    emotional_model=EmotionalModel(),
                ),
            ]
            block = svc.build_recall_block(
                "数据库优化",
                Shadow(content="上一轮提到服务器"),
                focus_role_ids=focus,
                focus_role_entries=ents,
            )
        finally:
            uninstall()

        assert trace["stream_a"], "stream A should run"
        assert trace["stream_b"], "stream B should run"
        assert trace["rrf"], "RRF merge should run"
        assert block.items, "expected at least one recalled item"

        a_ids = {h["event_id"] for h in trace["stream_a"][0]["hits"] if h.get("event_id")}
        assert e_shared.event_id in a_ids or e_noise.event_id in a_ids

        merged_ids = [x[0] for x in trace["rrf"][0]["merged_top"]]
        block_ids = [it.event_id for it in block.items]
        assert merged_ids[0] == block_ids[0], "top RRF item should be first in block order"

    def test_reinforcement_raises_forgetting_factor(self, trace_recall_service):
        svc, _, role_repo, _, rid, e_shared, _ = trace_recall_service
        wp_before = role_repo.get_white_painting_by_event(rid, e_shared.event_id)
        assert wp_before is not None
        factor_before = wp_before.forgetting_factor

        block = svc.build_recall_block(
            "数据库",
            shadow=None,
            focus_role_ids={rid},
            focus_role_entries=None,
        )
        assert any(it.event_id == e_shared.event_id for it in block.items)

        wp_after = role_repo.get_white_painting_by_event(rid, e_shared.event_id)
        assert wp_after is not None
        assert wp_after.forgetting_factor >= min(
            max(factor_before, 1.0) * svc._config.recall_reinforce_multiplier,
            svc._config.recall_forgetting_factor_cap,
        ) - 1e-6


class TestTieredStreamAWhereClause:
    def test_large_store_emits_role_or_filter_for_old_slice(self, config: REMSConfig, db, tmp_dir):
        config.recall_max_capacity = 4
        config.recall_global_ratio = 0.7
        event_repo = EventRepository(db)
        role_repo = RoleRepository(db)
        vector_store = VectorStore(config)
        role = Role(role_id="ROL-tier", name="TierActor")
        role_repo.save(role)
        base = datetime.now() - timedelta(hours=6)
        for i in range(5):
            _seed_event_with_wp(
                event_repo=event_repo,
                role_repo=role_repo,
                vector_store=vector_store,
                content=f"事件{i} TierActor 在场 关键词alpha",
                index_text=f"事件{i} TierActor 在场 关键词alpha",
                role_id=role.role_id,
                wp_summary=f"快照{i}",
                create_time=base + timedelta(minutes=i),
            )

        svc = RecallService(config, event_repo, role_repo, vector_store)
        trace, uninstall = install_recall_trace(svc)
        try:
            svc.build_recall_block("关键词alpha", focus_role_ids={role.role_id})
        finally:
            uninstall()

        assert vector_store.count() > config.recall_max_capacity
        searches = trace["vector_searches"]
        assert len(searches) >= 2, "tiered path should query recent + (optional) older slice"
        wheres = [s["kwargs"].get("where") for s in searches if s["kwargs"].get("where")]
        assert any(_where_has_role_anchor(w) for w in wheres), f"expected role_* in where, got {wheres}"


class TestSummaryTierWithFocus:
    def test_focused_primary_role_gets_more_detailed_summary_level(self, config: REMSConfig, db, tmp_dir):
        """Higher importance + in focus => lower tier offset => earlier index => longer summary when space allows."""
        event_repo = EventRepository(db)
        role_repo = RoleRepository(db)
        vector_store = VectorStore(config)
        rid_a = "ROL-star"
        rid_b = "ROL-bg"
        role_repo.save(Role(role_id=rid_a, name="主角"))
        role_repo.save(Role(role_id=rid_b, name="配角"))
        entry_star = EventRoleEntry(role_id=rid_a, importance=Importance.S, role_snapshot=RoleSnapshot())
        entry_bg = EventRoleEntry(role_id=rid_b, importance=Importance.D, role_snapshot=RoleSnapshot())
        long_l1 = "x" * 400
        mid_l2 = "y" * 120
        short_l3 = "z" * 40
        ev = Event(
            content_raw="raw",
            summaries={"L1": long_l1, "L2": mid_l2, "L3": short_l3},
            role_list=[entry_bg, entry_star],
        )
        event_repo.save(ev)
        vector_store.add_event(
            ev.event_id,
            long_l1,
            {
                "is_abstract": False,
                "create_time": datetime.now().timestamp(),
                f"role_{rid_a}": True,
                "role_ids": [rid_a],
            },
        )
        wp = WhitePaintingEntry(event_id=ev.event_id, role_summary="白描", forgetting_factor=1.0)
        role_repo.add_white_painting_entry(rid_a, wp)
        vector_store.add_white_painting(rid_a, ev.event_id, "白描")

        svc = RecallService(config, event_repo, role_repo, vector_store)
        block_focus_star = svc.build_recall_block("anything", focus_role_ids={rid_a})
        block_focus_other = svc.build_recall_block("anything", focus_role_ids={"ROL-nobody"})

        item_star = next(it for it in block_focus_star.items if it.event_id == ev.event_id)
        item_other = next(it for it in block_focus_other.items if it.event_id == ev.event_id)
        assert item_star.summary_level == "L1"
        assert item_other.summary_level == "L3"
