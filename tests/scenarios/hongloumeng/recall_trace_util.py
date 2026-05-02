"""Shared helpers to capture dual-stream recall traces and write ``recall_trace.json``.

Used by scenario scripts under ``tests/scenarios/hongloumeng/`` (not production code).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from rems.services.recall_service import RecallService


def _hit_preview(h: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if h.get("event_id") is not None:
        out["event_id"] = h["event_id"]
    if h.get("wp_id") is not None:
        out["wp_id"] = h["wp_id"]
    if h.get("distance") is not None:
        out["distance"] = h["distance"]
    doc = h.get("document") or ""
    out["document_preview"] = doc[:220]
    return out


def install_recall_hooks(svc: RecallService) -> tuple[dict[str, Any], Callable[[], None]]:
    """Wrap Stream A/B and ``_rrf_merge`` to append lightweight snapshots to *trace* lists."""

    trace: dict[str, Any] = {"stream_a": [], "stream_b": [], "rrf": []}
    orig_a = svc._get_stream_a_hits
    orig_b = svc._get_stream_b_hits
    orig_rrf = svc._rrf_merge

    def wrap_a(query: str, focus_ids: set[str]):
        hits = orig_a(query, focus_ids)
        trace["stream_a"].append({
            "query_prefix": query[:200],
            "n_focus_roles": len(focus_ids),
            "hits": [_hit_preview(x) for x in hits[:24]],
        })
        return hits

    def wrap_b(q: str):
        hits = orig_b(q)
        trace["stream_b"].append({
            "query_prefix": q[:260],
            "hits": [_hit_preview(x) for x in hits[:24]],
        })
        return hits

    def wrap_rrf(stream_a, stream_b, focus_entries):
        merged = orig_rrf(stream_a, stream_b, focus_entries)
        trace["rrf"].append({
            "merged_top": [
                {"event_id": e.event_id, "score": round(float(s), 6)}
                for e, s in merged[:36]
            ],
        })
        return merged

    svc._get_stream_a_hits = wrap_a  # type: ignore[method-assign]
    svc._get_stream_b_hits = wrap_b  # type: ignore[method-assign]
    svc._rrf_merge = wrap_rrf  # type: ignore[method-assign]

    def uninstall():
        svc._get_stream_a_hits = orig_a  # type: ignore[method-assign]
        svc._get_stream_b_hits = orig_b  # type: ignore[method-assign]
        svc._rrf_merge = orig_rrf  # type: ignore[method-assign]

    return trace, uninstall


def recall_block_to_json(block) -> list[dict[str, Any]]:
    return [
        {
            "event_id": it.event_id,
            "summary_level": it.summary_level,
            "score": it.score,
            "content_preview": (it.content or "")[:500],
            "content_chars": len(it.content or ""),
        }
        for it in block.items
    ]


def latest_recall_log_row(recall_log_repo) -> dict[str, Any] | None:
    rows = recall_log_repo.list_all()
    if not rows:
        return None
    rid, eids = rows[-1]
    return {"recall_id": rid, "event_ids": list(eids)}


def write_recall_trace_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
