"""Hongloumeng chunk-by-chunk ingest runner (2026.06)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rems.pipeline import ProcessingMode, REMSPipeline

from tests.scenarios.common.harness import (
    ChunkIngestSnapshot,
    ScenarioWorkspace,
    build_pipeline,
    install_llm_file_logger,
    write_chunk_report,
)
from tests.scenarios.common.recall_hooks import install_recall_hooks
from tests.scenarios.hongloumeng.dataset import Chunk, iter_chunks


def _count_events(pipeline: REMSPipeline) -> tuple[int, int, int]:
    event_repo = pipeline.event_service._event_repo  # noqa: SLF001 — scenario probe
    all_events = event_repo.list_all(exclude_tombstoned=True)
    basics = sum(1 for e in all_events if not e.is_abstract)
    abstracts = sum(1 for e in all_events if e.is_abstract)
    return len(all_events), basics, abstracts


def ingest_one_chunk(
    pipeline: REMSPipeline,
    chunk: Chunk,
    *,
    workspace: ScenarioWorkspace,
    enable_llm_log: bool = True,
    enable_recall_trace: bool = True,
) -> tuple[ChunkIngestSnapshot, dict[str, Any] | None]:
    """Ingest a single dataset chunk; return snapshot + optional recall trace."""
    uninstall_llm = None
    if enable_llm_log:
        uninstall_llm, _ = install_llm_file_logger(workspace.log_dir, chunk.id)

    recall_trace: dict[str, Any] | None = None
    uninstall_recall = None
    if enable_recall_trace:
        recall_trace, uninstall_recall = install_recall_hooks(pipeline.recall_service)

    events_before, _, _ = _count_events(pipeline)
    meta_repo = pipeline.metabolism_service._repo  # noqa: SLF001
    shadow_before = meta_repo.get_shadow()

    t0 = time.perf_counter()
    try:
        pipeline.ingest(chunk.content, mode=ProcessingMode.DIALOGUE)
    finally:
        if uninstall_recall:
            uninstall_recall()
        if uninstall_llm:
            uninstall_llm()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    events_total, _, abstracts = _count_events(pipeline)
    role_repo = pipeline.role_service._repo  # noqa: SLF001
    roles_total = len(role_repo.list_all())
    tier1 = pipeline.recall_service._active_pool  # noqa: SLF001
    pool_size = len(tier1.fetch()) if tier1 else 0
    qdrant_n = pipeline.recall_service._vector.count()  # noqa: SLF001
    recall_log_repo = pipeline.abstraction_service._recall_log_repo  # noqa: SLF001
    recall_rows = len(recall_log_repo.list_all())
    ucs = meta_repo.get_unclosed_events()
    shadow_after = meta_repo.get_shadow()

    intent_weights = None
    tri_hits = 0
    bypass = False
    if recall_trace:
        intent = recall_trace.get("intent") or {}
        w = intent.get("weights")
        if w:
            intent_weights = tuple(w)
        for row in recall_trace.get("tri_band_searches") or []:
            tri_hits = max(tri_hits, int(row.get("n_hits") or 0))
        bypass = bool(recall_trace.get("bypass"))

    recall_block_n = 0
    if recall_trace and pipeline.recall_service:
        # Last ingest already ran recall inside pipeline; trace has search hits.
        recall_block_n = tri_hits  # proxy; full block in trace file if needed

    snap = ChunkIngestSnapshot(
        chunk_id=chunk.id,
        chunk_chars=len(chunk.content),
        events_total=events_total,
        events_created=events_total - events_before,
        roles_total=roles_total,
        abstracts_total=abstracts,
        tier1_pool_size=pool_size,
        qdrant_points=qdrant_n,
        recall_log_rows=recall_rows,
        unclosed_count=len(ucs),
        shadow_chars=len(shadow_after.content or ""),
        recall_block_items=recall_block_n,
        intent_weights=intent_weights,
        tri_band_hits=tri_hits,
        bypass_triggered=bypass,
        elapsed_ms=elapsed_ms,
    )
    write_chunk_report(workspace, snap)

    if recall_trace is not None:
        trace_path = workspace.base_dir / f"chunk_{chunk.id}_recall_trace.json"
        trace_path.write_text(json.dumps(recall_trace, ensure_ascii=False, indent=2), encoding="utf-8")

    return snap, recall_trace


def run_chunks(
    *,
    workspace: ScenarioWorkspace,
    count: int = 1,
    chunk_id: int | None = None,
    reset: bool = False,
    offline: bool = False,
    config_overrides: dict[str, Any] | None = None,
) -> list[ChunkIngestSnapshot]:
    """Main entry: ingest *count* chunks into *workspace*."""
    if reset:
        workspace.reset()

    overrides = dict(config_overrides or {})
    if offline:
        overrides.setdefault("embedding", {"provider": "hash"})

    pipeline = build_pipeline(workspace, **overrides)

    if chunk_id is not None:
        from tests.scenarios.hongloumeng.dataset import get_chunk
        chunks = [get_chunk(chunk_id)]
        start_idx = chunk_id - 1 if chunk_id > 0 else 0
    else:
        start_idx = workspace.load_last_chunk_idx()
        chunks = list(iter_chunks(start_after_id=start_idx, count=count))

    if not chunks:
        raise RuntimeError(
            f"No chunks to process (last_chunk_idx={start_idx}, count={count}). "
            "Use --reset or lower last_chunk_idx."
        )

    results: list[ChunkIngestSnapshot] = []
    for chunk in chunks:
        print(f"\n=== Chunk {chunk.id} ({chunk.size} chars) ===", flush=True)
        snap, _ = ingest_one_chunk(
            pipeline,
            chunk,
            workspace=workspace,
            enable_llm_log=not offline,
            enable_recall_trace=True,
        )
        workspace.save_last_chunk_idx(chunk.id, extra={"last_report": f"chunk_{chunk.id}_report.json"})
        results.append(snap)
        print(
            f"  events +{snap.events_created} (total {snap.events_total}), "
            f"tier1={snap.tier1_pool_size}, qdrant={snap.qdrant_points}, "
            f"{snap.elapsed_ms:.0f}ms",
            flush=True,
        )

    return results
