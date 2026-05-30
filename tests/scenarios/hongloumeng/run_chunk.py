#!/usr/bin/env python3
"""Ingest one or more Hongloumeng chunks (2026.06 runner).

Examples (from repo root)::

    python tests/scenarios/hongloumeng/run_chunk.py
    python tests/scenarios/hongloumeng/run_chunk.py --count 3
    python tests/scenarios/hongloumeng/run_chunk.py --chunk-id 1 --reset
    python tests/scenarios/hongloumeng/run_chunk.py --offline --count 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as script without installing package in editable mode.
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.scenarios.common.harness import ScenarioWorkspace, find_repo_root
from tests.scenarios.hongloumeng.runner import run_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Hongloumeng chunk ingest (2026.06)")
    parser.add_argument("--count", type=int, default=1, help="Number of chunks to ingest")
    parser.add_argument("--chunk-id", type=int, default=None, help="Force a specific chunk id")
    parser.add_argument("--reset", action="store_true", help="Clear workspace before run")
    parser.add_argument(
        "--run-name",
        default="continuous_run",
        help="Subdirectory under outputs/ (default: continuous_run)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Hash embedding + no LLM file logs (still uses real LLM unless mocked)",
    )
    parser.add_argument(
        "--no-debug-files",
        action="store_true",
        help="Console trace only; skip debug/ JSON artifacts",
    )
    parser.add_argument(
        "--compact-llm-log",
        action="store_true",
        help="Write compact LLM logs (no full messages/response); default is full payload",
    )
    args = parser.parse_args()

    repo = find_repo_root()
    ws = ScenarioWorkspace.create(
        repo / "tests" / "scenarios" / "hongloumeng" / "outputs" / args.run_name,
        run_name=args.run_name,
    )

    print(f"Workspace: {ws.base_dir}", flush=True)
    print(f"Last chunk idx (before): {ws.load_last_chunk_idx()}", flush=True)

    results = run_chunks(
        workspace=ws,
        count=args.count,
        chunk_id=args.chunk_id,
        reset=args.reset,
        offline=args.offline,
        write_debug_files=not args.no_debug_files,
        full_llm_log=not args.compact_llm_log,
    )

    print(f"\nDone. last_chunk_idx={ws.load_last_chunk_idx()}, processed={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
