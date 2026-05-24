"""红楼梦「一块」冒烟：直接执行 ``continuous_simulation.py``，不改动该脚本。

输出与状态仍写在 ``outputs/continuous_run/``（与其它手动长跑共用）；需要干净起点时请自行备份或清空该目录。

启用与其它 ``live_llm`` 相同::

    $env:REMS_RUN_LIVE_METABOLISM_TEST = "1"
    conda run -n py3125 python -m pytest tests/scenarios/hongloumeng/test_continuous_simulation_chunk.py -v -s
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rems.config import REMSConfig

from tests.integration.metrics.live_common import live_llm_skip_if_disabled


@pytest.mark.live_llm
def test_hongloumeng_one_chunk_via_continuous_simulation_script() -> None:
    live_llm_skip_if_disabled()

    cfg = REMSConfig()
    if not (cfg.llm.api_key or "").strip():
        pytest.skip("REMS_LLM__API_KEY is empty — configure .env before running this test.")

    repo_root = Path(__file__).resolve().parents[3]
    dataset = repo_root / "data" / "hongloumeng_dataset.json"
    assert dataset.is_file(), f"红楼梦数据集缺失: {dataset}"

    script = Path(__file__).resolve().parent / "continuous_simulation.py"
    proc = subprocess.run(
        [sys.executable, str(script), "1"],
        cwd=str(repo_root),
        timeout=900,
    )
    assert proc.returncode == 0, "continuous_simulation.py 1 exited non-zero"
