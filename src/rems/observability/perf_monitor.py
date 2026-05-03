"""Rolling-window latency monitor for non-LLM algorithm phases.

白皮书 §2.3 "高负载自我保护（遗忘加速）" 的工程实现入口。

设计目标
--------
- 给系统中**主要的非 LLM 算法**（向量检索 / 回忆块组装 / 抽象事件挖掘 / 叙事线判重 等）
  挂一个轻量的"耗时观测器"——不依赖 OpenTelemetry，零运行时依赖。
- 每个 phase 维护一个**滚动窗口**的耗时样本（默认 32 次），算移动平均；超过该 phase 的
  ``tolerance_ms`` 视为"过载"。
- 暴露一个**全局负载因子** ``load_factor() ∈ [1.0, perf_load_factor_max]``：
  - 1.0 = 系统空闲；
  - >1.0 = 至少有一个 phase 的均值越线，越线越多倍数越大。
- 提供两条**驱动力**接口给下游使用：
  - ``adjusted_silence_threshold(base)``：把"白描遗忘静默阈值"按负载比例**抬高**——负载越大、
    silence 越严，更多老条目被踢出活跃池（白皮书 §2.3 高负载自我保护）。
  - ``adjusted_recent_k(default_k, min_k)``：把"叙事线判重比对窗口 K"按负载**收紧**——负载越大、
    K 越小，避免在已经卡的算法上再花算力。

线程模型
--------
当前 ingest 单线程串行，PerfMonitor 不上锁。如果未来引入多线程消费同一 monitor，需要给
``record`` 加 mutex；接口语义保持不变。

零开销关闭
----------
默认 ``enabled=True``。测试 / 性能对比时 ``enabled=False`` 后所有 timer 退化为 no-op，
``load_factor()`` 恒为 1.0。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class PhaseStat:
    """Rolling stats for a single phase.

    保留最近 ``window_size`` 次耗时（毫秒），用 ``samples`` 去算 mean。
    ``tolerance_ms`` 为该 phase 的"健康上限"——mean 越过它就开始为 load_factor 贡献。
    """

    name: str
    tolerance_ms: float
    window_size: int = 32
    samples: deque[float] = field(default_factory=deque)

    def record(self, elapsed_ms: float) -> None:
        if self.window_size <= 0:
            return
        self.samples.append(elapsed_ms)
        # 维持滚动窗口
        while len(self.samples) > self.window_size:
            self.samples.popleft()

    @property
    def mean_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    @property
    def overload_ratio(self) -> float:
        """Return mean / tolerance, or 0.0 when no samples / no tolerance.

        > 1.0 表示该 phase 越线；越大越严重。
        """
        if self.tolerance_ms <= 0 or not self.samples:
            return 0.0
        return self.mean_ms / self.tolerance_ms


class PerfMonitor:
    """Aggregator over multiple :class:`PhaseStat` objects with load-factor projection.

    用法（约定一）::

        with monitor.timer("rag_search"):
            results = vector_store.search(...)

    用法（约定二）::

        t0 = time.perf_counter()
        do_work()
        monitor.record("abstraction_mining", (time.perf_counter() - t0) * 1000)

    任意时刻 ``monitor.load_factor()`` 给出当前过载倍率：
        - 所有 phase 都未越线 → 1.0
        - 任一 phase 越线 → ``max(1.0, max_overload_ratio)``，并夹到 ``load_factor_max``
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        window_size: int = 32,
        tolerances_ms: dict[str, float] | None = None,
        load_factor_max: float = 4.0,
        silence_boost_at_max: float = 4.0,
    ) -> None:
        """
        ``tolerances_ms``：phase 名 → 健康上限（毫秒）。新 phase 在第一次 ``record`` 时按
            ``_default_tolerance_ms`` 自动注册（避免跑出不在表里的 phase 时崩）。
        ``load_factor_max``：``load_factor()`` 输出的硬上限——避免极端尖峰把下游阈值放大到无意义。
        ``silence_boost_at_max``：``adjusted_silence_threshold`` 在满载时把 base 阈值放大的最大倍数；
            线性插值到 ``load_factor_max``。
        """
        self.enabled = enabled
        self._window_size = window_size
        self._load_factor_max = max(1.0, float(load_factor_max))
        self._silence_boost_at_max = max(1.0, float(silence_boost_at_max))
        self._stats: dict[str, PhaseStat] = {}
        for name, tol in (tolerances_ms or {}).items():
            self._stats[name] = PhaseStat(
                name=name, tolerance_ms=float(tol), window_size=window_size,
            )
        # 兜底容忍度：未在表中的 phase 给一个相对宽松的默认值（200ms），第一次 record 时注册。
        self._default_tolerance_ms = 200.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, phase: str, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        stat = self._stats.get(phase)
        if stat is None:
            stat = PhaseStat(
                name=phase,
                tolerance_ms=self._default_tolerance_ms,
                window_size=self._window_size,
            )
            self._stats[phase] = stat
            logger.debug("perf_monitor: auto-registered phase %s with default tolerance", phase)
        stat.record(elapsed_ms)

    @contextmanager
    def timer(self, phase: str) -> Iterator[None]:
        """Context manager that records wall-clock elapsed time of the inner block.

        ``with monitor.timer("rag_search"): ...`` 等价于在出入两端 ``time.perf_counter()``
        然后 ``monitor.record("rag_search", elapsed_ms)``。出现异常也会记录耗时（避免漏样本
        让 mean 偏低，错估系统负载）。
        """
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record(phase, elapsed_ms)

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    def stats_snapshot(self) -> dict[str, PhaseStat]:
        return dict(self._stats)

    def mean_ms(self, phase: str) -> float:
        stat = self._stats.get(phase)
        return stat.mean_ms if stat else 0.0

    def is_overloaded(self) -> bool:
        return self.load_factor() > 1.0

    def load_factor(self) -> float:
        """Maximum overload ratio across all known phases, clamped to ``[1.0, load_factor_max]``.

        某个 phase 没有样本时 ``overload_ratio = 0`` 不参与 max。整个系统在 load_factor=1.0
        与 load_factor=load_factor_max 之间线性映射其它"应对动作"的强度。
        """
        if not self.enabled or not self._stats:
            return 1.0
        max_ratio = max((s.overload_ratio for s in self._stats.values()), default=0.0)
        if max_ratio <= 1.0:
            return 1.0
        return min(self._load_factor_max, max_ratio)

    # ------------------------------------------------------------------
    # Driving knobs (exposed to downstream subsystems)
    # ------------------------------------------------------------------

    def adjusted_silence_threshold(self, base_threshold: float) -> float:
        """Tighten the white-painting silence threshold under load.

        线性插值：
            load_factor=1.0      → 阈值 = base_threshold（不动）
            load_factor=max      → 阈值 = base_threshold * silence_boost_at_max
        负载越高，阈值越高，更多老条目跌破阈值进入 SILENT，等价于"系统主动遗忘加速"
        （白皮书 §2.3 高负荷自我保护）。
        """
        lf = self.load_factor()
        if lf <= 1.0 or self._load_factor_max <= 1.0:
            return base_threshold
        # t ∈ (0, 1]
        t = (lf - 1.0) / (self._load_factor_max - 1.0)
        boost = 1.0 + (self._silence_boost_at_max - 1.0) * t
        return base_threshold * boost

    def adjusted_recent_k(self, default_k: int, min_k: int) -> int:
        """Shrink any "compare with N most recent items" window under load.

        典型用法：抽象事件叙事线判重时，比对最近 K 条已有抽象事件——负载高时把 K 从
        ``default_k`` 收到 ``min_k``，按 ``load_factor`` 线性插值。即使 ``min_k=0``
        也保证返回值 ≥ 1（永远至少比 1 条，避免完全跳过判重导致重复抽象）。
        """
        lf = self.load_factor()
        floor_k = max(1, int(min_k))
        if lf <= 1.0 or self._load_factor_max <= 1.0:
            return max(floor_k, int(default_k))
        t = (lf - 1.0) / (self._load_factor_max - 1.0)
        k = int(default_k - (default_k - floor_k) * t)
        return max(floor_k, k)
