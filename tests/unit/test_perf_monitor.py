"""Unit tests for PerfMonitor: rolling stats, load_factor, and load-driven knobs."""

from __future__ import annotations

from rems.observability import PerfMonitor


def test_disabled_monitor_is_inert():
    pm = PerfMonitor(enabled=False, tolerances_ms={"x": 1.0})
    with pm.timer("x"):
        pass
    assert pm.load_factor() == 1.0
    assert pm.is_overloaded() is False
    assert pm.adjusted_silence_threshold(0.02) == 0.02
    assert pm.adjusted_recent_k(200, 20) == 200


def test_record_and_mean():
    pm = PerfMonitor(tolerances_ms={"phase_a": 100.0})
    for v in [10.0, 20.0, 30.0]:
        pm.record("phase_a", v)
    assert pm.mean_ms("phase_a") == 20.0


def test_rolling_window_evicts_oldest():
    pm = PerfMonitor(window_size=3, tolerances_ms={"p": 100.0})
    for v in [100.0, 200.0, 300.0, 400.0]:
        pm.record("p", v)
    # window=3 → 应只保留 [200, 300, 400]
    assert pm.mean_ms("p") == 300.0


def test_load_factor_normal_when_below_tolerance():
    pm = PerfMonitor(tolerances_ms={"p": 100.0}, load_factor_max=4.0)
    pm.record("p", 50.0)
    assert pm.load_factor() == 1.0
    assert pm.is_overloaded() is False


def test_load_factor_scales_with_overload_ratio():
    pm = PerfMonitor(tolerances_ms={"p": 100.0}, load_factor_max=4.0)
    pm.record("p", 250.0)  # ratio = 2.5
    assert pm.load_factor() == 2.5
    assert pm.is_overloaded()


def test_load_factor_capped_at_max():
    pm = PerfMonitor(tolerances_ms={"p": 100.0}, load_factor_max=4.0)
    pm.record("p", 1000.0)  # ratio = 10
    assert pm.load_factor() == 4.0


def test_load_factor_uses_max_across_phases():
    pm = PerfMonitor(
        tolerances_ms={"a": 100.0, "b": 100.0}, load_factor_max=4.0,
    )
    pm.record("a", 50.0)   # ratio 0.5 → 1.0 floor
    pm.record("b", 200.0)  # ratio 2.0
    assert pm.load_factor() == 2.0


def test_unknown_phase_auto_registers_with_default_tolerance():
    pm = PerfMonitor(tolerances_ms={})
    pm.record("never_seen_phase", 50.0)
    # 默认 tolerance 200ms → 50/200 = 0.25 < 1.0
    assert pm.load_factor() == 1.0


def test_adjusted_silence_threshold_linear_interpolation():
    pm = PerfMonitor(
        tolerances_ms={"p": 100.0},
        load_factor_max=4.0,
        silence_boost_at_max=4.0,
    )
    base = 0.02
    # 无负载
    assert pm.adjusted_silence_threshold(base) == base

    # 满载 (lf=4): boost=4.0 → base*4
    pm.record("p", 1000.0)  # ratio 10 → clamp to lf=4
    assert pm.adjusted_silence_threshold(base) == base * 4.0


def test_adjusted_silence_threshold_at_midload():
    # 健康上限 100ms；样本均值 250ms → ratio=2.5 → load_factor=2.5
    # max=4.0 → t=(2.5-1)/(4-1)=0.5 → boost=1+(4-1)*0.5=2.5
    pm = PerfMonitor(
        tolerances_ms={"p": 100.0},
        load_factor_max=4.0,
        silence_boost_at_max=4.0,
    )
    pm.record("p", 250.0)
    assert pm.adjusted_silence_threshold(0.02) == 0.02 * 2.5


def test_adjusted_recent_k_shrinks_with_load():
    pm = PerfMonitor(tolerances_ms={"p": 100.0}, load_factor_max=4.0)

    # No load: returns default
    assert pm.adjusted_recent_k(200, 20) == 200

    # Full overload: returns min
    pm.record("p", 1000.0)
    assert pm.adjusted_recent_k(200, 20) == 20


def test_adjusted_recent_k_floor_is_at_least_one():
    pm = PerfMonitor(tolerances_ms={"p": 100.0}, load_factor_max=4.0)
    pm.record("p", 1000.0)
    # 即使 min_k=0 也会返回 1，避免完全跳过判重
    assert pm.adjusted_recent_k(200, 0) == 1


def test_timer_context_records_elapsed():
    import time

    pm = PerfMonitor(tolerances_ms={"sleep_phase": 1000.0})
    with pm.timer("sleep_phase"):
        time.sleep(0.01)  # 10ms
    samples = pm.stats_snapshot()["sleep_phase"].samples
    assert len(samples) == 1
    # 至少 5ms（粗略下限，避免 OS 调度抖动），不会超过 1s
    assert 5.0 < samples[0] < 1000.0


def test_timer_records_even_on_exception():
    pm = PerfMonitor(tolerances_ms={"err": 100.0})
    try:
        with pm.timer("err"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    samples = pm.stats_snapshot()["err"].samples
    assert len(samples) == 1
