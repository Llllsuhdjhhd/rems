---
name: 白皮书指标 Live 报告套件
overview: 以已有的 test_metabolism_live_first_20_inputs.py 为母版，构建三条 Live LLM 驱动的场景测试，按白皮书章节产出可读 Markdown 指标报告（含原文引用 + 现场输入输出 + 实测值 + pass/fail 判定）。
todos:
  - id: metrics-pkg
    content: 建 tests/metrics/ 子包：reporter.py / probes.py / instrumentation.py
    status: completed
  - id: single-mode-report
    content: tests/test_wp_live_single_mode_narrative.py + tests/reports/wp_live_single_mode.md 剧本与报告渲染
    status: completed
  - id: multi-modes-report
    content: tests/test_wp_live_multi_and_modes.py + wp_live_multi_and_modes.md 单个文件覆盖 §2.2/§5.1-5.3
    status: completed
  - id: abstraction-recall-report
    content: tests/test_wp_live_abstraction_and_recall.py + wp_live_abstraction_and_recall.md：手构历史事件，测 §3.2/3.3/§4.3/4.4
    status: completed
  - id: index
    content: 报告结束时自动写 wp_live_INDEX.md 总入口，附 pass/fail 汇总
    status: completed
  - id: docs
    content: 在 tests/reports/ 存放一份 README 说明运行三条 live 套件的升级命令和 env 开关
    status: completed
isProject: false
---

# 白皮书指标 Live 报告套件

## 设计原则

- **一切围绕白皮书可观察的字段与阈值**：每个指标要能映射到一个代码中的属性或 config 键，方便事后人工核对。
- **报告先行**：每条测试的 `assert` 只保证流程不崩，真正的"是否达标"通过渲染到 Markdown 的指标核对块呈现，以便人眼扫读。
- **最小 LLM 调用**：单文件内尽量多覆盖章节，避免反复构造 pipeline。
- **保持与现有测试并存**：新文件全部以 `test_wp_live_*.py` 命名；现有 `tests/test_metabolism_live_first_20_inputs.py` 保留不动，作为代谢子场景的深度报告，新套件作为"白皮书体检"总入口。

## 文件结构

```
tests/
├── metrics/
│   ├── __init__.py
│   ├── reporter.py          # WPReporter：章节/指标累积 + Markdown 渲染
│   ├── probes.py            # 每个白皮书指标的探针函数（返回 dict）
│   └── instrumentation.py   # LLM trace（system prompt 前 500 字也记录）
├── test_wp_live_single_mode_narrative.py
├── test_wp_live_multi_and_modes.py
├── test_wp_live_abstraction_and_recall.py
└── reports/
    ├── wp_live_INDEX.md                      # 自动生成：总入口，列出各报告链接与 pass/fail 摘要
    ├── wp_live_single_mode.md
    ├── wp_live_multi_and_modes.md
    └── wp_live_abstraction_and_recall.md
```

## 报告骨架（每份报告）

```
# WP-LIVE · <场景名>
- 生成时间 / 配置快照（context_window、len_msg、user_mode、ae_*、wp_* 等）
- LLM/嵌入环境摘要
- 指标核对表（一行一个指标，链接到下方各章节）

## §X.Y.Z <指标名>
### 白皮书原文
> ……（从 readme.md 对应章节抽取 1-3 句）
### 检查方法
- 触发方式、涉及字段、阈值
### 现场数据
- 输入 / 模型调用（task_type、prompt 摘要、耗时）
- 实测值 vs 期望（绿色 PASS / 红色 FAIL 标签）
- 关键对象 JSON 摘录（折叠）
### 备注（必要时）

## 运行摘要
## 环境 & 复现命令
```

## 测试剧本三件套

### 1. `tests/test_wp_live_single_mode_narrative.py` → `wp_live_single_mode.md`

单人模式全景剧本（≈15-20 轮叙述输入），一次跑完覆盖最多指标。

- **§1.1.1** `event_id` 字典序递增 · 字段探针 `probes.event_id_monotonic`
- **§1.1.3** 递归摘要熔断：`actual_max_level` 生效、最末级字数 < `summary_fuse_min_chars`
- **§1.1.7** 物理上限 `event_length ≤ len_msg` · 防碎片化聚合（第 i 轮用"我去跑了步→回家→打扫→看小说" 特意诱导合并）
- **§2.2 单人模式注入**：instrumentation 捕获首个 `role_extraction` 调用的 system prompt，确认出现"单人隔离模式"与 `core_user_role_id` · 报告里折叠展示前 300 字
- **§2.3 白描收集粒度**：在配置 `core_user_role_id = ROL-user` 的前提下抽 2-3 条样本白描，断言其 `role_summary` 与来源事件的 `l3_decision` 一致（主角→L3），其他配角应落在 `l2_interaction`
- **§2.3 动态遗忘**：构造若干天前 + 今日各 N 条白描，调用 `role_service.get_white_painting_summary` 观察高 AE 条目留存数 vs 低 AE 衰减数；报告给出半衰公式代入值
- **§2.4 语义卡片**：每轮后读 `role_service.get_semantic_card(core_user_role_id)`，断言 `len(data.keys()) ≤ semantic_card_max_keys`；报告最后 3 次 data 快照
- **§2.5 EMA 指数平滑**：对同一 `role_id` 连续两轮比较 `entry.emotional_model.vedana.joy` 前后差 = `α·current + (1-α)·history`；报告以表格列出"新事件 LLM 粗估 / 历史均值 / 平滑后 / 偏差"
- **§2.5 activation_energy 硬绑定**：统计 `AE ≥ ae_high_threshold` 的事件，验证其 `activation_energy ≥ ae * activation_energy_gain * 1.5 - ε`
- **§4.4 混合打分**：任选一轮在 DIALOGUE 模式下保存 `RecallBlock`，报告展示 top-K 的 `{event_id, score, summary_level}` 与 `_hybrid_score` 四项分量（通过 monkey-patch 记录 cosine/time_decay/role_boost/ae/act 中间值）
- **§4.4 懒索引降级 / ultra fallback**：调大 `recall_cluster_threshold` 或构造超长事件，观察 `summary_level` 是否出现 `ultra`

### 2. `tests/test_wp_live_multi_and_modes.py` → `wp_live_multi_and_modes.md`

覆盖多人模式 + 其余两种输出场景。

- **§2.2 多人模式注入**：`config.user_mode = MULTI`、`active_participants = ["Alice","Bob","Carol"]`；一条包含多人代词的输入（"她把杯子递给他，然后我们都笑了"），断言 system prompt 出现花名册；抽取结果 `roles` 长度 ≥ 2 且未把所有无主语都归到单一核心用户
- **§2.3 配角 L2 收集**：多人模式下 `core_user_role_id` 为空，任何 role 的白描 summary 应取 `l2_interaction`
- **§5.1 DIALOGUE**：`ProcessingResult.context_package` 非空，`recall_block.total_length ≤ physical_redline`
- **§5.2 PASSIVE_LOG**：同一输入切到 PASSIVE_LOG，断言 `context_package is None` 且 `sealed_events` 仍产生（"只记不说"）
- **§5.3 NPC_AGENT**：构造一条历史事件描述 NPC 被玩家伤害，触发 ingest with `npc_role_id=ROL-npc`，断言 `npc_directives[0]["action"]` ∈ {`watch`,`flee`} 且含 `klesha_delta`

### 3. `tests/test_wp_live_abstraction_and_recall.py` → `wp_live_abstraction_and_recall.md`

抽象 + 回忆 + 墓碑的专项剧本。这里用"手工构造多条相关基本事件（直接走 `event_repo.save` + `vector_store.add_event`）+ 真实 LLM 做抽象与回忆"的组合，能用很少的 LLM 调用覆盖最多指标。

- **§3.2 召回聚集触发**：塞入 6 条主题相近（例如都围绕"张三在会议讨论项目进度"）的基本事件，调用 `abstraction_service.check_and_abstract`；报告展示触发前后 `count(recall_cluster) ≥ recall_cluster_threshold`、生成的抽象事件 `abstraction_level`、`source_events`、`insight`
- **§3.3 防幻觉锚定**：把 `hallucination_anchor_prob` 临时设为 `1.0`，重复构造抽象一次，抽取 instrumentation 捕获的 prompt 文本断言出现"L1（锚定事实层）"标签；再设为 `0.0` 观察使用 `mid_summary_key`
- **§4.3 墓碑化**：对其中一条事件 `tombstone`，再用相同 query 调用 `build_recall_block`，断言结果不含该 `event_id`；报告展示墓碑前后的 top-K 对比
- **§4.4 回忆上限**：刻意构造 L1 摘要极长的事件若干条，触发 `_ultra_concise_fallback`；报告展示 `block.total_length ≤ physical_redline` 与 `summary_level == "ultra"` 的条目
- **§4.4 混合打分四项分量导出**：对每个 hit 打印 `cosine / time_decay / role_boost / ae / act` 的加权贡献饼图式文本（如 `cosine:0.42 | time:0.08 | role:0.06 | ae:0.09 | act:0.04`）

## 核心支持模块

### `tests/metrics/reporter.py`

```python
class WPReporter:
    def __init__(self, report_path: Path, title: str) -> None: ...
    def set_config_snapshot(self, cfg: REMSConfig) -> None: ...
    def section(self, clause: str, title: str) -> "SectionCtx": ...
    # SectionCtx 上下文管理器：支持 .quote(...), .method(...), .metric(name, actual, expected, ok), .json_block(...), .note(...)
    def finalize(self) -> None: ...  # 写完整报告 + 更新 wp_live_INDEX.md
```

指标以 `(name, actual, expected, ok: bool, evidence: str | dict)` 元组累积，最终渲染为章节顶部的**指标核对表**和下方的详细段落。

### `tests/metrics/instrumentation.py`

将 `pipeline.llm.complete` 与 `vector_store.search` 两个入口 monkey-patch，把调用记录写入 `WPReporter` 的环境上下文。默认只保留**首 500 字 system prompt 摘要**、response 前 72 字与耗时（复用现有 live 测试的打印风格）。

### `tests/metrics/probes.py`

纯函数形态，对 Event / Role / RecallBlock 做读数：

```python
def event_length_vs_budget(event, cfg) -> dict: ...
def summary_fuse_state(event, cfg) -> dict: ...
def ema_smoothing_check(prev_history_mean, current_snapshot, smoothed, alpha) -> dict: ...
def activation_energy_binding(event, cfg) -> dict: ...
def recall_block_budget(block, cfg) -> dict: ...
def prompt_contains_user_mode_block(captured_system_prompt, mode) -> dict: ...
def white_painting_tier_check(wp_entry, source_event, is_primary, cfg) -> dict: ...
def abstraction_level_chain(abstract_event, cluster) -> dict: ...
def tombstone_exclusion(recall_items, tombstoned_id) -> dict: ...
```

每个探针返回 `{"name": ..., "expected": ..., "actual": ..., "ok": bool, "detail": ...}`，直接喂给 `WPReporter`。

## 运行约定

- 所有 Live 测试都带 `@pytest.mark.live_llm` 与现有一致的 env gate（`REMS_RUN_LIVE_METABOLISM_TEST=1`）；另加 `REMS_WP_REPORT_DIR` 以便重定向报告目录。
- `tests/metrics/reporter.py::finalize()` 退出时自动重写 `tests/reports/wp_live_INDEX.md`，列出最新每份报告的 pass/fail 汇总（供 PR 审阅时一眼看完整体状态）。
- 为了**复现失败**，每份报告末尾附上：生成时间、Python 解释器、pytest 命令、`.env` 关键键名（值不落盘）、种子（如有）。

## 与现有 live 测试的关系

- `tests/test_metabolism_live_first_20_inputs.py` 保留为"代谢专项深度日志"，不改动。
- 新套件的 `test_wp_live_single_mode_narrative.py` 与其共享类似的 20 轮剧本思路，但关注点从"代谢落盘细节"转为"白皮书条款核对"，两者互补。

## 非 Live 能直接观察的补充指标（本方案内的降级处理）

用户选择只做 L3，但以下指标严格意义上是**非 Live 可观察**的（与模型输出无关、或需要大量重复采样才能稳定）：

- **§3.3 锚定概率的长期分布**：一次 Live 测试无法统计 `hallucination_anchor_prob = 0.3` 的频率。方案内通过**直接 monkey-patch 把它先后设成 0/1** 各跑一次来断言"行为开关"而非分布。
- **§4.2 0.8:0.2 模糊截断**：阈值在 prompt 层由 LLM 执行，非确定；报告里**展示实测比例**但不以其作为 pass/fail 判定依据，只做陈列。
- **§2.2 单人/多人提示词的实际效力**（模型是否真的按指示做共指消解）：报告里展示**抽取结果**供人工主观判断，不做自动 assert（只自动断言"提示词内容确实注入"这一硬条件）。

如果以后要严格量化这几项，建议补一条 L2 脚本化离线测试（本方案不覆盖，保留扩展空间）。
