# WP-LIVE 白皮书指标报告

本目录存放 **`test_wp_live_*.py`** 生成的 Markdown 报告与总索引 **`wp_live_INDEX.md`**。

## 运行前提

- 配置 **`.env`** / 环境变量中的 **`REMS_LLM__API_KEY`** 与兼容 OpenAI 的 **`REMS_LLM__BASE_URL`**（与主项目一致）。
- 启用 Live 门闸：

```powershell
$env:REMS_RUN_LIVE_METABOLISM_TEST = "1"
```

- **可选**：将报告输出到其他目录（默认为本目录 `tests/reports/`）：

```powershell
$env:REMS_WP_REPORT_DIR = "D:\rems_wp_reports"
```

## 三条套件命令

在项目根目录执行（需已 `pip install -e ".[dev]"` 或等价安装）：

```powershell
# 单人全景叙述（§1 / §2 / §4.4 等）
python -m pytest tests/test_wp_live_single_mode_narrative.py -v -s

# 多人 + DIALOGUE / PASSIVE_LOG / NPC_AGENT（§2.2–2.3 / §5）
python -m pytest tests/test_wp_live_multi_and_modes.py -v -s

# 手造历史 + 抽象 / 墓碑 / ultra 回忆（§3 / §4.3–4.4）
python -m pytest tests/test_wp_live_abstraction_and_recall.py -v -s
```

一次跑完全部 WP-LIVE：

```powershell
python -m pytest tests/test_wp_live_single_mode_narrative.py tests/test_wp_live_multi_and_modes.py tests/test_wp_live_abstraction_and_recall.py -v -s
```

## 嵌入与 PyTorch

- 默认使用 **Chroma + SentenceTransformer** 本地嵌入，需要可用的 **PyTorch** 环境。
- 若当前解释器无法加载 `torch`（常见于 Windows 下选错 Python），请切换到已验证环境，或使用代谢测试相同的逃逸开关（**仅假向量，语义检索无效**）：

```powershell
$env:REMS_LIVE_ALLOW_FAKE_EMBEDDING = "1"
```

## 产出文件

| 文件 | 说明 |
| --- | --- |
| `wp_live_single_mode.md` | 单人模式剧本报告 |
| `wp_live_multi_and_modes.md` | 多人 + 多输出模式 |
| `wp_live_abstraction_and_recall.md` | 抽象 / 墓碑 / 回忆专项 |
| `wp_live_INDEX.md` | 各报告 **PASS/FAIL/INFO** 计数总览（由 `WPReporter.finalize()` 自动刷新） |

> 与 **`tests/test_metabolism_live_first_20_inputs.py`** 生成的 `metabolism_live_20_last_run.md` **互不覆盖**；代谢文件为「深度代谢日志」，WP-LIVE 为「白皮书条款体检」。
