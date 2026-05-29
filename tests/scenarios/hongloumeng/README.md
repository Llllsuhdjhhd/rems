# 红楼梦场景测试（2026.06）

长文本分块 ingest 的推荐入口与目录说明见仓库根 [`tests/TESTING.md`](../../TESTING.md)。

## 快速开始

```powershell
# 每次 ingest 下一块（状态保存在 outputs/continuous_run/）
python tests/scenarios/hongloumeng/run_chunk.py

# 连续 3 块
python tests/scenarios/hongloumeng/run_chunk.py --count 3

# 从第 1 块重来
python tests/scenarios/hongloumeng/run_chunk.py --reset --chunk-id 1

# 离线冒烟（pytest 用 FakeLLM，见 test_chunk_runner.py）
pytest tests/scenarios/hongloumeng/test_chunk_runner.py::test_offline_ingest_one_chunk -q
```

## 模块

| 文件 | 职责 |
|------|------|
| `dataset.py` | 读 `data/hongloumeng_dataset.json`，按 id 取块 |
| `runner.py` | 单块/多块 ingest，写 report + recall_trace |
| `run_chunk.py` | CLI |
| `test_chunk_runner.py` | pytest：离线一块 + live 一块 |
| `test_dataset.py` | 数据集格式校验 |

## 旧脚本（已归档）

以下文件在 `tests/_archive/hongloumeng/`，含 2026.05 双流诊断逻辑，与三频段主路径不一致：

- `continuous_simulation.py`、`simulation.py`、`diagnostic_recall.py`
- `ingest_one_chunk_recall_report.py`、`visualize_recall.py`、`phase2_metabolism_diagnostic.py`
- `inspection.py`、`probe_ROLE_SKILL_continuous_db_readonly.py`

新诊断请查看每块产出：

- `outputs/<run>/chunk_<id>_report.json`
- `outputs/<run>/chunk_<id>_recall_trace.json`（intent_weights、tri_band 命中）
