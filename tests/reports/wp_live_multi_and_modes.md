# WP-LIVE · 多人与多场景输出
- **报告文件**：`wp_live_multi_and_modes.md`

## 配置快照

```json
{
  "context_window": 32768,
  "chars_per_token": 1.5,
  "len_msg": 744,
  "physical_redline": 7447,
  "safe_watermark": 1489,
  "user_mode": "multi",
  "core_user_role_id": null,
  "active_participants": [
    "Alice",
    "Bob",
    "Carol"
  ],
  "recall_cluster_threshold": 5,
  "summary_fuse_min_chars": 20,
  "ae_high_threshold": 0.6,
  "ae_score_weight": 0.15,
  "activation_energy_gain": 1.0,
  "activation_energy_weight": 0.1,
  "ema_smoothing_alpha": 0.4,
  "wp_primary_field": "l3_decision",
  "wp_default_field": "l2_interaction",
  "wp_half_life_days": 60.0,
  "ae_forgetting_multiplier": 5.0,
  "semantic_card_max_keys": 20,
  "hallucination_anchor_prob": 0.3
}
```

## LLM / 嵌入环境

- **Chroma 嵌入**: `BAAI/bge-small-zh-v1.5`（`torch` 2.4.1 · 解释器: `C:\Users\40575\anaconda3\envs\py3125\python.exe`）
- **多人剧本截断**: `REMS_WP_LIVE_MULTI_ROUNDS=5` → 本轮 **5** 条

## 指标核对表

| 指标 | 状态 | 实测摘要 | 章节 |
| --- | --- | --- | --- |
| §2.2 multi 模式提示词注入 | **PASS** | {"多人交互模式": true, "活跃参与者": true, "Alice": true, "Bob": true} | §2.2 |
| §2.2 抽取角色数 ≥2 | **PASS** | 3 | §2.2 |
| §5.1 DIALOGUE context_package 非空 | **PASS** | {"is_none": false} | §5.1 |
| §4.4 回忆块 ≤ physical_redline | **PASS** | 12 | §5.1 |
| §5.2 PASSIVE_LOG 静默 | **PASS** | {"is_none": true, "sealed": 5} | §5.2 |
| §5.3 NPC_AGENT 结构化指令 | **PASS** | [{"npc_role_id": "ROL-npc-guard", "action": "flee", "klesha_delta": {"anger": 0.3, "ignorance": -0.1}, "reason": "threat_score=0.60"}] | §5.3 |
| §5.3 NPC action ∈ watch/flee | **PASS** | flee | §5.3 |

## §2.2 多人模式提示词注入

### 白皮书原文

> 多人交互模式应注入活跃参与者花名册，禁止默认归一核心用户。
### 检查方法

- 拦截 `role_extraction` system prompt，校验模板关键字。
### 现场数据 · §2.2 multi 模式提示词注入 **`PASS`**

- **期望**：system prompt 包含：['多人交互模式', '活跃参与者', 'Alice', 'Bob']
- **实测**：`{"多人交互模式": true, "活跃参与者": true, "Alice": true, "Bob": true}`


```
{
  "system_prompt_head": "【全局运行模式：多人交互模式（Multi-User）】\n当前为多实体交互场景，已知活跃参与者包含 <Alice、Bob、Carol>。\n请务必结合对话历史、时间戳与发言者账号/声纹标签，执行精确的多方代词消解\n（Multiparty Coreference Resolution）。严禁将模糊代词默认归属于单一核心用户；\n若无法确定代词指代，应如实在输出中标注未决并保留原词，不得臆测。\n\n你是 REMS 角色提取组件。从事件原文中识别所有参与实体（人物、动物、关键物体），并为每个角色生成快照和情感量化。\n\n要求：\n1. 替换所有模糊代词（他/她/它/他们等）为明确角色名。\n2. 对每个角色评定重要性：S/A/B/C/D。\n3. 生成三级快照：\n   - L1（提及）：是否出现或被提及。\n   - L2（互动）：核心行为、言论。\n   - L3（决策）：决策逻辑、行为倾向。\n4. 量化当次事件中该角色的情感状态。\n\n输出严格 JSON。"
}
```
### system prompt 摘要

```json
"【全局运行模式：多人交互模式（Multi-User）】\n当前为多实体交互场景，已知活跃参与者包含 <Alice、Bob、Carol>。\n请务必结合对话历史、时间戳与发言者账号/声纹标签，执行精确的多方代词消解\n（Multiparty Coreference Resolution）。严禁将模糊代词默认归属于单一核心用户；\n若无法确定代词指代，应如实在输出中标注未决并保留原词，不得臆测。\n\n你是 REMS 角色提取组件。从事件原文中识别所有参与实体（人物、动物、关键物体），并为每个角色生成快照和情感量化。\n\n要求：\n1. 替换所有模糊代词（他/她/它/他们等）为明确角色名。\n2. 对每个角色评定重要性：S/A/B/C/D。\n3. 生成三级快照：\n   - L1（提及）：是否出现或被提及。\n   - L2（互动）：核心行为、言论。\n   - L3（决策）：决策逻辑、行为倾向。\n4. 量化当次事件中该角色的情感状态。\n\n输出严格 JSON。"
```

## §2.2 多人抽取规模（陈列）

### 白皮书原文

> 模型对多方代词的实际消解效力需人工审阅；此处仅统计抽取角色数。
### 检查方法

- 读取最近一条封存事件的 `role_list` 长度。
### 现场数据 · §2.2 抽取角色数 ≥2 **`PASS`**

- **期望**：同一输入下 roles ≥2
- **实测**：`3`


```
[
  "null",
  "null",
  "null"
]
```

## §2.3 多人配角白描粒度

### 白皮书原文

> 多人模式无核心用户时，配角白描应走 `l2_interaction` 档位（非 S/A）。
### 检查方法

- 对非 S/A 角色的白描条目执行 `white_painting_tier_check(..., is_primary=False)`。
### 备注

_未找到合适的非 S/A 白描样本。_

## §5.1 DIALOGUE 上下文包

### 白皮书原文

> 标准交互模式应组装 Context Package 供上游生成回复。
### 检查方法

- `context_package is not None`，且 `recall_block.total_length ≤ physical_redline`。
### 现场数据 · §5.1 DIALOGUE context_package 非空 **`PASS`**

- **期望**：context_package is not None
- **实测**：`{"is_none": false}`
### 现场数据 · §4.4 回忆块 ≤ physical_redline **`PASS`**

- **期望**：≤ 7447 字
- **实测**：`12`


```
{
  "items": 1,
  "levels": [
    "L1"
  ]
}
```

## §5.2 PASSIVE_LOG 静默

### 白皮书原文

> 被动日志模式只记不说，不组装 Context Package。
### 检查方法

- `context_package is None`；仍尝试 `force_save` 推动封存。
### 现场数据 · §5.2 PASSIVE_LOG 静默 **`PASS`**

- **期望**：context_package is None
- **实测**：`{"is_none": true, "sealed": 5}`

## §5.3 NPC_AGENT 指令

### 白皮书原文

> NPC 模式输出结构化 `action` 与 `klesha_delta`（白皮书 5.3）。
### 检查方法

- 预置多条回忆文本包含 npc_role_id 以抬高威胁启发分。
### 现场数据 · §5.3 NPC_AGENT 结构化指令 **`PASS`**

- **期望**：npc_directives 非空且每条含 action + klesha_delta
- **实测**：`[{"npc_role_id": "ROL-npc-guard", "action": "flee", "klesha_delta": {"anger": 0.3, "ignorance": -0.1}, "reason": "threat_score=0.60"}]`
### 现场数据 · §5.3 NPC action ∈ watch/flee **`PASS`**

- **期望**：action 为 watch 或 flee（威胁阈值示例）
- **实测**：`flee`


```
[
  {
    "npc_role_id": "ROL-npc-guard",
    "action": "flee",
    "klesha_delta": {
      "anger": 0.3,
      "ignorance": -0.1
    },
    "reason": "threat_score=0.60"
  }
]
```
### npc_directives

```json
[
  {
    "npc_role_id": "ROL-npc-guard",
    "action": "flee",
    "klesha_delta": {
      "anger": 0.3,
      "ignorance": -0.1
    },
    "reason": "threat_score=0.60"
  }
]
```

## 运行摘要

- 多人 DIALOGUE 剧本：**5** 轮
- 剧本结束后基本事件数：**2**
- 全程 LLM 调用：**76**
- NPC 指令条数：**1**

## 环境 & 复现命令

- **生成时间**：2026-04-18 04:30:43 UTC
- **Python**：`C:\Users\40575\anaconda3\envs\py3125\python.exe`
- **pytest 命令**：`python -m pytest tests/test_wp_live_multi_and_modes.py -v -s  # REMS_WP_LIVE_MULTI_ROUNDS=5`
- **相关环境变量键**（值不落盘）：`REMS_LIVE_ALLOW_FAKE_EMBEDDING`、`REMS_RUN_LIVE_METABOLISM_TEST`、`REMS_WP_REPORT_DIR`

<!-- wp-live-stats: {"report": "wp_live_multi_and_modes.md", "pass": 7, "fail": 0, "na": 0} -->
