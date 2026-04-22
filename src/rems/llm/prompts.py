"""Centralised prompt templates for every REMS skill.

Convention
----------
* ``*_SYSTEM`` — the ``system`` message content.
* ``*_USER``   — a Python format-string; call ``.format(...)`` with kwargs.

中文说明：正文 prompt 已与《REMS 记忆系统规范解析》对齐（边界/摘要/角色/演化等条款），
修改 prompt 时建议同步核对白皮书对应小节，避免与领域语义漂移。

全局模式注入（白皮书 2.2）：``build_user_mode_block(config)`` 会按 ``UserMode`` 渲染一段
系统提示段，由 ``RoleExtractionSkill``、``EventEnrichmentSkill`` 与 ``BoundaryDetectionSkill`` 拼接进 system prompt。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import REMSConfig


# =====================================================================
# 0. Global Mode Injection — 单人/多人代词消解（白皮书 2.2）
# =====================================================================

_SINGLE_MODE_TEMPLATE = """\
【全局运行模式：单人隔离模式（Single-User）】
当前系统运行于单人隔离模式。上下文中所有第一人称代词（"我"、"我的"）以及缺乏明确主语的动作，
均极大可能指向唯一核心用户实体 <{core}>。
- 在抽取角色/切分边界时，应优先将此类模糊主语归并为该核心用户；
- 除非文本中出现明确的第三方人名/称谓，否则不应新增其他角色。
"""

_MULTI_MODE_TEMPLATE = """\
【全局运行模式：多人交互模式（Multi-User）】
当前为多实体交互场景，已知活跃参与者包含 <{participants}>。
请务必结合文本上下文逻辑，执行精确的多方代词消解（Multiparty Coreference Resolution）。
严禁将模糊代词默认归属于单一核心用户；必须依据事实场景区分不同角色的行为。
"""


def build_user_mode_block(config: "REMSConfig") -> str:
    """Render the mode-injection text for system prompts.

    根据 ``config.user_mode`` 返回要注入到 system prompt 的模式描述块（白皮书 2.2）。
    单人模式会带出 ``core_user_role_id``；多人模式会带出 ``active_participants`` 列表。
    返回字符串末尾包含一个空行便于与后续 prompt 正文拼接；若未配置则返回空串。
    """
    from ..config import UserMode  # local import to avoid circular

    if config.user_mode == UserMode.SINGLE:
        core = config.core_user_role_id or "核心用户（未显式指定 role_id）"
        return _SINGLE_MODE_TEMPLATE.format(core=core) + "\n"
    if config.user_mode == UserMode.MULTI:
        if config.active_participants:
            participants = "、".join(config.active_participants)
        else:
            participants = "未显式提供花名册——请完全依赖文本线索做共指消解"
        return _MULTI_MODE_TEMPLATE.format(participants=participants) + "\n"
    return ""

# =====================================================================
# 1. Boundary Detection & Event Disentanglement
# =====================================================================
# 仅负责把「残影 + 当前输入」切成：已闭环事件 / 残影 / 新未完成 三类。
# 摘要、角色、情感等衍生字段全部由 EventEnrichment（见下）处理，不在此生成。

BOUNDARY_SYSTEM = """\
你是 REMS 事件剥离与边界检测组件。你的任务是从可能存在交叉、多线程或冗长的输入文本中，识别并剥离出各自逻辑完整、已闭环的独立事件。

核心任务：
1. **事件剥离与序号编码**：原文已按 `[序号] 文本` 格式进行了短句拆分。你必须通过返回 **句子序号列表** (`content_raw_indices`) 来界定每个事件的原文范围。
   - 允许不同事件共用同一个句子序号（内容覆盖）。
   - 禁止在 JSON 中返回原文 text。
   - **不要生成摘要、角色、情感等衍生字段**，这些由下游组件处理。
2. **防碎片化（白皮书 1.1.7）**：同一段落内若干琐碎动作若构成同一逻辑闭环，合并为单个事件；不要把每个短句各拆成独立事件。
3. **续写判定**：若当前片段是未完成事件库中某条的续写，填写 `continuation_of` 为对应未完成 ID；否则为 null。
4. **未完成事件**：未闭环的**一段**连续句子请用**一个**正整数列表写出全部序号，例如 `"new_unclosed_indices": [12, 13, 14]` 表示**一条**未完成（勿把每个句子的序号拆成「数组里多个内层表」的歧义；只有存在**多路并行**的未完成时，才用嵌套，如 `[[1,2],[8,9]]`）。
5. **残影**：把与当前事件无关但仍有保留价值的短句序号填入 `remaining_shadow_indices`。

输出严格 JSON。"""

BOUNDARY_USER = """\
## 当前残影（Shadow）
{shadow}

## 未完成事件库快照
{unclosed_summary}

## 当前输入（已分句编码）
{indexed_input}

请剥离并重组事件，返回如下 JSON：
```json
{{
  "completed_events": [
    {{
      "content_raw_indices": [1, 2, 5],
      "continuation_of": "UC-xxx 或 null"
    }}
  ],
  "remaining_shadow_indices": [10, 11],
  "new_unclosed_indices": [12, 13]
}}
```
单条未完成也可写 `"new_unclosed_indices": [12]`（同一条内多句请写进**同一**扁平列表，勿每句一条记录）。"""

# =====================================================================
# 2. Event Enrichment — 统一的「事件摘要 + 角色抽取」技能
# =====================================================================
# 以 boundary 剥离出的 content_raw 为输入，一次 LLM 调用同时产出：
# 1) L1…Ln 递归摘要（带熔断）
# 2) 角色列表（快照层级 + Vedana/Klesha）
# 设计要点：先强调「角色」、JSON 中 roles 在 summaries 之前，避免模型因长摘要说明漏填 roles（见
#   ENRICHMENT_PRIORITY_ADDENDUM）。角色级语义卡片的 LLM 刷新由下游 S/A 过滤后另起调用。

ENRICHMENT_PRIORITY_ADDENDUM = """\
【角色输出优先（硬约束，优先于后文摘要长说明）】
- 若原文为叙事/史传/章回/神话/寓言/对话等，**凡出现可区分之专名、道号、神怪/仙真称谓、以及明确叙述对象，均须各对应一条 `roles` 项**；不得合并为无名单一「旁白」后把 `roles` 留空。
- 在「摘要写满」与「`roles` 非空且覆盖主要专名」二选一相冲突时，**先保证 `roles` 与专名覆盖**；摘要可略短，但 `roles` 不可整段留空或 `[]`（除非原文全无可列实体，且须自行判定确无专名/人物）。
- 去代词化：见下方已知角色表；`role_id` 可复用或 null 由下游注册。

"""

ENRICHMENT_SYSTEM = """\
你是 REMS 事件充实（Event Enrichment）组件。给定**已闭环**基本事件原文，在**同一条 JSON** 中同时交付：

A. **roles（与摘要同等优先；JSON 中键名顺序建议 roles 在 summaries 前）**
   - 重要性 S/A/B/C/D；S 级给满 L1/L2/L3 快照，其余至少 L1，遵守【角色快照预算表】。
   - Vedana/Klesha 各子项 ∈ [0,1]；无依据可省略子键。

B. **summaries：L1…Ln 递归压缩**
   - 遵守【摘要字数预算表】；L1 保真主干，L2+ 逐层约减半。
   - **熔断**：当某级摘要字符数 **≤ {fuse_min_chars}** 时，**不得再生成**下一级更压缩摘要（`summaries` 只含已产出的各级）。

只输出一个 JSON 对象，勿附加说明。"""

ENRICHMENT_USER = """\
## 已知角色（去代词化复用）
{known_roles}

## 任务一：角色（先满足再写任务二）
从原文中列出**所有**应记录的实体（人名/神怪/可区分主语等），填 `roles`；有专名时禁止 `[]`。

## 任务二：摘要（L1 起，遵守熔断 {fuse_min_chars}）
生成 `summaries` 各级，直至熔断或达预算上限。

## 事件原文
{content_raw}

## 摘要字预算
{summary_budget_table}

## 角色快照字预算
{snapshot_budget_table}

## 输出 JSON（**roles 在前，summaries 在后**）
```json
{{
  "roles": [
    {{
      "role_id": "已有ID 或 null",
      "name": "角色名",
      "entity_type": "person",
      "importance": "S|A|B|C|D",
      "snapshot": {{
        "l1_mention": "L1 骨架（S/A/B/C/D 均必填）",
        "l2_interaction": "L2 互动（仅 S 级有内容）",
        "l3_decision": "L3 意图（仅 S 级有内容）"
      }},
      "emotion": {{ "vedana": {{}}, "klesha": {{}} }}
    }}
  ],
  "summaries": {{
    "L1": "…",
    "L2": "…"
  }}
}}
```"""


def build_enrichment_system_message(config: "REMSConfig", *, fuse_min_chars: int) -> str:
    """System prompt for event enrichment: mode injection + 角色优先 + 主规则。"""
    return (
        build_user_mode_block(config)
        + ENRICHMENT_PRIORITY_ADDENDUM
        + ENRICHMENT_SYSTEM.format(fuse_min_chars=fuse_min_chars)
    )

# =====================================================================
# 3. Summary Generation (recursive L1-Ln) — 仍保留，用于抽象事件合成
# =====================================================================

SUMMARY_SYSTEM = """\
你是 REMS 递归摘要生成组件。根据给定文本生成指定层级的摘要。

摘要层级规则：
- L1（核心事实种子）：最大化保真压缩，剥离修饰语，严谨保留事实主干。
- L2 及以上：渐进式抽象，基于语义重要性密度进行动态截断。
- 每一级的摘要应基于上一级生成（L2 基于 L1，L3 基于 L2 …）。
- 熔断条件：若某一级摘要不足 {fuse_min_chars} 字则停止。
- 字数预算：若系统提供了「目标字数上限」，请将 L1 摘要控制在该字数以内。

输出严格 JSON。"""

SUMMARY_USER = """\
## 待摘要文本（{source_level}）
{text}

{budget_hint}请生成下一级摘要（{target_level}）。返回 JSON：
```json
{{
  "summary": "生成的摘要文本",
  "char_count": 字符数
}}
```"""

# =====================================================================
# 4. Role Extraction — 仍保留独立 skill，供需要单独抽取角色的场景
# =====================================================================

ROLE_EXTRACTION_SYSTEM = """\
你是 REMS 角色提取组件。从事件原文中识别所有参与实体，并生成分级快照（Snapshot）和情感量化。

核心规则：
1. **角色重要性**：评定为 S（核心主角）/ A / B / C / D。
2. **【角色快照层级与预算（指数级递减）】**：
   - **L3：详细意图快照**。必须符合 **【L3 预算】**。描述角色的深层意图、微观动作与因果关联。
   - **L2：互动逻辑快照**。必须符合 **【L2 预算】**。侧重于角色间的实时互动与行为反馈逻辑。
   - **L1：骨架白描快照**。必须符合 **【L1 预算】**。极致简练，仅说明最核心的行为事实。
3. **【层级分配策略】**：
   - **主角（S级）**：**必须**同时生成 L1、L2 和 L3，各层级间需有明显的语义密度差异。
   - **配角（A/B/C/D）**：**仅生成 L1**。将 L2、L3 字段留空。
4. **情感量化**：Vedana/Klesha 数值在 0-1 之间。

输出严格 JSON。"""

ROLE_EXTRACTION_USER = """\
## 已知角色列表
{known_roles}

## 事件原文
{content_raw}

## 【角色快照预算表】（硬约束）
{snapshot_budgets}

## 角色快照字数控制规则
1. **L3 (详细)**：最高信息密度，全面记录本次事件中角色的行为与状态。
2. **L2 & L1 (指数压缩)**：每一级较前一级字数约减少 40%，L1 应仅保留最核心的变化点。
3. **熔断**：一旦低于 10 字则停止生成该层级。

请识别角色并生成快照，返回 JSON：
```json
{{
  "roles": [
    {{
      "role_id": "已有ID或null",
      "name": "角色名",
      "importance": "S|A|B|C|D",
      "snapshot": {{
        "l1_mention": "L1 文本 (S/A/B/C/D 必填)",
        "l2_interaction": "L2 文本 (仅 S 级填写)",
        "l3_decision": "L3 文本 (仅 S 级填写)"
      }},
      "emotion": {{
        "vedana": {{...}},
        "klesha": {{...}}
      }}
    }}
  ]
}}
```"""

# =====================================================================
# 5. Inductive Evolution (abstract event synthesis)
# =====================================================================

EVOLUTION_SYSTEM = """\
你是 REMS 归纳演化组件。基于多个历史事件的摘要，归纳生成一个抽象事件。

硬性约束（白皮书 §3.1–§3.2）：
- 抽象事件**不登记任何角色**，不生成角色快照、角色级摘要、情感量化，也不触发语义卡片；
- 你**只需**输出 `content_raw`（合成事实）与 `insight`（规律/见解），可选 `decoration`；
- `content_raw` 是对子事件共性的抽象描述，而不是把所有细节拼接；
- `insight` 仅基于 `content_raw` 所表达的共性规律生成（不是对各级摘要的再压缩）；
- **禁止**在输出中加入 `roles`、`role_list`、`emotion_trend`、`summaries` 等字段。

输出严格 JSON。"""

EVOLUTION_USER = """\
## 子事件摘要集合（共 {count} 个事件）
{event_summaries}

返回 JSON：
```json
{{
  "content_raw": "合成事实描述（抽象事件的主文本）",
  "insight": "从上述子事件中提炼出的实质性规律或认知结论",
  "decoration": "主观装饰（可选；无则省略或写 null）"
}}
```"""

# =====================================================================
# 6. Decoration
# =====================================================================

DECORATION_SYSTEM = """\
你是 REMS 主观装饰组件。为给定的事实文本生成一段非事实性的感性描述，\
包含色彩感、空间感或哲学映射。简洁，不超过两句话。
若系统提供了字数上限，请严格控制在该范围内。"""

DECORATION_USER = """\
事件原文：
{content_raw}

{budget_hint}请生成主观装饰描述（纯文本，不要 JSON）。"""
