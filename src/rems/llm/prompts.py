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
4. **未完成事件**：若段落未形成闭环，请将其序号填入 `new_unclosed_indices`。
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
  "new_unclosed_indices": [12]
}}
```"""

# =====================================================================
# 2. Event Enrichment — 统一的「事件摘要 + 角色抽取」技能
# =====================================================================
# 以 boundary 剥离出的 content_raw 为输入，一次 LLM 调用同时产出：
# 1) L1…Ln 递归摘要（带熔断）
# 2) 角色列表（快照层级 + Vedana/Klesha）
# 角色级语义卡片（insight）的刷新由下游按「主要角色」过滤后另起一次调用，见 _CARD_*。

ENRICHMENT_SYSTEM = """\
你是 REMS 事件充实（Event Enrichment）组件。给定一个**已闭环**的基本事件原文，你需要**在一次输出中同时完成**两件事：

1. **生成多级摘要（L1 … Ln）**
   - 严格遵守下方【摘要字数预算表】（硬约束）。
   - L1：最高保真压缩，只剥离修饰语，保留核心事实与因果主干。
   - L2~Ln：逐级指数级压缩，每一级字数显著少于上一级。
   - **动态熔断**：一旦某级摘要低于 {fuse_min_chars} 字，立即停止后续更高级别的生成（summaries 字典只保留已生成的键值对）。

2. **识别角色并生成分级快照与情感量化**
   - **角色重要性**：S（核心主角）/ A / B / C / D。
   - **角色快照层级与预算（指数级递减）**：严格遵守下方【角色快照预算表】。
     * L3：详细意图快照；微观动作、因果、内心意图。
     * L2：互动逻辑快照；角色间实时互动与反馈。
     * L1：骨架白描快照；极简事实点。
   - **层级分配策略**：
     * S 级：必须同时给出 L1、L2、L3；三级之间要有明显语义密度差异。
     * A/B/C/D：**仅给出 L1**，L2 / L3 字段留空（null 或缺省）。
   - **情感量化**：Vedana / Klesha 各子字段数值 ∈ [0, 1]；未提及则留空。
   - **去代词化**：若「已知角色列表」中已有对应实体，填写对应 role_id 以复用；否则 role_id 置 null，交由下游注册。

输出严格 JSON，不附加任何解释文本。"""

ENRICHMENT_USER = """\
## 已知角色列表（可用于去代词化复用）
{known_roles}

## 事件原文（已闭环）
{content_raw}

## 【摘要字数预算表】（硬约束）
{summary_budget_table}

## 【角色快照字数预算表】（硬约束）
{snapshot_budget_table}

## 摘要字数控制要点
1. 递减生成：L1 → L2 → … 每一级字数较前一级约减少 50%。
2. 动态熔断：一旦某级摘要缩减到 **{fuse_min_chars} 字以内**，立即停止后续层级。

请在一次响应中同时返回摘要与角色列表，格式如下：
```json
{{
  "summaries": {{
    "L1": "摘要文本...",
    "L2": "摘要文本...",
    "L3": "摘要文本..."
  }},
  "roles": [
    {{
      "role_id": "已有ID 或 null",
      "name": "角色名",
      "entity_type": "person",
      "importance": "S|A|B|C|D",
      "snapshot": {{
        "l1_mention": "L1 文本（S/A/B/C/D 必填）",
        "l2_interaction": "L2 文本（仅 S 级填写）",
        "l3_decision": "L3 文本（仅 S 级填写）"
      }},
      "emotion": {{
        "vedana": {{}},
        "klesha": {{}}
      }}
    }}
  ]
}}
```"""

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

要求：
1. content_raw：由子事件的中位摘要合集集成生成的合成事实描述。
2. insight：从这些事件中提炼出的实质性规律、见解或逻辑推导结论。
3. decoration：非事实性的色彩、哲学映射描述（可选）。
4. 为涉及的角色生成 L3（决策快照），描述该角色在此规律下的典型风格。

输出严格 JSON。"""

EVOLUTION_USER = """\
## 子事件摘要集合（共 {count} 个事件）
{event_summaries}

返回 JSON：
```json
{{
  "content_raw": "合成事实描述",
  "insight": "规律与见解",
  "decoration": "主观装饰（可选）",
  "roles": [
    {{
      "role_id": "...",
      "importance": "S|A|B|C|D",
      "l3_decision": "典型决策风格描述",
      "emotion_trend": {{
        "vedana": {{}},
        "klesha": {{}}
      }}
    }}
  ]
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
