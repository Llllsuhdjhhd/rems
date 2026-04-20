"""Centralised prompt templates for every REMS skill.

Convention
----------
* ``*_SYSTEM`` — the ``system`` message content.
* ``*_USER``   — a Python format-string; call ``.format(...)`` with kwargs.

中文说明：正文 prompt 已与《REMS 记忆系统规范解析》对齐（边界/摘要/角色/演化等条款），
修改 prompt 时建议同步核对白皮书对应小节，避免与领域语义漂移。

全局模式注入（白皮书 2.2）：``build_user_mode_block(config)`` 会按 ``UserMode`` 渲染一段
系统提示段，由 ``RoleExtractionSkill`` 与 ``BoundaryDetectionSkill`` 拼接进 system prompt。
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
# 1. Boundary Detection & Event Disentanglement (合并了摘要生成)
# =====================================================================

BOUNDARY_SYSTEM = """\
你是 REMS 事件剥离与边界检测组件。你的任务是从可能存在交叉、多线程或冗长的输入文本中，识别并剥离出各自逻辑完整、已闭环的独立事件。

核心任务：
1. **事件剥离**：如果一段原文同时讲述了多条并行的故事线，你必须将它们剥离为独立的不同事件。
2. **提取纯段落**：为剥离出的每一个独立事件提取出专属的原始文本片段供系统作为 `content_raw` 使用。
3. **合并生成多级摘要**：直接为每个剥离出的事件**同步生成 L1 至 L10 各层级的摘要**：
   - **层级规则**：从 L1 到 L10 遵循**从长到短**的递减规律。
   - L1：最高保真压缩，剔除修饰语，保留核心事实。
   - L2~L9：渐进式压缩，每级字数应明显少于上一级。
   - **动态熔断机制**（核心）：摘要生成遵循 **“20 字熔断”** 原则。即从 L1 开始进行逐级压缩，一旦某级摘要的字数已经缩减到 **20 字左右**，即视为触及“信息熵极限”，必须**立即停止**后续所有更高级别的生成。
   - **结果表现**：summaries 字典中的键值对数量将取决于语义饱读速度，可能只有 L1-L5，不再强凑 L6-L10。
4. "未完成事件"：如果段落未结束，请将其填入未完成项中。

输出严格 JSON。"""

BOUNDARY_USER = """\
## 当前残影（Shadow）
{shadow}

## 未完成事件列表
{unclosed_summary}

## 当前输入
{current_input}

{budget_hint}请剥离并重组事件，返回如下 JSON：
```json
{{
  "completed_events": [
    {{
      "content_raw": "...",
      "summaries": {{
        "L1": "最详细摘要",
        "L10": "极致缩减摘要"
      }},
      "continuation_of": null
    }}
  ],
  "remaining_shadow": "...",
  "new_unclosed": []
}}
```"""

# =====================================================================
# 2. Summary Generation (recursive L1-Ln)
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
# 3. Role Extraction
# =====================================================================

ROLE_EXTRACTION_SYSTEM = """\
你是 REMS 角色提取组件。从事件原文中识别所有参与实体，并生成快照和情感量化。

要求：
1. 【绝对指令】"name" 必须使用原文名。禁止使用无效占位符。
2. 对每个角色评定重要性：S（主角/绝对核心）/A/B/C/D（配角）。
3. **【角色快照层级规则（从小到大）】**
   - **L1：骨架白描**。必须控制在 **10 字以内**。仅描述最核心动作（如“甄士隐递英莲”）。
   - **L2：互动白描**。描述具体的行为逻辑与上下文意图。
   - **L3：详细微观白描**。保留微观动作、环境交互与因果细节。
4. **【分配策略】**
   - **配角（A/B/C/D）**：**仅允许生成 L1**。必须将 L2、L3 字段留空或返回 null。
   - **主角（S级）**：必须同时生成 L1、L2 和 L3，展现完整的粒度梯度。
5. 情感量化：Vedana/Klesha 数值在 0-1 之间。

输出严格 JSON。"""

ROLE_EXTRACTION_USER = """\
## 已知角色列表
{known_roles}

## 事件原文
{content_raw}

{budget_hint}返回 JSON：
```json
{{
  "roles": [
    {{
      "role_id": "已有ID或null（新角色）",
      "name": "角色名",
      "entity_type": "person|animal|object",
      "importance": "S|A|B|C|D",
      "snapshot": {{
        "l1_mention": "配角A/B/C/D仅填此项。S主角简写该项",
        "l2_interaction": "(只有S级填写，其他填null或直接省略该字段)",
        "l3_decision": "(只有S级填写，其他填null或直接省略该字段)"
      }},
      "emotion": {{
        "vedana": {{"joy":0,"suffering":0,"happiness":0,"worry":0,"equanimity":0}},
        "klesha": {{"greed":0,"anger":0,"ignorance":0,"pride":0,"doubt":0,"wrong_view":0}}
      }}
    }}
  ]
}}
```"""

# =====================================================================
# 4. Inductive Evolution (abstract event synthesis)
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
# 5. Decoration
# =====================================================================

DECORATION_SYSTEM = """\
你是 REMS 主观装饰组件。为给定的事实文本生成一段非事实性的感性描述，\
包含色彩感、空间感或哲学映射。简洁，不超过两句话。
若系统提供了字数上限，请严格控制在该范围内。"""

DECORATION_USER = """\
事件原文：
{content_raw}

{budget_hint}请生成主观装饰描述（纯文本，不要 JSON）。"""
