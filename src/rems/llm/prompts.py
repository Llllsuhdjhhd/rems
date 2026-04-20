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
# 1. Boundary Detection
# =====================================================================

BOUNDARY_SYSTEM = """\
你是 REMS 事件边界检测组件。你的任务是分析输入文本，判断其中是否包含逻辑完整、已闭环的事实片段。

规则：
1. "完整事件"指一件事实从发生到结束的描述已闭环，可独立理解。
2. 如果某段文字描述的事实尚未结束（如"他开始做……"但没有结果），标记为"未完成"。
3. 篇幅极短（< 20 字）且逻辑过于简单的完整事实可以标记为 short_complete，允许暂留。
4. 如果输入本身就是对某个已有未完成事件的延续，在 continuation_of 中填入对应 ID。

【防碎片化聚合原则（重要）】
5. 对于描述同一连续时间段内、性质相近的琐碎日常动作（如跑步→回家→打扫→看小说），
   不得将其拆分为多个极短的独立事件。应在 content 中将其聚合为一条完整事件描述，
   如："用户去跑步后回家，依次打扫了卫生并阅读了一小时小说。"
6. 聚合判定标准：若多个动作在同一"生活段落"内发生，时间跨度合理（<数小时），
   且彼此之间无实质性的情感断裂或场景跳转，则归为同一基本事件。
7. 只有涉及不同场景、不同主体、或存在明显情感/逻辑转折的动作，才拆分为独立事件。

输出严格 JSON，不要附加解释。"""

BOUNDARY_USER = """\
## 当前残影（Shadow）
{shadow}

## 未完成事件列表
{unclosed_summary}

## 当前输入
{current_input}

请分析以上文本，返回如下 JSON（注意不要返回原文本的正文，以节约 token）：
```json
{{
  "completed_events": [
    {{
      "end_snippet": "该事件在此处结束的精确原文片段（请严格从原文复制最后的10-15个及以上字符），用于系统截断定位，严禁发挥。",
      "continuation_of": null 或 "未完成事件ID"
    }}
  ],
  "remaining_shadow": "剩余的尚未闭合、无头无尾的残影文本（必须是提取自原文的精确后缀）",
  "new_unclosed": [
    {{
      "content": "新识别到但未闭环的事实片段句子（精简）",
      "logical_gaps": "缺什么信息才能闭环"
    }}
  ]
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
你是 REMS 角色提取组件。从事件原文中识别所有参与实体（人物、动物、关键物体），\
并为每个角色生成快照和情感量化。

要求：
1. 【绝对指令】"name" 字段必须填写原文中角色的姓名、称谓或具体的身份标识（如：“甄士隐”、“贾雨村”、“英莲”）。
2. 【严禁行为】禁止在 "name" 中填写数字（如 "1", "2"）、"null"、"未知人物"、"核心用户" 等任何非真实名字的占位符。
3. 替换所有模糊代词（他/她/它/他们等）为上述明确角色名。
4. 对每个角色评定重要性：S（核心）/A（主要）/B（次要）/C（边缘）/D（背景）。
5. 生成快照：对 S/A/B 级的重要角色生成完整三级快照（L1提及、L2互动、L3决策）；对 C/D 级的不重要角色仅生成 L1（提及）即可，禁止对其虚构 L2/L3。
6. 量化当次事件中该角色的情感状态（Vedana/Klesha 数值在 0-1 之间）。

输出严格 JSON。"""

ROLE_EXTRACTION_USER = """\
## 已知角色列表
{known_roles}

## 事件原文
{content_raw}

返回 JSON：
```json
{{
  "roles": [
    {{
      "role_id": "已有ID或null（新角色）",
      "name": "角色名",
      "entity_type": "person|animal|object",
      "importance": "S|A|B|C|D",
      "snapshot": {{
        "l1_mention": "...",
        "l2_interaction": "...(仅重要角色S/A/B填写，C/D级直接省略该字段或留空)",
        "l3_decision": "...(同上，仅重要角色填写)"
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
