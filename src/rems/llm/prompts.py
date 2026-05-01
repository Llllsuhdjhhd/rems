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
4. **残影 / 当前输入边界**：用户消息会显式给出"既有残影序号区间"与"本轮新输入序号区间"。
   - 残影序号对应的句子来自此前已挂起的未完成事件；如果它们应当继续保留为未完成，请把它们出现在 `new_unclosed_indices` 中（系统会**用这份列表完整重建未完成库**——未列出的旧残影内容将按白皮书 §4.2.2 机制 3 视为无主碎屑被丢弃）。
   - 残影中已可与本轮输入闭环的部分，请放入对应 `completed_events.content_raw_indices`，并视情况填 `continuation_of`。
5. **未完成事件与残影统一**：与当前事件无关、或尚未闭环的逻辑片段，请统一填入 `new_unclosed_indices`。系统后续将这些片段的拼接定义为"残影"。

输出严格 JSON。"""

BOUNDARY_USER = """\
## 当前残影（Shadow）
{shadow}

## 未完成事件库快照
{unclosed_summary}

## 当前输入（已分句编码）{range_hint}
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
  "new_unclosed_indices": [12, 13]
}}
```
单条未完成也可写 `"new_unclosed_indices": [12]`（同一条内多句请写进**同一**扁平列表，勿每句一条记录）。与当前事件无关但需保留的句子，也请放入 `new_unclosed_indices`。**残影中应继续挂起的句子也必须重新出现在 `new_unclosed_indices`，否则系统会丢弃它们**。"""

# =====================================================================
# 2. Event Enrichment — 统一的「事件摘要 + 角色抽取」技能
# =====================================================================
# 以 boundary 剥离出的 content_raw 为输入，生成 L1…Ln 递归摘要（带熔断）。

ENRICHMENT_SUMMARY_ONLY_SYSTEM = """\
你是 REMS 事件充实（Event Enrichment）组件。给定**已闭环**基本事件原文，交付事件摘要：

**summaries：L1…Ln 递归压缩**
- 遵守【摘要字数预算表】；L1 保真主干，L2+ 逐层约减半。
- **熔断**：当某级摘要字符数 **≤ {fuse_min_chars}** 时，**不得再生成**下一级更压缩摘要（`summaries` 只含已产出的各级）。

只输出一个 JSON 对象，包含 `summaries` 字典，勿附加说明。"""

ENRICHMENT_SUMMARY_ONLY_USER = """\
## 事件原文
{content_raw}

## 摘要字预算
{summary_budget_table}

## 任务：摘要（L1 起，遵守熔断 {fuse_min_chars}）
生成 `summaries` 各级，直至熔断或达预算上限。

## 输出 JSON
```json
{{
  "summaries": {{
    "L1": "…",
    "L2": "…"
  }}
}}
```"""


# 真正"一次调用同时输出摘要 + 角色"的合并 prompt：
ENRICHMENT_FULL_SYSTEM = """\
你是 REMS 事件充实（Event Enrichment）组件。给定**已闭环**基本事件原文，**单次输出**两类衍生数据：

A. **summaries：L1…Ln 递归压缩**
   - 遵守【摘要字数预算表】；L1 保真主干，L2+ 逐层约减半；
   - **熔断**：当某级摘要字符数 **≤ {fuse_min_chars}** 时，**不得再生成**下一级（`summaries` 仅含已产出层级）。

B. **roles：参与角色识别 + 分级快照 + 8 维基础情绪**
   1. **角色重要性**：评定为 S（核心主角）/ A / B / C / D；
   2. **角色快照层级与预算（指数级递减）**：
      - **L3：详细意图快照**，遵守【L3 预算】，描述深层意图、微观动作与因果；
      - **L2：互动逻辑快照**，遵守【L2 预算】，侧重实时互动与行为反馈；
      - **L1：骨架白描快照**，遵守【L1 预算】，仅说明最核心行为事实；
   3. **层级分配策略**：
      - **S 级**：必须同时生成 L1 + L2 + L3；
      - **A 级**：必须生成 L1 + L2，L3 留空；
      - **B/C/D 级**：仅生成 L1，L2/L3 留空；
   4. **情感量化**：8 维基础情绪 anger / fear / joy / sadness / surprise / disgust / trust / anticipation，数值 0-1；后端自行合成 arousal / valence，**不要**输出这两个字段。
   5. **去代词化对齐**：若已知角色列表非空，应优先重用其 `role_id`；遇到代词指代请尝试映射到最可能的已知角色。

输出严格 JSON。"""

ENRICHMENT_FULL_USER = """\
## 事件原文
{content_raw}

## 已知角色列表（用于去代词化对齐）
{known_roles}

## 摘要字预算
{summary_budget_table}

## 角色快照预算表（硬约束）
{snapshot_budgets}

## 任务
1. 生成 `summaries`（L1 起，熔断 {fuse_min_chars}）；
2. 识别参与角色 + 快照 + 8 维情绪。

## 输出 JSON
```json
{{
  "summaries": {{
    "L1": "…",
    "L2": "…"
  }},
  "roles": [
    {{
      "role_id": "已有ID或null",
      "name": "角色名",
      "importance": "S|A|B|C|D",
      "snapshot": {{
        "l1_mention": "L1 文本 (S/A/B/C/D 必填)",
        "l2_interaction": "L2 文本 (仅 S/A 级填写)",
        "l3_decision": "L3 文本 (仅 S 级填写)"
      }},
      "emotion": {{
        "anger": 0.0,
        "fear": 0.0,
        "joy": 0.0,
        "sadness": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "trust": 0.0,
        "anticipation": 0.0
      }}
    }}
  ]
}}
```"""


def build_enrichment_system_message(
    config: "REMSConfig",
    *,
    fuse_min_chars: int,
    full_mode: bool = True,
) -> str:
    """System prompt for event enrichment: mode injection + 主规则。

    ``full_mode=True`` 走"摘要 + 角色"单次合并调用；``False`` 退回纯摘要变体，
    用于外部已经传入 ``role_entries`` 时（pipeline pre-recall 已完成角色提取）跳过角色字段。
    """
    base = build_user_mode_block(config)
    body = (
        ENRICHMENT_FULL_SYSTEM if full_mode else ENRICHMENT_SUMMARY_ONLY_SYSTEM
    ).format(fuse_min_chars=fuse_min_chars)
    return base + body


# 旧符号保留（兼容外部 import）：
ENRICHMENT_SYSTEM = ENRICHMENT_SUMMARY_ONLY_SYSTEM
ENRICHMENT_USER = ENRICHMENT_SUMMARY_ONLY_USER

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
你是 REMS 角色提取组件。从事件原文中识别所有参与实体，并生成分级快照（Snapshot）和 8 维基础情绪。

核心规则：
1. **角色重要性**：评定为 S（核心主角）/ A / B / C / D。
2. **【角色快照层级与预算（指数级递减）】**：
   - **L3：详细意图快照**。必须符合 **【L3 预算】**。描述角色的深层意图、微观动作与因果关联。
   - **L2：互动逻辑快照**。必须符合 **【L2 预算】**。侧重于角色间的实时互动与行为反馈逻辑。
   - **L1：骨架白描快照**。必须符合 **【L1 预算】**。极致简练，仅说明最核心的行为事实。
3. **【层级分配策略】**：
   - **主角（S级）**：**必须**同时生成 L1、L2 和 L3，各层级间需有明显的语义密度差异。
   - **主要互动对象（A级）**：**必须**生成 L1 和 L2，将 L3 字段留空。
   - **B/C/D 级角色**：**仅生成 L1**。将 L2、L3 字段留空。
4. **情感量化**：输出 8 维基础情绪，键为 anger/fear/joy/sadness/surprise/disgust/trust/anticipation，数值在 0-1 之间。
   后端会自行合成 arousal 与 valence，不要输出这两个字段。

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
        "anger": 0.0,
        "fear": 0.0,
        "joy": 0.0,
        "sadness": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
        "trust": 0.0,
        "anticipation": 0.0
      }}
    }}
  ]
}}
```"""

# =====================================================================
# 5. Inductive Evolution (abstract event synthesis)
# =====================================================================

EVOLUTION_SYSTEM = """\
你是 REMS 抽象事件压缩组件。输入是若干叶子基本事件的 `content_raw` 及其角色线索。

硬性约束：
- 抽象事件**不登记任何角色**，不生成角色快照、角色级摘要、情感量化，也不触发语义卡片；
- 但压缩 `content_raw` 时**不能漏掉关键角色/实体**：角色线索只用于帮助保留主体、行为与关系，不作为 `role_list` 输出；
- `content_raw` 不是高阶泛化口号，也不是罗列所有原文；它应像基本事件一样抓住重点做事实压缩，保留共现事件中的主干人物、动作、对象与结果；
- `content_raw` 的目标长度约等于输入叶子基本事件 `content_raw` 平均长度的 1.2 倍；
- L1-L10 摘要不在此处生成，外层会按基本事件同规则继续递归摘要；
- 仅当用户消息明确要求 `insight` 字段时才输出 insight；insight 是跨事件提炼出的认知/规律，不是对事实文本的再摘要；
- **禁止**在输出中加入 `roles`、`role_list`、`emotion_trend`、`summaries` 等字段。

输出严格 JSON。"""

EVOLUTION_USER = """\
## 基本事件 content_raw 证据集合（共 {count} 个事件）
{event_contents}

## 输出控制
- `content_raw`：抓住重点压缩，目标约 {target_content_len} 字（叶子基本事件均值的 1.2 倍）；保留关键角色/实体，不做角色对象输出。
{insight_instruction}

返回 JSON：
```json
{json_schema}
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
