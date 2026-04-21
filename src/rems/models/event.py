from __future__ import annotations

import secrets
import time
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# 事件与情感量化领域模型；对齐《REMS 记忆系统规范解析》第 1 章（L0/L1~Ln、角色快照、
# Vedana/Klesha、抽象字段、墓碑）及 AE 工程化抗遗忘逻辑。


import uuid

def generate_event_id() -> str:
    """Stateless unique event identifier.

    废除字典序，改用无状态分布式的唯一标识（目前使用 uuid4 模拟），
    前缀 ``EVT-``，满足海量并发和高可用等无状态工程需求（白皮书 1.1.1）。
    """
    return f"EVT-{uuid.uuid4().hex}"


class Importance(str, Enum):
    # S = 至关重要（主角、核心行为推动者）
    # A = 非常重要（直接推动事件发展、关键互动）
    # B = 比较重要（有明确互动，但非核心）
    # C = 次要参与（轻度出现或被动关联）
    # D = 可忽略角色（路人、背景板等）
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class EventStatus(str, Enum):
    # 事件在回忆与生命周期中的状态（白皮书 1.1.5）。
    UNCLOSED = "unclosed"
    ACTIVE = "active"
    SILENT = "silent"


# ---------- Emotional model ----------
# 白皮书 1.1.4：五受、六烦恼的工程量化字段名采用英文键，与 LLM JSON 输出对齐。

class Vedana(BaseModel):
    """五受量化 — joy/suffering/happiness/worry/equanimity.

    佛学五受（乐、苦、喜、忧、舍）在工程上映射为上述英文字段，便于 LLM JSON 键稳定输出（白皮书 1.1.4）。
    """
    joy: float = 1.0
    suffering: float = 1.0
    happiness: float = 1.0
    worry: float = 1.0
    equanimity: float = 1.0


class Klesha(BaseModel):
    """六根本烦恼量化 — greed/anger/ignorance/pride/doubt/wrong_view.

    贪、嗔、痴、慢、疑、恶见映射为上述英文键；与 Vedana 一起构成事件内情感量化，驱动 AE 与 NPC 增量等（1.1.4）。
    """
    greed: float = 1.0
    anger: float = 1.0
    ignorance: float = 1.0
    pride: float = 1.0
    doubt: float = 1.0
    wrong_view: float = 1.0


class EmotionalModel(BaseModel):
    vedana: Vedana = Field(default_factory=Vedana)
    klesha: Klesha = Field(default_factory=Klesha)


# ---------- Role within an event ----------

class RoleSnapshot(BaseModel):
    # 白皮书 1.1.4：L1 提及、L2 互动白描、L3 决策/意图快照。
    l1_mention: Optional[str] = None
    l2_interaction: Optional[str] = None
    l3_decision: Optional[str] = None


# ---------- Compression budget (白皮书 1.2) ----------

class CompressionBudget(BaseModel):
    """Pre-computed character budgets for each derived-data component.

    在事件封存流程最前端，根据已知的 ``raw_len`` 确定性地计算出各衍生数据项的字符预算，
    并注入后续 LLM 提示词中作为硬约束（白皮书 1.2.2）。
    支持多层级指数级递减预算。
    """
    raw_len: int                   # L0 原文字符长度
    total_budget: int              # raw_len * target_ratio * multiplier
    
    # [L1..L10] 摘要字数上限字典 (key: "L1"..."L10")
    summary_level_budgets: dict[str, int] = Field(default_factory=dict)
    
    # [L1..L3] 角色快照字数上限字典 (key: "L1"..."L3")
    snapshot_level_budgets: dict[str, int] = Field(default_factory=dict)

    wp_budget_per_role: int        # 每个角色白描条目的字数上限
    decoration_budget: int         # 装饰的字数上限
    role_count_estimate: int = 1   # 预估角色数（用于分摍计算）

    @property
    def summary_budget(self) -> int:
        """Backward compatibility: returns L1 budget."""
        return self.summary_level_budgets.get("L1", 0)

    @property
    def snapshot_budget_per_role(self) -> int:
        """Backward compatibility: returns L3 (base) budget."""
        return self.snapshot_level_budgets.get("L3", 0)


class EventRoleEntry(BaseModel):
    # 单个角色在一次事件中的瞬时记录（之后会被写入角色白描时间线）。
    role_id: str
    importance: Importance = Importance.C
    role_snapshot: RoleSnapshot = Field(default_factory=RoleSnapshot)
    emotional_model: EmotionalModel = Field(default_factory=EmotionalModel)


# ---------- Event ----------
# 基本事件与抽象事件共用同一结构；抽象事件 is_abstract=True 且带 abstraction_level/source_events。

class Event(BaseModel):
    event_id: str = Field(default_factory=generate_event_id)
    create_time: datetime = Field(default_factory=datetime.now)
    input_id: Optional[str] = None  # 记录产生此事件的原始输入/对话轮次唯一标识
    content_raw: str  # L0：基本事件为已闭环事实原文；抽象事件为自子事件归纳的合成描述（白皮书 1.1.2、3.1）。

    summaries: dict[str, str] = Field(default_factory=dict)  # L1~Ln 递归摘要（白皮书 1.1.3）。
    summary_lengths: dict[str, int] = Field(default_factory=dict)
    actual_max_level: int = 0  # 熔断后的实际最高摘要层级。

    role_list: list[EventRoleEntry] = Field(default_factory=list)  # 当次事件涉及角色与情感快照（1.1.4）。

    is_abstract: bool = False  # 架构级区分基本/抽象事件（1.1.5、3.1）。
    is_abstracted: bool = False  # 是否已被更高阶认知吸收（1.1.5）。
    status: EventStatus = EventStatus.ACTIVE # unclosed/active/silent（1.1.5）。

    decoration: Optional[str] = None  # 非事实装饰，支撑「梦见」等（1.1.6）。
    insight: Optional[str] = None  # 基本事件宜为空；抽象事件存规律；墓碑审计可追加（1.1.6、4.3）。
    event_length: int = 0  # L0 字符长度，用于约束与检索预估（1.1.7）。

    # 仅抽象事件：演化树深度与证据链 source_events（白皮书 1.1.8、3.1）。
    abstraction_level: Optional[int] = None
    source_events: Optional[list[str]] = None

    # 墓碑化：逻辑上被修正覆盖，保留审计但回忆排除（白皮书 4.3）。
    is_tombstoned: bool = False

    # 激活能量（Activation Energy，白皮书 2.5 与记忆初始值硬绑定）：
    # 由 EMA + 事件级 AE 计算；重大情感事件（极乐/大苦）在封存时写入较高初值，
    # 作为进入回忆混合打分的独立权重（与 AE 组合但不等同：AE 是事件瞬时最大值，
    # activation_energy 是与角色长期心境做动态调节后的"落地权重"）。
    activation_energy: float = 0.0

    # 动态压缩率（白皮书 1.2）：封存后实际的 sum_len / raw_len，用于审计与追踪。
    compression_ratio: float = 0.0

    def model_post_init(self, __context: object) -> None:
        if not self.event_length:
            self.event_length = len(self.content_raw)

    # ------------------------------------------------------------------
    # Affective Energy helpers
    # ------------------------------------------------------------------

    @property
    def affective_energy(self) -> float:
        """Event-level AE based on deviation from baseline (1.0).

        [接口占位] 当前仅提供临时算法：计算全要素偏离基础值 1.0 的最大波动（绝对差值）平均作为初步实现。
        ※ TODO: 这里的算法留出接口，后续需由业务或数据科学重新评估及重新实写。
        """
        if not self.role_list:
            return 0.0
            
        max_ae = 0.0
        for entry in self.role_list:
            v = entry.emotional_model.vedana
            k = entry.emotional_model.klesha
            
            # 使用与 1.0 基础值的偏离程度（绝对值）作为波动能量的粗略估计
            vedana_peak_diff = max(abs(v.joy-1.0), abs(v.suffering-1.0), abs(v.happiness-1.0), abs(v.worry-1.0), abs(v.equanimity-1.0))
            klesha_peak_diff = max(abs(k.greed-1.0), abs(k.anger-1.0), abs(k.ignorance-1.0), abs(k.pride-1.0), abs(k.doubt-1.0), abs(k.wrong_view-1.0))
            
            ae = (vedana_peak_diff + klesha_peak_diff) / 2.0
            max_ae = max(max_ae, ae)
            
        # 限制在某一合理上限或继续返回 0~1 的百分比
        return min(max_ae, 1.0)

    @property
    def mid_summary_key(self) -> str:
        """Return the key for the middle-level summary (used by recall / abstraction).

        将 ``summaries`` 的键按 L1、L2… 数值排序后取中位下标对应的级别，作为「中间粒度」摘要键；
        若无任何摘要则与实现一致地退化为 ``L1``。供回忆组装时在长度预算下做懒索引降级，
        以及归纳技能非锚定路径选用。
        """
        if not self.summaries:
            return "L1"
        levels = sorted(self.summaries.keys(), key=lambda k: int(k[1:]))
        return levels[len(levels) // 2]
