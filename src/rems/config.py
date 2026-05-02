from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# REMS 全局配置（对照《REMS 记忆系统规范解析》白皮书）。
# - len_msg / physical_redline / safe_watermark：1.1.7 事件长度与物理防御、4.2 触发与截断；
# - abstract_subset_min_size / abstract_subset_min_support：3.2 频繁极大子集挖掘触发抽象；
# - ae_*、wp_*：1.1.4、2.2 情感能量（AE）与白描动态遗忘；
# - hallucination_anchor_prob：兼容旧配置；当前抽象合成始终使用叶子基本事件 content_raw；
# - tombstone_prefix：4.3 墓碑化时在 insight 中的审计标记前缀；
# - user_mode / core_user_role_id / active_participants：2.2 单人/多人模式与提示词注入引擎。
# 加载：环境变量前缀 REMS_，嵌套键用 __ 分隔；可选 .env。


class UserMode(str, Enum):
    """Global pronoun-resolution mode (白皮书 2.2).

    SINGLE：单人模式——第一人称与无主语动作默认归属 ``core_user_role_id``，降低代词消解幻觉。
    MULTI：多人模式——提供 ``active_participants`` 花名册，模型需做多方共指消解，禁止默认归一。
    """

    SINGLE = "single"
    MULTI = "multi"


class TaskModelMapping(BaseModel):
    """Maps each REMS skill/task type to a specific LLM model name.

    将 REMS 中各技能任务类型（summary、boundary_detection、role_extraction、abstraction、insight 等）
    映射到具体模型名称字符串，便于在成本、延迟与能力之间做分流；未识别类型回退 ``default``。
    """

    # 以下为各技能默认模型名；可按任务强度分流成本（摘要/边界/抽取/抽象等）。

    summary: str = "deepseek-v4-flash"
    boundary_detection: str = "deepseek-v4-flash"
    # 超长未完成事件的分裂修复；通常可直接复用 boundary_detection 的模型，也可指定更强模型。
    overlong_uc_split: str = "deepseek-v4-flash"
    # 可选：对 boundary_detection 输出的二级 LLM 评估器；默认同主模型。
    boundary_evaluation: str = "deepseek-v4-flash"
    role_extraction: str = "deepseek-v4-flash"
    # 事件充实：一次调用产出摘要 + 角色；与 summary 同主模型时便于在 DashScope 侧统一配额。
    event_enrichment: str = "deepseek-v4-flash"
    abstraction: str = "deepseek-v4-flash"
    insight: str = "deepseek-v4-flash"
    default: str = "deepseek-v4-flash"


class LLMConfig(BaseModel):
    # 兼容 OpenAI 协议的对话 API（DeepSeek / DashScope 等）。密钥与基址用环境变量覆盖（见 .env.example）。
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    task_models: TaskModelMapping = Field(default_factory=TaskModelMapping)
    temperature: float = 0.3
    max_retries: int = 3


class EmbeddingConfig(BaseModel):
    # pydantic v2 的 ``model_`` 命名空间会对 ``model_name`` 产生 UserWarning，这里显式放通。
    model_config = {"protected_namespaces": ()}

    # 向量嵌入：默认本地 SentenceTransformer；可扩展为远程 API。
    provider: str = "local"
    model_name: str = "BAAI/bge-small-zh-v1.5"
    api_base_url: str | None = None
    api_key: str | None = None


class StorageConfig(BaseModel):
    # 关系型数据（事件/角色/代谢状态）与 Chroma 向量持久化路径。
    database_url: str = "sqlite:///rems.db"
    chromadb_path: str = "./chroma_data"


class REMSConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REMS_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # 由 token 窗口折算字符预算（白皮书 1.1.7：单条与缓冲区与上下文比例关系）。
    # 默认 88000：以 chars_per_token=1.5 折算，使 len_msg = 88000 * 1.5 / 66 ≈ 2000。
    context_window: int = 88000
    chars_per_token: float = 1.5

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # ---- Recall: role-aware summary tier selection (白皮书 4.4 Lazy Index) ----
    # 默认召回摘要起点：0 = 中间级（mid），-N = 向 L1 偏移 N 档（更精细），+N = 向最高级偏移 N 档（更压缩）。
    recall_default_tier_offset: int = 0
    # 关注角色（S/A 重要性）：在默认档基础上向 L1 方向再偏移的档数（更精细）。
    recall_primary_role_detail_shift: int = 1
    # 次要/无关角色（C/D 重要性或未出现）：在默认档基础上向高压缩方向偏移的档数。
    recall_minor_role_compress_shift: int = 1
    # 摘要最低字数门槛：某档摘要字数 ≤ 此值时视为已最大压缩，不再向上推进档位。
    recall_summary_min_chars: int = 20
    
    # ---- Recall Capacity & Tiered Retrieval (70/30 Rule) ----
    # 系统级全量检索容量上限（基本事件数）。低于此值全量可见；超过后开启 70/30 分层。
    recall_max_capacity: int = 1000
    # 分层比例：最近 N% 的条目对全局可见，剩余部分仅焦点角色关联可见。
    recall_global_ratio: float = 0.7

    # ---- Abstraction: Frequent Maximal Subset Mining (白皮书 §3.2 唯一触发) ----
    # 每次回忆产生的回忆块 event_id 集合被登记到 ``recall_log``；在集合族中找满足：
    #   - 子集大小 >= abstract_subset_min_size（默认 6；可配置）
    #   - 支持度（跨多少条回忆块被整体覆盖） >= abstract_subset_min_support（默认 5）
    # 的 **极大子集**，对其合成抽象事件。合成后把 ``recall_log`` 中该子集替换为抽象事件 id，
    # 以便后续更高阶抽象继续在同一命名空间演进（"用抽象事件 id 代替原来的子集，逻辑保持统一"）。
    abstract_subset_min_size: int = 6
    abstract_subset_min_support: int = 5

    # 未闭合事件总长超过 len_msg * 该比例则触发 80/20 强制分裂（详见 boundary_split_* 设置）。
    # 白皮书 4.2 的物理红线（physical_redline）仍在 ``_check_physical_redline`` 兜底，
    # 但"认知层"的默认路径不再做盲目强制封存——只做评估 → 修复 → 保留 oversized UC。
    # 是否「可疑」由角色抽取与 RoleService 仲裁判断。
    unclosed_force_ratio: float = 1.2
    # ---- 80/20 Forced Split (2026-05, 白皮书 4.2 升级) ----
    # 评估到 oversized_uc 后，OverlongUCSplitSkill 的目标切分比例与可接受区间。
    # 切点仍由模型基于"逻辑闭环"选择，此处只给数量级指引。
    boundary_split_ratio_target: float = 0.8
    boundary_split_ratio_min: float = 0.7
    boundary_split_ratio_max: float = 0.9
    # 是否启用 OverlongUCSplitRemediator（修复器）。关闭时 oversized UC 将保留为 oversized=True
    # 但不再 force-seal，直接让物理红线层在极端情况下兜底。
    boundary_remediation_enabled: bool = True
    # 硬编码 1：评估只做一层，修复后的输出不再评估。
    boundary_max_remediation_passes: int = 1
    # 是否启用 LLM 评估器（对切分质量做二级评估，成本敏感）。默认关闭。
    boundary_enable_llm_eval: bool = False
    # 回忆时前缀链展开的最大深度——防止"前缀的前缀的前缀……"把回忆块撑爆。
    recall_split_prefix_max_depth: int = 3
    # 递归摘要熔断：某级摘要字符数 **低于** 该阈值则不再生成更高级（白皮书 1.1.3）；按产品约定为 20 字。
    summary_fuse_min_chars: int = 20

    # 情绪唤醒度映射遗忘因子的幂指数：base_forgetting_factor = 100 * arousal ** gamma。
    emotion_arousal_gamma: float = 2.0

    # 白描时间线遗忘半衰期（天）。
    wp_half_life_days: float = 60.0
    
    # 事件静默阈值：当所有角色的有效遗忘因子均低于此值时，事件被设为 SILENT。
    event_silence_threshold: float = 0.02
    forgetting_silence_threshold: float = 0.02

    # ---- Dynamic Recall Compression (白皮书 4.4 扩展与分级压缩) ----
    # 初始向量检索的目标长度倍率（相对于 physical_redline）。
    recall_expansion_factor: float = 2.0
    # 情感精排后的中间过滤目标长度倍率（相对于 physical_redline）。
    recall_intermediate_filter_factor: float = 1.2
    # 初始保持高保真摘要的头部条目比例（由条目数决定）。
    recall_head_ratio: float = 0.66

    # 兼容旧配置：当前实现已改为始终展开到叶子基本事件并使用 content_raw 作为抽象证据。
    hallucination_anchor_prob: float = 0.3

    # 抽象事件是否额外生成 ``insight``。关闭时抽象合成只产出压缩后的 ``content_raw`` 与可选 decoration；
    # 打开时才要求模型提炼跨事件规律，且该 insight 不是事实摘要本身。
    enable_abstract_insight: bool = False

    # ---- Role capacity & soft-forgetting (白皮书 2.3 容量分配与软遗忘) ----
    # 角色白描容量上限倍率：实际容量 = context_chars / wp_role_capacity_divisor（默认 6.6，即与 len_msg 同阶）。
    # 容量内白描不施加遗忘惩罚；超出后旧条目在 Recall 打分中被遗忘因子惩罚。数据永不物理删除。
    wp_role_capacity_divisor: float = 6.6

    # ---- Dynamic compression ratio control (白皮书 1.2 前置预算计算) ----
    # 事件封存时 sum_len / raw_len 的目标比值，默认 1/6.6。
    compression_target_ratio: float = 0.1515
    # 递归摘要指数衰减因子 (L1 -> L10)；每一级相对于上一级的字数限制比例。
    summary_decay_factor: float = 0.5
    # 角色快照指数衰减因子 (L3 -> L2 -> L1)；L2 = L3 * factor, L1 = L2 * factor。
    snapshot_decay_factor: float = 0.6
    # 缩放因子：在目标压缩率基础上的全局缩放（> 1.0 放松字数，< 1.0 收紧字数）。
    # 既有递归摘要熔断等逻辑保持不变，预算约束通过该因子叠加于其上。
    compression_budget_multiplier: float = 1.0
    # 分项预算占总预算的比例（合计应为 1.0）。
    budget_ratio_summary: float = 0.40     # 默认级摘要
    budget_ratio_snapshot: float = 0.25    # 角色快照（各角色均分）
    budget_ratio_wp: float = 0.25          # 角色白描条目（各角色均分）
    budget_ratio_decoration: float = 0.10  # 装饰
    
    # 是否启用情节装饰（Decoration）：开启后会额外调用一次 LLM 为事件生成感性描述。
    enable_decoration: bool = False

    # 墓碑化时写入 insight 的审计前缀（白皮书 4.3）。
    tombstone_prefix: str = "[TOMBSTONE]"

    # ---- Global pronoun-resolution mode (白皮书 2.2) ----
    # 默认单人模式；生产环境按场景切换为多人。
    user_mode: UserMode = UserMode.SINGLE
    # 单人模式下第一人称与无主语动作默认归属的核心用户 role_id；留空则仍由模型自推。
    core_user_role_id: str | None = None
    # 多人模式下的已知活跃参与者花名册（role_id 或角色名），会被注入 system prompt 做共指消解。
    active_participants: list[str] = Field(default_factory=list)

    # ---- Emotion: EMA (Emotion & Adaptation) dynamic evolution (白皮书 2.5) ----
    # 情绪滚动状态的时间衰减核（小时）：phi(delta) = exp(-lambda * delta_hours)。
    emotion_decay_lambda_per_hour: float = 0.1
    # 情绪滚动状态回溯读取白描尾部的条数。
    ema_history_window: int = 10
    # 事件 activation_energy 由角色滚动 energy 聚合后可选放大。
    activation_energy_gain: float = 1.0

    # ---- RRF recall modifiers (白皮书 4.4) ----
    recall_rrf_k: int = 60
    recall_factor_alpha: float = 0.5
    recall_mood_beta: float = 0.2
    recall_reinforce_multiplier: float = 1.5
    recall_forgetting_factor_cap: float = 300.0

    # ---- White-painting collection tier (白皮书 2.3 动态粒度路由·收集端) ----
    # 白描收集时对主要角色（S/A 或单人核心用户）落盘的默认档位字段：l3_decision / l2_interaction / l1_mention。
    wp_primary_field: str = "l3_decision"
    wp_default_field: str = "l2_interaction"
    wp_minor_field: str = "l2_interaction"

    @property
    def context_chars(self) -> int:
        """Effective character budget derived from token window.

        用 ``context_window * chars_per_token`` 估算整段上下文可承载的字符上限，作为后续 1/66、10/66 比例约束的基数。
        """
        return int(self.context_window * self.chars_per_token)

    @property
    def len_msg(self) -> int:
        """1/66 of context — max single input / event length (chars).

        单条消息或单事件 L0 的推荐/硬上限比例：约为 ``context_chars`` 的 1/66（白皮书 1.1.7 物理防御与4.2 触发语义）。
        """
        return int(self.context_chars / 66)

    @property
    def wp_role_capacity(self) -> int:
        """Per-role white-painting capacity in characters (白皮书 2.3).

        角色白描时间线在「全保真无遗忘」模式下可承载的最大字符总量。
        默认等于 ``context_chars / 6.6``（与 ``len_msg`` 同阶）。
        容量内不施加遗忘惩罚；超出后旧条目仅在 Recall 打分中被遗忘因子惩罚，数据永不删除。
        """
        return int(self.context_chars / self.wp_role_capacity_divisor)

    @property
    def physical_redline(self) -> int:
        """10/66 of context — absolute physical safety ceiling (chars).

        回忆块等组装内容的物理安全上限：约为 ``context_chars`` 的 10/66，对应白皮书 4.4 所述约 1/6.6「主权约束」，
        用于防止单次上下文爆炸。
        """
        return int(self.context_chars * 10 / 66)

    @property
    def safe_watermark(self) -> int:
        """2/66 of context — target watermark after forced compaction (chars).

        触发物理红线强制整理后，残影内容希望保留的尾部目标长度（约 2/66上下文），用于代谢模块截断策略。
        """
        return int(self.context_chars * 2 / 66)
