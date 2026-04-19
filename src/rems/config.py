from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# REMS 全局配置（对照《REMS 记忆系统规范解析》白皮书）。
# - len_msg / physical_redline / safe_watermark：1.1.7 事件长度与物理防御、4.2 触发与截断；
# - recall_cluster_threshold：3.2、4.4 召回聚集与记忆重组触发阈值；
# - ae_*、wp_*：1.1.4、2.2 情感能量（AE）与白描动态遗忘；
# - hallucination_anchor_prob：3.3 递归抽象时锚定子事件摘要的概率；
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

    summary: str = "tongyi-xiaomi-analysis-pro"
    boundary_detection: str = "qwen3.6-flash-2026-04-16"
    role_extraction: str = "qwen3.6-flash-2026-04-16"
    abstraction: str = "MiniMax-M2.5"
    insight: str = "tongyi-xiaomi-analysis-pro"
    default: str = "qwen3.6-flash-2026-04-16"


class LLMConfig(BaseModel):
    # 兼容 OpenAI 协议的对话 API（如 DashScope 兼容模式）。
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    task_models: TaskModelMapping = Field(default_factory=TaskModelMapping)
    temperature: float = 0.3
    max_retries: int = 3


class EmbeddingConfig(BaseModel):
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
    # 调整为 128k (131072) 以支持 len_msg ≈ 3000。
    context_window: int = 131072
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

    # 回忆块中「同主题」基本事件数达到阈值则触发抽象/再巩固（白皮书 3.2、4.4）。
    recall_cluster_threshold: int = 5
    # 物理红线触发时，未闭合事件总长超过 len_msg * 该比例则强制封存（代谢防溢出）。
    unclosed_force_ratio: float = 0.8
    # 递归摘要熔断：低于该字数则不再生成更高级摘要（白皮书 1.1.3 actual_max_level）。
    summary_fuse_min_chars: int = 20

    # AE（Affective Energy，情感能量）：高于 ae_high_threshold 时增强抗遗忘权重（白皮书 1.1.4、2.2）。
    ae_high_threshold: float = 0.6
    # 回忆混合打分中来自事件级 AE 的权重（余下与时间衰减、角色重要性、相似度分配）。
    ae_score_weight: float = 0.15

    # 白描时间线遗忘：低 AE 条目半衰期（天）；高 AE 条目半衰期乘以 ae_forgetting_multiplier。
    wp_half_life_days: float = 60.0
    ae_forgetting_multiplier: float = 5.0

    # 抽象演化时以该概率强制用子事件 L1/原文锚定，抑制「推理当事实」闭环（白皮书 3.3）。
    hallucination_anchor_prob: float = 0.3

    # 角色语义卡片最多保留的键数量（白皮书 2.3）。
    semantic_card_max_keys: int = 20

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
    # EMA 指数平滑系数；越大越偏向"新事件"，越小越延续"历史心境"。
    ema_smoothing_alpha: float = 0.4
    # EMA 回溯读取白描尾部的条数（用于生成历史心境基线）。
    ema_history_window: int = 10
    # 将事件 AE 映射为 ``activation_energy`` 的增益；超过 ae_high_threshold 触发重大事件硬绑定。
    activation_energy_gain: float = 1.0
    # 回忆混合打分中 ``activation_energy`` 的权重（从余弦相似度份额中扣除）。
    activation_energy_weight: float = 0.10

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
