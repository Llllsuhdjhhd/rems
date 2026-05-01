from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .config import REMSConfig, UserMode
from .llm.provider import LLMProvider
from .models.event import Event
from .models.metabolism import ContextPackage
from .services.abstraction_service import AbstractionService
from .services.belief_revision_service import BeliefRevisionService
from .services.emotion_service import EMAEvolver
from .services.event_service import EventService
from .services.metabolism_service import MetabolismService
from .services.recall_service import RecallService
from .services.role_service import RoleService
from .skills.boundary_detection import BoundaryDetectionSkill
from .skills.event_enrichment import EventEnrichmentSkill
from .skills.inductive_evolution import InductiveEvolutionSkill
from .skills.role_extraction import RoleExtractionSkill
from .skills.summary_generation import SummaryGenerationSkill
from .storage.database import Database
from .storage.repository import (
    AbstractedSubsetRepository,
    EventRepository,
    MetabolismRepository,
    RecallLogRepository,
    RoleRepository,
)
from .storage.vector_store import VectorStore

# 顶层编排：ingest 串联回忆（ContextPackage）、代谢封存、角色白描/语义卡片、
# 记忆再巩固抽象与 NPC 指令；对照《REMS 记忆系统规范解析》4.4、5.1–5.3。

logger = logging.getLogger(__name__)


# =====================================================================
# Processing Modes (§5 Multi-scenario Output)
# =====================================================================

class ProcessingMode(str, Enum):
    """Output mode controlling what the pipeline **returns** to the caller.

    重要设计约定（与白皮书 §5 一致）：**模式只影响"输出形态"，不影响"内部记忆动力学"**。
    回忆块组装、``recall_log`` 登记、频繁子集挖掘 (§3.2) 等一律无条件执行——因为：
        (a) 抽象事件的唯一触发路径是 ``recall_log``，跳过回忆等于放弃所有抽象合成；
        (b) 白描、AE、语义卡片等后台巩固机制都依赖对事件的持续检索相关性；
        (c) "静默倾听"的语义是「不对用户可见」，而不是「不要构建记忆」。

    所以模式的差异收敛到一个布尔属性 :pyattr:`returns_context_package` 以及少量输出
    装配逻辑（例如 NPC 的 Directive）。新增模式时只需在枚举中追加成员、覆盖属性或
    扩展 ``_build_*_directives``，不需要触动 ``ingest`` 的主干流程。

    DIALOGUE
        标准强输出交互。ingest 返回完整 ContextPackage，供上游 LLM 生成自然语言回复。
    PASSIVE_LOG
        "静默倾听"：穿戴录音、会议转写等「只记不说」场景。回忆、代谢、抽象合成**仍然执行**
        且登记入库，只是 ``ContextPackage`` 不对外暴露（:pyattr:`returns_context_package`
        ``= False``），不产出人类可见的文本回复。
    NPC_AGENT
        生成式 NPC / 沙盒 Agent。除了 ContextPackage 外，还输出结构化 ``npc_directives``
        （动作 + 情绪增量等），交给下游 Directive Parser 转为引擎调用。
    """
    DIALOGUE = "dialogue"
    PASSIVE_LOG = "passive_log"
    NPC_AGENT = "npc_agent"

    @property
    def returns_context_package(self) -> bool:
        """Whether ``ingest`` should expose the ContextPackage to the caller.

        默认对外暴露；被动日志等"静默"模式重写为 False。新增对外不回复、只沉淀记忆的模式
        时复写此属性即可，内部回忆与抽象管线不需要改动。
        """
        return self != ProcessingMode.PASSIVE_LOG


# =====================================================================
# Result dataclasses
# =====================================================================

@dataclass
class ProcessingResult:
    """Outcome of a single ``ingest`` call across processing modes.

    sealed_events — newly sealed basic events from this metabolism pass.
    abstract_events — abstract events synthesised via reconsolidation or post-seal clustering.
    context_package — recall + shadow + current input (omitted in PASSIVE_LOG).
    npc_directives — structured NPC actions when ``mode == NPC_AGENT``.

    单次 ``ingest`` 的返回结果（多场景共用同一结构）：
    ``sealed_events`` 为本轮代谢新封存的基本事件；
    ``abstract_events`` 为记忆再巩固或封存后聚类产生的抽象事件；
    ``context_package`` 为回忆块+残影+当前输入（被动日志模式下为 None）；
    ``npc_directives`` 为 NPC 模式下的结构化行为指令列表。
    """
    # 一次 ingest 的产出：新封存基本事件、（可选）新抽象事件、对话用上下文包、NPC 指令。
    sealed_events: list[Event] = field(default_factory=list)
    abstract_events: list[Event] = field(default_factory=list)
    context_package: Optional[ContextPackage] = None
    mode: ProcessingMode = ProcessingMode.DIALOGUE

    # NPC 模式下的结构化行为输出（白皮书 5.3）。
    npc_directives: list[dict] = field(default_factory=list)


# =====================================================================
# Pipeline
# =====================================================================

class REMSPipeline:
    """Top-level orchestrator wiring all REMS components together.

    Usage::

        pipeline = REMSPipeline.from_config(config)

        # Interactive dialogue
        result = pipeline.ingest("今天发生了一件事……")

        # Passive logging (silent, no ctx package)
        result = pipeline.ingest("…", mode=ProcessingMode.PASSIVE_LOG)

        # NPC agent
        result = pipeline.ingest("玩家靠近", mode=ProcessingMode.NPC_AGENT,
                                 npc_role_id="ROL-xxx")

    顶层编排器：将配置、LLM、持久化、向量库、各仓储与领域服务串联为可调用管线。
    典型用法见上；``from_config`` 负责一次性构造全部依赖（技能与服务的依赖注入入口）。
    """

    def __init__(
        self,
        config: REMSConfig,
        llm: LLMProvider,
        db: Database,
        vector_store: VectorStore,
        event_repo: EventRepository,
        role_repo: RoleRepository,
        meta_repo: MetabolismRepository,
        recall_log_repo: RecallLogRepository,
        event_service: EventService,
        role_service: RoleService,
        metabolism_service: MetabolismService,
        recall_service: RecallService,
        abstraction_service: AbstractionService,
        belief_revision_service: BeliefRevisionService,
        role_skill: RoleExtractionSkill | None = None,
    ):
        self.config = config
        self.llm = llm
        self.db = db
        self.vector_store = vector_store
        self.event_repo = event_repo
        self.role_repo = role_repo
        self.meta_repo = meta_repo
        self.recall_log_repo = recall_log_repo
        self.event_service = event_service
        self.role_service = role_service
        self.metabolism_service = metabolism_service
        self.recall_service = recall_service
        self.abstraction_service = abstraction_service
        self.belief_revision_service = belief_revision_service
        self.role_skill = role_skill

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: REMSConfig | None = None) -> REMSPipeline:
        # Factory: wire storage, skills, and domain services in one place for API/scripts.
        # 工厂方法：集中实例化存储、技能与领域服务，供 API 与脚本共用。
        if config is None:
            config = REMSConfig()

        llm = LLMProvider(config)
        db = Database(config.storage.database_url)
        db.create_tables()
        vector_store = VectorStore(config)

        event_repo = EventRepository(db)
        role_repo = RoleRepository(db)
        meta_repo = MetabolismRepository(db)
        recall_log_repo = RecallLogRepository(db)
        abstracted_subset_repo = AbstractedSubsetRepository(db)

        # SummaryGenerationSkill 仍用于抽象事件（inductive evolution 后的总结）。
        # 基本事件流已被 EventEnrichmentSkill 接管（一次调用产出摘要 + 角色）。
        summary_skill = SummaryGenerationSkill(llm, config)
        role_skill = RoleExtractionSkill(llm, config)
        boundary_skill = BoundaryDetectionSkill(llm, config)
        enrichment_skill = EventEnrichmentSkill(llm, config, role_fallback=role_skill)
        evolution_skill = InductiveEvolutionSkill(llm, config)

        emotion_evolver = EMAEvolver(config, role_repo)
        # Pass llm to role_service so semantic cards can be refreshed in-process
        role_service = RoleService(config, role_repo, role_skill, llm=llm, vector_store=vector_store)

        event_service = EventService(
            config,
            llm,
            event_repo,
            vector_store,
            enrichment_skill,
            role_service=role_service,
            emotion_evolver=emotion_evolver,
        )
        
        metabolism_service = MetabolismService(config, meta_repo, boundary_skill, event_service)
        recall_service = RecallService(config, event_repo, role_repo, vector_store)
        abstraction_service = AbstractionService(
            config,
            event_repo,
            vector_store,
            evolution_skill,
            summary_skill,
            recall_log_repo,
            abstracted_subset_repo,
        )
        belief_revision_service = BeliefRevisionService(config, event_repo, role_repo)

        return cls(
            config=config,
            llm=llm,
            db=db,
            vector_store=vector_store,
            event_repo=event_repo,
            role_repo=role_repo,
            meta_repo=meta_repo,
            recall_log_repo=recall_log_repo,
            event_service=event_service,
            role_service=role_service,
            metabolism_service=metabolism_service,
            recall_service=recall_service,
            abstraction_service=abstraction_service,
            belief_revision_service=belief_revision_service,
            role_skill=role_skill,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def ingest(
        self,
        raw_input: str,
        *,
        force_save: bool = False,
        mode: ProcessingMode = ProcessingMode.DIALOGUE,
        npc_role_id: str | None = None,
        input_id: str | None = None,
    ) -> ProcessingResult:
        """Full processing cycle.

        **内部记忆动力学（所有模式一律执行）**：
            1. 取残影 + 焦点角色；
            2. 组装 ``ContextPackage``（回忆块 + 残影 + 当前输入）；
            3. 把回忆块真实事件 ID 登记到 ``recall_log``——白皮书 §3.2 规定这是抽象事件的**唯一**触发路径，
               因此即使被动日志模式"不对用户说话"，也必须完成回忆与登记，否则系统永远不会演化出抽象规律；
            4. 代谢：边界检测、封存基本事件、维护残影与未完成库（§4.1–§4.2）；
            5. 角色更新：对每个新基本事件追加白描时间线、必要时刷新语义卡片（§2.2–§2.3，抽象事件自动跳过）；
            6. 抽象合成：扫 ``recall_log`` 全量历史做极大频繁子集挖掘（§3.2）。

        **外部输出形态（由 ``mode`` 决定）**：
            - ``returns_context_package == True``：对外返回 ContextPackage 供上游 LLM 生成回复；
            - ``returns_context_package == False``：ContextPackage 留在内部，对外返回 ``None``
              （"静默倾听"）；
            - ``NPC_AGENT``：额外基于回忆块派生 ``npc_directives``；
            - 新增模式只需在 :class:`ProcessingMode` 追加成员并复写 ``returns_context_package``
              （或扩展专属装配步骤），不需要改动主干流程。
        """
        shadow = self.meta_repo.get_shadow()

        # Step 1: LLM Character Extraction (Before Recall)
        # 提前进行人物提取（输入+残影），用于提取焦点角色辅助召回（不直接传给封存阶段）
        combined_text = (shadow.content + "\n" + raw_input).strip()
        print(f"  [DEBUG] role_skill exists: {self.role_skill is not None}, combined_text length: {len(combined_text)}")
        role_entries = []
        if self.role_skill and combined_text:
            extraction_result = self.role_skill.extract(combined_text)
            extracted_roles = list(extraction_result.roles)
            if extracted_roles:
                id_mapping = self.role_service.resolve_and_register(extracted_roles, is_suspicious=False)
                for er in extracted_roles:
                    lookup_key = er.role_id or er.name
                    assigned_id = id_mapping.get(lookup_key, lookup_key)
                    role_entries.append(self.role_skill.to_event_role_entry(er, assigned_id))

        # 从当前输入、残影以及刚抽取的角色中提取焦点角色
        focus_role_ids = set(r.role_id for r in role_entries)
        if self.config.user_mode == UserMode.SINGLE and self.config.core_user_role_id:
            focus_role_ids.add(self.config.core_user_role_id)
        focus_role_ids.update(self._extract_focus_roles(raw_input, shadow.content if shadow else ""))

        # 回忆（所有模式必须执行）：
        # 抽象事件的唯一触发路径是 recall_log 的频繁子集挖掘，跳过回忆等于放弃所有归纳演化；
        # "静默倾听"等模式只是不把 ctx 交给外部，内部仍然完整组装、登记与挖掘。
        ctx: ContextPackage = self.recall_service.build_context_package(
            raw_input, shadow, focus_role_ids=focus_role_ids, focus_role_entries=role_entries
        )

        # 登记本次回忆块 event_id 到 recall_log（白皮书 §3.2 唯一抽象触发路径的输入流）。
        # 只记录真实 basic/abstract 事件；抽象事件 id 同样进入命名空间，
        # 便于后续更高阶抽象在同一空间继续挖掘。
        if ctx.recall_block.items:
            recall_event_ids: list[str] = []
            seen_ids: set[str] = set()
            for it in ctx.recall_block.items:
                eid = it.event_id
                if not eid or eid in seen_ids:
                    continue
                seen_ids.add(eid)
                recall_event_ids.append(eid)
            if recall_event_ids:
                self.recall_log_repo.append(
                    recall_id=f"RCL-{uuid.uuid4().hex}",
                    event_ids=recall_event_ids,
                )

        # 代谢：边界检测、封存基本事件、维护残影与未完成库（第 4.1–4.2）。
        sealed = self.metabolism_service.process_input(
            raw_input, 
            force_save=force_save, 
            input_id=input_id,
        )

        # 角色：每个新事件更新白描时间线并刷新语义卡片（第 2.2–2.3）。
        # RoleService 对 is_abstract=True 的事件自带早退，抽象事件不会污染白描。
        for event in sealed:
            self.role_service.update_from_event(event)

        # 抽象事件触发 —— 唯一路径：``recall_log`` 中的极大频繁子集（白皮书 §3.2）。
        # 所有模式都跑：被动日志场景下仍然需要持续演化出抽象规律供未来检索或审计。
        abstract_events: list[Event] = self.abstraction_service.mine_and_synthesize()

        # NPC：由回忆与威胁启发式生成行为指令（第 5.3）。
        npc_directives: list[dict] = []
        if mode == ProcessingMode.NPC_AGENT and npc_role_id:
            npc_directives = self._build_npc_directives(npc_role_id, ctx, sealed)

        # 输出路由：``returns_context_package`` 控制 ctx 是否对外暴露；内部管线与产物不变。
        exposed_ctx: Optional[ContextPackage] = ctx if mode.returns_context_package else None

        return ProcessingResult(
            sealed_events=sealed,
            abstract_events=abstract_events,
            context_package=exposed_ctx,
            mode=mode,
            npc_directives=npc_directives,
        )

    # ------------------------------------------------------------------
    # Focus role extraction (for recall tier selection)
    # ------------------------------------------------------------------

    def _extract_focus_roles(self, raw_input: str, shadow_content: str = "") -> set[str]:
        """Identify role IDs that should be treated as 'primary' in this cycle.

        Strategy (no LLM call — pure heuristic for low latency):
        1. In single-user mode, always include ``core_user_role_id`` (白皮书 2.2/2.3:
           单人模式下核心用户恒为主角，配合动态粒度路由强制拉取 L3 详细白描)。
        2. Collect all registered role names / aliases.
        3. Check which names appear (case-insensitive substring) in the combined
           text of *raw_input* + *shadow_content*.
        4. Return the matching role IDs.

        This gives the recall assembler enough signal to prefer detailed summaries
        for roles the user is currently talking about, and compress others.
        """
        combined = (shadow_content + " " + raw_input).lower()
        focus: set[str] = set()

        # 单人模式：核心用户恒在 focus（白皮书 2.2）。
        if self.config.user_mode == UserMode.SINGLE and self.config.core_user_role_id:
            focus.add(self.config.core_user_role_id)

        try:
            for role in self.role_repo.list_all():
                names_to_check = [role.name] + (role.aliases or [])
                if any(n.lower() in combined for n in names_to_check):
                    focus.add(role.role_id)
        except Exception:
            pass
        return focus

    # ------------------------------------------------------------------
    # NPC / Generative-Agent directives
    # ------------------------------------------------------------------

    def _build_npc_directives(
        self,
        npc_role_id: str,
        ctx: Optional[ContextPackage],
        sealed: list[Event],
    ) -> list[dict]:
        """Derive structured action directives for an NPC role.

        Looks for memory of player interactions stored in the role's
        white-painting and the current recall block, then outputs a simple
        action + emotion-delta dict.  The downstream Directive Parser
        translates this to engine calls.

        为指定 ``npc_role_id`` 生成结构化 NPC 指令：在当前 ``ContextPackage`` 的回忆条目中，
        用启发式统计该 ID 在文本中的出现次数以估计威胁程度，并映射为 ``idle`` / ``watch`` /
        ``flee`` 等 ``action`` 及 ``emotion_delta``。下游 Directive Parser 将其翻译为
        寻路、动画或状态机变更（白皮书 5.3）；本实现为轻量示例，非完整游戏 AI。
        """
        directives: list[dict] = []
        if not ctx:
            return directives

        # Gather anger/threat signals from recall
        threat_score = 0.0
        for item in ctx.recall_block.items:
            if npc_role_id in item.content:
                threat_score += 0.2  # heuristic bump per mention

        action = "idle"
        emotion_delta: dict[str, float] = {}
        if threat_score >= 0.6:
            action = "flee"
            emotion_delta = {"fear": 0.4, "anger": 0.2}
        elif threat_score >= 0.3:
            action = "watch"
            emotion_delta = {"anticipation": 0.2, "fear": 0.1}

        if action != "idle":
            directives.append({
                "npc_role_id": npc_role_id,
                "action": action,
                "emotion_delta": emotion_delta,
                "reason": f"threat_score={threat_score:.2f}",
            })
        return directives

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def query_role(self, name_or_id: str) -> dict:
        role = self.role_repo.get(name_or_id) or self.role_service.find_role(name_or_id)
        if not role:
            return {"error": f"Role '{name_or_id}' not found"}
        summary = self.role_service.get_white_painting_summary(role.role_id)
        return {
            "role": role.model_dump(mode="json", exclude={"white_painting"}),
            "white_painting_summary": summary,
        }

    def run_evolution(self) -> list[Event]:
        """Manually drive the subset-mining pass (白皮书 §3.2).

        Equivalent to what ``ingest`` already runs after each recall; exposed for batch jobs.
        """
        return self.abstraction_service.mine_and_synthesize()

    def resolve_to_basic_events(self, event_id: str) -> list[Event]:
        """Resolve an abstract event to the flat list of basic events it ultimately derives from.

        从某条事件出发，沿 ``source_events`` 的嵌套链下钻直到叶子层（基本事件），
        返回 ``Event`` 模型列表（白皮书 §1.1.8 / §3.2）。
        输入若是基本事件，直接返回含自身的单元素列表；抽象事件会跨多层抽象逐级展开；
        墓碑事件不会进入结果。
        """
        basic_ids = self.event_repo.resolve_basic_event_ids(event_id)
        events: list[Event] = []
        for eid in basic_ids:
            ev = self.event_repo.get(eid)
            if ev is not None:
                events.append(ev)
        return events

    def tombstone(self, event_id: str, reason: str, replacement_id: str | None = None) -> bool:
        return self.belief_revision_service.tombstone_event(
            event_id, reason=reason, replacement_event_id=replacement_id
        )
