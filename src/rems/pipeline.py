from __future__ import annotations

import logging
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
from .storage.repository import EventRepository, MetabolismRepository, RoleRepository
from .storage.vector_store import VectorStore

# 顶层编排：ingest 串联回忆（ContextPackage）、代谢封存、角色白描/语义卡片、
# 记忆再巩固抽象与 NPC 指令；对照《REMS 记忆系统规范解析》4.4、5.1–5.3。

logger = logging.getLogger(__name__)


# =====================================================================
# Processing Modes (§5 Multi-scenario Output)
# =====================================================================

class ProcessingMode(str, Enum):
    """Output mode controlling pipeline behaviour.

    DIALOGUE
        Standard interactive mode.  Assembles full Context Package for LLM
        response generation.  Metabolism (sealing, indexing) runs in the
        same call (in-process; use async/worker in production for latency).

    PASSIVE_LOG
        "Silent listener" mode for wearables, meeting recordings, etc.
        No context package is assembled.  Input is fed directly to shadow
        buffer and sealed when thresholds are met.  No text output expected.

    NPC_AGENT
        Generative-agent / virtual-sandbox mode.  Output is a structured
        dict with ``action`` and updated Klesha/Vedana deltas, intended
        for a downstream Directive Parser.  No natural-language reply.

    中文（对照白皮书第 5 章）：
        DIALOGUE：标准强输出交互；组装完整 Context Package供 LLM 生成自然语言回复；
        代谢（封存、索引）可与本次调用同进程执行（生产环境建议异步/队列以降低延迟）。
        PASSIVE_LOG：被动日志/静默倾听；不组装上下文包；输入进入残影并按阈值封存；
        不向用户输出任何文本（穿戴设备、会议转写等「只记不说」场景）。
        NPC_AGENT：生成式智能体/沙盒 NPC；输出含 ``action`` 与 Klesha/Vedana 增量等的结构化字典，
        交由下游 Directive Parser 转为引擎调用，无需人类可见的对话文本。
    """
    DIALOGUE = "dialogue"
    PASSIVE_LOG = "passive_log"
    NPC_AGENT = "npc_agent"


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
        event_service: EventService,
        role_service: RoleService,
        metabolism_service: MetabolismService,
        recall_service: RecallService,
        abstraction_service: AbstractionService,
        belief_revision_service: BeliefRevisionService,
    ):
        self.config = config
        self.llm = llm
        self.db = db
        self.vector_store = vector_store
        self.event_repo = event_repo
        self.role_repo = role_repo
        self.meta_repo = meta_repo
        self.event_service = event_service
        self.role_service = role_service
        self.metabolism_service = metabolism_service
        self.recall_service = recall_service
        self.abstraction_service = abstraction_service
        self.belief_revision_service = belief_revision_service

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

        # SummaryGenerationSkill 仍用于抽象事件（inductive evolution 后的总结）。
        # 基本事件流已被 EventEnrichmentSkill 接管（一次调用产出摘要 + 角色）。
        summary_skill = SummaryGenerationSkill(llm, config)
        role_skill = RoleExtractionSkill(llm, config)
        boundary_skill = BoundaryDetectionSkill(llm, config)
        enrichment_skill = EventEnrichmentSkill(llm, config)
        evolution_skill = InductiveEvolutionSkill(llm, config)

        emotion_evolver = EMAEvolver(config, role_repo)
        # Pass llm to role_service so semantic cards can be refreshed in-process
        role_service = RoleService(config, role_repo, role_skill, llm=llm)

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
        abstraction_service = AbstractionService(config, event_repo, vector_store, evolution_skill, summary_skill)
        belief_revision_service = BeliefRevisionService(config, event_repo, role_repo)

        return cls(
            config=config,
            llm=llm,
            db=db,
            vector_store=vector_store,
            event_repo=event_repo,
            role_repo=role_repo,
            meta_repo=meta_repo,
            event_service=event_service,
            role_service=role_service,
            metabolism_service=metabolism_service,
            recall_service=recall_service,
            abstraction_service=abstraction_service,
            belief_revision_service=belief_revision_service,
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
        """Full processing cycle with multi-scenario output.

        PASSIVE_LOG:
            Skip context package assembly; feed directly to metabolism.

        DIALOGUE:
            Assemble context package first, then metabolism.

        NPC_AGENT:
            Assemble context package, derive action directives from recalled
            memories, run metabolism in background.

        完整处理周期（多场景输出形态由 ``mode`` 决定）：

        PASSIVE_LOG：
            不组装 Context Package；输入直接进入代谢（残影/边界/封存），适用于「只记不说」的被动日志场景。

        DIALOGUE：
            先基于残影与当前输入组装 Context Package（供上游 LLM 生成回复），
            再执行代谢封存；两者在同一调用内顺序执行（生产可改为后台代谢）。

        NPC_AGENT：
            同样组装 Context Package；根据回忆块等生成 NPC 结构化指令；
            代谢仍执行以持久化环境事件（注释中所述 background 指与「对话生成」解耦的语义，
            本实现仍为同进程顺序调用，部署时可拆分为异步工作者）。
        """
        shadow = self.meta_repo.get_shadow()

        # 从当前输入和残影中提取焦点角色（用于回忆时的角色感知摘要档位选择）。
        focus_role_ids = self._extract_focus_roles(raw_input, shadow.content if shadow else "")

        # 对话/NPC：用「残影 + 当前输入」检索并组装 ContextPackage；被动日志跳过（白皮书 5.2）。
        ctx: Optional[ContextPackage] = None
        if mode != ProcessingMode.PASSIVE_LOG:
            ctx = self.recall_service.build_context_package(
                raw_input, shadow, focus_role_ids=focus_role_ids
            )

        # 代谢：边界检测、封存基本事件、维护残影与未完成库（第 4.1–4.2）。
        sealed = self.metabolism_service.process_input(raw_input, force_save=force_save, input_id=input_id)

        # 角色：每个新事件更新白描时间线并刷新语义卡片（第 2.2–2.3）。
        for event in sealed:
            self.role_service.update_from_event(event)

        # 记忆再巩固：回忆块中同一主题基本事件数量达阈值则归纳抽象（第 4.4）。
        abstract_events: list[Event] = []

        if ctx and ctx.recall_block.items and mode != ProcessingMode.PASSIVE_LOG:
            abstract_events.extend(self._reconsolidate(ctx, sealed))

        # 封存后再以新事件为锚做一次向量聚类抽象（第 3.2 演化驱动触发的一种实现路径）。
        for event in sealed:
            abstract = self.abstraction_service.check_and_abstract(event)
            if abstract:
                abstract_events.append(abstract)

        # NPC：由回忆与威胁启发式生成行为指令（第 5.3）。
        npc_directives: list[dict] = []
        if mode == ProcessingMode.NPC_AGENT and npc_role_id:
            npc_directives = self._build_npc_directives(npc_role_id, ctx, sealed)

        return ProcessingResult(
            sealed_events=sealed,
            abstract_events=abstract_events,
            context_package=ctx,
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
    # Memory Reconsolidation
    # ------------------------------------------------------------------

    def _reconsolidate(
        self,
        ctx: ContextPackage,
        newly_sealed: list[Event],
    ) -> list[Event]:
        """Trigger abstract event synthesis based on space pressure (Whitepaper 4.4).
        
        Uses the ``abstraction_candidate_ids`` pre-calculated by RecallService 
        (which targets the top 1/6.6 context pressure zone).
        """
        candidates = []
        for eid in ctx.recall_block.abstraction_candidate_ids:
            e = self.event_repo.get(eid)
            if e and not e.is_abstract and not e.is_tombstoned and not e.is_abstracted:
                candidates.append(e)

        if not candidates:
            return []

        logger.info(
            "Memory Reconsolidation triggered: %d candidates from recall pressure zone",
            len(candidates),
        )

        abstract_evt = self.abstraction_service.abstract_event_cluster(candidates)
        return [abstract_evt] if abstract_evt else []

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
        ``flee`` 等 ``action`` 及 ``klesha_delta``。下游 Directive Parser 将其翻译为
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
        klesha_delta: dict[str, float] = {}
        if threat_score >= 0.6:
            action = "flee"
            klesha_delta = {"anger": 0.3, "ignorance": -0.1}
        elif threat_score >= 0.3:
            action = "watch"
            klesha_delta = {"doubt": 0.2}

        if action != "idle":
            directives.append({
                "npc_role_id": npc_role_id,
                "action": action,
                "klesha_delta": klesha_delta,
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
        card = self.role_service.get_semantic_card(role.role_id)
        return {
            "role": role.model_dump(mode="json", exclude={"white_painting", "semantic_card"}),
            "white_painting_summary": summary,
            "semantic_card": card.data if card else {},
        }

    def run_evolution(self) -> list[Event]:
        return self.abstraction_service.run_background_evolution()

    def tombstone(self, event_id: str, reason: str, replacement_id: str | None = None) -> bool:
        return self.belief_revision_service.tombstone_event(
            event_id, reason=reason, replacement_event_id=replacement_id
        )
