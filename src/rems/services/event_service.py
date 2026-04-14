from __future__ import annotations

import logging
from typing import Optional

# 基本事件封存：L0 → 递归摘要 → 角色抽取 → decoration → 持久化与向量索引。
# 对照《REMS 记忆系统规范解析》1.1（事件字段）、1.1.6（decoration）。

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import DECORATION_SYSTEM, DECORATION_USER
from ..models.event import Event, EventRoleEntry, EventStatus
from ..skills.role_extraction import RoleExtractionSkill
from ..skills.summary_generation import SummaryGenerationSkill
from ..storage.repository import EventRepository
from ..storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EventService:
    """Orchestrates the full lifecycle of a Basic Event:

    content_raw -> summaries -> role extraction -> decoration -> persist.

    基本事件全生命周期编排：从 ``content_raw`` 生成多级摘要，抽取角色与情感快照，
    生成非事实 ``decoration``，将完整 ``Event`` 持久化并写入向量索引（供回忆检索）。
    """

    def __init__(
        self,
        config: REMSConfig,
        llm: LLMProvider,
        event_repo: EventRepository,
        vector_store: VectorStore,
        summary_skill: SummaryGenerationSkill,
        role_skill: RoleExtractionSkill,
    ):
        self._config = config
        self._llm = llm
        self._event_repo = event_repo
        self._vector = vector_store
        self._summary_skill = summary_skill
        self._role_skill = role_skill

    # ------------------------------------------------------------------
    def seal_event(
        self,
        content_raw: str,
        *,
        role_entries: list[EventRoleEntry] | None = None,
        skip_roles: bool = False,
        known_roles: list | None = None,
    ) -> Event:
        """Create, enrich, persist and index a new basic event.

        创建、丰富字段、持久化并索引一条新的基本事件（``is_abstract`` 默认为 False）。
        可选传入已构造好的 ``role_entries`` 或 ``skip_roles`` 跳过角色抽取；
        ``known_roles`` 供角色技能做去代词化对齐。超长 ``content_raw`` 会在 ``len_msg`` 处截断。
        """
        length_cap = self._config.len_msg
        if len(content_raw) > length_cap:
            logger.warning(
                "content_raw (%d chars) exceeds len_msg (%d), truncating",
                len(content_raw), length_cap,
            )
            content_raw = content_raw[:length_cap]

        summary_result = self._summary_skill.generate(content_raw)

        if role_entries is None and not skip_roles:
            extraction = self._role_skill.extract(content_raw, known_roles=known_roles)
            role_entries = []
            for er in extraction.roles:
                rid = er.role_id or er.name
                role_entries.append(RoleExtractionSkill.to_event_role_entry(er, rid))

        decoration = self._generate_decoration(content_raw)

        event = Event(
            content_raw=content_raw,
            summaries=summary_result.summaries,
            summary_lengths=summary_result.summary_lengths,
            actual_max_level=summary_result.actual_max_level,
            role_list=role_entries or [],
            status=EventStatus.ACTIVE,
            decoration=decoration,
        )

        self._event_repo.save(event)
        self._index_event(event)

        logger.info("Sealed event %s (%d chars, %d roles)", event.event_id, event.event_length, len(event.role_list))
        return event

    # ------------------------------------------------------------------
    def get_event(self, event_id: str) -> Optional[Event]:
        return self._event_repo.get(event_id)

    def list_basic_events(self, *, is_abstracted: bool | None = None) -> list[Event]:
        return self._event_repo.list_all(is_abstract=False, is_abstracted=is_abstracted)

    def mark_abstracted(self, event_id: str) -> None:
        self._event_repo.update_status(event_id, is_abstracted=True)

    # ------------------------------------------------------------------
    def _generate_decoration(self, content_raw: str) -> str:
        try:
            return self._llm.complete(
                "summary",
                [
                    {"role": "system", "content": DECORATION_SYSTEM},
                    {"role": "user", "content": DECORATION_USER.format(content_raw=content_raw)},
                ],
            ).strip()
        except Exception:
            logger.debug("Decoration generation failed, skipping", exc_info=True)
            return ""

    def _index_event(self, event: Event) -> None:
        # 检索主键优先 L1（保真压缩），无则退回原文（与白皮书 1.1.3 一致）。
        index_text = event.summaries.get("L1", event.content_raw)
        metadata: dict = {
            "is_abstract": event.is_abstract,
            "status": event.status.value,
            "event_length": event.event_length,
        }
        if event.role_list:
            metadata["role_ids"] = ",".join(r.role_id for r in event.role_list)
        self._vector.add_event(event.event_id, index_text, metadata)
