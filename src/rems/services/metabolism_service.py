from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

# 事件代谢：残影合并、边界检测、封存、未完成库维护、物理红线强制整理。
# 对照《REMS 记忆系统规范解析》4.1–4.2。

from ..config import REMSConfig
from ..models.event import Event
from ..models.metabolism import Shadow, UnclosedEvent
from ..services.event_service import EventService
from ..skills.boundary_detection import BoundaryDetectionSkill, BoundaryResult
from ..storage.repository import MetabolismRepository

logger = logging.getLogger(__name__)


class MetabolismService:
    """Manages shadow buffer, unclosed-event library and event trigger logic.

    Lifecycle:
        1. Accept raw input.
        2. Merge with shadow.
        3. Run boundary detection.
        4. Seal completed events; update unclosed library.
        5. Enforce physical-redline compaction when needed.

    管理残影（Shadow）、未完成事件库及封存触发逻辑。典型生命周期为：接收原始输入并与残影合并；
    调用边界检测技能划分已闭环片段与剩余残影；对已闭环内容调用 ``EventService.seal_event``；
    更新未完成库；当残影与未完成总长超过 ``config.physical_redline`` 时执行强制压缩与遗忘策略。
    """

    def __init__(
        self,
        config: REMSConfig,
        meta_repo: MetabolismRepository,
        boundary_skill: BoundaryDetectionSkill,
        event_service: EventService,
    ):
        self._config = config
        self._repo = meta_repo
        self._boundary = boundary_skill
        self._event_svc = event_service

    # ------------------------------------------------------------------
    # Main entry: process a raw input string
    # ------------------------------------------------------------------

    def process_input(
        self,
        raw_input: str,
        *,
        force_save: bool = False,
        input_id: str | None = None,
    ) -> list[Event]:
        """Ingest *raw_input*, return list of newly sealed events (may be empty).

        摄入字符串 *raw_input*，返回本轮新封存的基本事件列表（可能为空列表）。
        ``force_save=True`` 时跳过边界模型，立即合并残影与未完成项并封存（手动 /save 类触发）。

        白皮书 4.2 说明：``msg_len × 1.2`` 的兜底阈值作用于**未闭环事件的累积长度**，
        而非整体输入。边界检测不会因「残影 + 当前输入」过长被跳过；超长输入仅在边界剥离
        得到的单条未闭环片段越过该红线时，才会按「可疑事件」强制封存（见 ``_apply_boundary_result``）。
        """
        if len(raw_input) > self._config.len_msg:
            logger.warning(
                "Input length %d exceeds len_msg %d; boundary detection will still run",
                len(raw_input),
                self._config.len_msg,
            )

        shadow = self._repo.get_shadow()
        unclosed = self._repo.get_unclosed_events()

        if force_save:
            return self._force_save_all(shadow, raw_input, unclosed, input_id=input_id, role_entries=role_entries)

        # 边界检测仅负责事件切分；摘要/角色等衍生字段由 EventEnrichment 在 seal 时生成。
        result = self._boundary.detect(shadow.content, raw_input, unclosed)
        return self._apply_boundary_result(result, unclosed, input_id=input_id, role_entries=role_entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_boundary_result(
        self,
        result: BoundaryResult,
        unclosed: list[UnclosedEvent],
        input_id: str | None = None,
        role_entries: list["EventRoleEntry"] | None = None,
    ) -> list[Event]:
        sealed: list[Event] = []

        for frag in result.completed_events:
            if frag.continuation_of:
                ue = self._find_unclosed(unclosed, frag.continuation_of)
                if ue:
                    content = ue.merged_content + "\n" + frag.content_raw
                    self._repo.delete_unclosed_event(ue.id)
                else:
                    content = frag.content_raw
            else:
                content = frag.content_raw

            event = self._event_svc.seal_event(
                content,
                input_id=input_id,
            )
            sealed.append(event)

        self._repo.update_shadow(Shadow(content=result.remaining_shadow, updated_at=datetime.now()))

        # 白皮书 4.2 底线兜底：对单条未闭环片段逐项判定；若长度越过 ``len_msg × 1.2`` 红线，
        # 立即强制封存为事件，防止内存/计算爆炸。是否「可疑」交由 EventService 内部的
        # 角色抽取与 RoleService 仲裁判断（角色清晰时不应被盲目标脏）。
        force_threshold = int(self._config.len_msg * self._config.unclosed_force_ratio)
        for nu in result.new_unclosed:
            if len(nu.content) > force_threshold:
                logger.warning(
                    "Unclosed fragment length %d exceeds %.2f×len_msg (%d), force-sealing",
                    len(nu.content),
                    self._config.unclosed_force_ratio,
                    force_threshold,
                )
                event = self._event_svc.seal_event(
                    nu.content,
                    input_id=input_id,
                )
                sealed.append(event)
                continue

            ue = UnclosedEvent(
                id=f"UC-{secrets.token_hex(4)}",
                content_fragments=[nu.content],
                logical_gaps=nu.logical_gaps,
            )
            self._repo.save_unclosed_event(ue)

        self._check_physical_redline()

        return sealed

    # ------------------------------------------------------------------
    def _force_save_all(
        self,
        shadow: Shadow,
        raw_input: str,
        unclosed: list[UnclosedEvent],
        *,
        is_suspicious: bool = False,
        input_id: str | None = None,
    ) -> list[Event]:
        """Manual trigger (/save, /mem) or length-based fallback: seal everything immediately.

        手动触发或长度被迫兜底：将残影与当前输入合并后尽可能封存；遍历未完成库中已有内容的条目逐一封存并删除；
        最后清空残影。用于用户显式「保存记忆」或文本溢出边界强制回收。
        """
        sealed: list[Event] = []

        combined = (shadow.content + "\n" + raw_input).strip()
        if combined:
            event = self._event_svc.seal_event(combined, is_suspicious=is_suspicious, input_id=input_id)
            sealed.append(event)

        for ue in unclosed:
            if ue.total_length > 0:
                event = self._event_svc.seal_event(ue.merged_content, is_suspicious=is_suspicious, input_id=input_id)
                sealed.append(event)
            self._repo.delete_unclosed_event(ue.id)

        self._repo.update_shadow(Shadow(content="", updated_at=datetime.now()))
        return sealed

    # ------------------------------------------------------------------
    def _check_physical_redline(self) -> None:
        """If shadow + unclosed exceed physical redline, force-compact.

        当残影长度与所有未完成事件片段长度之和超过 ``physical_redline`` 时触发：
        对超过 ``len_msg * unclosed_force_ratio`` 的未完成条目强制封存；按策略遗忘陈旧未完成项；
        若残影仍高于 ``safe_watermark`` 则截断保留尾部，防止缓冲区无限增长（白皮书 4.2 工程化防溢出）。
        """
        shadow = self._repo.get_shadow()
        unclosed = self._repo.get_unclosed_events()
        total = shadow.length + sum(ue.total_length for ue in unclosed)

        if total <= self._config.physical_redline:
            return

        logger.warning("Physical redline hit (%d > %d), forcing compaction", total, self._config.physical_redline)

        for ue in unclosed:
            force_threshold = self._config.len_msg * self._config.unclosed_force_ratio
            if ue.total_length >= force_threshold:
                event = self._event_svc.seal_event(ue.merged_content)
                self._repo.delete_unclosed_event(ue.id)
                logger.info("Force-sealed unclosed %s as event %s", ue.id, event.event_id)

        self._forget_stale_unclosed(unclosed)

        remaining_shadow = self._repo.get_shadow()
        if remaining_shadow.length > self._config.safe_watermark:
            trimmed = remaining_shadow.content[-self._config.safe_watermark:]
            self._repo.update_shadow(Shadow(content=trimmed, updated_at=datetime.now()))

    # ------------------------------------------------------------------
    def _forget_stale_unclosed(self, unclosed: list[UnclosedEvent], stale_hours: int = 24) -> None:
        # 长时间未续写的未完成条目丢弃，避免库无限膨胀（工程策略）。
        cutoff = datetime.now() - timedelta(hours=stale_hours)
        for ue in unclosed:
            if ue.last_hit_time < cutoff:
                logger.info("Forgetting stale unclosed event %s", ue.id)
                self._repo.delete_unclosed_event(ue.id)

    # ------------------------------------------------------------------
    @staticmethod
    def _find_unclosed(events: list[UnclosedEvent], ue_id: str) -> Optional[UnclosedEvent]:
        for ue in events:
            if ue.id == ue_id:
                return ue
        return None
