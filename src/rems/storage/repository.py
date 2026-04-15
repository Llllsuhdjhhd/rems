from __future__ import annotations

from typing import Optional

# 仓储层：Event/Role/Metabolism 的 CRUD 与 ORM ↔ Pydantic 模型转换。

from ..models.event import EmotionalModel, Event, EventRoleEntry, EventStatus
from ..models.metabolism import Shadow, UnclosedEvent
from ..models.role import Role, SemanticCard, WhitePaintingEntry
from .database import (
    Database,
    EventRecord,
    RoleRecord,
    SemanticCardRecord,
    ShadowRecord,
    UnclosedEventRecord,
    WhitePaintingRecord,
)


# =====================================================================
# Event CRUD
# =====================================================================

class EventRepository:
    def __init__(self, db: Database):
        self._db = db

    def save(self, event: Event) -> None:
        with self._db.session() as s:
            record = EventRecord(
                event_id=event.event_id,
                create_time=event.create_time,
                content_raw=event.content_raw,
                summaries=event.summaries,
                summary_lengths=event.summary_lengths,
                actual_max_level=event.actual_max_level,
                role_list=[r.model_dump(mode="json") for r in event.role_list],
                is_abstract=event.is_abstract,
                is_abstracted=event.is_abstracted,
                status=event.status.value,
                decoration=event.decoration,
                insight=event.insight,
                event_length=event.event_length,
                abstraction_level=event.abstraction_level,
                source_events=event.source_events,
                is_tombstoned=event.is_tombstoned,
            )
            s.merge(record)
            s.commit()

    def get(self, event_id: str) -> Optional[Event]:
        with self._db.session() as s:
            r = s.get(EventRecord, event_id)
            return self._to_model(r) if r else None

    def list_all(
        self,
        *,
        is_abstract: bool | None = None,
        is_abstracted: bool | None = None,
        status: EventStatus | None = None,
        exclude_tombstoned: bool = True,
    ) -> list[Event]:
        with self._db.session() as s:
            q = s.query(EventRecord)
            if is_abstract is not None:
                q = q.filter(EventRecord.is_abstract == is_abstract)
            if is_abstracted is not None:
                q = q.filter(EventRecord.is_abstracted == is_abstracted)
            if status is not None:
                q = q.filter(EventRecord.status == status.value)
            if exclude_tombstoned:
                q = q.filter(EventRecord.is_tombstoned == False)  # noqa: E712
            return [self._to_model(r) for r in q.order_by(EventRecord.create_time).all()]

    def update_status(
        self,
        event_id: str,
        *,
        is_abstracted: bool | None = None,
        status: EventStatus | None = None,
        is_tombstoned: bool | None = None,
    ) -> None:
        with self._db.session() as s:
            r = s.get(EventRecord, event_id)
            if not r:
                return
            if is_abstracted is not None:
                r.is_abstracted = is_abstracted
            if status is not None:
                r.status = status.value
            if is_tombstoned is not None:
                r.is_tombstoned = is_tombstoned
            s.commit()

    # ------------------------------------------------------------------
    @staticmethod
    def _to_model(r: EventRecord) -> Event:
        role_list = [EventRoleEntry(**rd) for rd in (r.role_list or [])]
        return Event(
            event_id=r.event_id,
            create_time=r.create_time,
            content_raw=r.content_raw,
            summaries=r.summaries or {},
            summary_lengths=r.summary_lengths or {},
            actual_max_level=r.actual_max_level or 0,
            role_list=role_list,
            is_abstract=r.is_abstract,
            is_abstracted=r.is_abstracted,
            status=EventStatus(r.status),
            decoration=r.decoration,
            insight=r.insight,
            event_length=r.event_length or 0,
            abstraction_level=r.abstraction_level,
            source_events=r.source_events,
            is_tombstoned=bool(r.is_tombstoned),
        )


# =====================================================================
# Role CRUD
# =====================================================================

class RoleRepository:
    def __init__(self, db: Database):
        self._db = db

    def save(self, role: Role) -> None:
        with self._db.session() as s:
            record = RoleRecord(
                role_id=role.role_id,
                name=role.name,
                entity_type=role.entity_type,
                aliases=role.aliases,
                created_at=role.created_at,
            )
            s.merge(record)
            s.commit()

    def get(self, role_id: str) -> Optional[Role]:
        with self._db.session() as s:
            record = s.get(RoleRecord, role_id)
            if not record:
                return None
            entries = (
                s.query(WhitePaintingRecord)
                .filter(WhitePaintingRecord.role_id == role_id)
                .order_by(WhitePaintingRecord.create_time)
                .all()
            )
            card_rec = s.get(SemanticCardRecord, role_id)
            return self._build_role(record, entries, card_rec)

    def list_all(self) -> list[Role]:
        with self._db.session() as s:
            records = s.query(RoleRecord).order_by(RoleRecord.created_at).all()
            result: list[Role] = []
            for rec in records:
                entries = (
                    s.query(WhitePaintingRecord)
                    .filter(WhitePaintingRecord.role_id == rec.role_id)
                    .order_by(WhitePaintingRecord.create_time)
                    .all()
                )
                card_rec = s.get(SemanticCardRecord, rec.role_id)
                result.append(self._build_role(rec, entries, card_rec))
            return result

    def find_by_name(self, name: str) -> Optional[Role]:
        with self._db.session() as s:
            record = s.query(RoleRecord).filter(RoleRecord.name == name).first()
            if record:
                return self.get(record.role_id)
            for r in s.query(RoleRecord).all():
                if name in (r.aliases or []):
                    return self.get(r.role_id)
            return None

    def add_white_painting_entry(self, role_id: str, entry: WhitePaintingEntry) -> None:
        with self._db.session() as s:
            importance_val = entry.importance.value if hasattr(entry.importance, "value") else str(entry.importance)
            record = WhitePaintingRecord(
                role_id=role_id,
                event_id=entry.event_id,
                role_summary=entry.role_summary,
                emotional_model=entry.emotional_model.model_dump(mode="json"),
                importance=importance_val,
                create_time=entry.create_time,
                memory_weight=entry.memory_weight,
            )
            s.add(record)
            s.commit()

    def get_white_painting(
        self,
        role_id: str,
        *,
        limit: int | None = None,
        min_memory_weight: float | None = None,
    ) -> list[WhitePaintingEntry]:
        with self._db.session() as s:
            q = (
                s.query(WhitePaintingRecord)
                .filter(WhitePaintingRecord.role_id == role_id)
                .order_by(WhitePaintingRecord.create_time)
            )
            if min_memory_weight is not None:
                q = q.filter(WhitePaintingRecord.memory_weight >= min_memory_weight)
            if limit:
                q = q.limit(limit)
            return [self._to_wp_entry(e) for e in q.all()]

    def save_semantic_card(self, card: SemanticCard) -> None:
        with self._db.session() as s:
            record = SemanticCardRecord(
                role_id=card.role_id,
                updated_at=card.updated_at,
                data=card.data,
            )
            s.merge(record)
            s.commit()

    def get_semantic_card(self, role_id: str) -> Optional[SemanticCard]:
        with self._db.session() as s:
            r = s.get(SemanticCardRecord, role_id)
            if not r:
                return None
            return SemanticCard(role_id=r.role_id, updated_at=r.updated_at, data=r.data or {})

    # ------------------------------------------------------------------
    @staticmethod
    def _build_role(
        rec: RoleRecord,
        entries: list[WhitePaintingRecord],
        card_rec: SemanticCardRecord | None,
    ) -> Role:
        card = None
        if card_rec:
            card = SemanticCard(role_id=card_rec.role_id, updated_at=card_rec.updated_at, data=card_rec.data or {})
        return Role(
            role_id=rec.role_id,
            name=rec.name,
            entity_type=rec.entity_type,
            aliases=rec.aliases or [],
            created_at=rec.created_at,
            white_painting=[RoleRepository._to_wp_entry(e) for e in entries],
            semantic_card=card,
        )

    @staticmethod
    def _to_wp_entry(e: WhitePaintingRecord) -> WhitePaintingEntry:
        em = EmotionalModel(**(e.emotional_model or {}))
        return WhitePaintingEntry(
            event_id=e.event_id,
            role_summary=e.role_summary,
            emotional_model=em,
            importance=e.importance,
            create_time=e.create_time,
            memory_weight=float(e.memory_weight or 0.0),
        )


# =====================================================================
# Metabolism state (shadow + unclosed events)
# =====================================================================

class MetabolismRepository:
    def __init__(self, db: Database):
        self._db = db

    def get_shadow(self) -> Shadow:
        with self._db.session() as s:
            record = s.query(ShadowRecord).first()
            if not record:
                return Shadow()
            return Shadow(content=record.content, updated_at=record.updated_at)

    def update_shadow(self, shadow: Shadow) -> None:
        with self._db.session() as s:
            record = s.query(ShadowRecord).first()
            if record:
                record.content = shadow.content
                record.updated_at = shadow.updated_at
            else:
                s.add(ShadowRecord(content=shadow.content, updated_at=shadow.updated_at))
            s.commit()

    def save_unclosed_event(self, event: UnclosedEvent) -> None:
        with self._db.session() as s:
            record = UnclosedEventRecord(
                id=event.id,
                content_fragments=event.content_fragments,
                identified_roles=event.identified_roles,
                logical_gaps=event.logical_gaps,
                created_at=event.created_at,
                updated_at=event.updated_at,
                last_hit_time=event.last_hit_time,
            )
            s.merge(record)
            s.commit()

    def get_unclosed_events(self) -> list[UnclosedEvent]:
        with self._db.session() as s:
            return [self._to_model(r) for r in s.query(UnclosedEventRecord).all()]

    def get_unclosed_event(self, event_id: str) -> Optional[UnclosedEvent]:
        with self._db.session() as s:
            r = s.get(UnclosedEventRecord, event_id)
            return self._to_model(r) if r else None

    def delete_unclosed_event(self, event_id: str) -> None:
        with self._db.session() as s:
            s.query(UnclosedEventRecord).filter(UnclosedEventRecord.id == event_id).delete()
            s.commit()

    @staticmethod
    def _to_model(r: UnclosedEventRecord) -> UnclosedEvent:
        return UnclosedEvent(
            id=r.id,
            content_fragments=r.content_fragments or [],
            identified_roles=r.identified_roles or [],
            logical_gaps=r.logical_gaps,
            created_at=r.created_at,
            updated_at=r.updated_at,
            last_hit_time=r.last_hit_time,
        )
