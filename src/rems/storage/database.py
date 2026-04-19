from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

# SQLAlchemy ORM：事件、角色、白描条目、残影、未完成事件、语义卡片的持久化表定义与 Database 门面。
# 领域含义见《REMS 记忆系统规范解析》第 1–2 章与第 4.1 节。

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------
# ORM table definitions
# ------------------------------------------------------------------

class EventRecord(Base):
    __tablename__ = "events"
    # 与 models.event.Event 字段一一对应，JSON 列存 summaries、role_list、source_events 等。

    event_id = Column(String, primary_key=True)
    create_time = Column(DateTime, nullable=False)
    content_raw = Column(Text, nullable=False)
    summaries = Column(JSON, default=dict)
    summary_lengths = Column(JSON, default=dict)
    actual_max_level = Column(Integer, default=0)
    role_list = Column(JSON, default=list)
    is_abstract = Column(Boolean, default=False)
    is_abstracted = Column(Boolean, default=False)
    status = Column(String, default="active")
    decoration = Column(Text, nullable=True)
    insight = Column(Text, nullable=True)
    event_length = Column(Integer, default=0)
    abstraction_level = Column(Integer, nullable=True)
    source_events = Column(JSON, nullable=True)
    is_tombstoned = Column(Boolean, default=False)
    activation_energy = Column(Float, default=0.0)  # 白皮书 2.5 记忆初始值硬绑定


class RoleRecord(Base):
    __tablename__ = "roles"

    role_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    entity_type = Column(String, default="person")
    aliases = Column(JSON, default=list)
    created_at = Column(DateTime, nullable=False)


class WhitePaintingRecord(Base):
    __tablename__ = "white_painting_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    role_id = Column(String, ForeignKey("roles.role_id"), nullable=False, index=True)
    event_id = Column(String, ForeignKey("events.event_id"), nullable=False)
    role_summary = Column(Text, nullable=False)
    emotional_model = Column(JSON, default=dict)
    importance = Column(String, default="C")
    create_time = Column(DateTime, nullable=False)
    memory_weight = Column(Float, default=0.0)


class ShadowRecord(Base):
    __tablename__ = "shadow"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False, default="")
    updated_at = Column(DateTime, nullable=False)


class UnclosedEventRecord(Base):
    __tablename__ = "unclosed_events"

    id = Column(String, primary_key=True)
    content_fragments = Column(JSON, default=list)
    identified_roles = Column(JSON, default=list)
    logical_gaps = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    last_hit_time = Column(DateTime, nullable=False)


class SemanticCardRecord(Base):
    __tablename__ = "semantic_cards"

    role_id = Column(String, ForeignKey("roles.role_id"), primary_key=True)
    updated_at = Column(DateTime, nullable=False)
    data = Column(JSON, default=dict)


# ------------------------------------------------------------------
# Database facade
# ------------------------------------------------------------------

import json
from functools import partial

# ------------------------------------------------------------------
# Database facade
# ------------------------------------------------------------------

class Database:
    def __init__(self, url: str):
        # 强制 json_serializer 使用 ensure_ascii=False，确保中文在 SQLite 数据库中以明文存储
        self.engine = create_engine(
            url, 
            echo=False, 
            json_serializer=partial(json.dumps, ensure_ascii=False)
        )
        self._session_factory = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._session_factory()
        try:
            yield s
        finally:
            s.close()
