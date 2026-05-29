from __future__ import annotations

from rems.config import REMSConfig
from rems.embedding.tri_band import TriBandEncoder
from rems.models.event import Event, EventRoleEntry, Importance
from rems.storage.database import Database
from rems.storage.repository import EventRepository, RoleRepository
from rems.models.role import Role


def test_abstract_ent_uses_leaf_roles(config, db):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    role_repo.save(Role(role_id="ROL-a", name="Alice", aliases=["阿丽"]))
    basic = Event(
        content_raw="Alice met Bob",
        role_list=[EventRoleEntry(role_id="ROL-a", importance=Importance.A)],
    )
    event_repo.save(basic)
    abstract = Event(
        content_raw="pattern",
        is_abstract=True,
        source_events=[basic.event_id],
        role_list=[],
    )
    event_repo.save(abstract)

    enc = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    ent_text = enc._entity_text(abstract)
    assert "ROL-a" in ent_text
    assert "Alice" in ent_text
    vecs = enc.encode(abstract)
    assert len(vecs.vector_ent) == config.tri_band.vector_dim_ent


def test_basic_event_ent_from_role_list(config, db):
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    role_repo.save(Role(role_id="ROL-b", name="Bob"))
    ev = Event(
        content_raw="Bob spoke",
        role_list=[EventRoleEntry(role_id="ROL-b", importance=Importance.B)],
    )
    enc = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
    assert "Bob" in enc._entity_text(ev)
