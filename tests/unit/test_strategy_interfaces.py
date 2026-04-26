from __future__ import annotations

from datetime import datetime, timedelta

from rems.config import REMSConfig
from rems.llm.prompt_registry import PromptRegistry, PromptTemplate
from rems.models.event import Event, EventRoleEntry, Importance
from rems.models.role import WhitePaintingEntry
from rems.strategies.forgetting import DefaultWhitePaintingRetentionStrategy
from rems.strategies.recall import DefaultRecallScoringStrategy, DefaultSummaryTierPolicy


def test_default_recall_scoring_exposes_breakdown():
    config = REMSConfig()
    strategy = DefaultRecallScoringStrategy(config)
    event = Event(
        content_raw="张三推进项目。",
        role_list=[EventRoleEntry(role_id="ROL-1", importance=Importance.A)],
        activation_energy=0.2,
    )

    breakdown = strategy.score(event, cosine_sim=0.7)

    assert breakdown.cosine == 0.7
    assert breakdown.role_boost == 0.2
    assert 0.0 <= breakdown.total <= 1.0


def test_default_summary_tier_policy_is_role_aware():
    config = REMSConfig()
    policy = DefaultSummaryTierPolicy(config)
    event = Event(
        content_raw="张三推进项目。",
        role_list=[EventRoleEntry(role_id="ROL-1", importance=Importance.A)],
    )

    focused = policy.tier_offset(event, {"ROL-1"})
    unfocused = policy.tier_offset(event, {"ROL-other"})

    assert focused < unfocused


def test_default_forgetting_strategy_penalizes_old_entries():
    config = REMSConfig()
    strategy = DefaultWhitePaintingRetentionStrategy(config)
    entry = WhitePaintingEntry(
        event_id="EVT-1",
        role_summary="旧白描",
        create_time=datetime.now() - timedelta(days=120),
        memory_weight=0.1,
    )

    unpenalized = strategy.score(entry, is_penalized=False)
    penalized = strategy.score(entry, is_penalized=True, now=datetime.now())

    assert unpenalized.score == 1.0
    assert penalized.score < 1.0
    assert penalized.is_penalized is True


def test_prompt_registry_supports_versions_and_defaults():
    registry = PromptRegistry()
    v1 = PromptTemplate(task_type="abstraction", version="v1", system="s1", user="u1")
    v2 = PromptTemplate(task_type="abstraction", version="v2", system="s2", user="u2")

    registry.register(v1, make_default=True)
    registry.register(v2)

    assert registry.get("abstraction").version == "v1"
    assert registry.get("abstraction", "v2").system == "s2"
    assert registry.default_version("abstraction") == "v1"
