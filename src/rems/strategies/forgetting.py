from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..config import REMSConfig
from ..models.role import WhitePaintingEntry


@dataclass(frozen=True)
class ForgettingScore:
    """Observable retention score for a white-painting entry."""

    is_penalized: bool
    retention: float
    score: float
    effective_forgetting: float
    is_silenced: bool


class WhitePaintingRetentionStrategy(Protocol):
    """Score white-painting entries for capacity-aware forgetting."""

    def score(
        self,
        entry: WhitePaintingEntry,
        *,
        is_penalized: bool,
        now: datetime | None = None,
    ) -> ForgettingScore:
        """Return a retention score for one timeline entry."""
        ...


class DefaultWhitePaintingRetentionStrategy:
    """Current FIFO + time half-life + AE policy as a swappable strategy."""

    def __init__(self, config: REMSConfig):
        self._config = config

    def score(
        self,
        entry: WhitePaintingEntry,
        *,
        is_penalized: bool,
        now: datetime | None = None,
    ) -> ForgettingScore:
        now = now or datetime.now()

        # 计算动态遗忘因子
        age_since_access = max((now - getattr(entry, "last_accessed_time", entry.create_time)).total_seconds() / 86400, 0.0)
        half_life = self._config.wp_half_life_days * (
            self._config.ae_forgetting_multiplier
            if entry.memory_weight >= self._config.ae_high_threshold
            else 1.0
        )
        forgetting_decay = math.exp(-0.693 * age_since_access / half_life)
        effective_forgetting = float(getattr(entry, "forgetting_factor", 1.0)) * forgetting_decay
        is_silenced = effective_forgetting < 0.02

        if not is_penalized:
            return ForgettingScore(
                is_penalized=False, 
                retention=1.0, 
                score=1.0, 
                effective_forgetting=effective_forgetting, 
                is_silenced=is_silenced
            )

        age_days = max((now - entry.create_time).total_seconds() / 86400, 0.0)
        retention = math.exp(-0.693 * age_days / half_life)
        score = entry.memory_weight * 0.4 + retention * 0.6
        return ForgettingScore(
            is_penalized=True, 
            retention=retention, 
            score=score * effective_forgetting,  # 动态遗忘因子影响最终评分
            effective_forgetting=effective_forgetting,
            is_silenced=is_silenced
        )
