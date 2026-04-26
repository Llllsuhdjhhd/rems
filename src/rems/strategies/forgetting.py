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
        if not is_penalized:
            return ForgettingScore(is_penalized=False, retention=1.0, score=1.0)

        now = now or datetime.now()
        age_days = max((now - entry.create_time).total_seconds() / 86400, 0.0)
        half_life = self._config.wp_half_life_days * (
            self._config.ae_forgetting_multiplier
            if entry.memory_weight >= self._config.ae_high_threshold
            else 1.0
        )
        retention = math.exp(-0.693 * age_days / half_life)
        score = entry.memory_weight * 0.4 + retention * 0.6
        return ForgettingScore(is_penalized=True, retention=retention, score=score)
