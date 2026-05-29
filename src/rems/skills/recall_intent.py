from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import REMSConfig


@dataclass(frozen=True)
class RecallIntent:
    weights: tuple[float, float, float]
    label: str


_ENTITY_PATTERNS = re.compile(
    r"谁|人物|角色|轨迹|做过什么|干什么|where|who|character|person",
    re.I,
)
_EMO_PATTERNS = re.compile(
    r"感觉|心情|情绪|害怕|开心|难过|feel|emotion|mood|afraid|happy|sad",
    re.I,
)


class RecallIntentClassifier:
    """Rule-based intent router for tri-band weights (§4.4.0 v1)."""

    def __init__(self, config: REMSConfig):
        self._config = config

    def classify(self, query: str) -> RecallIntent:
        tb = self._config.tri_band
        if _ENTITY_PATTERNS.search(query):
            return RecallIntent(weights=tb.weight_entity, label="entity_trace")
        if _EMO_PATTERNS.search(query):
            return RecallIntent(weights=tb.weight_emotion, label="emotion")
        return RecallIntent(weights=tb.weight_fact, label="fact")
