from __future__ import annotations

import logging
from typing import Iterable

# Emotion & Adaptation (EMA) 动态演化（白皮书 2.5）。
# 本轮不做 OCC Appraisal：情感基准仍由 role_extraction 的 LLM 产出；EMAEvolver 负责在时间线上做
# 指数平滑（历史心境 + 新事件），并把事件级 AE 映射为 ``Event.activation_energy``（记忆初始值硬绑定）。
# 在未来接入 OCC 时，只需把 LLM 粗估替换为基于目标/偏好的规则化评估，签名保持不变。

from ..config import REMSConfig
from ..models.event import EmotionalModel, Event, EventRoleEntry, Klesha, Vedana
from ..storage.repository import RoleRepository

logger = logging.getLogger(__name__)


_VEDANA_FIELDS = ("joy", "suffering", "happiness", "worry", "equanimity")
_KLESHA_FIELDS = ("greed", "anger", "ignorance", "pride", "doubt", "wrong_view")


class EMAEvolver:
    """Evolve per-role emotion state via exponential smoothing and compute ``activation_energy``.

    针对一个 ``Event`` 的 ``role_list`` 中每个角色条目：
    1. 从仓储读取该 ``role_id`` 白描尾部 ``ema_history_window`` 条，聚合历史均值作为"基线心境"。
    2. 以 ``alpha = ema_smoothing_alpha`` 对"历史均值"与"当前快照情感"做指数平滑，
       写回 ``entry.emotional_model`` —— 这就是"情感随时间的动态演化"（白皮书 2.5 EMA 机制）。
    3. 事件级：以演化后的 ``event.affective_energy`` × ``activation_energy_gain`` 写入
       ``event.activation_energy``；高于 ``ae_high_threshold`` 视为重大事件硬绑定，施加 1.5× 增强。
    """

    def __init__(self, config: REMSConfig, role_repo: RoleRepository):
        self._config = config
        self._repo = role_repo

    # ------------------------------------------------------------------
    def evolve_event(self, event: Event) -> None:
        """Mutate ``event`` in place: adapt each role's emotion + set ``activation_energy``.

        同步演化所有角色条目的情感量化与事件激活能量；抽象事件跳过（抽象事件 role_list 为模型生成的
        "风格/趋势"而非具体快照，不做 EMA 叠加）。
        """
        if event.is_abstract:
            return

        for entry in event.role_list:
            self._evolve_role_entry(entry)

        ae = event.affective_energy
        gain = self._config.activation_energy_gain
        if ae >= self._config.ae_high_threshold:
            gain *= 1.5  # 重大事件硬绑定：抗遗忘与回忆权重的初值加成（白皮书 2.5）。
        event.activation_energy = min(ae * gain, 1.0)

    # ------------------------------------------------------------------
    def _evolve_role_entry(self, entry: EventRoleEntry) -> None:
        try:
            recent = self._repo.get_white_painting(
                entry.role_id, limit=self._config.ema_history_window
            )
        except Exception:
            logger.debug("EMA: no history for %s", entry.role_id, exc_info=True)
            recent = []

        if not recent:
            return  # 新角色无历史心境，保留当前 LLM 快照作为首个基线。

        baseline = self._average_emotion(e.emotional_model for e in recent)
        smoothed = self._smooth(baseline, entry.emotional_model, self._config.ema_smoothing_alpha)
        entry.emotional_model = smoothed

    # ------------------------------------------------------------------
    @staticmethod
    def _average_emotion(models: Iterable[EmotionalModel]) -> EmotionalModel:
        models = list(models)
        n = max(len(models), 1)
        v_sum = {f: 0.0 for f in _VEDANA_FIELDS}
        k_sum = {f: 0.0 for f in _KLESHA_FIELDS}
        for m in models:
            for f in _VEDANA_FIELDS:
                v_sum[f] += float(getattr(m.vedana, f, 0.0))
            for f in _KLESHA_FIELDS:
                k_sum[f] += float(getattr(m.klesha, f, 0.0))
        return EmotionalModel(
            vedana=Vedana(**{f: v_sum[f] / n for f in _VEDANA_FIELDS}),
            klesha=Klesha(**{f: k_sum[f] / n for f in _KLESHA_FIELDS}),
        )

    @staticmethod
    def _smooth(history: EmotionalModel, current: EmotionalModel, alpha: float) -> EmotionalModel:
        """Exponential smoothing: ``smoothed = alpha * current + (1 - alpha) * history``.

        ``alpha`` 越大越偏向当前事件，越小越延续历史心境；默认 0.4 给历史更多权重，
        避免单次强刺激把角色长期心境一次性推翻（白皮书 2.5 所述"持续评估目标进度与现实预期的差异"）。
        """
        beta = 1.0 - alpha
        v = {
            f: alpha * float(getattr(current.vedana, f, 0.0))
               + beta * float(getattr(history.vedana, f, 0.0))
            for f in _VEDANA_FIELDS
        }
        k = {
            f: alpha * float(getattr(current.klesha, f, 0.0))
               + beta * float(getattr(history.klesha, f, 0.0))
            for f in _KLESHA_FIELDS
        }
        return EmotionalModel(vedana=Vedana(**v), klesha=Klesha(**k))
