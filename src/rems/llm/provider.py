from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

# OpenAI 兼容 Chat Completions 封装；按 task_type 选择模型名（见 config.llm.task_models）。

from openai import OpenAI

from ..config import REMSConfig
from .metrics import LLMInvocationMetrics

logger = logging.getLogger(__name__)


class LLMProvider:
    """Thin wrapper around the OpenAI-compatible chat API.

    Routes each *task_type* to the model specified in ``config.llm.task_models``.

    对兼容 OpenAI 协议的 Chat Completions 做薄封装；根据 *task_type*（如 ``summary``、
    ``boundary_detection``）在 ``TaskModelMapping`` 中选取具体模型，供各 Skill 复用。
    ``complete_json`` 在文本回复中抽取 JSON（含 markdown 代码围栏容错）。
    """

    def __init__(self, config: REMSConfig):
        self.config = config
        self._client = OpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            max_retries=config.llm.max_retries,
        )
        # 每次 ``complete`` 追加一条，供观测账单、延迟与测试报告（非流式 usage）。
        self._invocations: list[LLMInvocationMetrics] = []

    def _get_model(self, task_type: str) -> str:
        mapping = self.config.llm.task_models
        return getattr(mapping, task_type, None) or mapping.default

    def invocation_history(self) -> list[LLMInvocationMetrics]:
        """Copy of all completed calls since this provider was constructed."""
        return list(self._invocations)

    def clear_invocation_history(self) -> None:
        """Drop recorded metrics (e.g. before a new benchmark window)."""
        self._invocations.clear()

    @staticmethod
    def _usage_int(usage: Any, name: str) -> int | None:
        if usage is None:
            return None
        v = getattr(usage, name, None)
        return int(v) if v is not None else None

    # ------------------------------------------------------------------
    def complete(
        self,
        task_type: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> str:
        model = self._get_model(task_type)
        temp = temperature if temperature is not None else self.config.llm.temperature

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        self._invocations.append(
            LLMInvocationMetrics(
                task_type=task_type,
                model=model,
                latency_ms=round(latency_ms, 3),
                prompt_tokens=self._usage_int(usage, "prompt_tokens"),
                completion_tokens=self._usage_int(usage, "completion_tokens"),
                total_tokens=self._usage_int(usage, "total_tokens"),
            ),
        )
        logger.debug(
            "LLM [%s/%s] latency_ms=%.2f usage=%s",
            task_type,
            model,
            latency_ms,
            usage,
        )
        return content

    # ------------------------------------------------------------------
    def complete_json(
        self,
        task_type: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        raw = self.complete(task_type, messages, temperature=temperature or 0.1)
        return self._extract_json(raw)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise ValueError(f"Cannot parse JSON from LLM output: {text[:300]}")
