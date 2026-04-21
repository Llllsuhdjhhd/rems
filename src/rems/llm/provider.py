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
            timeout=300.0,
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
        # 1. Isolating JSON from markdown fences (handles leading/trailing LLM talk)
        fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        
        # 2. Basic cleanup for common artifacts
        def clean_basic(s: str) -> str:
            # Trailing commas: {"a":1,} -> {"a":1}
            s = re.sub(r",\s*([\]}])", r"\1", s)
            return s.strip()

        # 3. Structural recovery (Handles unescaped quotes in narrative content)
        def structural_fix(s: str) -> str:
            # Look for property patterns: "key": "value"
            # This regex identifies values that might contain unescaped quotes by 
            # greedy-matching until we see the "next" property or the end of object.
            # Warning: heuristic, but powerful for LLM narrative outputs.
            # Example fix: "content": "He said "Hi"" -> "content": "He said \"Hi\""
            
            # Identify the outermost JSON block
            start = s.find("{")
            end = s.rfind("}") + 1
            if start == -1: return s
            json_str = s[start:end]
            
            # Simple escape of internal quotes: matches " inside a value string
            # This is complex in pure regex, so we use a more targeted approach if loads fails.
            return json_str

        candidate = clean_basic(text)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Fallback 1: Truncate to outermost {}
            start = candidate.find("{")
            end = candidate.rfind("}") + 1
            if start >= 0 and end > start:
                inner = candidate[start:end]
                try:
                    return json.loads(clean_basic(inner))
                except json.JSONDecodeError:
                    # Fallback 2: Structural Scraper (Regex)
                    # This targets specific keys to rebuild the dictionary manually
                    # Useful when LLM fails to escape internal quotes.
                    recovered = {}
                    # Pattern for string values: "key": "value"
                    # Matches "key" : " ... any text ... " followed by , or }
                    # Uses negative lookahead to not stop at internal escaped quotes (though here we handle unescaped too)
                    for key in ["remaining_shadow", "summary", "logical_gaps", "content"]:
                        match = re.search(f'"{key}"\\s*:\\s*"(.*?)"\\s*(?:,|}})', inner, re.DOTALL)
                        if match:
                            recovered[key] = match.group(1).replace('\\"', '"') # Basic unescape
                    
                    # Pattern for list of objects (like completed_events or new_unclosed)
                    for key in ["completed_events", "new_unclosed", "roles", "sealed_events"]:
                        # Extract the array content between [...]
                        match = re.search(f'"{key}"\\s*:\\s*\\[(.*?)\\]\\s*(?:,|}})', inner, re.DOTALL)
                        if match:
                            array_str = match.group(1).strip()
                            if not array_str:
                                recovered[key] = []
                                continue
                            
                            # Heuristic: split objects by "}," or "}\s*," pattern
                            items = []
                            obj_matches = re.finditer(r"\{(.*?)\}", array_str, re.DOTALL)
                            for m in obj_matches:
                                obj_inner = m.group(1)
                                obj_recovered = {}
                                # Recover simple keys inside the object
                                for subkey in ["content", "end_snippet", "continuation_of", "name", "role_id", "entity_type", "importance", "logical_gaps"]:
                                    # Very loose match for "key": "value"
                                    # This handles even garbage around the quotes
                                    sub_match = re.search(f'"{subkey}"\\s*:\\s*"(.*?)"', obj_inner, re.DOTALL)
                                    if sub_match:
                                        # Fix: replace internal unescaped quotes if they exist 
                                        # (though here we just captured everything between the first and last ")
                                        obj_recovered[subkey] = sub_match.group(1).replace('\\"', '"')
                                if obj_recovered:
                                    items.append(obj_recovered)
                            recovered[key] = items
                    
                    if recovered:
                        logger.warning("JSON recovered via structural scraper. HEURISTIC mode used for objects.")
                        return recovered

                    with open("failed_llm_json.txt", "w", encoding="utf-8") as f:
                        f.write(text)
                    raise ValueError(f"JSON Recovery failed. Source saved to failed_llm_json.txt. Snippet: {inner[:200]}")
            
            with open("failed_llm_json.txt", "w", encoding="utf-8") as f:
                f.write(text)
            raise ValueError(f"No JSON structure detected. Snippet: {text[:200]}")
