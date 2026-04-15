"""Shared fixtures: in-memory DB, mock LLM provider, temp vector store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from rems.config import REMSConfig
from rems.llm.provider import LLMProvider
from rems.storage.database import Database


# ── Config ────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def config(tmp_dir: Path) -> REMSConfig:
    return REMSConfig(
        context_window=4096,
        chars_per_token=1.0,
        llm={"api_key": "test-key", "base_url": "http://localhost:11434/v1"},
        storage={
            "database_url": "sqlite:///:memory:",
            "chromadb_path": str(tmp_dir / "chroma"),
        },
    )


# ── Database ──────────────────────────────────────────────────────────

@pytest.fixture()
def db(config: REMSConfig) -> Database:
    database = Database(config.storage.database_url)
    database.create_tables()
    return database


# ── Mock LLM ──────────────────────────────────────────────────────────

class FakeLLM(LLMProvider):
    """LLMProvider that returns pre-configured responses without network calls."""

    def __init__(self, config: REMSConfig):
        self.config = config
        self._client = MagicMock()
        self._responses: list[str] = []
        self._call_count = 0

    def push_response(self, obj: dict[str, Any] | str) -> None:
        if isinstance(obj, dict):
            self._responses.append(json.dumps(obj, ensure_ascii=False))
        else:
            self._responses.append(obj)

    def complete(self, task_type: str, messages: list[dict[str, str]], **kw: Any) -> str:
        if self._responses:
            resp = self._responses.pop(0)
        else:
            resp = json.dumps({"summary": "mock summary", "char_count": 12})
        self._call_count += 1
        return resp


@pytest.fixture()
def fake_llm(config: REMSConfig) -> FakeLLM:
    return FakeLLM(config)


class FakeEmbeddingFunction:
    """Minimal embedding function with the signature chromadb expects."""

    is_legacy = False

    def __call__(self, input):  # noqa: A002
        return [[0.0] * 4 for _ in input]

    def name(self) -> str:
        return "default"
