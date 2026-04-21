"""Integration tests for the full REMS pipeline."""

from __future__ import annotations

from unittest.mock import patch


from rems.config import REMSConfig
from rems.pipeline import REMSPipeline
from .conftest import FakeEmbeddingFunction


class TestPipelineConstruction:
    def test_from_config(self, config: REMSConfig):
        """Pipeline can be constructed from config (with mock embedding to avoid downloads)."""
        with patch("rems.storage.vector_store.SentenceTransformerEmbeddingFunction") as mock_ef:
            mock_ef.return_value = FakeEmbeddingFunction()
            pipeline = REMSPipeline.from_config(config)

        assert pipeline.config is config
        assert pipeline.event_service is not None
        assert pipeline.role_service is not None
        assert pipeline.metabolism_service is not None
        assert pipeline.recall_service is not None
        assert pipeline.abstraction_service is not None


class TestQueryRole:
    def test_role_not_found(self, config: REMSConfig):
        with patch("rems.storage.vector_store.SentenceTransformerEmbeddingFunction") as mock_ef:
            mock_ef.return_value = FakeEmbeddingFunction()
            pipeline = REMSPipeline.from_config(config)

        result = pipeline.query_role("nonexistent")
        assert "error" in result

    def test_role_found_after_register(self, config: REMSConfig):
        with patch("rems.storage.vector_store.SentenceTransformerEmbeddingFunction") as mock_ef:
            mock_ef.return_value = FakeEmbeddingFunction()
            pipeline = REMSPipeline.from_config(config)

        pipeline.role_service.register_role("TestUser")
        result = pipeline.query_role("TestUser")
        assert "error" not in result
        assert result["role"]["name"] == "TestUser"
