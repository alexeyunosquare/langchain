"""Tests for Agentic RAG configuration."""

import pytest

from src.agentic_rag.config import AgenticRAGConfig


class TestAgenticRAGConfig:
    """Test cases for AgenticRAGConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgenticRAGConfig()
        assert config.temperature == 0.7
        assert config.top_k == 5
        assert config.similarity_threshold == 0.7
        assert config.max_iterations == 3
        assert config.retriever_type == "vectorstore"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AgenticRAGConfig(
            temperature=0.9,
            top_k=10,
            similarity_threshold=0.8,
            max_iterations=5,
            retriever_type="web",
        )
        assert config.temperature == 0.9
        assert config.top_k == 10
        assert config.similarity_threshold == 0.8
        assert config.max_iterations == 5
        assert config.retriever_type == "web"

    def test_config_to_dict(self):
        """Test config serialization."""
        config = AgenticRAGConfig(temperature=0.8)
        config_dict = config.to_dict()
        assert config_dict["temperature"] == 0.8
        assert config_dict["top_k"] == 5

    def test_config_from_dict(self):
        """Test config deserialization."""
        config_dict = {
            "temperature": 0.95,
            "top_k": 15,
            "similarity_threshold": 0.85,
        }
        config = AgenticRAGConfig.from_dict(config_dict)
        assert config.temperature == 0.95
        assert config.top_k == 15
        assert config.similarity_threshold == 0.85
        assert config.max_iterations == 3

    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            AgenticRAGConfig(temperature=1.5)

        with pytest.raises(ValueError):
            AgenticRAGConfig(top_k=-1)
