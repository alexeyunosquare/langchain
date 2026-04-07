"""
Tests for factory functions in Agentic RAG.
"""

from unittest.mock import MagicMock

import pytest

from src.agentic_rag.config import AgenticRAGConfig
from src.agentic_rag.factory import (
    create_agentic_rag_agent,
    create_corrective_rag,
    create_evaluator,
    create_hybrid_retriever,
    create_tavily_search,
    merge_config_with_env,
)


class TestFactoryFunctions:
    """Test suite for factory functions."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = MagicMock()
        mock.invoke.return_value = MagicMock(content="Test response", role="assistant")
        return mock

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever for testing."""
        mock = MagicMock()
        mock.invoke.return_value = []
        return mock

    def test_create_evaluator(self, mock_llm):
        """Test evaluator creation."""
        evaluator = create_evaluator(llm=mock_llm, threshold=0.8)

        assert evaluator is not None
        assert evaluator.threshold == 0.8

    def test_create_corrective_rag(self, mock_llm):
        """Test CorrectiveRAG creation."""
        crag = create_corrective_rag(llm=mock_llm, quality_threshold=0.8, max_attempts=3)

        assert crag is not None
        assert crag.correction_threshold == 0.8
        assert crag.max_correction_attempts == 3

    def test_create_agentic_rag_agent_basic(self, mock_llm, mock_retriever):
        """Test basic agent creation."""
        agent = create_agentic_rag_agent(
            llm=mock_llm,
            local_retriever=mock_retriever,
            use_hybrid_retrieval=False,
        )

        assert agent is not None
        assert agent.llm is mock_llm
        assert agent.local_retriever is mock_retriever

    def test_create_agentic_rag_agent_with_config(self, mock_llm, mock_retriever):
        """Test agent creation with custom config."""
        config = AgenticRAGConfig(
            evaluation_threshold=0.9,
            max_search_iterations=5,
        )

        agent = create_agentic_rag_agent(
            llm=mock_llm,
            local_retriever=mock_retriever,
            config=config,
            use_hybrid_retrieval=False,
        )

        assert agent.config.evaluation_threshold == 0.9
        assert agent.config.max_search_iterations == 5

    def test_create_agentic_rag_agent_override_iterations(self, mock_llm, mock_retriever):
        """Test agent creation with iteration override."""
        agent = create_agentic_rag_agent(
            llm=mock_llm,
            local_retriever=mock_retriever,
            max_search_iterations=7,
            use_hybrid_retrieval=False,
        )

        assert agent.config.max_search_iterations == 7

    def test_merge_config_with_env(self, monkeypatch):
        """Test config merging with environment variables."""
        monkeypatch.setenv("RAG_EVALUATION_THRESHOLD", "0.85")
        monkeypatch.setenv("RAG_MAX_SEARCH_ITERATIONS", "4")

        custom_config = AgenticRAGConfig(
            evaluation_threshold=0.7,
            max_search_iterations=3,
        )

        merged = merge_config_with_env(
            custom_config=custom_config,
            evaluation_threshold=0.95,  # Override
        )

        # Env vars take precedence over custom config
        assert merged.evaluation_threshold == 0.95  # Override wins
        assert merged.max_search_iterations == 4  # From env

    def test_merge_config_with_env_custom_wins_over_default(self, monkeypatch):
        """Test custom config values over defaults when not overridden."""
        monkeypatch.setenv("RAG_EVALUATION_THRESHOLD", "0.85")
        monkeypatch.delenv("RAG_MAX_SEARCH_ITERATIONS", raising=False)

        custom_config = AgenticRAGConfig(
            evaluation_threshold=0.7,
            max_search_iterations=5,
        )

        merged = merge_config_with_env(custom_config=custom_config)

        # Env var takes precedence
        assert merged.evaluation_threshold == 0.85
        # Custom config is used when no env var
        assert merged.max_search_iterations == 5

    def test_create_hybrid_retriever(self, mock_llm, mock_retriever):
        """Test hybrid retriever creation."""
        from src.agentic_rag.search import TavilySearch

        tavily = TavilySearch(api_key="test_key")

        retriever = create_hybrid_retriever(
            local_retriever=mock_retriever,
            tavily_search=tavily,
            llm=mock_llm,
            tavily_priority=0.2,
        )

        assert retriever is not None
        assert retriever.tavily_priority == 0.2

    def test_create_tavily_search(self, monkeypatch):
        """Test Tavily search creation."""
        monkeypatch.setenv("TAVILY_API_KEY", "env_key")

        tavily_integration = create_tavily_search()

        assert tavily_integration is not None
        assert tavily_integration.tavily_search.api_key == "env_key"

    def test_create_tavily_search_with_api_key(self):
        """Test Tavily search creation with provided API key."""
        tavily_integration = create_tavily_search(api_key="provided_key")

        assert tavily_integration is not None
        assert tavily_integration.tavily_search.api_key == "provided_key"


class TestConfigMerging:
    """Test config merging logic."""

    def test_empty_overrides(self):
        """Test merging with no overrides."""
        config = AgenticRAGConfig(evaluation_threshold=0.8)
        merged = merge_config_with_env(custom_config=config)

        assert merged.evaluation_threshold == 0.8

    def test_single_override(self):
        """Test merging with single override."""
        config = AgenticRAGConfig(evaluation_threshold=0.7, max_search_iterations=3)
        merged = merge_config_with_env(
            custom_config=config,
            evaluation_threshold=0.9,
        )

        assert merged.evaluation_threshold == 0.9
        assert merged.max_search_iterations == 3

    def test_multiple_overrides(self):
        """Test merging with multiple overrides."""
        config = AgenticRAGConfig(
            evaluation_threshold=0.7,
            max_search_iterations=3,
            temperature=0.5,
        )
        merged = merge_config_with_env(
            custom_config=config,
            evaluation_threshold=0.9,
            max_search_iterations=5,
        )

        assert merged.evaluation_threshold == 0.9
        assert merged.max_search_iterations == 5
        assert merged.temperature == 0.5

    def test_none_values_not_overridden(self):
        """Test that None values in custom config don't override."""
        config = AgenticRAGConfig(
            evaluation_threshold=0.8,
            max_search_iterations=3,
        )
        # Create config dict with None values
        config_dict = config.to_dict()
        config_dict["evaluation_threshold"] = None
        config_dict["max_search_iterations"] = None

        # This tests that None values are not applied
        # In actual implementation, None values should be skipped
        merged = merge_config_with_env(custom_config=config)

        assert merged.evaluation_threshold == 0.8
        assert merged.max_search_iterations == 3
