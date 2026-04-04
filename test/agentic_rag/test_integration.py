"""
Integration tests for Agentic RAG.

These tests require external dependencies (Tavily API, LLM) and should
be marked with @pytest.mark.integration.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Force reimport
if "agentic_rag" in sys.modules:
    del sys.modules["agentic_rag"]


@pytest.mark.integration
class TestAgenticRAGIntegration:
    """Integration test suite for Agentic RAG."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        from agentic_rag.state import Document

        return [
            Document(
                page_content="LangChain is a framework for developing applications powered by language models.",
                metadata={"source": "langchain_docs.txt", "page": 1},
            ),
            Document(
                page_content="Python is a high-level programming language.",
                metadata={"source": "python_docs.txt", "page": 1},
            ),
        ]

    def test_config_to_dict_conversion(self, _sample_documents):
        """Test configuration serialization."""
        from src.agentic_rag.config import AgenticRAGConfig

        config = AgenticRAGConfig(evaluation_threshold=0.8)
        config_dict = config.to_dict()

        assert config_dict["evaluation_threshold"] == 0.8
        assert "max_search_iterations" in config_dict

    def test_state_serialization(self, _sample_documents):
        """Test agent state serialization."""
        from src.agentic_rag.state import AgentState, Message, MessageRole

        state = AgentState(
            query="Test query",
            messages=[Message(role=MessageRole.USER, content="Hello")],
        )

        state_dict = state.to_dict()
        restored_state = AgentState.from_dict(state_dict)

        assert restored_state.query == state.query
        assert len(restored_state.messages) == len(state.messages)
