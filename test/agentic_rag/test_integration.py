"""
Integration tests for Agentic RAG.

These tests require external dependencies (Tavily API, LLM) and should
be marked with @pytest.mark.integration.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

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
        from src.agentic_rag.state import Document

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

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        from langchain_core.messages import BaseMessage

        mock = MagicMock()
        mock.invoke.return_value = BaseMessage(
            content="This is a valid answer based on the documents.",
            role="assistant",
        )
        return mock

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever for testing."""
        from langchain_core.retrievers import BaseRetriever

        mock = MagicMock(spec=BaseRetriever)
        mock.invoke.return_value = []
        return mock

    @pytest.mark.integration
    def test_config_to_dict_conversion(self):
        """Test configuration serialization."""
        from src.agentic_rag.config import AgenticRAGConfig

        config = AgenticRAGConfig(evaluation_threshold=0.8)
        config_dict = config.to_dict()

        assert config_dict["evaluation_threshold"] == 0.8
        assert "max_search_iterations" in config_dict

    @pytest.mark.integration
    def test_state_serialization(self):
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

    @pytest.mark.integration
    def test_factory_create_agent(self, mock_llm, mock_retriever):
        """Test factory function creates agent correctly."""
        from src.agentic_rag.factory import create_agentic_rag_agent

        agent = create_agentic_rag_agent(
            llm=mock_llm,
            local_retriever=mock_retriever,
            use_hybrid_retrieval=False,
        )

        assert agent is not None
        assert agent.llm is mock_llm
        assert agent.local_retriever is mock_retriever

    @pytest.mark.integration
    def test_validation_result_parsing(self, mock_llm):
        """Test validation result parsing from LLM response."""
        from src.agentic_rag.corrective import (
            AnswerValidator,
            ValidationDetail,
            ValidationResult,
            ValidationStatus,
        )

        # Setup mock to return valid JSON
        mock_llm.invoke.return_value = MagicMock(
            content='{"status": "valid", "quality_score": 0.9, "validation_details": [], "issues": [], "corrective_action": "none"}'
        )

        validator = AnswerValidator(llm=mock_llm)
        result = validator.validate(
            answer="Test answer",
            documents=[],
            query="Test query",
        )

        assert isinstance(result, ValidationResult)
        assert result.status == ValidationStatus.VALID
        assert result.quality_score == 0.9

    @pytest.mark.integration
    def test_hallucination_detection(self, mock_llm):
        """Test hallucination detection in CorrectiveRAG."""
        from src.agentic_rag.corrective import CorrectiveRAG
        from src.agentic_rag.state import Document

        # Setup mock to indicate hallucination
        mock_llm.invoke.return_value = MagicMock(
            content='{"status": "hallucinated", "quality_score": 0.3, "validation_details": [], "issues": ["Answer contains unsupported claims"], "corrective_action": "rephrase"}'
        )

        crag = CorrectiveRAG(llm=mock_llm)
        is_hallucinated, confidence = crag.check_hallucination(
            answer="Hallucinated answer",
            documents=[
                Document(
                    page_content="Different content",
                    metadata={"source": "doc.txt"},
                )
            ],
        )

        assert isinstance(is_hallucinated, bool)
        assert 0 <= confidence <= 1

    @pytest.mark.integration
    def test_correction_flow(self, mock_llm):
        """Test complete correction workflow."""
        from src.agentic_rag.corrective import CorrectiveRAG
        from src.agentic_rag.state import Document

        # Setup LLM to return corrected answer
        mock_llm.invoke.return_value = MagicMock(
            content='{"corrected_answer": "Corrected answer based on documents."}'
        )

        crag = CorrectiveRAG(llm=mock_llm)
        documents = [
            Document(page_content="Context", metadata={"source": "doc.txt"})
        ]

        corrected_answer = crag.validate_and_correct(
            answer="Original answer",
            documents=documents,
            query="Test query",
        )

        assert isinstance(corrected_answer, str)
        assert len(corrected_answer) > 0

    @pytest.mark.integration
    def test_quality_score_calculation(self, mock_llm):
        """Test quality score calculation."""
        from src.agentic_rag.corrective import CorrectiveRAG
        from src.agentic_rag.state import Document

        # Setup mock with high quality score
        mock_llm.invoke.return_value = MagicMock(
            content='{"status": "valid", "quality_score": 0.95, "validation_details": [], "issues": [], "corrective_action": "none"}'
        )

        crag = CorrectiveRAG(llm=mock_llm)
        quality = crag.evaluate_answer_quality(
            answer="High quality answer",
            documents=[Document(page_content="Supporting content", metadata={})],
        )

        assert 0 <= quality <= 1
        assert quality == 0.95

    @pytest.mark.integration
    def test_evaluate_relevance(self, mock_llm):
        """Test document relevance evaluation."""
        from src.agentic_rag.evaluator import RelevanceEvaluator, EvaluationResult
        from src.agentic_rag.state import Document

        # Setup mock to return relevant evaluation
        mock_llm.invoke.return_value = MagicMock(
            content='{"is_relevant": true, "confidence": 0.9, "reason": "Documents contain relevant information"}'
        )

        evaluator = RelevanceEvaluator(llm=mock_llm, threshold=0.7)
        result = evaluator.evaluate(
            query="Test query",
            documents=[
                Document(
                    page_content="Relevant content",
                    metadata={"source": "doc.txt"},
                )
            ],
        )

        assert isinstance(result, EvaluationResult)
        assert result.is_relevant is True
        assert result.confidence >= 0.7

    @pytest.mark.integration
    def test_hybrid_retrieval_result(self, mock_llm, mock_retriever):
        """Test hybrid retrieval result structure."""
        from src.agentic_rag.search import HybridRetrievalResult

        result = HybridRetrievalResult(
            documents=[
                {
                    "content": "Local doc",
                    "metadata": {"source": "local"},
                    "score": 0.9,
                    "source": "local",
                }
            ],
            local_count=1,
            tavily_count=0,
            search_time=0.5,
        )

        assert result.local_count == 1
        assert result.tavily_count == 0
        assert result.search_time == 0.5

    @pytest.mark.integration
    def test_search_results_structure(self, mock_llm):
        """Test search results structure."""
        from src.agentic_rag.search import DocumentResult, SearchResults

        doc = DocumentResult(
            url="https://example.com",
            title="Example",
            content="Content",
            score=0.9,
        )

        results = SearchResults(
            query="Test query",
            documents=[doc],
            total_results=1,
        )

        assert results.query == "Test query"
        assert len(results.documents) == 1
        assert results.total_results == 1

    @pytest.mark.integration
    def test_agent_result_structure(self, mock_llm, mock_retriever):
        """Test agent result structure."""
        from src.agentic_rag.agent import AgentResult
        from src.agentic_rag.state import Document

        result = AgentResult(
            answer="Test answer",
            documents=[Document(page_content="Doc", metadata={})],
            search_count=1,
            validation_passed=True,
            search_iterations=1,
            hallucination_score=0.1,
            tavily_used=False,
            tavily_document_count=0,
            local_document_count=1,
            total_documents=1,
        )

        assert result.answer == "Test answer"
        assert result.search_count == 1
        assert result.validation_passed is True
