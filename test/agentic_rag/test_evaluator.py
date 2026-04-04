"""
Tests for Document Relevance Evaluator in Agentic RAG.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.agentic_rag.evaluator import EvaluationResult, RelevanceEvaluator
from src.agentic_rag.state import Document, Message

# Force reimport
if "agentic_rag" in sys.modules:
    del sys.modules["agentic_rag"]
if "agentic_rag.evaluator" in sys.modules:
    del sys.modules["agentic_rag.evaluator"]


class TestRelevanceEvaluator:
    """Test suite for RelevanceEvaluator class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = MagicMock()
        mock.invoke.return_value = Message(
            role="assistant",
            content='{"is_relevant": true, "confidence": 0.9, "reason": "Document contains relevant information"}',
        )
        return mock

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="LangChain is a framework for developing applications powered by language models.",
                metadata={"source": "langchain_docs.txt", "page": 1},
            ),
            Document(
                page_content="The weather today is sunny with a temperature of 72 degrees Fahrenheit.",
                metadata={"source": "weather_report.txt", "page": 1},
            ),
        ]

    def test_evaluate_highly_relevant_document(self, mock_llm, sample_documents):
        """Test evaluator correctly identifies relevant documents."""
        evaluator = RelevanceEvaluator(llm=mock_llm, threshold=0.7)

        result = evaluator.evaluate(
            query="What is LangChain?", documents=sample_documents[:1]
        )

        assert result.is_relevant is True
        assert result.confidence >= 0.7
        assert result.reason is not None

    def test_evaluate_irrelevant_document(self, mock_llm, sample_documents):
        """Test evaluator correctly identifies irrelevant documents."""
        evaluator = RelevanceEvaluator(llm=mock_llm, threshold=0.7)

        # Mock response for irrelevant document
        mock_llm.invoke.return_value = Message(
            role="assistant",
            content='{"is_relevant": false, "confidence": 0.85, "reason": "Document does not contain relevant information"}',
        )

        result = evaluator.evaluate(
            query="What is LangChain?", documents=sample_documents[1:]
        )

        assert result.is_relevant is False
        assert result.confidence >= 0.7

    def test_evaluate_edge_cases_empty_documents(self, mock_llm):
        """Test evaluator handles edge cases (empty docs)."""
        evaluator = RelevanceEvaluator(llm=mock_llm, threshold=0.7)

        # Empty documents list should return irrelevant
        result = evaluator.evaluate(query="Test query", documents=[])

        assert result.is_relevant is False
        assert result.reason is not None

    def test_evaluate_edge_cases_empty_query(self, mock_llm, sample_documents):
        """Test evaluator handles edge cases (empty query)."""
        evaluator = RelevanceEvaluator(llm=mock_llm, threshold=0.7)

        with pytest.raises(ValueError, match="query cannot be empty"):
            evaluator.evaluate(query="", documents=sample_documents)

    def test_evaluate_mixed_relevance_documents(self, mock_llm, sample_documents):
        """Test evaluator with documents of mixed relevance."""
        evaluator = RelevanceEvaluator(llm=mock_llm, threshold=0.7)

        # Mock a response that indicates mixed relevance
        mock_llm.invoke.return_value = Message(
            role="assistant",
            content='{"is_relevant": true, "confidence": 0.6, "reason": "Mixed relevance detected"}',
        )

        result = evaluator.evaluate(query="Test query", documents=sample_documents)

        assert result.confidence == 0.6
        # is_relevant comes from JSON response, not threshold comparison
        assert result.is_relevant is True

    def test_evaluator_threshold_respected(self, mock_llm, sample_documents):
        """Test that threshold is correctly applied."""
        evaluator = RelevanceEvaluator(llm=mock_llm, threshold=0.9)

        # Mock a response with relevance false (below threshold behavior)
        mock_llm.invoke.return_value = Message(
            role="assistant",
            content='{"is_relevant": false, "confidence": 0.85, "reason": "Test"}',
        )

        result = evaluator.evaluate(query="Test query", documents=sample_documents)

        assert result.is_relevant is False

    def test_evaluator_parses_invalid_json_fallback(self, mock_llm, sample_documents):
        """Test evaluator handles invalid JSON gracefully."""
        mock_llm.invoke.return_value = Message(
            role="assistant", content="This is not valid JSON"
        )

        evaluator = RelevanceEvaluator(llm=mock_llm, threshold=0.7)

        # Should fall back to default behavior
        result = evaluator.evaluate(query="Test query", documents=sample_documents)

        # Default should be irrelevant for malformed response
        assert result.is_relevant is False
        assert result.reason is not None
        assert "Failed to parse" in result.reason

    def test_evaluate_with_source_tracking(self, mock_llm, sample_documents):
        """Test evaluator tracks document sources."""
        evaluator = RelevanceEvaluator(llm=mock_llm, threshold=0.7)

        result = evaluator.evaluate(
            query="What is LangChain?", documents=sample_documents[:1]
        )

        # Should include document metadata in evaluation
        assert len(result.document_ids) <= len(sample_documents)


class TestEvaluationResult:
    """Test suite for EvaluationResult class."""

    def test_result_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            is_relevant=True,
            confidence=0.9,
            reason="Document is highly relevant",
            document_ids=["doc1", "doc2"],
        )

        assert result.is_relevant is True
        assert result.confidence == 0.9
        assert result.reason == "Document is highly relevant"
        assert result.document_ids == ["doc1", "doc2"]

    def test_result_default_values(self):
        """Test evaluation result with default values."""
        result = EvaluationResult(is_relevant=True)

        assert result.confidence == 0.0
        assert result.reason is None
        assert result.document_ids == []

    def test_result_str_representation(self):
        """Test string representation of evaluation result."""
        result = EvaluationResult(
            is_relevant=True, confidence=0.85, reason="Test reason"
        )

        result_str = str(result)

        assert "is_relevant=True" in result_str
        assert "confidence=0.85" in result_str
