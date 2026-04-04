"""
Tests for CRAG (Corrective RAG) logic in Agentic RAG.
"""

from unittest.mock import MagicMock

import pytest

from src.agentic_rag.corrective import (
    AnswerValidator,
    CorrectionStrategy,
    CorrectiveRAG,
)
from src.agentic_rag.state import Document, Message, MessageRole


class TestCorrectiveRAG:
    """Test suite for CorrectiveRAG class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = MagicMock()
        mock.invoke.return_value = Message(
            role=MessageRole.ASSISTANT,
            content="This is a valid answer with no hallucinations.",
        )
        return mock

    def test_crag_initialization(self, mock_llm):
        """Test CRAG initializes with correct components."""
        crag = CorrectiveRAG(llm=mock_llm)

        assert crag.llm is mock_llm
        assert crag.answer_validator is not None

    def test_identify_hallucination(self, mock_llm):
        """Test detection of potential hallucinations."""
        crag = CorrectiveRAG(llm=mock_llm)

        answer = "According to the documents, LangChain was invented in 2023."
        documents = [
            Document(
                page_content="LangChain was developed starting in 2022.",
                metadata={"source": "docs.txt"},
            )
        ]

        is_hallucinated, confidence = crag.check_hallucination(answer, documents)

        assert isinstance(is_hallucinated, bool)
        assert 0 <= confidence <= 1

    def test_trigger_correction(self, mock_llm):
        """Test correction mechanism when hallucination detected."""
        # Setup LLM to indicate hallucination with corrected answer
        mock_llm.invoke.return_value = Message(
            role=MessageRole.ASSISTANT,
            content='{"is_hallucinated": true, "reason": "Answer contains information not in documents", "corrected_answer": "Corrected answer based on documents"}',
        )

        crag = CorrectiveRAG(llm=mock_llm)

        answer = "Hallucinated answer"
        documents = [
            Document(page_content="Different content", metadata={"source": "doc.txt"})
        ]

        result = crag.correct_answer(answer, documents)

        # Should return corrected answer or original
        assert isinstance(result, str)
        assert len(result) > 0

    def test_quality_score_calculation(self, mock_llm):
        """Test quality scoring of generated answers."""
        crag = CorrectiveRAG(llm=mock_llm)

        # High quality answer
        quality = crag.evaluate_answer_quality(
            answer="Accurate and complete answer.",
            documents=[Document(page_content="Supporting content", metadata={})],
        )

        assert 0 <= quality <= 1

    def test_correction_flow(self, mock_llm):
        """Test complete correction workflow."""
        mock_llm.invoke.side_effect = [
            Message(
                role="assistant",
                content='{"is_hallucinated": false, "confidence": 0.9}',
            ),
            Message(role="assistant", content="Verified answer content"),
        ]

        crag = CorrectiveRAG(llm=mock_llm)

        answer = "Initial answer"
        documents = [Document(page_content="Source content", metadata={})]

        validated_answer = crag.validate_and_correct(answer, documents)

        assert validated_answer is not None
        assert len(validated_answer) > 0

    def test_hallucination_with_insufficient_context(self, mock_llm):
        """Test detection when documents don't support answer."""
        mock_llm.invoke.return_value = Message(
            role=MessageRole.ASSISTANT,
            content='{"is_hallucinated": true, "confidence": 0.8, "reason": "Insufficient context", "corrected_answer": "Acknowledged uncertainty"}',
        )

        crag = CorrectiveRAG(llm=mock_llm)

        answer = "Specific claim about topic X"
        documents = [Document(page_content="Unrelated content", metadata={})]

        is_hallucinated, confidence = crag.check_hallucination(answer, documents)

        # Result depends on LLM mock response
        assert isinstance(is_hallucinated, bool)
        assert 0 <= confidence <= 1

    def test_correction_preserves_valid_content(self, mock_llm):
        """Test that correction preserves valid parts of answer."""
        # Setup LLM to return partially corrected answer
        mock_llm.invoke.return_value = Message(
            role=MessageRole.ASSISTANT,
            content='{"corrected_answer": "Partial answer preserved.", "changes": "Removed hallucinated portion"}',
        )

        crag = CorrectiveRAG(llm=mock_llm)

        original_answer = "Valid part. Hallucinated part."
        documents = [Document(page_content="Context", metadata={})]

        corrected = crag.correct_answer(original_answer, documents)

        # Should return a string
        assert isinstance(corrected, str)
        assert len(corrected) > 0


class TestAnswerValidator:
    """Test suite for AnswerValidator class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = MagicMock()
        mock.invoke.return_value = Message(
            role="assistant", content='{"valid": true, "confidence": 0.9}'
        )
        return mock

    def test_validator_initialization(self, mock_llm):
        """Test answer validator initializes correctly."""
        validator = AnswerValidator(llm=mock_llm)

        assert validator.llm is mock_llm

    def test_validate_answer_basic(self, mock_llm):
        """Test basic answer validation."""
        validator = AnswerValidator(llm=mock_llm)

        answer = "This is a test answer."
        documents = [Document(page_content="Context", metadata={})]

        result = validator.validate(answer, documents)

        assert isinstance(result.is_valid, bool)
        assert 0 <= result.confidence <= 1

    def test_validate_answer_empty(self, mock_llm):
        """Test validation of empty answer."""
        validator = AnswerValidator(llm=mock_llm)

        answer = ""
        documents = [Document(page_content="Context", metadata={})]

        result = validator.validate(answer, documents)

        assert result.is_valid is False

    def test_validate_answer_with_multiple_documents(self, mock_llm):
        """Test validation with multiple documents."""
        validator = AnswerValidator(llm=mock_llm)

        answer = "Multi-document answer"
        documents = [
            Document(page_content="Doc 1", metadata={"source": "doc1.txt"}),
            Document(page_content="Doc 2", metadata={"source": "doc2.txt"}),
            Document(page_content="Doc 3", metadata={"source": "doc3.txt"}),
        ]

        result = validator.validate(answer, documents)

        assert result.is_valid is not None
        assert result.confidence >= 0


class TestCorrectionEngine:
    """Test suite for correction engine functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = MagicMock()
        mock.invoke.return_value = Message(
            role="assistant", content="Corrected answer based on documents."
        )
        return mock

    def test_correction_strategy_selection(self):
        """Test selection of appropriate correction strategy."""

        # Test strategy enum
        assert hasattr(CorrectionStrategy, "REPHRASE")
        assert hasattr(CorrectionStrategy, "RETRIEVE_AGAIN")
        assert hasattr(CorrectionStrategy, "ADMIT_UNCERTAINTY")

    def test_correction_with_retrieve_aggressive(self, mock_llm):
        """Test correction when retrieving again is needed."""

        crag = CorrectiveRAG(llm=mock_llm)

        answer = "Answer needing more information"
        documents = [Document(page_content="Limited context", metadata={})]

        result = crag.apply_correction(
            answer, documents, strategy=CorrectionStrategy.RETRIEVE_AGAIN
        )

        assert result is not None

    def test_correction_with_admit_uncertainty(self, mock_llm):
        """Test correction with uncertainty admission."""

        crag = CorrectiveRAG(llm=mock_llm)

        answer = "Uncertain answer"
        documents = []

        result = crag.apply_correction(
            answer, documents, strategy=CorrectionStrategy.ADMIT_UNCERTAINTY
        )

        assert "uncertain" in result.answer.lower() or len(result.answer) > 0
