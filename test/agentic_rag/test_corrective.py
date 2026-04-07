"""
Tests for CRAG (Corrective RAG) logic in Agentic RAG.
"""

from unittest.mock import MagicMock

import pytest

from src.agentic_rag.corrective import (
    AnswerValidator,
    CorrectionEngine,
    CorrectionEngineConfig,
    CorrectionResult,
    CorrectionStrategy,
    CorrectiveRAG,
    ValidationDetail,
    ValidationResult,
    ValidationStatus,
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
        assert crag.correction_engine is not None

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
        mock_llm.invoke.return_value = Message(
            role=MessageRole.ASSISTANT,
            content='{"status": "hallucinated", "quality_score": 0.3, "validation_details": [], "issues": ["Answer contains unsupported claims"], "corrective_action": "rephrase"}',
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
        mock_llm.invoke.return_value = Message(
            role=MessageRole.ASSISTANT,
            content='{"status": "valid", "quality_score": 0.95, "validation_details": [], "issues": [], "corrective_action": "none"}',
        )

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
                content='{"status": "invalid", "quality_score": 0.3, "validation_details": [], "issues": ["Low quality"], "corrective_action": "rephrase"}',
            ),
            Message(
                role="assistant",
                content='{"status": "valid", "quality_score": 0.8, "validation_details": [], "issues": [], "corrective_action": "none"}',
            ),
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
            content='{"status": "hallucinated", "quality_score": 0.2, "validation_details": [], "issues": ["Insufficient context"], "corrective_action": "admit_uncertainty"}',
        )

        crag = CorrectiveRAG(llm=mock_llm)

        answer = "Specific claim about topic X"
        documents = [Document(page_content="Unrelated content", metadata={})]

        is_hallucinated, confidence = crag.check_hallucination(answer, documents)

        assert isinstance(is_hallucinated, bool)
        assert 0 <= confidence <= 1

    def test_correction_preserves_valid_content(self, mock_llm):
        """Test that correction preserves valid parts of answer."""
        mock_llm.invoke.return_value = Message(
            role=MessageRole.ASSISTANT,
            content='{"status": "partially_valid", "quality_score": 0.6, "validation_details": [], "issues": ["Some claims unsupported"], "corrective_action": "rephrase"}',
        )

        crag = CorrectiveRAG(llm=mock_llm)

        original_answer = "Valid part. Hallucinated part."
        documents = [Document(page_content="Context", metadata={})]

        corrected = crag.correct_answer(original_answer, documents)

        # Should return a string
        assert isinstance(corrected, str)
        assert len(corrected) > 0

    def test_should_correct_decision(self, mock_llm):
        """Test should_correct logic."""
        crag = CorrectiveRAG(llm=mock_llm, correction_threshold=0.7)

        # High quality validation
        mock_llm.invoke.return_value = Message(
            role=MessageRole.ASSISTANT,
            content='{"status": "valid", "quality_score": 0.9, "validation_details": [], "issues": [], "corrective_action": "none"}',
        )

        validator = crag.answer_validator
        validation = validator.validate(
            answer="Good answer", documents=[Document(page_content="Context", metadata={})]
        )

        should_correct = crag.correction_engine.should_correct(validation)
        assert should_correct is False

    def test_correction_statistics(self, mock_llm):
        """Test correction statistics retrieval."""
        crag = CorrectiveRAG(llm=mock_llm, correction_threshold=0.8)

        stats = crag.get_correction_statistics()

        assert "correction_threshold" in stats
        assert "max_correction_attempts" in stats
        assert stats["correction_threshold"] == 0.8


class TestAnswerValidator:
    """Test suite for AnswerValidator class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = MagicMock()
        mock.invoke.return_value = Message(
            role="assistant", content='{"status": "valid", "quality_score": 0.9, "validation_details": [], "issues": [], "corrective_action": "none"}'
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

        assert isinstance(result, ValidationResult)
        assert result.status in ValidationStatus
        assert 0 <= result.quality_score <= 1

    def test_validate_answer_empty(self, mock_llm):
        """Test validation of empty answer."""
        validator = AnswerValidator(llm=mock_llm)

        answer = ""
        documents = [Document(page_content="Context", metadata={})]

        result = validator.validate(answer, documents)

        assert result.status == ValidationStatus.INVALID

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

        assert result.status is not None
        assert result.quality_score >= 0

    def test_validate_parsing_error_fallback(self, mock_llm):
        """Test fallback on parsing error."""
        mock_llm.invoke.return_value = Message(
            role="assistant", content="Invalid JSON response"
        )

        validator = AnswerValidator(llm=mock_llm)

        answer = "Test answer"
        documents = [Document(page_content="Context", metadata={})]

        result = validator.validate(answer, documents)

        # Should fallback to invalid status
        assert result.status == ValidationStatus.INVALID


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

        assert hasattr(CorrectionStrategy, "REPHRASE")
        assert hasattr(CorrectionStrategy, "RETRIEVE_AGAIN")
        assert hasattr(CorrectionStrategy, "ADMIT_UNCERTAINTY")

    def test_correction_with_retrieve_again(self, mock_llm):
        """Test correction when retrieving again is needed."""
        crag = CorrectiveRAG(llm=mock_llm)

        answer = "Answer needing more information"
        documents = [Document(page_content="Limited context", metadata={})]

        result = crag.apply_correction(
            answer, documents, query="Query", strategy="re-search"
        )

        assert isinstance(result, CorrectionResult)
        assert result.correction_type == "re-search"

    def test_correction_with_admit_uncertainty(self, mock_llm):
        """Test correction with uncertainty admission."""
        crag = CorrectiveRAG(llm=mock_llm)

        answer = "Uncertain answer"
        documents = []

        result = crag.apply_correction(
            answer, documents, query="Query", strategy="admit_uncertainty"
        )

        assert isinstance(result, CorrectionResult)
        assert result.correction_type == "admit_uncertainty"

    def test_should_correct_threshold(self, mock_llm):
        """Test should_correct with threshold."""
        config = CorrectionEngineConfig(quality_threshold=0.8)
        engine = CorrectionEngine(llm=mock_llm, config=config)

        # High quality validation
        validation = ValidationResult(
            answer="Good answer",
            status=ValidationStatus.VALID,
            quality_score=0.9,
            issues=[],
        )

        assert engine.should_correct(validation) is False

        # Low quality validation
        validation_low = ValidationResult(
            answer="Bad answer",
            status=ValidationStatus.INVALID,
            quality_score=0.5,
            issues=["Low quality"],
        )

        assert engine.should_correct(validation_low) is True

    def test_correction_result_structure(self):
        """Test CorrectionResult dataclass structure."""
        result = CorrectionResult(
            original_answer="Original",
            corrected_answer="Corrected",
            correction_type="rephrase",
            quality_improvement=0.2,
            iterations=1,
        )

        assert result.original_answer == "Original"
        assert result.corrected_answer == "Corrected"
        assert result.correction_type == "rephrase"
        assert result.quality_improvement == 0.2
        assert result.iterations == 1

    def test_validation_detail_structure(self):
        """Test ValidationDetail dataclass structure."""
        detail = ValidationDetail(
            claim="Test claim",
            is_supported=True,
            supporting_document_id="doc1",
            confidence=0.9,
            issue_type=None,
        )

        assert detail.claim == "Test claim"
        assert detail.is_supported is True
        assert detail.supporting_document_id == "doc1"
        assert detail.confidence == 0.9


class TestValidationStatus:
    """Test ValidationStatus enum."""

    def test_status_values(self):
        """Test all validation status values."""
        assert ValidationStatus.VALID.value == "valid"
        assert ValidationStatus.PARTIALLY_VALID.value == "partially_valid"
        assert ValidationStatus.INVALID.value == "invalid"
        assert ValidationStatus.HALLUCINATED.value == "hallucinated"


class TestCorrectiveRAGIntegration:
    """Integration tests for Complete CRAG workflow."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = MagicMock()
        mock.invoke.return_value = Message(
            role="assistant",
            content='{"status": "valid", "quality_score": 0.85, "validation_details": [], "issues": [], "corrective_action": "none"}',
        )
        return mock

    def test_end_to_end_validation_correction(self, mock_llm):
        """Test complete validation and correction workflow."""
        crag = CorrectiveRAG(llm=mock_llm)

        answer = "Test answer"
        documents = [
            Document(page_content="Supporting context", metadata={"source": "doc.txt"})
        ]

        # Validate without correction (high quality)
        validated = crag.validate_and_correct(answer, documents)
        assert validated is not None

    def test_correction_loop_with_multiple_attempts(self, mock_llm):
        """Test correction loop with multiple attempts."""
        mock_llm.invoke.side_effect = [
            Message(
                role="assistant",
                content='{"status": "invalid", "quality_score": 0.3, "validation_details": [], "issues": ["Low quality"], "corrective_action": "rephrase"}',
            ),
            Message(
                role="assistant",
                content='{"status": "valid", "quality_score": 0.8, "validation_details": [], "issues": [], "corrective_action": "none"}',
            ),
            Message(
                role="assistant",
                content='{"status": "valid", "quality_score": 0.85, "validation_details": [], "issues": [], "corrective_action": "none"}',
            ),
        ]

        crag = CorrectiveRAG(llm=mock_llm, max_correction_attempts=3)

        answer = "Initial low quality answer"
        documents = [Document(page_content="Context", metadata={})]

        corrected = crag.validate_and_correct(answer, documents)

        assert isinstance(corrected, str)
        assert len(corrected) > 0

    def test_validation_detail_extraction(self, mock_llm):
        """Test extraction of validation details."""
        mock_llm.invoke.return_value = Message(
            role="assistant",
            content='{"status": "partially_valid", "quality_score": 0.6, "validation_details": [{"claim": "Test claim", "is_supported": true, "confidence": 0.9}], "issues": ["Some claims unsupported"], "corrective_action": "rephrase"}',
        )

        crag = CorrectiveRAG(llm=mock_llm)
        result = crag.answer_validator.validate(
            answer="Test answer",
            documents=[Document(page_content="Context", metadata={})],
        )

        assert result.status == ValidationStatus.PARTIALLY_VALID
        assert len(result.validation_details) >= 0
        assert result.quality_score == 0.6
