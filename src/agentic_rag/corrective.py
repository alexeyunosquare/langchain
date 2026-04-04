"""
CRAG (Corrective RAG) logic for hallucination detection and correction.

This module provides the CorrectiveRAG class that implements corrective
RAG techniques to detect and reduce hallucinations in generated answers.

Phase 4: Adaptive Learning and Optimization
- Quality scoring and validation
- Hallucination detection and correction
- Correction strategies (rephrase, admit uncertainty, retrieve again)
- Adaptive correction with feedback loop
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from langchain_core.language_models import BaseLanguageModel

from .state import Document


class CorrectionStrategy(str, Enum):
    """Strategies for correcting hallucinated answers."""

    REPHRASE = "rephrase"
    RETRIEVE_AGAIN = "retrieve_again"
    ADMIT_UNCERTAINTY = "admit_uncertainty"
    USE_EXTERNAL_KNOWLEDGE = "use_external_knowledge"


@dataclass
class CorrectionResult:
    """
    Result of answer correction.

    Attributes:
        answer: Corrected answer
        strategy_used: Strategy applied for correction
        confidence: Confidence in the corrected answer (0-1)
        changes_made: Description of changes made
    """

    answer: str
    strategy_used: CorrectionStrategy
    confidence: float = 1.0
    changes_made: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Result of answer validation.

    Attributes:
        is_valid: Whether answer is supported by documents
        confidence: Confidence in validation result (0-1)
        reason: Explanation of validation result
        issues: List of identified issues
        supporting_claims: List of claims supported by documents
    """

    is_valid: bool
    confidence: float
    reason: str
    issues: List[str] = field(default_factory=list)
    supporting_claims: List[str] = field(default_factory=list)


class AnswerValidator:
    """
    Validates generated answers for potential hallucinations.

    This validator checks if an answer is supported by the retrieved
    documents and identifies potential hallucinations.

    Attributes:
        llm: Language model for validation
        validation_threshold: Threshold for considering answer valid (0-1)
    """

    VALIDATION_PROMPT = """
You are a fact-checking assistant. Verify if this answer is supported by
the provided context and documents.

Question: {query}
Answer: {answer}

Context:
{context}

Respond with JSON containing:
- is_valid: boolean (true if answer is supported by context)
- confidence: float (0-1, confidence in validity)
- reason: string (brief explanation)
- issues: list of strings (any issues identified)
- supporting_claims: list of strings (claims supported by context)

Respond ONLY with valid JSON.
"""

    def __init__(
        self, llm: BaseLanguageModel, validation_threshold: float = 0.7
    ) -> None:
        """Initialize the answer validator."""
        self.llm = llm
        self.validation_threshold = validation_threshold

    def validate(
        self,
        answer: str,
        documents: List[Document],
        query: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate if answer is supported by documents.

        Args:
            answer: Generated answer to validate
            documents: Documents used to generate answer
            query: Original query (optional)

        Returns:
            ValidationResult with validation details
        """
        if not answer.strip():
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                reason="Answer is empty",
                issues=["Empty answer"],
            )

        # Build context
        context = "\n\n".join([doc.page_content for doc in documents])

        # Create validation prompt
        prompt = self.VALIDATION_PROMPT.format(
            query=query or "Unknown",
            answer=answer,
            context=context if context else "No context provided.",
        )

        try:
            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Parse validation result
            result = self._parse_validation_response(content)

            return result

        except Exception:
            # Fallback: return conservative validation
            return ValidationResult(
                is_valid=False,
                confidence=0.5,
                reason="Validation error",
                issues=["Validation failed"],
            )

    def _parse_validation_response(self, response: str) -> ValidationResult:
        """
        Parse validation response from LLM.

        Args:
            response: LLM response string

        Returns:
            ValidationResult
        """
        import json
        import re

        try:
            json_str = response.strip()
            if json_str.startswith("{"):
                data = json.loads(json_str)
            else:
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                else:
                    raise ValueError("No JSON found")

            is_valid = bool(data.get("is_valid", True))
            confidence = float(data.get("confidence", 0.5))
            reason = str(data.get("reason", "No reason provided"))
            issues = data.get("issues", [])
            supporting_claims = data.get("supporting_claims", [])

            # Ensure lists are proper
            if not isinstance(issues, list):
                issues = [str(issues)] if issues else []
            if not isinstance(supporting_claims, list):
                supporting_claims = (
                    [str(supporting_claims)] if supporting_claims else []
                )

            return ValidationResult(
                is_valid=is_valid,
                confidence=max(0.0, min(1.0, confidence)),
                reason=reason,
                issues=issues,
                supporting_claims=supporting_claims,
            )

        except Exception:
            return ValidationResult(
                is_valid=False,
                confidence=0.5,
                reason="Failed to parse response",
                issues=["Parsing error"],
            )

    def validate_simple(
        self,
        answer: str,
        documents: List[Document],
        query: Optional[str] = None,
    ) -> Tuple[bool, float]:
        """
        Simplified validation returning tuple for backwards compatibility.

        Args:
            answer: Generated answer to validate
            documents: Documents used to generate answer
            query: Original query (optional)

        Returns:
            Tuple of (is_valid, confidence)
        """
        result = self.validate(answer, documents, query)
        return result.is_valid, result.confidence


class CorrectiveRAG:
    """
    Corrective RAG implementation for hallucination reduction.

    This class implements CRAG techniques to detect and correct
    hallucinations in generated answers, improving reliability.

    Attributes:
        llm: Language model for correction operations
        answer_validator: Validator for answer quality
        correction_threshold: Threshold for triggering correction (0-1)
        max_correction_attempts: Maximum number of correction attempts
    """

    CORRECTION_PROMPT = """
You are a fact-checking assistant. The following answer may contain
hallucinations (information not supported by the documents).

Question: {query}
Potential Hallucinated Answer: {answer}

Documents:
{documents}

Identify if there are hallucinations and provide a corrected answer.

Respond with JSON containing:
- is_hallucinated: boolean
- confidence: float (0-1)
- reason: string (explanation)
- corrected_answer: string (corrected version, same as original if no changes)

Respond ONLY with valid JSON.
"""

    REPHRASE_PROMPT = """
Rephrase the following answer to be more accurate based on the context.
Ensure all claims are supported by the provided documents.

Context:
{context}

Original answer: {answer}

Provide a corrected, more accurate version:
"""

    UNCERTAINTY_PROMPT = """
The available information is insufficient to fully answer this question.

Question: {query}

Provide an answer that honestly acknowledges the uncertainty while
providing the most helpful response possible based on available context.
"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        correction_threshold: float = 0.7,
        max_correction_attempts: int = 2,
    ) -> None:
        """
        Initialize Corrective RAG.

        Args:
            llm: Language model for correction operations
            correction_threshold: Threshold for triggering correction (0-1)
            max_correction_attempts: Maximum correction attempts
        """
        self.llm = llm
        self.answer_validator = AnswerValidator(llm=llm)
        self.correction_threshold = correction_threshold
        self.max_correction_attempts = max_correction_attempts

    def validate_and_correct(
        self, answer: str, documents: List[Document], query: Optional[str] = None
    ) -> str:
        """
        Validate and correct an answer in one step.

        Args:
            answer: Answer to validate and potentially correct
            documents: Documents used to generate answer
            query: Original query

        Returns:
            Validated and potentially corrected answer
        """
        validation = self.answer_validator.validate(answer, documents, query)

        if not validation.is_valid or validation.confidence < self.correction_threshold:
            return self.correct_answer(answer, documents, query)

        return answer

    def check_hallucination(
        self,
        answer: str,
        documents: List[Document],
        query: Optional[str] = None,
    ) -> Tuple[bool, float]:
        """
        Check if answer contains hallucinations.

        Args:
            answer: Answer to check
            documents: Documents used to generate answer
            query: Original query

        Returns:
            Tuple of (is_hallucinated, confidence)
        """
        if not answer.strip():
            return False, 0.0

        # Build documents text
        doc_text = "\n\n".join([doc.page_content for doc in documents])

        # Create correction prompt
        prompt = self.CORRECTION_PROMPT.format(
            query=query or "Unknown",
            answer=answer,
            documents=doc_text if doc_text else "No documents provided.",
        )

        try:
            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Parse correction result
            is_hallucinated, confidence = self._parse_correction_response(content)

            return is_hallucinated, confidence

        except Exception:
            # Fallback: assume no hallucination on error
            return False, 0.0

    def correct_answer(
        self,
        answer: str,
        documents: List[Document],
        query: Optional[str] = None,
        strategy: Optional[CorrectionStrategy] = None,
    ) -> str:
        """
        Correct a potentially hallucinated answer.

        Args:
            answer: Answer to correct
            documents: Documents for context
            query: Original query
            strategy: Correction strategy (auto-selects if None)

        Returns:
            Corrected answer
        """
        if strategy is None:
            # Auto-select strategy based on documents
            if documents:
                strategy = CorrectionStrategy.REPHRASE
            else:
                strategy = CorrectionStrategy.ADMIT_UNCERTAINTY

        result = self.apply_correction(answer, documents, query, strategy)
        return result.answer

    def apply_correction(
        self,
        answer: str,
        documents: List[Document],
        query: Optional[str] = None,
        strategy: CorrectionStrategy = CorrectionStrategy.REPHRASE,
    ) -> CorrectionResult:
        """
        Apply a specific correction strategy.

        Args:
            answer: Answer to correct
            documents: Documents for context
            query: Original query
            strategy: Correction strategy to apply

        Returns:
            CorrectionResult with corrected answer
        """
        if strategy == CorrectionStrategy.REPHRASE:
            return self._rephrase_answer(answer, documents, query)
        elif strategy == CorrectionStrategy.ADMIT_UNCERTAINTY:
            return self._admit_uncertainty(answer, documents, query)
        elif strategy == CorrectionStrategy.RETRIEVE_AGAIN:
            return CorrectionResult(
                answer=answer,
                strategy_used=strategy,
                confidence=0.5,
                changes_made="Requires additional retrieval",
            )
        else:
            return CorrectionResult(
                answer=answer,
                strategy_used=strategy,
                confidence=1.0,
            )

    def _rephrase_answer(
        self,
        answer: str,
        documents: List[Document],
        query: Optional[str],  # noqa: ARG002
    ) -> CorrectionResult:
        """Rephrase answer to be more accurate."""
        context = "\n\n".join([doc.page_content for doc in documents])

        prompt = self.REPHRASE_PROMPT.format(
            context=context if context else "No context available.",
            answer=answer,
        )

        try:
            response = self.llm.invoke(prompt)
            corrected = (
                response.content if hasattr(response, "content") else str(response)
            )

            return CorrectionResult(
                answer=corrected,
                strategy_used=CorrectionStrategy.REPHRASE,
                confidence=0.8,
                changes_made="Rephrased for accuracy",
            )
        except Exception:
            return CorrectionResult(
                answer=answer,
                strategy_used=CorrectionStrategy.REPHRASE,
                confidence=0.5,
            )

    def _admit_uncertainty(
        self,
        answer: str,  # noqa: ARG002
        documents: List[Document],  # noqa: ARG002
        query: Optional[str],  # noqa: ARG002
    ) -> CorrectionResult:
        """Generate answer that acknowledges uncertainty."""
        prompt = self.UNCERTAINTY_PROMPT.format(
            query=query or "Unknown question",
        )

        try:
            response = self.llm.invoke(prompt)
            corrected = (
                response.content if hasattr(response, "content") else str(response)
            )

            return CorrectionResult(
                answer=corrected,
                strategy_used=CorrectionStrategy.ADMIT_UNCERTAINTY,
                confidence=0.7,
                changes_made="Acknowledged uncertainty",
            )
        except Exception:
            return CorrectionResult(
                answer="I don't have enough information to provide a confident answer.",
                strategy_used=CorrectionStrategy.ADMIT_UNCERTAINTY,
                confidence=0.5,
            )

    def evaluate_answer_quality(
        self,
        answer: str,
        documents: List[Document],
        query: Optional[str] = None,
    ) -> float:
        """
        Evaluate quality score of an answer.

        Args:
            answer: Answer to evaluate
            documents: Documents used
            query: Original query

        Returns:
            Quality score (0-1)
        """
        validation = self.answer_validator.validate(answer, documents, query)

        # Combine validation with other factors
        quality_score = (
            validation.confidence * 0.7  # Validation confidence
            + (1.0 if answer.strip() else 0.0) * 0.3  # Answer completeness
        )

        return min(1.0, max(0.0, quality_score))

    def _parse_correction_response(self, response: str) -> Tuple[bool, float]:
        """
        Parse correction response from LLM.

        Args:
            response: LLM response string

        Returns:
            Tuple of (is_hallucinated, confidence)
        """
        import json
        import re

        try:
            json_str = response.strip()
            if json_str.startswith("{"):
                data = json.loads(json_str)
            else:
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                else:
                    raise ValueError("No JSON found")

            is_hallucinated = bool(data.get("is_hallucinated", False))
            confidence = float(data.get("confidence", 0.5))

            return is_hallucinated, max(0.0, min(1.0, confidence))

        except Exception:
            return False, 0.5
