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


class ValidationStatus(str, Enum):
    """Validation status for answer validation."""

    VALID = "valid"
    PARTIALLY_VALID = "partially_valid"
    INVALID = "invalid"
    HALLUCINATED = "hallucinated"


@dataclass
class ValidationDetail:
    """
    Detail of a validation check.

    Attributes:
        claim: The claim being validated
        is_supported: Whether claim is supported by documents
        supporting_document_id: ID of supporting document if any
        confidence: Confidence in support verification (0-1)
        issue_type: Type of issue if not supported (e.g., hallucination)
    """

    claim: str
    is_supported: bool
    supporting_document_id: Optional[str] = None
    confidence: float = 0.0
    issue_type: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Result of answer validation.

    Attributes:
        answer: The validated answer
        status: Overall validation status
        quality_score: Quality score (0-1)
        validation_details: List of validation details
        issues: List of identified issues
        corrective_action: Suggested corrective action
    """

    answer: str
    status: ValidationStatus
    quality_score: float
    validation_details: List[ValidationDetail] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    corrective_action: Optional[str] = None


@dataclass
class CorrectionResult:
    """
    Result of answer correction.

    Attributes:
        original_answer: Original answer before correction
        corrected_answer: Corrected answer
        correction_type: Type of correction applied
        quality_improvement: Quality improvement from correction (0-1)
        iterations: Number of correction iterations
    """

    original_answer: str
    corrected_answer: str
    correction_type: str
    quality_improvement: float = 0.0
    iterations: int = 1


@dataclass
class CorrectionEngineConfig:
    """Configuration for CorrectionEngine."""

    quality_threshold: float = 0.7
    max_correction_attempts: int = 2


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
- status: one of "valid", "partially_valid", "invalid", "hallucinated"
- quality_score: float (0-1, quality of the answer)
- validation_details: list of objects with claim, is_supported, confidence
- issues: list of strings (any issues identified)
- corrective_action: "none", "regenerate", "re-search", "admit_uncertainty"

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
                answer=answer,
                status=ValidationStatus.INVALID,
                quality_score=0.0,
                issues=["Empty answer"],
                corrective_action="regenerate",
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
            result = self._parse_validation_response(content, answer)

            return result

        except Exception:
            # Fallback: return conservative validation
            return ValidationResult(
                answer=answer,
                status=ValidationStatus.INVALID,
                quality_score=0.5,
                issues=["Validation failed"],
                corrective_action="re-search",
            )

    def _parse_validation_response(
        self, response: str, answer: str
    ) -> ValidationResult:
        """
        Parse validation response from LLM.

        Args:
            response: LLM response string
            answer: Original answer for fallback

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

            # Parse status
            status_str = data.get("status", "invalid")
            try:
                status = ValidationStatus(status_str)
            except ValueError:
                status = ValidationStatus.INVALID

            # Parse quality score
            quality_score = float(data.get("quality_score", 0.5))
            quality_score = max(0.0, min(1.0, quality_score))

            # Parse issues
            issues = data.get("issues", [])
            if not isinstance(issues, list):
                issues = [str(issues)] if issues else []

            # Parse validation details
            validation_details = []
            for detail_data in data.get("validation_details", []):
                if isinstance(detail_data, dict):
                    detail = ValidationDetail(
                        claim=detail_data.get("claim", ""),
                        is_supported=bool(detail_data.get("is_supported", False)),
                        supporting_document_id=detail_data.get(
                            "supporting_document_id"
                        ),
                        confidence=float(detail_data.get("confidence", 0.0)),
                        issue_type=detail_data.get("issue_type"),
                    )
                    validation_details.append(detail)

            # Parse corrective action
            corrective_action = data.get("corrective_action", "none")
            if corrective_action not in ["none", "regenerate", "re-search", "admit_uncertainty"]:
                corrective_action = "none"
            corrective_action = corrective_action if corrective_action != "none" else None

            return ValidationResult(
                answer=answer,
                status=status,
                quality_score=quality_score,
                validation_details=validation_details,
                issues=issues,
                corrective_action=corrective_action,
            )

        except Exception:
            return ValidationResult(
                answer=answer,
                status=ValidationStatus.INVALID,
                quality_score=0.5,
                issues=["Parsing error"],
                corrective_action="re-search",
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
        return result.status in [ValidationStatus.VALID, ValidationStatus.PARTIALLY_VALID], result.quality_score


class CorrectionEngine:
    """
    Handles corrective actions when validation fails.

    This engine applies appropriate correction strategies based on
    the validation result and the available context.

    Attributes:
        llm: Language model for correction operations
        config: Configuration for correction engine
    """

    REPHRASE_PROMPT = """
Rephrase the following answer to be more accurate based on the context.
Ensure all claims are supported by the provided documents.

Context:
{context}

Original answer: {answer}

Identify which claims are not supported and remove or modify them.
Provide only the corrected answer, nothing else.
"""

    UNCERTAINTY_PROMPT = """
The available information is insufficient to fully answer this question.

Question: {query}

Provide an answer that honestly acknowledges the uncertainty while
providing the most helpful response possible based on available context.
"""

    RESEARCH_PROMPT = """
The previous search did not find relevant documents. Please provide
alternative search terms or query variations that might find better results.

Original query: {query}

Suggest 3-5 alternative search queries that might find more relevant information.
Respond as a JSON list of strings.
"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        config: Optional[CorrectionEngineConfig] = None,
    ) -> None:
        """
        Initialize CorrectionEngine.

        Args:
            llm: Language model for correction operations
            config: Optional configuration (uses defaults if None)
        """
        self.llm = llm
        self.config = config or CorrectionEngineConfig()

    def should_correct(self, validation_result: ValidationResult) -> bool:
        """
        Determine if correction is needed.

        Args:
            validation_result: Result from validator

        Returns:
            True if correction should be applied
        """
        if validation_result.status in [
            ValidationStatus.INVALID,
            ValidationStatus.HALLUCINATED,
        ]:
            return True

        if validation_result.quality_score < self.config.quality_threshold:
            return True

        return False

    def correct(
        self,
        validation_result: ValidationResult,
        query: str,
        documents: List[Document],
        search_available: bool = True,
    ) -> CorrectionResult:
        """
        Apply corrective actions.

        Args:
            validation_result: Result from validator
            query: Original query
            documents: Current documents
            search_available: Can we re-search?

        Returns:
            CorrectionResult with improved answer
        """
        original_answer = validation_result.answer
        iterations = 0
        corrected_answer = original_answer
        best_quality = validation_result.quality_score

        while iterations < self.config.max_correction_attempts:
            iterations += 1

            # Choose correction strategy
            strategy = self._choose_correction_strategy(
                validation_result, documents, search_available
            )

            # Apply correction
            correction_result = self.apply_correction(
                corrected_answer, documents, query, strategy, validation_result
            )
            corrected_answer = correction_result.answer

            # Calculate quality improvement
            quality_improvement = (
                max(0.0, 1.0 - (iterations * 0.1)) if iterations > 0 else 0.0
            )

            if quality_improvement > best_quality:
                best_quality = quality_improvement

            # Check if correction is sufficient
            if correction_result.quality_improvement >= 0.8:
                break

        return CorrectionResult(
            original_answer=original_answer,
            corrected_answer=corrected_answer,
            correction_type=strategy,
            quality_improvement=best_quality,
            iterations=iterations,
        )

    def _choose_correction_strategy(
        self,
        validation_result: ValidationResult,
        documents: List[Document],
        search_available: bool,
    ) -> str:
        """
        Choose appropriate correction strategy.

        Args:
            validation_result: Validation result
            documents: Available documents
            search_available: Can we re-search?

        Returns:
            Strategy name
        """
        if validation_result.status == ValidationStatus.HALLUCINATED:
            if search_available and not documents:
                return "re-search"
            return "rephrase"

        if validation_result.status == ValidationStatus.INVALID:
            if not documents:
                return "admit_uncertainty"
            return "rephrase"

        if validation_result.quality_score < 0.5:
            if search_available:
                return "re-search"
            return "admit_uncertainty"

        return "rephrase"

    def apply_correction(
        self,
        answer: str,
        documents: List[Document],
        query: str,
        strategy: str,
        validation_result: Optional[ValidationResult] = None,
    ) -> CorrectionResult:
        """
        Apply a specific correction strategy.

        Args:
            answer: Answer to correct
            documents: Documents for context
            query: Original query
            strategy: Correction strategy to apply
            validation_result: Optional validation result for context

        Returns:
            CorrectionResult with corrected answer
        """
        if strategy == "rephrase":
            return self._rephrase_answer(answer, documents)
        elif strategy == "admit_uncertainty":
            return self._admit_uncertainty(query, documents)
        elif strategy == "re-search":
            return self._suggest_search_alternatives(query, validation_result)
        else:
            return CorrectionResult(
                original_answer=answer,
                corrected_answer=answer,
                correction_type=strategy,
                quality_improvement=0.0,
            )

    def _rephrase_answer(
        self, answer: str, documents: List[Document]
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
                original_answer=answer,
                corrected_answer=corrected.strip(),
                correction_type="rephrase",
                quality_improvement=0.7,
                iterations=1,
            )
        except Exception:
            return CorrectionResult(
                original_answer=answer,
                corrected_answer=answer,
                correction_type="rephrase",
                quality_improvement=0.0,
            )

    def _admit_uncertainty(
        self, query: str, documents: List[Document]
    ) -> CorrectionResult:
        """Generate answer that acknowledges uncertainty."""
        prompt = self.UNCERTAINTY_PROMPT.format(query=query or "Unknown question")

        try:
            response = self.llm.invoke(prompt)
            corrected = (
                response.content if hasattr(response, "content") else str(response)
            )

            return CorrectionResult(
                original_answer="",
                corrected_answer=corrected.strip(),
                correction_type="admit_uncertainty",
                quality_improvement=0.6,
                iterations=1,
            )
        except Exception:
            return CorrectionResult(
                original_answer="",
                corrected_answer="I don't have enough information to provide a confident answer.",
                correction_type="admit_uncertainty",
                quality_improvement=0.5,
                iterations=1,
            )

    def _suggest_search_alternatives(
        self, query: str, validation_result: Optional[ValidationResult]
    ) -> CorrectionResult:
        """Suggest alternative search queries."""
        prompt = self.RESEARCH_PROMPT.format(query=query)

        try:
            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Try to parse as JSON list
            import json
            import re

            try:
                suggestions = json.loads(content)
                if isinstance(suggestions, list):
                    search_suggestions = suggestions
                else:
                    # Extract list from JSON
                    match = re.search(r"\[.*\]", content, re.DOTALL)
                    if match:
                        suggestions = json.loads(match.group())
                        if isinstance(suggestions, list):
                            search_suggestions = suggestions
                        else:
                            search_suggestions = [content]
                    else:
                        search_suggestions = [content]
            except Exception:
                search_suggestions = [content]

            return CorrectionResult(
                original_answer="",
                corrected_answer=f"Search suggestions: {', '.join(search_suggestions[:3])}",
                correction_type="re-search",
                quality_improvement=0.5,
                iterations=1,
            )
        except Exception:
            return CorrectionResult(
                original_answer="",
                corrected_answer="Unable to generate search alternatives.",
                correction_type="re-search",
                quality_improvement=0.0,
                iterations=1,
            )


class CorrectiveRAG:
    """
    Corrective RAG implementation for hallucination reduction.

    This class implements CRAG techniques to detect and correct
    hallucinations in generated answers, improving reliability.

    Attributes:
        llm: Language model for correction operations
        answer_validator: Validator for answer quality
        correction_engine: Engine for applying corrections
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
        self.correction_engine = CorrectionEngine(
            llm=llm,
            config=CorrectionEngineConfig(
                quality_threshold=correction_threshold,
                max_correction_attempts=max_correction_attempts,
            ),
        )
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

        if self.correction_engine.should_correct(validation):
            correction = self.correction_engine.correct(
                validation_result=validation,
                query=query or "",
                documents=documents,
                search_available=len(documents) > 0,
            )
            return correction.corrected_answer

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
        validation = self.answer_validator.validate(answer, documents, query)

        # Determine if hallucinated based on status
        is_hallucinated = validation.status in [
            ValidationStatus.INVALID,
            ValidationStatus.HALLUCINATED,
        ]

        # Calculate confidence
        confidence = 1.0 - validation.quality_score if is_hallucinated else validation.quality_score

        return is_hallucinated, confidence

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
        # For backwards compatibility, support CorrectionStrategy enum
        if strategy is not None:
            strategy_map = {
                CorrectionStrategy.REPHRASE: "rephrase",
                CorrectionStrategy.ADMIT_UNCERTAINTY: "admit_uncertainty",
                CorrectionStrategy.RETRIEVE_AGAIN: "re-search",
                CorrectionStrategy.USE_EXTERNAL_KNOWLEDGE: "re-search",
            }
            strategy = strategy_map.get(strategy, "rephrase")

            # Manual correction with specific strategy
            if strategy == "rephrase":
                result = self.correction_engine._rephrase_answer(answer, documents)
            elif strategy == "admit_uncertainty":
                result = self.correction_engine._admit_uncertainty(query or "", documents)
            elif strategy == "re-search":
                validation = self.answer_validator.validate(answer, documents, query)
                result = self.correction_engine._suggest_search_alternatives(query or "", validation)
            else:
                result = self.correction_engine.apply_correction(
                    answer, documents, query or "", strategy
                )

            return result.corrected_answer

        # Auto-select strategy
        validation = self.answer_validator.validate(answer, documents, query)

        if self.correction_engine.should_correct(validation):
            correction = self.correction_engine.correct(
                validation_result=validation,
                query=query or "",
                documents=documents,
                search_available=len(documents) > 0,
            )
            return correction.corrected_answer

        return answer

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
        # Map CorrectionStrategy enum to string
        strategy_map = {
            CorrectionStrategy.REPHRASE: "rephrase",
            CorrectionStrategy.ADMIT_UNCERTAINTY: "admit_uncertainty",
            CorrectionStrategy.RETRIEVE_AGAIN: "re-search",
            CorrectionStrategy.USE_EXTERNAL_KNOWLEDGE: "re-search",
        }
        strategy_str = strategy_map.get(strategy, "rephrase")

        validation = self.answer_validator.validate(answer, documents, query)

        return self.correction_engine.apply_correction(
            answer=answer,
            documents=documents,
            query=query or "",
            strategy=strategy_str,
            validation_result=validation,
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
        return validation.quality_score

    def get_correction_statistics(self) -> dict:
        """
        Get statistics about corrections.

        Returns:
            Dictionary with correction statistics
        """
        return {
            "correction_threshold": self.correction_threshold,
            "max_correction_attempts": self.max_correction_attempts,
        }

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
        return validation.quality_score

    def get_correction_statistics(self) -> dict:
        """
        Get statistics about corrections.

        Returns:
            Dictionary with correction statistics
        """
        return {
            "correction_threshold": self.correction_threshold,
            "max_correction_attempts": self.max_correction_attempts,
        }

    def _parse_correction_response(self, response: str) -> Tuple[bool, float]:
        """
        Parse correction response from LLM.

        DEPRECATED: Use ValidationResult instead.

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
