"""
Document Relevance Evaluator for Agentic RAG.

This module provides the RelevanceEvaluator class that assesses whether
retrieved documents are relevant to the user query, enabling the agent
to decide whether to search again or proceed with answer generation.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.language_models import BaseLanguageModel

from .state import Document


@dataclass
class EvaluationResult:
    """
    Result of document relevance evaluation.

    Attributes:
        is_relevant: Whether documents are considered relevant
        confidence: Confidence score in the evaluation (0-1)
        reason: Explanation of the evaluation decision
        document_ids: List of document IDs that were evaluated
    """

    is_relevant: bool
    confidence: float = 0.0
    reason: Optional[str] = None
    document_ids: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"EvaluationResult(is_relevant={self.is_relevant}, "
            f"confidence={self.confidence:.2f}, reason='{self.reason}')"
        )


class RelevanceEvaluator:
    """
    Evaluates relevance of retrieved documents to a query.

    This evaluator uses an LLM to assess whether the retrieved documents
    contain information relevant to answering the user's query. The result
    informs whether the agent should:
    - Proceed with answer generation (relevant documents)
    - Perform another search (irrelevant documents)

    Attributes:
        llm: Language model used for evaluation
        threshold: Relevance threshold for binary decision (0-1)
        evaluation_prompt: Prompt template for evaluation
    """

    EVALUATION_PROMPT = """
You are an expert evaluator. Given a user query and a set of documents,
determine if the documents are relevant to answering the query.

Query: {query}

Documents:
{documents}

Return a JSON response with:
- is_relevant: boolean (true if documents help answer the query)
- confidence: float (0-1, how confident you are in your assessment)
- reason: string (brief explanation of your assessment)

Respond ONLY with valid JSON, no other text.
"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        threshold: float = 0.7,
    ) -> None:
        """
        Initialize the relevance evaluator.

        Args:
            llm: Language model for evaluation
            threshold: Threshold for considering documents relevant (0-1)

        Raises:
            ValueError: If threshold is outside valid range
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

        self.llm = llm
        self.threshold = threshold

    def evaluate(
        self,
        query: str,
        documents: List[Document],
    ) -> EvaluationResult:
        """
        Evaluate whether documents are relevant to the query.

        Args:
            query: The user's query
            documents: List of documents to evaluate

        Returns:
            EvaluationResult with relevance assessment

        Raises:
            ValueError: If query is empty
        """
        if not query.strip():
            raise ValueError("query cannot be empty")

        if not documents:
            return EvaluationResult(
                is_relevant=False,
                confidence=0.0,
                reason="No documents provided",
            )

        # Format documents for evaluation
        doc_text = "\n\n".join([doc.page_content for doc in documents])

        # Create evaluation prompt
        prompt = self.EVALUATION_PROMPT.format(
            query=query,
            documents=doc_text,
        )

        # Invoke LLM for evaluation
        try:
            response = self.llm.invoke(prompt)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Parse JSON response
            result = self._parse_evaluation_response(content)

            # Add document IDs to result
            result.document_ids = [
                doc.metadata.get("id", str(i)) for i, doc in enumerate(documents)
            ]

            return result

        except Exception as e:
            # Fallback for parsing errors
            return EvaluationResult(
                is_relevant=False,
                confidence=0.0,
                reason=f"Evaluation error: {str(e)}",
            )

    def _parse_evaluation_response(self, response: str) -> EvaluationResult:
        """
        Parse LLM response into EvaluationResult.

        Args:
            response: Raw LLM response string

        Returns:
            Parsed EvaluationResult

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Extract JSON from response if needed
            json_str = response.strip()
            if json_str.startswith("{"):
                data = json.loads(json_str)
            else:
                # Try to find JSON in response
                import re

                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                else:
                    raise ValueError("No JSON found in response")

            return EvaluationResult(
                is_relevant=bool(data.get("is_relevant", False)),
                confidence=float(data.get("confidence", 0.0)),
                reason=data.get("reason", "No reason provided"),
            )

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback for malformed JSON
            return EvaluationResult(
                is_relevant=False,
                confidence=0.0,
                reason=f"Failed to parse response: {str(e)}",
            )

    def should_search_again(self, result: EvaluationResult) -> bool:
        """
        Determine if another search is needed based on evaluation result.

        Args:
            result: EvaluationResult from evaluate()

        Returns:
            True if another search should be performed
        """
        return not result.is_relevant or result.confidence < self.threshold

    def get_relevant_documents(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """
        Filter documents to only those marked as relevant.

        Args:
            query: The user's query
            documents: List of documents to filter

        Returns:
            List of relevant documents
        """
        evaluation = self.evaluate(query, documents)

        if evaluation.is_relevant:
            return documents
        return []
