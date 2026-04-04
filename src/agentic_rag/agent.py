"""
Agentic RAG Agent orchestration.

This module provides the main AgenticRAGAgent class that orchestrates
the complete RAG workflow, including retrieval, evaluation, and answer
generation with self-correction capabilities.
"""

from dataclasses import dataclass
from typing import Generator, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever

from .config import AgenticRAGConfig
from .corrective import CorrectiveRAG
from .evaluator import EvaluationResult, RelevanceEvaluator
from .state import AgentState, Document, MessageRole


@dataclass
class AgentResult:
    """
    Result of agent execution.

    Attributes:
        answer: The final generated answer
        documents: Documents used in generating the answer
        search_count: Number of searches performed
        validation_passed: Whether answer passed quality validation
        search_iterations: Number of search iterations before final answer
        hallucination_score: Score indicating potential hallucination (0-1)
    """

    answer: str
    documents: List[Document]
    search_count: int = 0
    validation_passed: bool = True
    search_iterations: int = 0
    hallucination_score: Optional[float] = None


class AgenticRAGAgent:
    """
    Main agent for Agentic RAG workflow.

    This agent orchestrates the complete RAG pipeline:
    1. Retrieves documents based on user query
    2. Evaluates document relevance
    3. Decides whether to search again or generate answer
    4. Generates and validates answer
    5. Applies correction if hallucination detected

    Attributes:
        llm: Language model for query processing and answer generation
        retriever: Document retriever for initial search
        evaluator: Document relevance evaluator
        corrective: CRAG component for validation and correction
        state: Current agent state
        config: Configuration settings
    """

    GENERATION_PROMPT = """
You are an expert assistant. Answer the following question based only on
the provided context. If the context doesn't contain enough information,
state that clearly and provide the best answer you can.

Question: {query}

Context:
{context}

Answer:
"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        evaluator: RelevanceEvaluator,
        corrective: Optional[CorrectiveRAG] = None,
        config: Optional[AgenticRAGConfig] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Initialize the agentic RAG agent.

        Args:
            llm: Language model for processing
            retriever: Document retriever for search
            evaluator: Document relevance evaluator
            corrective: Optional corrective RAG component
            config: Optional configuration (uses defaults if None)
            max_iterations: Override max search iterations (deprecated, use config)
        """
        self.llm = llm
        self.retriever = retriever
        self.evaluator = evaluator
        self.corrective = corrective or CorrectiveRAG(llm=llm)
        self.config = config or AgenticRAGConfig()
        if max_iterations is not None:
            self.config.max_search_iterations = max_iterations
        self.state = AgentState()

    def run(
        self,
        query: str,
        max_iterations: Optional[int] = None,
    ) -> AgentResult:
        """
        Execute the complete RAG workflow for a query.

        Args:
            query: User's question
            max_iterations: Override max search iterations (uses config default)

        Returns:
            AgentResult with answer and metadata
        """
        max_iterations = max_iterations or self.config.max_search_iterations

        # Initialize state
        self.state = AgentState(query=query)
        self.state.add_message(MessageRole.USER, query)

        search_count = 0
        iteration = 0
        final_documents: List[Document] = []

        try:
            while iteration < max_iterations:
                iteration += 1
                self.state.iteration = iteration

                # Retrieve documents
                search_results = self._retrieve_documents(query)
                search_count += 1
                self.state.search_count = search_count
                self.state.search_results = search_results

                # Convert to Document objects
                documents = self._convert_to_documents(search_results)
                self.state.documents = documents

                # Evaluate relevance
                evaluation = self.evaluator.evaluate(query, documents)
                self.state.is_relevant = evaluation.is_relevant

                # Check if we should continue searching
                if not self.evaluator.should_search_again(evaluation):
                    # Documents are relevant, proceed to answer generation
                    final_documents = documents
                    break

                # Documents not relevant, prepare for next search
                query = self._refine_query(query, evaluation)
                self.state.search_query = query

            # Generate answer with retrieved documents
            answer = self._generate_answer(
                query=query,
                documents=final_documents,
            )

            # Validate and potentially correct answer
            is_hallucinated, hallucination_score = self.corrective.check_hallucination(
                answer,
                final_documents,
            )
            self.state.hallucination_score = hallucination_score

            if is_hallucinated:
                answer = self.corrective.correct_answer(answer, final_documents)
                self.state.correction_triggered = True

            self.state.answer = answer
            self.state.validation_passed = True

            return AgentResult(
                answer=answer,
                documents=final_documents,
                search_count=search_count,
                validation_passed=True,
                search_iterations=iteration,
                hallucination_score=hallucination_score,
            )

        except Exception as e:
            # Handle errors gracefully
            self.state.error = str(e)
            self.state.answer = f"Error processing query: {str(e)}"

            return AgentResult(
                answer=self.state.answer,
                documents=[],
                search_count=search_count,
                validation_passed=False,
                search_iterations=iteration,
                hallucination_score=1.0,
            )

    def _retrieve_documents(self, query: str) -> List[dict]:
        """
        Retrieve documents using the retriever.

        Args:
            query: Search query

        Returns:
            List of search results as dictionaries
        """
        try:
            results = self.retriever.invoke(query)

            # Convert to list of dicts
            search_results = []
            for doc in results:
                search_results.append(
                    {
                        "id": doc.metadata.get("id", str(len(search_results))),
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, "score", None),
                    }
                )

            return search_results

        except Exception:
            return []

    def _convert_to_documents(
        self,
        search_results: List[dict],
    ) -> List[Document]:
        """
        Convert search results to Document objects.

        Args:
            search_results: List of search result dictionaries

        Returns:
            List of Document objects
        """
        documents = []
        for result in search_results:
            doc = Document(
                page_content=result.get("content", ""),
                metadata=result.get("metadata", {}),
                score=result.get("score"),
            )
            documents.append(doc)
        return documents

    def _generate_answer(
        self,
        query: str,
        documents: List[Document],
    ) -> str:
        """
        Generate answer from retrieved documents.

        Args:
            query: User's query
            documents: Retrieved documents

        Returns:
            Generated answer
        """
        # Build context
        context = "\n\n".join([doc.page_content for doc in documents])

        # Create generation prompt
        prompt = self.GENERATION_PROMPT.format(
            query=query,
            context=context if context else "No relevant documents found.",
        )

        # Generate answer
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        self.state.add_message(MessageRole.ASSISTANT, answer)

        return answer

    def _refine_query(
        self,
        query: str,
        evaluation: EvaluationResult,  # noqa: ARG002
    ) -> str:
        """
        Refine query for next search iteration.

        Args:
            query: Original query
            evaluation: Evaluation result explaining why search failed

        Returns:
            Refined query for better results
        """
        # For now, use the original query
        # Could be extended to use LLM to generate better query
        return query

    def stream(
        self,
        query: str,
        max_iterations: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Stream agent results as they become available.

        Args:
            query: User's question
            max_iterations: Override max search iterations

        Yields:
            Chunks of the generated answer
        """
        result = self.run(query, max_iterations)

        # Stream the answer in chunks
        chunk_size = 50
        for i in range(0, len(result.answer), chunk_size):
            yield result.answer[i : i + chunk_size]

        # Yield metadata
        yield f"\n---\nSearch iterations: {result.search_iterations}\n"
        yield f"Documents used: {len(result.documents)}\n"
        if result.hallucination_score is not None:
            yield f"Hallucination score: {result.hallucination_score:.2f}"
