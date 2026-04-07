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
from .search import (
    DocumentResult,
    HybridRetrievalResult,
    HybridRetriever,
    TavilySearch,
    TavilySearchIntegration,
)
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
        tavily_used: Whether Tavily web search was used
        tavily_document_count: Number of documents from Tavily
        local_document_count: Number of documents from local source
        total_documents: Total number of documents used
    """

    answer: str
    documents: List[Document]
    search_count: int = 0
    validation_passed: bool = True
    search_iterations: int = 0
    hallucination_score: Optional[float] = None
    tavily_used: bool = False
    tavily_document_count: int = 0
    local_document_count: int = 0
    total_documents: int = 0


class AgenticRAGAgent:
    """
    Main agent for Agentic RAG workflow.

    This agent orchestrates the complete RAG pipeline:
    1. Retrieves documents based on user query (local + Tavily)
    2. Evaluates document relevance
    3. Decides whether to search again or generate answer
    4. Generates and validates answer
    5. Applies correction if hallucination detected

    Attributes:
        llm: Language model for query processing and answer generation
        retriever: Document retriever for initial search
        tavily_search: Tavily search integration for web search
        evaluator: Document relevance evaluator
        corrective: CRAG component for validation and correction
        hybrid_retriever: Hybrid retriever combining local + Tavily
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
        local_retriever: BaseRetriever,
        evaluator: RelevanceEvaluator,
        tavily_search: Optional[TavilySearch] = None,
        corrective: Optional[CorrectiveRAG] = None,
        config: Optional[AgenticRAGConfig] = None,
        max_iterations: Optional[int] = None,
        use_hybrid_retrieval: bool = True,
        tavily_priority: float = 0.3,
    ) -> None:
        """
        Initialize the agentic RAG agent.

        Args:
            llm: Language model for processing
            local_retriever: Local document retriever for search
            evaluator: Document relevance evaluator
            tavily_search: Optional Tavily search for web search
            corrective: Optional corrective RAG component
            config: Optional configuration (uses defaults if None)
            max_iterations: Override max search iterations (deprecated, use config)
            use_hybrid_retrieval: Whether to use hybrid local+Tavily retrieval
            tavily_priority: Weight for Tavily results (0-1)
        """
        self.llm = llm
        self.local_retriever = local_retriever
        self.evaluator = evaluator
        self.tavily_search = tavily_search
        self.corrective = corrective or CorrectiveRAG(llm=llm)
        self.config = config or AgenticRAGConfig()
        if max_iterations is not None:
            self.config.max_search_iterations = max_iterations

        # Initialize hybrid retriever if Tavily is available
        self.use_hybrid_retrieval = use_hybrid_retrieval
        if use_hybrid_retrieval and tavily_search:
            from .search import QueryRefiner

            query_refiner = QueryRefiner(llm=llm)
            self.hybrid_retriever = HybridRetriever(
                local_retriever=local_retriever,
                tavily_search=tavily_search,
                query_refiner=query_refiner,
                tavily_priority=tavily_priority,
            )
        else:
            self.hybrid_retriever = None

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
                if self.use_hybrid_retrieval and self.hybrid_retriever:
                    # Use hybrid retrieval (local + Tavily)
                    search_result = self._retrieve_documents_hybrid(query)
                    search_count = self._get_search_count_from_hybrid(search_result)
                    documents = self._convert_hybrid_to_documents(search_result)
                else:
                    # Use local retrieval only
                    search_results = self._retrieve_documents_local(query)
                    search_count += 1
                    documents = self._convert_to_documents(search_results)

                self.state.search_count = search_count
                self.state.search_results = (
                    search_result.to_dict()
                    if self.use_hybrid_retrieval
                    else search_results
                )
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
                tavily_used=search_result.tavily_count > 0
                if self.use_hybrid_retrieval
                else False,
                tavily_document_count=search_result.tavily_count
                if self.use_hybrid_retrieval
                else 0,
                local_document_count=search_result.local_count
                if self.use_hybrid_retrieval
                else len(final_documents),
                total_documents=len(final_documents),
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
                tavily_used=False,
                tavily_document_count=0,
                local_document_count=0,
                total_documents=0,
            )

    def _retrieve_documents_local(self, query: str) -> List[dict]:
        """
        Retrieve documents using local retriever only.

        Args:
            query: Search query

        Returns:
            List of search results as dictionaries
        """
        try:
            results = self.local_retriever.invoke(query)

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

    def _retrieve_documents_hybrid(
        self, query: str
    ) -> HybridRetrievalResult:
        """
        Retrieve documents using hybrid retriever (local + Tavily).

        Args:
            query: Search query

        Returns:
            HybridRetrievalResult with combined documents
        """
        if not self.hybrid_retriever:
            # Fallback to local retrieval if hybrid not available
            local_docs = self._retrieve_documents_local(query)
            return HybridRetrievalResult(
                documents=[
                    {
                        "content": d["content"],
                        "metadata": d["metadata"],
                        "score": d["score"],
                        "source": "local",
                    }
                    for d in local_docs
                ],
                local_count=len(local_docs),
                tavily_count=0,
            )

        # Get evaluation feedback from state if available
        eval_feedback = None
        if hasattr(self.state, "evaluation_result") and self.state.evaluation_result:
            eval_feedback = self.state.evaluation_result.to_dict()

        # Perform hybrid retrieval
        result = self.hybrid_retriever.retrieve(
            query=query,
            search_history=getattr(self.state, "search_history", []),
            eval_feedback=eval_feedback,
        )

        return result

    def _get_search_count_from_hybrid(
        self, result: HybridRetrievalResult
    ) -> int:
        """
        Get search count from hybrid retrieval result.

        Args:
            result: HybridRetrievalResult

        Returns:
            Search count (1 for local, 1 for Tavily if used)
        """
        count = 1  # Local search always happens
        if result.tavily_count > 0:
            count += 1
        return count

    def _convert_hybrid_to_documents(
        self, result: HybridRetrievalResult
    ) -> List[Document]:
        """
        Convert hybrid retrieval result to Document objects.

        Args:
            result: HybridRetrievalResult

        Returns:
            List of Document objects
        """
        documents = []
        for doc_data in result.documents:
            # Determine source
            source = doc_data.get("source", "local")

            # Create Document
            doc = Document(
                page_content=doc_data.get("content", ""),
                metadata={
                    **doc_data.get("metadata", {}),
                    "source": source,
                },
                score=doc_data.get("score", 0.5),
            )
            documents.append(doc)

        return documents

    def _retrieve_documents(self, query: str) -> List[dict]:
        """
        Retrieve documents using the retriever.

        DEPRECATED: Use _retrieve_documents_local or _retrieve_documents_hybrid
        instead. This method is kept for backwards compatibility.

        Args:
            query: Search query

        Returns:
            List of search results as dictionaries
        """
        return self._retrieve_documents_local(query)

    def _convert_to_documents(
        self,
        search_results: List[dict],
    ) -> List[Document]:
        """
        Convert search results to Document objects.

        DEPRECATED: Use _convert_hybrid_to_documents for hybrid results.
        This method is kept for backwards compatibility.

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
