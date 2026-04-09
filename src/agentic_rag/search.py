"""
Tavily Search Integration for Agentic RAG.

This module provides the TavilySearch class for integrating web search
capabilities using Tavily API, enabling the agent to search external
sources when local documents are insufficient.

Enhanced with proper hybrid retrieval integration and query refinement.
"""

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel


@dataclass
class DocumentResult:
    """
    Represents a single search result document.

    Attributes:
        url: URL of the source
        title: Document title
        content: Content/snippet of the document
        score: Relevance score (0-1)
        metadata: Additional metadata
    """

    url: str
    title: str
    content: str
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentResult":
        """Create DocumentResult from dictionary."""
        return cls(
            url=data.get("url", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            score=float(data.get("score", 1.0)),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class SearchResults:
    """
    Results from a search operation.

    Attributes:
        query: The search query
        documents: List of result documents
        total_results: Total number of results found
        error: Error message if search failed
        search_metadata: Additional search metadata
    """

    query: str
    documents: List[DocumentResult] = field(default_factory=list)
    total_results: int = 0
    error: Optional[str] = None
    search_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResults":
        """Create SearchResults from dictionary."""
        documents = [
            DocumentResult.from_dict(doc) if isinstance(doc, dict) else doc
            for doc in data.get("documents", [])
        ]

        return cls(
            query=data.get("query", ""),
            documents=documents,
            total_results=data.get("total_results", len(documents)),
            error=data.get("error"),
            search_metadata=data.get("search_metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "documents": [doc.to_dict() for doc in self.documents],
            "total_results": self.total_results,
            "error": self.error,
            "search_metadata": self.search_metadata,
        }


class TavilySearch:
    """
    Wrapper for Tavily API search functionality.

    This class provides an interface to Tavily's search API for retrieving
    web-based information when local documents are insufficient.

    Attributes:
        api_key: Tavily API key
        client: Tavily client instance
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        tavily_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize TavilySearch.

        Args:
            api_key: Tavily API key (falls back to environment variable)
            tavily_client: Optional pre-initialized Tavily client
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

        if tavily_client:
            self.client = tavily_client
        else:
            self.client = self._create_client()
            if self.api_key is None:
                self.client = None

    def _create_client(self) -> Any:
        """Create Tavily client instance."""
        try:
            from tavily import TavilyClient

            return TavilyClient(api_key=self.api_key)
        except ImportError as err:
            raise ImportError(
                "tavily-python package is required. "
                "Install with: pip install tavily-python"
            ) from err

    def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        timeout: int = 30,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> SearchResults:
        """
        Perform web search using Tavily API.

        Args:
            query: Search query
            search_depth: "basic" or "advanced"
            max_results: Maximum number of results to return
            timeout: Request timeout in seconds
            include_domains: Optional list of domains to include
            exclude_domains: Optional list of domains to exclude

        Returns:
            SearchResults with query and documents

        Raises:
            Exception: If search fails (handled gracefully, returns empty results)
        """
        if not query.strip():
            return SearchResults(query=query, error="Empty query")

        try:
            # Build search parameters
            params = {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "timeout": timeout,
            }

            # Add domain filters if provided
            if include_domains:
                params["include_domains"] = include_domains
            if exclude_domains:
                params["exclude_domains"] = exclude_domains

            # Execute search
            response = self.client.search(**params)

            # Parse results
            results = self._parse_tavily_response(response, query)

            return results

        except Exception as e:
            # Return empty results on error
            return SearchResults(
                query=query,
                error=f"Search failed: {str(e)}",
            )

    def _parse_tavily_response(
        self, response: Dict[str, Any], query: str
    ) -> SearchResults:
        """
        Parse Tavily API response into SearchResults.

        Args:
            response: Raw response from Tavily API
            query: Original search query

        Returns:
            Parsed SearchResults
        """
        documents = []

        for result in response.get("results", []):
            doc = DocumentResult(
                url=result.get("url", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
                score=float(result.get("score", 1.0)),
                metadata={
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                },
            )
            documents.append(doc)

        return SearchResults(
            query=query,
            documents=documents,
            total_results=len(documents),
            search_metadata=response.get("metadata", {}),
        )

    def search_with_query_refinement(
        self,
        query: str,
        iterations: int = 3,
        query_refiner: Optional[Any] = None,
    ) -> SearchResults:
        """
        Perform multiple searches with query refinement.

        Args:
            query: Initial search query
            iterations: Number of refinement iterations
            query_refiner: Optional query refiner (uses default if None)

        Returns:
            SearchResults with aggregated results
        """
        all_documents: List[DocumentResult] = []
        seen_urls = set()

        current_query = query
        refiner = query_refiner or self._create_default_refiner()

        for i in range(iterations):
            results = self.search(current_query)

            # Add new documents
            for doc in results.documents:
                if doc.url not in seen_urls:
                    all_documents.append(doc)
                    seen_urls.add(doc.url)

            # Stop if we have enough results
            if len(all_documents) >= 10:
                break

            # Refine query if not last iteration
            if i < iterations - 1:
                feedback = "Continuing search for more information"
                current_query = refiner.refine(
                    current_query,
                    feedback,
                    [
                        {
                            "query": current_query,
                            "documents": [d.to_dict() for d in results.documents],
                        }
                    ],
                )

        return SearchResults(
            query=query,
            documents=all_documents,
            total_results=len(all_documents),
        )

    def _create_default_refiner(self) -> "QueryRefiner":
        """Create a default QueryRefiner instance."""
        from .search import QueryRefiner

        # Try to get LLM from globals or return a basic refiner
        return QueryRefiner(llm=None)


class QueryRefiner:
    """
    Refines queries for better search results using LLM.

    This class analyzes search failures and generates alternative queries
    that are more likely to retrieve relevant documents.

    Attributes:
        llm: Language model for query refinement
        refinement_prompt: Prompt template for query refinement
    """

    REFINEMENT_PROMPT = """
You are a search query refinement assistant. Your task is to improve
search queries based on previous search failures.

Previous query: {previous_query}
Search feedback: {search_feedback}
Search history: {search_history}

Analyze why the previous search failed to find relevant documents
and generate a refined query that addresses the issue. Consider:
- Using different keywords or synonyms
- Being more specific or more general
- Adding context or constraints
- Changing the query structure

Generate ONLY the refined query, nothing else.
"""

    def __init__(self, llm: Optional[BaseLanguageModel] = None) -> None:
        """
        Initialize QueryRefiner.

        Args:
            llm: Language model for query refinement (optional, uses heuristic if None)
        """
        self.llm = llm
        self._default_queries = []

    def refine(
        self,
        previous_query: str,
        search_feedback: str,
        search_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Refine a query based on search feedback.

        Args:
            previous_query: The query that failed
            search_feedback: Explanation of why search failed
            search_history: Previous search attempts and results

        Returns:
            Refined query string
        """
        search_history_str = (
            "\n".join(
                [
                    f"- Query: {h['query']}, Results: {len(h.get('documents', []))}"
                    for h in (search_history or [])
                ]
            )
            or "No previous searches"
        )

        prompt = self.REFINEMENT_PROMPT.format(
            previous_query=previous_query,
            search_feedback=search_feedback,
            search_history=search_history_str,
        )

        if self.llm:
            response = self.llm.invoke(prompt)
            refined_query = (
                response.content if hasattr(response, "content") else str(response)
            ).strip()
        else:
            # Heuristic-based refinement
            refined_query = self._heuristic_refine(previous_query, search_feedback)

        # Clean up the response
        return self._clean_query(refined_query)

    def _clean_query(self, query: str) -> str:
        """Clean up a query string by removing quotes and extra whitespace."""
        if not query:
            return query
        query = query.strip()
        if query.startswith('"') or query.startswith("'"):
            query = query[1:-1]
        return query.strip()

    def _heuristic_refine(self, query: str, feedback: str) -> str:
        """Apply heuristic-based query refinement without LLM.

        Args:
            query: Original search query
            feedback: Feedback explaining why previous search failed

        Returns:
            Refined query that addresses the feedback
        """
        # Use feedback to guide refinement strategy
        feedback_lower = feedback.lower()

        # Strategy 1: If feedback mentions "irrelevant" or "not relevant", make more specific
        if "irrelevant" in feedback_lower or "not relevant" in feedback_lower:
            return self._make_query_more_specific(query)

        # Strategy 2: If feedback mentions "no results" or "empty", try broader terms
        if (
            "no results" in feedback_lower
            or "empty" in feedback_lower
            or "none found" in feedback_lower
        ):
            return self._expand_query(query)

        # Strategy 3: If feedback mentions "limited" or "few", adjust scope
        if "limited" in feedback_lower or "few" in feedback_lower:
            return self._adjust_scope(query)

        # Default: Use keyword extraction with stopword removal
        words = query.lower().split()

        # Common words to remove
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "just",
            "don",
            "now",
        }

        # Filter out stopwords
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        if len(keywords) < 2:
            # If too few keywords, try different phrasing
            alternatives = self._generate_alternatives(query)
            return alternatives[0] if alternatives else query

        # Reconstruct query with keywords
        refined = " ".join(keywords)

        return refined

    def _make_query_more_specific(self, query: str) -> str:
        """Make query more specific by adding qualifiers.

        Args:
            query: Original query

        Returns:
            More specific query
        """
        # Add specificity keywords to narrow down results
        specifics = ["specifically", "detailed", "comprehensive", "in depth"]
        return f"{query} {specifics[0]}"

    def _expand_query(self, query: str) -> str:
        """Expand query with broader terms.

        Args:
            query: Original query

        Returns:
            Broader query
        """
        # Add broadening keywords
        broadenings = ["overview", "general", "introduction to"]
        return f"{broadenings[2]} {query}"

    def _adjust_scope(self, query: str) -> str:
        """Adjust query scope based on result count.

        Args:
            query: Original query

        Returns:
            Adjusted query
        """
        # Remove limiting qualifiers if present
        limiters = ["only", "just", "exactly", "specifically"]
        words = query.split()
        filtered_words = [w for w in words if w.lower() not in limiters]
        return " ".join(filtered_words) if filtered_words else query

    def _generate_alternatives(self, query: str) -> List[str]:
        """Generate alternative query phrasings."""
        alternatives = [query]

        # Add variations
        words = query.split()
        if len(words) > 1:
            # Try different word orders
            alternatives.append(" ".join(reversed(words)))

            # Try with "and" or "or"
            if "and" not in query.lower():
                alternatives.append(" and ".join(words))
            if "or" not in query.lower():
                alternatives.append(" or ".join(words))

        return list(set(alternatives))

    def suggest_alternative_keywords(self, query: str) -> List[str]:
        """
        Suggest alternative keywords for a query.

        Args:
            query: Original query

        Returns:
            List of alternative keyword suggestions
        """
        words = query.lower().split()
        variations = [query]

        # Add variations with different word orders
        if len(words) > 1:
            variations.append(" ".join(reversed(words)))

        # Add variations with synonyms (basic implementation)
        synonym_map = {
            "how": ["ways to", "methods of", "approaches to"],
            "what": ["explain", "describe", "define"],
            "why": ["reason for", "causes of", "factors behind"],
            "when": ["time of", "date of", "period of"],
            "where": ["location of", "situated at", "found in"],
            "which": ["select from", "choose between", "determine which"],
        }

        for key, values in synonym_map.items():
            if key in words:
                for value in values:
                    variations.append(query.replace(key, value, 1))

        return list(set(variations))

    def make_query_more_specific(self, query: str, context: str) -> str:
        """
        Make a vague query more specific based on context.

        Args:
            query: Original vague query
            context: Additional context to narrow down the query

        Returns:
            More specific query
        """
        # Simple implementation: append context if query is too short
        if len(query.split()) < 3:
            return f"{query} in the context of {context[:50]}..."

        return query

    def expand_query(self, query: str, available_context: List[str]) -> str:
        """
        Expand query using available context information.

        Args:
            query: Original query
            available_context: Related context snippets

        Returns:
            Expanded query with additional context
        """
        # Simple implementation: add context keywords to query
        if not available_context:
            return query

        # Extract key terms from context
        context_keywords = " ".join(available_context[:2])
        return f"{query} related to {context_keywords[:100]}"


@dataclass
class HybridRetrievalResult:
    """
    Results from hybrid retrieval combining multiple sources.

    Attributes:
        documents: Combined and ranked documents
        local_count: Number of documents from local source
        tavily_count: Number of documents from Tavily
        search_time: Total retrieval time
        error: Error message if any
    """

    documents: List[Document] = field(default_factory=list)
    local_count: int = 0
    tavily_count: int = 0
    search_time: float = 0.0
    error: Optional[str] = None


class HybridRetriever:
    """
    Combines local vector search with Tavily web search.

    This class implements a hybrid retrieval strategy that first attempts
    local retrieval, then falls back to Tavily web search when local
    documents are insufficient or of low quality.

    Attributes:
        local_retriever: Local document retriever
        tavily_search: Tavily search integration
        query_refiner: Query refinement component
        use_tavily_fallback: Whether to use Tavily as fallback
        tavily_priority: Weight for Tavily results (0-1)
    """

    def __init__(
        self,
        local_retriever: "BaseRetriever",
        tavily_search: TavilySearch,
        query_refiner: Optional[QueryRefiner] = None,
        use_tavily_fallback: bool = True,
        tavily_priority: float = 0.3,
        local_min_score: float = 0.5,
    ) -> None:
        """
        Initialize HybridRetriever.

        Args:
            local_retriever: Local document retriever
            tavily_search: Tavily search integration
            query_refiner: Optional query refiner
            use_tavily_fallback: Use Tavily when local fails
            tavily_priority: Weight for Tavily results
            local_min_score: Minimum local score to avoid Tavily
        """
        self.local_retriever = local_retriever
        self.tavily_search = tavily_search
        self.query_refiner = query_refiner or QueryRefiner(llm=None)
        self.use_tavily_fallback = use_tavily_fallback
        self.tavily_priority = tavily_priority
        self.local_min_score = local_min_score

    def retrieve(
        self,
        query: str,
        search_history: Optional[List[Dict[str, Any]]] = None,
        eval_feedback: Optional[Dict[str, Any]] = None,
        max_local_results: int = 5,
        max_tavily_results: int = 5,
    ) -> HybridRetrievalResult:
        """
        Perform hybrid retrieval combining local and web search.

        Args:
            query: Search query
            search_history: Previous searches for query refinement
            eval_feedback: Evaluation feedback from previous attempts
            max_local_results: Maximum local results to consider
            max_tavily_results: Maximum Tavily results to fetch

        Returns:
            HybridRetrievalResult with combined documents
        """
        start_time = time.time()

        try:
            # Step 1: Local retrieval
            local_docs = self._retrieve_local(query, max_results=max_local_results)

            # Step 2: Evaluate local results
            need_tavily = self._should_use_tavily(local_docs, eval_feedback)

            # Step 3: Tavily retrieval if needed
            tavily_docs = []
            if need_tavily and self.use_tavily_fallback:
                refined_query = query
                if search_history and len(search_history) > 0:
                    # Refine query based on history
                    feedback = (
                        eval_feedback.get("reason", "Insufficient results")
                        if eval_feedback
                        else "Retrieving additional information"
                    )
                    refined_query = self.query_refiner.refine(
                        query, feedback, search_history
                    )

                tavily_results = self.tavily_search.search(
                    query=refined_query, max_results=max_tavily_results
                )
                tavily_docs = tavily_results.documents

            # Step 4: Merge and rank
            merged_docs = self._merge_and_rank(local_docs, tavily_docs)

            search_time = time.time() - start_time

            return HybridRetrievalResult(
                documents=merged_docs,
                local_count=len(local_docs),
                tavily_count=len(tavily_docs),
                search_time=search_time,
            )

        except Exception as e:
            return HybridRetrievalResult(
                documents=[],
                error=str(e),
            )

    def _retrieve_local(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents from local source.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of local document results
        """
        try:
            results = self.local_retriever.invoke(query)

            local_docs = []
            for doc in results:
                local_docs.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, "score", 0.5),
                        "source": "local",
                        "document": doc,
                    }
                )

            # Sort by score and limit
            local_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
            return local_docs[:max_results]

        except Exception:
            return []

    def _should_use_tavily(
        self,
        local_docs: List[Dict[str, Any]],
        eval_feedback: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Determine if Tavily search should be used.

        Args:
            local_docs: Local document results
            eval_feedback: Evaluation feedback

        Returns:
            True if Tavily should be used
        """
        # Use Tavily if local results are insufficient
        if len(local_docs) == 0:
            return True

        # Use Tavily if local scores are too low
        avg_score = sum(d.get("score", 0) for d in local_docs) / len(local_docs)
        if avg_score < self.local_min_score:
            return True

        # Use Tavily if evaluation suggests it
        return bool(eval_feedback and eval_feedback.get("should_search_again", False))

    def _merge_and_rank(
        self,
        local_docs: List[Dict[str, Any]],
        tavily_docs: List[Any],
    ) -> List[Document]:
        """
        Merge local and Tavily documents with appropriate ranking.

        Args:
            local_docs: Local document results
            tavily_docs: Tavily document results

        Returns:
            Combined and ranked List[Document]
        """
        # Convert Tavily docs to compatible format
        tavily_converted = []
        for doc in tavily_docs:
            if hasattr(doc, "to_dict"):
                tavily_doc = Document(
                    page_content=doc.content,
                    metadata={
                        "url": doc.url,
                        "title": doc.title,
                        "source": "tavily",
                    },
                    score=float(doc.score),
                )
                tavily_converted.append(tavily_doc)
            elif isinstance(doc, dict):
                doc_copy = doc.copy()
                doc_copy["source"] = "tavily"
                tavily_doc = Document(
                    page_content=doc_copy.get("content", ""),
                    metadata=doc_copy.get("metadata", {}),
                    score=float(doc_copy.get("score", 0)),
                )
                tavily_converted.append(tavily_doc)

        # Apply Tavily priority weight
        if tavily_converted:
            for doc in tavily_converted:
                if hasattr(doc, "score"):
                    doc.score = (doc.score or 0) * (1 - self.tavily_priority)

        # Combine all documents - convert local to Document objects first
        all_docs = []

        # Convert local docs to Document objects
        for doc_data in local_docs:
            if isinstance(doc_data, Document):
                all_docs.append(doc_data)
            else:
                doc = Document(
                    page_content=doc_data.get("content", ""),
                    metadata=doc_data.get("metadata", {}),
                    score=float(doc_data.get("score", 0.5)),
                )
                all_docs.append(doc)

        # Add tavily docs
        all_docs.extend(tavily_converted)

        # Rank by score
        all_docs.sort(key=lambda x: x.score or 0, reverse=True)

        return all_docs

    def search_with_correction(
        self,
        query: str,
        previous_results: List[Dict[str, Any]],
        correction_feedback: str,
    ) -> HybridRetrievalResult:
        """
        Search with query correction based on feedback.

        Args:
            query: Original query
            previous_results: Previous search results
            correction_feedback: Feedback from correction engine

        Returns:
            HybridRetrievalResult with corrected search results
        """
        # Refine query based on correction feedback
        refined_query = self.query_refiner.refine(
            previous_query=query,
            search_feedback=correction_feedback,
            search_history=[{"query": query, "documents": previous_results}],
        )

        # Perform new search with refined query
        return self.retrieve(
            query=refined_query,
            search_history=[{"query": query, "documents": previous_results}],
        )


class TavilySearchIntegration:
    """
    Enhanced Tavily search integration with query refinement.

    This class provides advanced Tavily search capabilities including
    query refinement, search history management, and result consolidation.

    Attributes:
        tavily_search: Base TavilySearch instance
        query_refiner: Query refinement component
        search_history: History of search queries and results
    """

    def __init__(
        self,
        tavily_search: TavilySearch,
        query_refiner: Optional[QueryRefiner] = None,
    ) -> None:
        """
        Initialize TavilySearchIntegration.

        Args:
            tavily_search: Base TavilySearch instance
            query_refiner: Optional query refiner
        """
        self.tavily_search = tavily_search
        self.query_refiner = query_refiner or QueryRefiner(llm=None)
        self.search_history: List[Dict[str, Any]] = []

    def search(
        self,
        query: str,
        use_refinement: bool = True,
        max_iterations: int = 3,
        **search_params: Any,
    ) -> SearchResults:
        """
        Perform Tavily search with optional query refinement.

        Args:
            query: Search query
            use_refinement: Whether to use query refinement
            max_iterations: Maximum refinement iterations
            **search_params: Additional Tavily search parameters

        Returns:
            SearchResults with query and documents
        """
        all_documents: List[DocumentResult] = []
        seen_urls = set()

        current_query = query
        for iteration in range(max_iterations):
            # Perform search
            results = self.tavily_search.search(current_query, **search_params)

            # Add new documents
            for doc in results.documents:
                if doc.url not in seen_urls:
                    all_documents.append(doc)
                    seen_urls.add(doc.url)

            # Record in history
            self.search_history.append(
                {
                    "query": current_query,
                    "iteration": iteration + 1,
                    "documents": [doc.to_dict() for doc in results.documents],
                    "total": len(results.documents),
                }
            )

            # Stop if we have enough results
            if len(all_documents) >= 10 or iteration == max_iterations - 1:
                break

            # Refine query if needed
            if use_refinement and iteration < max_iterations - 1:
                current_query = self.query_refiner.refine(
                    previous_query=current_query,
                    search_feedback="Continuing search for more information",
                    search_history=self.search_history[-2:],
                )

        return SearchResults(
            query=query,
            documents=all_documents,
            total_results=len(all_documents),
        )

    def refine_and_search(
        self,
        query: str,
        evaluation_feedback: Dict[str, Any],
        **search_params: Any,
    ) -> SearchResults:
        """
        Refine query based on evaluation feedback and search.

        Args:
            query: Original query
            evaluation_feedback: Feedback from document evaluation
            **search_params: Additional Tavily search parameters

        Returns:
            SearchResults with refined query results
        """
        # Refine query based on feedback
        feedback_text = evaluation_feedback.get(
            "reason", "Insufficient document relevance"
        )
        refined_query = self.query_refiner.refine(
            previous_query=query,
            search_feedback=feedback_text,
            search_history=self.search_history[-3:],
        )

        # Perform search with refined query
        return self.search(refined_query, **search_params)

    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about search usage.

        Returns:
            Dictionary with search statistics
        """
        if not self.search_history:
            return {
                "total_searches": 0,
                "total_documents": 0,
                "avg_documents_per_search": 0,
            }

        total_docs = sum(h["total"] for h in self.search_history)

        return {
            "total_searches": len(self.search_history),
            "total_documents": total_docs,
            "avg_documents_per_search": total_docs / len(self.search_history),
            "last_query": (
                self.search_history[-1]["query"] if self.search_history else None
            ),
        }
