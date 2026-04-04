"""
Tavily Search Integration for Agentic RAG.

This module provides the TavilySearch class for integrating web search
capabilities using Tavily API, enabling the agent to search external
sources when local documents are insufficient.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    ) -> SearchResults:
        """
        Perform multiple searches with query refinement.

        Args:
            query: Initial search query
            iterations: Number of refinement iterations

        Returns:
            SearchResults with aggregated results
        """
        all_documents: List[DocumentResult] = []
        seen_urls = set()

        current_query = query
        for _i in range(iterations):
            results = self.search(current_query)

            # Add new documents
            for doc in results.documents:
                if doc.url not in seen_urls:
                    all_documents.append(doc)
                    seen_urls.add(doc.url)

            # Stop if we have enough results
            if len(all_documents) >= 10:
                break

            # Could refine query here based on results
            # For now, just continue with original query

        return SearchResults(
            query=query,
            documents=all_documents,
            total_results=len(all_documents),
        )
