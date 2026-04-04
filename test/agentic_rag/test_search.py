"""
Tests for Tavily Search Integration in Agentic RAG.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Force reimport
if "agentic_rag" in sys.modules:
    del sys.modules["agentic_rag"]

from src.agentic_rag.search import SearchResults, TavilySearch


class TestTavilySearch:
    """Test suite for TavilySearch class."""

    @pytest.fixture
    def mock_tavily_client(self):
        """Create a mock Tavily client."""
        mock = MagicMock()
        mock.search.return_value = {
            "results": [
                {
                    "url": "https://example.com/doc1",
                    "title": "Document 1",
                    "content": "Relevant content about the topic",
                    "score": 0.9,
                },
                {
                    "url": "https://example.com/doc2",
                    "title": "Document 2",
                    "content": "Additional relevant information",
                    "score": 0.8,
                },
            ]
        }
        return mock

    @pytest.fixture
    def tavily_search(self, mock_tavily_client):
        """Create TavilySearch instance with mock client."""
        from src.agentic_rag.search import TavilySearch

        return TavilySearch(tavily_client=mock_tavily_client)

    def test_search_initialization(self, mock_tavily_client):
        """Test TavilySearch initialization."""
        search = TavilySearch(tavily_client=mock_tavily_client)

        assert search.client is mock_tavily_client
        # api_key may be None if using mock client without real key
        assert search.api_key is None or search.api_key != ""

    def test_search_execution(self, tavily_search):
        """Test search returns results in expected format."""
        _results = tavily_search.search(query="Test query")

        assert isinstance(_results, SearchResults)
        assert len(_results.documents) == 2
        assert _results.query == "Test query"
        assert _results.total_results == 2

    def test_search_with_filters(self, tavily_search):
        """Test search with various query filters."""

        # Setup mock to return different results based on depth
        def side_effect(**_kwargs):
            return {
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Test",
                        "content": "Content",
                        "score": 0.9,
                    }
                ]
            }

        tavily_search.client.search.side_effect = side_effect

        _results = tavily_search.search(query="Test query", search_depth="advanced")

        assert len(_results.documents) == 1
        assert _results.total_results == 1

    def test_search_error_handling(self, tavily_search):
        """Test graceful handling of search failures."""
        # Mock client to raise exception
        tavily_search.client.search.side_effect = Exception("API error")

        _results = tavily_search.search(query="Test query")

        # Should return empty results, not raise exception
        assert _results.documents == []
        assert _results.error is not None

    def test_search_timeout_handling(self, tavily_search):
        """Test handling of search timeouts."""
        import time

        def slow_search(*_args, **_kwargs):
            time.sleep(10)  # Simulate timeout
            return {"results": []}

        tavily_search.client.search.side_effect = slow_search

        # Should timeout and return empty results
        _results = tavily_search.search(query="Test query", timeout=1)

        assert _results.documents == []

    def test_search_rate_limit_handling(self, tavily_search):
        """Test handling of rate limit errors."""
        tavily_search.client.search.side_effect = Exception("Rate limit exceeded")

        _results = tavily_search.search(query="Test query")

        assert _results.documents == []
        assert _results.error is not None

    def test_search_empty_query(self, tavily_search):
        """Test search with empty query."""
        results = tavily_search.search(query="")

        assert results.documents == []
        assert results.error is not None

    def test_search_custom_params(self, tavily_search, mock_tavily_client):
        """Test search with custom parameters."""
        # Clear the side_effect to use default mock
        mock_tavily_client.search.side_effect = None
        mock_tavily_client.search.return_value = {
            "results": [
                {
                    "url": "https://example.com",
                    "title": "Test",
                    "content": "Content",
                    "score": 0.9,
                }
            ]
        }

        results = tavily_search.search(
            query="Test query", max_results=5, include_domains=["example.com"]
        )

        assert len(results.documents) == 1

    def test_search_result_parsing(self, tavily_search, mock_tavily_client):
        """Test parsing of search results."""
        # Setup mock with various result fields
        mock_tavily_client.search.return_value = {
            "results": [
                {
                    "url": "https://example.com/article",
                    "title": "Article Title",
                    "content": "Article content here",
                    "score": 0.95,
                    "raw_content": "Full raw content",
                }
            ]
        }

        results = tavily_search.search(query="Test")

        assert len(results.documents) == 1
        doc = results.documents[0]
        assert doc.url == "https://example.com/article"
        assert doc.title == "Article Title"
        assert doc.score == 0.95

    def test_search_multiple_queries(self, tavily_search):
        """Test multiple sequential searches."""
        call_count = [0]

        def side_effect(*_args, **_kwargs):
            call_count[0] += 1
            return {
                "results": [
                    {
                        "url": f"https://example.com/{call_count[0]}",
                        "title": f"Result {call_count[0]}",
                        "content": f"Content {call_count[0]}",
                        "score": 0.9,
                    }
                ]
            }

        tavily_search.client.search.side_effect = side_effect

        # Perform multiple searches
        for i in range(3):
            results = tavily_search.search(query=f"Query {i}")
            assert len(results.documents) == 1
            assert results.documents[0].title == f"Result {i + 1}"


class TestSearchResults:
    """Test suite for SearchResults class."""

    def test_results_creation(self):
        """Test creating search results."""
        from src.agentic_rag.search import DocumentResult

        results = SearchResults(
            query="Test query",
            documents=[
                DocumentResult(
                    url="https://example.com",
                    title="Test",
                    content="Content",
                    score=0.9,
                )
            ],
        )

        assert results.query == "Test query"
        assert len(results.documents) == 1

    def test_results_empty(self):
        """Test empty search results."""
        results = SearchResults(query="Test query")

        assert results.documents == []
        assert results.total_results == 0
        assert results.error is None

    def test_results_with_error(self):
        """Test results with error information."""
        results = SearchResults(query="Test query", error="Search failed", documents=[])

        assert results.error == "Search failed"
        assert results.documents == []

    def test_results_total_count(self):
        """Test total results count."""
        from src.agentic_rag.search import DocumentResult

        results = SearchResults(
            query="Test",
            documents=[
                DocumentResult(
                    url="https://example.com/1", title="1", content="", score=0.9
                ),
                DocumentResult(
                    url="https://example.com/2", title="2", content="", score=0.8
                ),
                DocumentResult(
                    url="https://example.com/3", title="3", content="", score=0.7
                ),
            ],
            total_results=3,
        )

        assert results.total_results == 3

    def test_results_as_dict(self):
        """Test converting results to dictionary."""
        from src.agentic_rag.search import DocumentResult

        results = SearchResults(
            query="Test",
            documents=[
                DocumentResult(
                    url="https://example.com", title="Test", content="", score=0.9
                )
            ],
        )

        results_dict = results.to_dict()

        assert "query" in results_dict
        assert "documents" in results_dict
        assert "total_results" in results_dict

    def test_results_from_dict(self):
        """Test creating results from dictionary."""
        results_dict = {
            "query": "Test",
            "documents": [
                {
                    "url": "https://example.com",
                    "title": "Test",
                    "content": "Content",
                    "score": 0.9,
                }
            ],
            "total_results": 1,
            "error": None,
        }

        results = SearchResults.from_dict(results_dict)

        assert results.query == "Test"
        assert len(results.documents) == 1
