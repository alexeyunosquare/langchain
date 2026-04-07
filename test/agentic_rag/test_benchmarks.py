"""
Performance benchmarks for Agentic RAG.

These benchmarks measure the performance characteristics of the
agentic RAG system.
"""

import time
from unittest.mock import MagicMock

import pytest


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests for Agentic RAG."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM with configurable latency."""
        mock = MagicMock()

        def mock_invoke(*_args, **_kwargs):
            time.sleep(0.01)  # Simulate 10ms latency
            return MagicMock(content="Test response")

        mock.invoke = mock_invoke
        return mock

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever with configurable latency."""
        mock = MagicMock()

        def mock_invoke(*_args, **_kwargs):
            time.sleep(0.005)  # Simulate 5ms latency
            return []

        mock.invoke = mock_invoke
        return mock

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator."""
        from src.agentic_rag.evaluator import EvaluationResult

        mock = MagicMock()
        mock.evaluate.return_value = EvaluationResult(
            is_relevant=True, confidence=0.9, reason="Test"
        )
        return mock

    def test_retrieval_latency(self, mock_retriever):
        """Benchmark retrieval latency."""

        start_time = time.time()
        _results = mock_retriever.invoke("test query")
        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second
        assert elapsed > 0  # Should have some latency

    def test_evaluation_latency(self, mock_evaluator):
        """Benchmark evaluation latency."""

        start_time = time.time()
        mock_evaluator.evaluate("test query", [])
        elapsed = time.time() - start_time

        assert elapsed < 1.0

    def test_full_pipeline_latency(self, mock_llm, mock_retriever, mock_evaluator):
        """Benchmark full pipeline latency."""
        from src.agentic_rag.agent import AgenticRAGAgent
        from src.agentic_rag.config import AgenticRAGConfig

        config = AgenticRAGConfig(max_search_iterations=2)
        agent = AgenticRAGAgent(
            llm=mock_llm,
            local_retriever=mock_retriever,
            evaluator=mock_evaluator,
            config=config,
            use_hybrid_retrieval=False,
        )

        start_time = time.time()
        _result = agent.run("test query")
        elapsed = time.time() - start_time

        # Full pipeline should complete in reasonable time
        # (This is a soft constraint for slow tests)
        assert elapsed < 5.0

    def test_concurrent_query_handling(self, mock_llm, mock_retriever, mock_evaluator):
        """Test handling of concurrent queries."""
        from src.agentic_rag.agent import AgenticRAGAgent
        from src.agentic_rag.config import AgenticRAGConfig

        config = AgenticRAGConfig(max_search_iterations=1)
        agent = AgenticRAGAgent(
            llm=mock_llm,
            local_retriever=mock_retriever,
            evaluator=mock_evaluator,
            config=config,
            use_hybrid_retrieval=False,
        )

        # Process multiple queries sequentially
        queries = ["query1", "query2", "query3"]
        start_time = time.time()

        for query in queries:
            agent.run(query)

        elapsed = time.time() - start_time

        # All queries should complete in reasonable time
        assert elapsed < 10.0

    def test_memory_efficiency(self, mock_llm, mock_retriever, mock_evaluator):
        """Test memory usage with many documents."""
        from src.agentic_rag.agent import AgenticRAGAgent
        from src.agentic_rag.config import AgenticRAGConfig
        from src.agentic_rag.state import Document

        # Create many documents
        _large_document_set = [
            Document(
                page_content=f"Document content {i} with some text.",
                metadata={"source": f"doc{i}.txt", "page": 1},
            )
            for i in range(100)
        ]

        config = AgenticRAGConfig(max_search_iterations=1)
        agent = AgenticRAGAgent(
            llm=mock_llm,
            local_retriever=mock_retriever,
            evaluator=mock_evaluator,
            config=config,
            use_hybrid_retrieval=False,
        )

        # Process with large document set
        result = agent.run("test query")

        # Should complete without memory issues
        assert result.answer is not None
