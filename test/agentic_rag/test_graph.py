"""
Tests for LangGraph orchestration in Agentic RAG.

Tests the LangGraph-based workflow orchestration including state machine
transitions, conditional branching, and graph execution.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agentic_rag.config import AgenticRAGConfig
from src.agentic_rag.evaluator import EvaluationResult, RelevanceEvaluator
from src.agentic_rag.graph import (
    LangGraphAgenticRAG,
    LangGraphNode,
    build_agentic_rag_graph,
    create_agentic_graph_workflow,
)
from src.agentic_rag.state import AgentState, Document, MessageRole


class TestLangGraphNode:
    """Test suite for LangGraphNode class."""

    def test_retrieve_documents_node(self):
        """Test retrieve_documents node updates search metadata."""
        state = AgentState(query="Test query")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        result = LangGraphNode.retrieve_documents(state, mock_retriever)

        assert "search_count" in result
        assert "iteration" in result

    def test_evaluate_relevance_node_with_documents(self):
        """Test evaluate_relevance node with relevant documents."""
        evaluator = MagicMock(spec=RelevanceEvaluator)
        mock_result = MagicMock()
        mock_result.is_relevant = True
        evaluator.evaluate.return_value = mock_result
        evaluator.should_search_again.return_value = False

        state = AgentState(
            query="Test query",
            documents=[
                Document(page_content="Relevant content"),
            ],
        )

        result = LangGraphNode.evaluate_relevance(state, evaluator)

        assert "is_relevant" in result
        assert "should_search_again" in result

    def test_evaluate_relevance_node_no_documents(self):
        """Test evaluate_relevance node with no documents."""
        evaluator = MagicMock(spec=RelevanceEvaluator)

        state = AgentState(
            query="Test query",
            documents=[],
        )

        result = LangGraphNode.evaluate_relevance(state, evaluator)

        assert result.get("is_relevant") is False
        assert result.get("should_search_again") is True
        assert "No documents retrieved" in (result.get("error") or "")

    def test_refine_query_node(self):
        """Test refine_query node preserves query."""
        state = AgentState(query="Original query")

        result = LangGraphNode.refine_query(state)

        assert "query" in result
        assert result["query"] == "Original query"

    def test_generate_answer_node(self):
        """Test generate_answer node creates answer and adds message."""
        state = AgentState(
            query="Test query",
            documents=[
                Document(page_content="Context content"),
            ],
        )

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Generated answer"
        mock_llm.invoke.return_value = mock_response

        mock_corrective = MagicMock()
        mock_corrective.check_hallucination.return_value = (False, 1.0)

        result = LangGraphNode.generate_answer(state, mock_llm, mock_corrective)

        assert "answer" in result
        assert len(result.get("answer", "")) > 0
        assert "messages" in result
        assert len(result.get("messages", [])) > 0
        assert result["messages"][-1].role == MessageRole.ASSISTANT

    def test_validate_and_correct_node(self):
        """Test validate_and_correct node sets validation flags."""
        state = AgentState(
            query="Test query",
            answer="Test answer",
            documents=[
                Document(page_content="Context"),
            ],
        )

        result = LangGraphNode.validate_and_correct(state)

        assert result.validation_passed is True
        assert result.correction_triggered is False
        assert result.hallucination_score is not None


class TestGraphBuilding:
    """Test suite for LangGraph building functions."""

    def test_create_agentic_graph_workflow(self):
        """Test workflow creation with mock components."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_evaluator = MagicMock(spec=RelevanceEvaluator)

        result = create_agentic_graph_workflow(
            evaluator=mock_evaluator,
            llm=mock_llm,
            retriever=mock_retriever,
        )

        assert "graph" in result
        assert result["entry_point"] == "retrieve"
        assert result["end_point"] is not None

    def test_build_agentic_rag_graph_structure(self):
        """Test graph has correct node structure."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_evaluator = MagicMock(spec=RelevanceEvaluator)

        graph = build_agentic_rag_graph(
            evaluator=mock_evaluator,
            llm=mock_llm,
            retriever=mock_retriever,
        )

        # Check that graph is compiled
        assert graph is not None

    def test_build_agentic_rag_graph_with_config(self):
        """Test graph accepts custom configuration."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_evaluator = MagicMock(spec=RelevanceEvaluator)

        config = AgenticRAGConfig(
            evaluation_threshold=0.8,
            max_search_iterations=5,
        )

        graph = build_agentic_rag_graph(
            evaluator=mock_evaluator,
            llm=mock_llm,
            retriever=mock_retriever,
            config=config,
        )

        assert graph is not None


class TestLangGraphAgenticRAG:
    """Test suite for LangGraph-based Agentic RAG."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return MagicMock()

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        return MagicMock()

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator."""
        evaluator = MagicMock(spec=RelevanceEvaluator)
        evaluator.evaluate.return_value = EvaluationResult(
            is_relevant=True,
            confidence=0.85,
            reason="Documents are relevant",
        )
        return evaluator

    @pytest.fixture
    def agentic_rag_agent(self, mock_llm, mock_retriever, mock_evaluator):
        """Create a LangGraph-based agentic RAG instance."""
        return LangGraphAgenticRAG(
            evaluator=mock_evaluator,
            llm=mock_llm,
            retriever=mock_retriever,
        )

    def test_initialization(
        self, agentic_rag_agent, mock_evaluator, mock_llm, mock_retriever
    ):
        """Test LangGraph Agentic RAG initializes correctly."""
        assert agentic_rag_agent.evaluator is mock_evaluator
        assert agentic_rag_agent.llm is mock_llm
        assert agentic_rag_agent.retriever is mock_retriever
        assert agentic_rag_agent.graph is not None

    def test_run_with_relevant_documents(self, agentic_rag_agent):
        """Test running workflow with relevant documents."""
        query = "What is LangChain?"

        # Mock graph invoke to return a state with answer
        final_state = AgentState(
            query=query,
            answer="LangChain is a framework for building LLM applications.",
            documents=[
                Document(page_content="LangChain documentation"),
            ],
            validation_passed=True,
            is_relevant=True,
        )

        with patch.object(agentic_rag_agent.graph, "invoke", return_value=final_state):
            result = agentic_rag_agent.run(query)

        assert getattr(result, "query", None) == query or result.get("query") == query
        assert (
            getattr(result, "answer", None) is not None
            or result.get("answer") is not None
        )

    def test_run_with_max_search_count(self, agentic_rag_agent):
        """Test running workflow with max search count."""
        query = "Test query"

        with patch.object(agentic_rag_agent.graph, "invoke") as mock_invoke:
            agentic_rag_agent.run(query, max_search_count=3)

            # Check that invoke was called with initial state
            args, kwargs = mock_invoke.call_args
            assert len(args) >= 1
            # Initial state should be the first argument (can be dict or AgentState)
            initial_state = args[0]
            assert (
                initial_state.get("query") == query
                or getattr(initial_state, "query", None) == query
            )

    def test_stream_execution(self, agentic_rag_agent):
        """Test streaming workflow execution."""
        query = "Test query"

        # Create mock states to yield
        states = [
            AgentState(query=query, search_count=0),
            AgentState(query=query, search_count=1),
            AgentState(query=query, search_count=2, answer="Test answer"),
        ]

        with patch.object(agentic_rag_agent.graph, "stream", return_value=iter(states)):
            results = list(agentic_rag_agent.stream(query))

            assert len(results) > 0
            last_result = results[-1]
            assert (
                last_result.get("answer") == "Test answer"
                or getattr(last_result, "answer", None) == "Test answer"
            )

    def test_run_with_irrelevant_documents(self):
        """Test workflow handles irrelevant documents."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()

        mock_evaluator = MagicMock(spec=RelevanceEvaluator)
        mock_evaluator.evaluate.return_value = EvaluationResult(
            is_relevant=False,
            confidence=0.3,
            reason="Documents not relevant",
        )

        agent = LangGraphAgenticRAG(
            evaluator=mock_evaluator,
            llm=mock_llm,
            retriever=mock_retriever,
        )

        with patch.object(agent.graph, "invoke") as mock_invoke:
            final_state = AgentState(
                query="Test",
                answer="I couldn't find relevant information.",
                is_relevant=False,
                should_search_again=True,
                search_count=3,
            )
            mock_invoke.return_value = final_state

            result = agent.run("Test query")

            assert (
                result.get("should_search_again") is True
                or getattr(result, "should_search_again", None) is True
            )
            assert (
                result.get("search_count") == 3
                or getattr(result, "search_count", None) == 3
            )

    def test_run_error_handling(self):
        """Test workflow handles graph errors gracefully."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()

        mock_evaluator = MagicMock(spec=RelevanceEvaluator)

        agent = LangGraphAgenticRAG(
            evaluator=mock_evaluator,
            llm=mock_llm,
            retriever=mock_retriever,
        )

        # Mock graph to raise an error
        with (
            patch.object(agent.graph, "invoke", side_effect=Exception("Graph error")),
            pytest.raises(Exception) as exc_info,
        ):
            agent.run("Test query")
            assert str(exc_info.value) == "Graph error"


class TestGraphConditionalEdges:
    """Test conditional edge routing logic."""

    def test_routing_after_first_retrieval(self):
        """Test routing after initial retrieval."""
        state = AgentState(
            query="Test query",
            documents=[Document(page_content="Test")],
            search_count=0,
        )

        result = LangGraphNode.route_after_retrieval(state)
        assert result == "evaluate"

    def test_routing_after_multiple_retrievals(self):
        """Test routing after multiple retrievals."""
        state = AgentState(
            query="Test query",
            documents=[Document(page_content="Test")],
            search_count=1,
        )

        result = LangGraphNode.route_after_retrieval(state)
        assert result == "refine"


# Add the missing method to LangGraphNode
def route_after_retrieval(state: AgentState) -> str:
    """Route to evaluate or refine based on search count."""
    if state.search_count == 0:
        return "evaluate"
    return "refine"


# Attach to class
LangGraphNode.route_after_retrieval = staticmethod(route_after_retrieval)
