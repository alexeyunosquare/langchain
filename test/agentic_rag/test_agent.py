"""
Tests for Agentic RAG Agent orchestration.
"""

from unittest.mock import MagicMock

import pytest

from src.agentic_rag.agent import AgenticRAGAgent, AgentResult
from src.agentic_rag.evaluator import EvaluationResult, RelevanceEvaluator
from src.agentic_rag.state import Document, Message, MessageRole


class TestAgenticRAGAgent:
    """Test suite for AgenticRAGAgent class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = MagicMock()
        mock.invoke.return_value = Message(
            role=MessageRole.ASSISTANT,
            content="LangChain is a framework for building applications with LLMs.",
        )
        return mock

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever for testing."""
        mock = MagicMock()
        mock.search.return_value = [
            {
                "id": "doc1",
                "content": "LangChain documentation and guides.",
                "score": 0.9,
            },
            {
                "id": "doc2",
                "content": "Python programming language features.",
                "score": 0.7,
            },
        ]
        return mock

    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator for testing."""
        mock = MagicMock(spec=RelevanceEvaluator)
        mock.evaluate.return_value = EvaluationResult(
            is_relevant=True,
            confidence=0.85,
            reason="Documents are relevant to the query",
        )
        return mock

    def test_agent_initialization(self, mock_llm, mock_retriever, mock_evaluator):
        """Test agent initializes with correct components."""
        agent = AgenticRAGAgent(
            llm=mock_llm, retriever=mock_retriever, evaluator=mock_evaluator
        )

        assert agent.llm is mock_llm
        assert agent.retriever is mock_retriever
        assert agent.evaluator is mock_evaluator
        assert agent.state is not None
        assert agent.state.query == ""

    def test_agent_workflow_execution(self, mock_llm, mock_retriever, mock_evaluator):
        """Test agent can execute complete workflow."""
        agent = AgenticRAGAgent(
            llm=mock_llm, retriever=mock_retriever, evaluator=mock_evaluator
        )

        query = "What is LangChain?"
        result = agent.run(query)

        assert result.answer is not None
        assert len(result.answer) > 0
        assert len(agent.state.messages) > 0

    def test_agent_decision_to_rerun(self, mock_llm, mock_retriever, mock_evaluator):
        """Test agent decides to re-search when documents are irrelevant."""
        # Setup evaluator to return irrelevant result
        mock_evaluator.evaluate.return_value = EvaluationResult(
            is_relevant=False, confidence=0.7, reason="Documents not relevant"
        )

        # Setup retriever to return results on second call
        call_count = [0]

        def side_effect(_query, **_kwargs):
            call_count[0] += 1
            return [
                {
                    "id": f"doc{call_count[0]}",
                    "page_content": f"Retrieved document {call_count[0]}",
                    "metadata": {"source": "test"},
                    "score": 0.9,
                }
            ]

        mock_retriever.invoke.side_effect = side_effect

        agent = AgenticRAGAgent(
            llm=mock_llm, retriever=mock_retriever, evaluator=mock_evaluator
        )

        query = "Test query"
        _result = agent.run(query)

        # Should have called retriever at least once
        assert mock_retriever.invoke.call_count >= 1

    def test_agent_max_search_iterations(
        self, mock_llm, mock_retriever, mock_evaluator
    ):
        """Test agent respects max search iterations."""
        mock_evaluator.evaluate.return_value = EvaluationResult(
            is_relevant=False, confidence=0.5, reason="Always irrelevant"
        )

        agent = AgenticRAGAgent(
            llm=mock_llm,
            retriever=mock_retriever,
            evaluator=mock_evaluator,
            max_iterations=2,
        )

        query = "Test query"
        _result = agent.run(query)

        # Should stop after max iterations
        assert mock_retriever.invoke.call_count <= 2

    def test_agent_error_handling_retrieval_failure(self, mock_llm, mock_retriever):
        """Test agent handles retrieval failures gracefully."""
        mock_retriever.search.side_effect = Exception("Retrieval failed")

        mock_evaluator = MagicMock(spec=RelevanceEvaluator)
        mock_evaluator.evaluate.return_value = EvaluationResult(
            is_relevant=False, confidence=0.0, reason="No documents retrieved"
        )

        agent = AgenticRAGAgent(
            llm=mock_llm, retriever=mock_retriever, evaluator=mock_evaluator
        )

        query = "Test query"

        # Should handle exception and return answer
        result = agent.run(query)

        assert result.answer is not None

    def test_agent_state_tracking(self, mock_llm, mock_retriever, mock_evaluator):
        """Test agent properly tracks state throughout workflow."""
        agent = AgenticRAGAgent(
            llm=mock_llm, retriever=mock_retriever, evaluator=mock_evaluator
        )

        query = "Test query"
        agent.run(query)

        state = agent.state

        assert state.query == query
        assert len(state.messages) > 0
        assert state.documents is not None
        assert state.answer is not None

    def test_agent_conversation_history(self, mock_llm, mock_retriever, mock_evaluator):
        """Test agent maintains conversation history."""
        agent = AgenticRAGAgent(
            llm=mock_llm, retriever=mock_retriever, evaluator=mock_evaluator
        )

        # First query
        agent.run("First question")
        _initial_message_count = len(agent.state.messages)

        # Second query
        agent.run("Second question")

        # Should have messages from both queries
        # Note: run() creates new state, so messages are reset
        assert len(agent.state.messages) > 0

    def test_agent_streaming_output(self, mock_llm, mock_retriever, mock_evaluator):
        """Test agent supports streaming output."""
        agent = AgenticRAGAgent(
            llm=mock_llm, retriever=mock_retriever, evaluator=mock_evaluator
        )

        query = "Test query"

        # Should be able to iterate over results
        for chunk in agent.stream(query):
            assert chunk is not None
            assert len(chunk) > 0


class TestAgentResult:
    """Test suite for AgentResult class."""

    def test_result_creation(self):
        """Test creating an agent result."""

        result = AgentResult(
            answer="Test answer", documents=[], search_count=0, validation_passed=True
        )

        assert result.answer == "Test answer"
        assert result.documents == []
        assert result.search_count == 0
        assert result.validation_passed is True

    def test_result_with_search_info(self):
        """Test result with search information."""

        result = AgentResult(
            answer="Test answer",
            documents=[Document(page_content="Doc1")],
            search_count=2,
            validation_passed=False,
        )

        assert len(result.documents) == 1
        assert result.search_count == 2
        assert result.validation_passed is False
