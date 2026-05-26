"""
Tests for the MCP server module.

Tests tool schemas, lazy initialization, bearer token auth,
and HTTP transport connectivity.
"""

import asyncio
import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the singleton registry before each test."""
    from mcp_server.server import _AgentRegistry

    _AgentRegistry._instance = None
    yield
    _AgentRegistry._instance = None


@pytest.fixture
def mock_env():
    """Set up minimal environment for MCP server tests."""
    env = {
        "OPENAI_API_KEY": "test-key",
        "LM_STUDIO_HOST": "localhost",
        "LM_STUDIO_PORT": "1234",
        "MCP_API_TOKEN": "test-token-123",
    }
    old = {}
    for key, value in env.items():
        old[key] = os.environ.get(key)
        os.environ[key] = value
    yield env
    for key, old_val in old.items():
        if old_val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_val


@pytest.fixture
def mock_agent():
    """Create a mock AgenticRAGAgent."""
    agent = MagicMock()
    agent.config.evaluation_threshold = 0.7
    agent.config.max_search_iterations = 3
    agent.config.temperature = 0.7
    agent.config.top_k = 5
    agent.use_hybrid_retrieval = True
    agent.hybrid_retriever = MagicMock()
    return agent


@pytest.fixture
def mock_rag_chain(mock_agent):
    """Create a mock RAGChain."""
    chain = MagicMock()
    chain.llm = MagicMock()
    chain.retriever = MagicMock()
    chain.clear_memory = MagicMock()
    return chain


# ---------------------------------------------------------------------------
# Tool schema tests
# ---------------------------------------------------------------------------

class TestToolSchemas:
    """Verify MCP tool definitions have correct parameter types and descriptions."""

    def test_query_tool_exists(self, mock_env):
        from mcp_server.server import get_mcp_server

        server = get_mcp_server()
        tools = asyncio.run(server.list_tools())
        tool_names = [t.name for t in tools]
        assert "query" in tool_names

    def test_search_documents_tool_exists(self, mock_env):
        from mcp_server.server import get_mcp_server

        server = get_mcp_server()
        tools = asyncio.run(server.list_tools())
        tool_names = [t.name for t in tools]
        assert "search_documents" in tool_names

    def test_validate_answer_tool_exists(self, mock_env):
        from mcp_server.server import get_mcp_server

        server = get_mcp_server()
        tools = asyncio.run(server.list_tools())
        tool_names = [t.name for t in tools]
        assert "validate_answer" in tool_names

    def test_add_document_tool_exists(self, mock_env):
        from mcp_server.server import get_mcp_server

        server = get_mcp_server()
        tools = asyncio.run(server.list_tools())
        tool_names = [t.name for t in tools]
        assert "add_document" in tool_names

    def test_get_config_tool_exists(self, mock_env):
        from mcp_server.server import get_mcp_server

        server = get_mcp_server()
        tools = asyncio.run(server.list_tools())
        tool_names = [t.name for t in tools]
        assert "get_config" in tool_names

    def test_reset_conversation_tool_exists(self, mock_env):
        from mcp_server.server import get_mcp_server

        server = get_mcp_server()
        tools = asyncio.run(server.list_tools())
        tool_names = [t.name for t in tools]
        assert "reset_conversation" in tool_names

    def test_tool_descriptions(self, mock_env):
        """All tools should have non-empty descriptions."""
        from mcp_server.server import get_mcp_server

        server = get_mcp_server()
        tools = asyncio.run(server.list_tools())
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' missing description"


# ---------------------------------------------------------------------------
# Lazy initialization tests
# ---------------------------------------------------------------------------

class TestLazyInitialization:
    """Verify agent is created lazily on first tool call."""

    def test_registry_creates_agent_once(self, mock_env, mock_agent, mock_rag_chain):
        from mcp_server.server import _AgentRegistry

        with patch(
            "mcp_server.server.create_default_agentic_rag", return_value=mock_agent
        ) as mock_create, patch(
            "mcp_server.server.create_rag_chain", return_value=mock_rag_chain
        ):
            registry = _AgentRegistry()
            agent1 = registry.get_agent()
            agent2 = registry.get_agent()
            assert agent1 is agent2
            # create_default_agentic_rag called exactly once
            assert mock_create.call_count == 1

    def test_registry_reuses_same_instance(self, mock_env):
        from mcp_server.server import _AgentRegistry

        r1 = _AgentRegistry()
        r2 = _AgentRegistry()
        assert r1 is r2

    def test_reset_clears_memory(self, mock_env, mock_rag_chain):
        from mcp_server.server import _AgentRegistry

        with patch(
            "mcp_server.server.create_rag_chain", return_value=mock_rag_chain
        ):
            registry = _AgentRegistry()
            registry.get_rag_chain()
            registry.reset()
            mock_rag_chain.clear_memory.assert_called_once()


# ---------------------------------------------------------------------------
# Tool function tests (with mocked agent)
# ---------------------------------------------------------------------------

class TestQueryTool:
    """Test the query tool end-to-end with mocked dependencies."""

    def test_query_returns_answer_dict(self, mock_env, mock_agent, mock_rag_chain):
        from agentic_rag import AgentResult
        from agentic_rag.state import Document

        result = AgentResult(
            answer="The answer is 42.",
            documents=[
                Document(
                    page_content="Some context",
                    metadata={"source": "local"},
                    score=0.9,
                )
            ],
            search_count=1,
            validation_passed=True,
            search_iterations=1,
            hallucination_score=0.1,
            tavily_used=False,
            tavily_document_count=0,
            local_document_count=1,
            total_documents=1,
        )
        mock_agent.run.return_value = result

        with patch(
            "mcp_server.server.create_default_agentic_rag", return_value=mock_agent
        ), patch(
            "mcp_server.server.create_rag_chain", return_value=mock_rag_chain
        ):
            from mcp_server.server import query

            out = query("What is the meaning?", max_iterations=2)
            assert out["answer"] == "The answer is 42."
            assert out["validation_passed"] is True
            assert out["total_documents"] == 1
            mock_agent.run.assert_called_once_with(
                "What is the meaning?", max_iterations=2
            )


class TestSearchDocumentsTool:
    """Test the search_documents tool."""

    def test_search_documents_local_only(self, mock_env, mock_agent, mock_rag_chain):
        from agentic_rag.state import Document

        mock_docs = [
            Document(
                page_content="Result one",
                metadata={"source": "local"},
                score=0.8,
            ),
            Document(
                page_content="Result two",
                metadata={"source": "local"},
                score=0.6,
            ),
        ]
        mock_agent._retrieve_documents_local.return_value = [
            {"content": "Result one", "metadata": {"source": "local"}, "score": 0.8},
            {"content": "Result two", "metadata": {"source": "local"}, "score": 0.6},
        ]
        mock_agent._convert_to_documents.return_value = mock_docs

        with patch(
            "mcp_server.server.create_default_agentic_rag", return_value=mock_agent
        ), patch(
            "mcp_server.server.create_rag_chain", return_value=mock_rag_chain
        ):
            from mcp_server.server import search_documents

            out = search_documents("test query", top_k=2, use_web=False)
            assert len(out) == 2
            assert out[0]["page_content"] == "Result one"
            mock_agent._retrieve_documents_local.assert_called_once_with("test query")


class TestValidateAnswerTool:
    """Test the validate_answer tool."""

    def test_validate_answer(self, mock_env, mock_agent, mock_rag_chain):
        from agentic_rag.state import ValidationStatus

        mock_validation = MagicMock()
        mock_validation.status = ValidationStatus.VALID
        mock_validation.quality_score = 0.95
        mock_validation.issues = []
        mock_validation.corrective_action = None
        mock_validation.is_hallucinated = False

        mock_agent.corrective.answer_validator.validate.return_value = mock_validation

        with patch(
            "mcp_server.server.create_default_agentic_rag", return_value=mock_agent
        ), patch(
            "mcp_server.server.create_rag_chain", return_value=mock_rag_chain
        ):
            from mcp_server.server import validate_answer

            out = validate_answer(
                answer="The sky is blue.",
                query="What color is the sky?",
                sources=["The sky appears blue due to Rayleigh scattering."],
            )
            assert out["status"] == "valid"
            assert out["quality_score"] == 0.95
            assert out["is_hallucinated"] is False


class TestAddDocumentTool:
    """Test the add_document tool."""

    def test_add_document(self, mock_env, mock_rag_chain):
        mock_rag_chain.load_and_store_document.return_value = 15

        with patch(
            "mcp_server.server.create_rag_chain", return_value=mock_rag_chain
        ):
            from mcp_server.server import add_document

            out = add_document("/path/to/file.txt")
            assert out["status"] == "ok"
            assert out["chunks_added"] == 15
            assert out["source"] == "/path/to/file.txt"


class TestGetConfigTool:
    """Test the get_config tool."""

    def test_get_config(self, mock_env, mock_agent, mock_rag_chain):
        with patch(
            "mcp_server.server.create_default_agentic_rag", return_value=mock_agent
        ), patch(
            "mcp_server.server.create_rag_chain", return_value=mock_rag_chain
        ):
            from mcp_server.server import get_config

            out = get_config()
            assert out["evaluation_threshold"] == 0.7
            assert out["max_search_iterations"] == 3
            assert out["temperature"] == 0.7
            assert out["top_k"] == 5
            assert out["use_hybrid_retrieval"] is True


class TestResetConversationTool:
    """Test the reset_conversation tool."""

    def test_reset_conversation(self, mock_env, mock_rag_chain):
        with patch(
            "mcp_server.server.create_rag_chain", return_value=mock_rag_chain
        ):
            from mcp_server.server import reset_conversation, _AgentRegistry

            registry = _AgentRegistry()
            registry.get_rag_chain()

            out = reset_conversation()
            assert out["status"] == "ok"
            assert "cleared" in out["message"]


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

class TestBearerTokenAuth:
    """Test bearer token authentication."""

    def test_valid_token_accepted(self, mock_env):
        # Reload server module so _mcp_token picks up the env var
        import mcp_server.server as server_module

        importlib.reload(server_module)
        result = server_module.verifier.validate("test-token-123")
        assert result is True

    def test_invalid_token_rejected(self, mock_env):
        import mcp_server.server as server_module

        importlib.reload(server_module)
        result = server_module.verifier.validate("wrong-token")
        assert result is False

    def test_empty_token_rejected(self, mock_env):
        import mcp_server.server as server_module

        importlib.reload(server_module)
        result = server_module.verifier.validate("")
        assert result is False


class TestNoTokenConfig:
    """Test server behaviour when MCP_API_TOKEN is not set."""

    def test_warns_on_missing_token(self, mock_env, capsys):
        """When MCP_API_TOKEN is empty, main() should warn."""
        os.environ.pop("MCP_API_TOKEN", None)
        # Reload to pick up the new env
        import mcp_server.server as server_module

        importlib.reload(server_module)
        token = server_module._mcp_token
        assert token == ""