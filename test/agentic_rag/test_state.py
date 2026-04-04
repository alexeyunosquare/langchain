"""
Tests for LangGraph state definitions in Agentic RAG.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.agentic_rag.state import AgentState, Document, Message, MessageRole

# Force reimport
if "agentic_rag" in sys.modules:
    del sys.modules["agentic_rag"]
if "agentic_rag.state" in sys.modules:
    del sys.modules["agentic_rag.state"]


class TestAgentState:
    """Test suite for AgentState class."""

    def test_state_initialization(self):
        """Test that agent state initializes with correct default values."""
        state = AgentState()

        assert state.messages == []
        assert state.query == ""
        assert state.documents == []
        assert state.context == ""
        assert state.answer == ""
        assert state.is_relevant is None
        assert state.should_search_again is None
        assert state.search_query == ""
        assert state.search_results == []
        assert state.validation_passed is None
        assert state.correction_triggered is None
        assert state.hallucination_score is None

    def test_state_with_query(self):
        """Test state initialization with a query."""
        state = AgentState(query="What is LangChain?")

        assert state.query == "What is LangChain?"
        assert state.messages == []

    def test_state_serialization_deserialization(self):
        """Test state can be serialized and deserialized."""
        original_state = AgentState(
            query="Test query",
            messages=[Message(role=MessageRole.USER, content="Hello")],
        )

        # Test dict conversion
        state_dict = original_state.to_dict()

        assert "query" in state_dict
        assert "messages" in state_dict

        # Test reconstruction
        restored_state = AgentState.from_dict(state_dict)

        assert restored_state.query == original_state.query
        assert len(restored_state.messages) == len(original_state.messages)

    def test_state_messages_append(self):
        """Test appending messages to state."""
        state = AgentState(query="Test")

        user_msg = Message(role="user", content="Question")
        assistant_msg = Message(role="assistant", content="Answer")

        state.messages.append(user_msg)
        state.messages.append(assistant_msg)

        assert len(state.messages) == 2
        assert state.messages[0].role == "user"
        assert state.messages[1].role == "assistant"

    def test_state_empty_documents(self):
        """Test state with empty document list."""
        state = AgentState(query="Test")

        assert state.documents == []
        assert state.context == ""


class TestMessage:
    """Test suite for Message class."""

    def test_message_creation(self):
        """Test creating a message object."""
        msg = Message(role=MessageRole.USER, content="Hello world")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello world"

    def test_message_validation_invalid_role(self):
        """Test message rejects invalid role values."""
        with pytest.raises(ValueError, match="role must be one of"):
            Message(role="invalid", content="Test")

    def test_message_to_dict(self):
        """Test message serialization to dict."""
        msg = Message(role=MessageRole.ASSISTANT, content="Test content")

        msg_dict = msg.to_dict()

        assert msg_dict["role"] == "assistant"
        assert msg_dict["content"] == "Test content"
        assert msg_dict["metadata"] == {}

    def test_message_from_dict(self):
        """Test message deserialization from dict."""
        msg_dict = {"role": "user", "content": "Hello"}

        msg = Message.from_dict(msg_dict)

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"


class TestDocument:
    """Test suite for Document class."""

    def test_document_creation(self):
        """Test creating a document object."""
        doc = Document(
            page_content="Test content", metadata={"source": "test.txt", "page": 1}
        )

        assert doc.page_content == "Test content"
        assert doc.metadata["source"] == "test.txt"

    def test_document_default_metadata(self):
        """Test document with empty metadata."""
        doc = Document(page_content="Simple content")

        assert doc.page_content == "Simple content"
        assert doc.metadata == {}

    def test_document_to_dict(self):
        """Test document serialization to dict."""
        doc = Document(page_content="Content", metadata={"source": "doc.txt"})

        doc_dict = doc.to_dict()

        assert doc_dict["page_content"] == "Content"
        assert doc_dict["metadata"]["source"] == "doc.txt"

    def test_document_from_dict(self):
        """Test document deserialization from dict."""
        doc_dict = {
            "page_content": "Page content",
            "metadata": {"source": "test.pdf", "page": 5},
        }

        doc = Document.from_dict(doc_dict)

        assert doc.page_content == "Page content"
        assert doc.metadata["source"] == "test.pdf"
        assert doc.metadata["page"] == 5
