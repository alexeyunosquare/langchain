"""
LangGraph state definitions for Agentic RAG.

This module defines the state structures used by LangGraph to orchestrate
the agentic RAG workflow, including messages, documents, and evaluation results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict


class MessageRole(str, Enum):
    """Valid message roles in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    """
    Represents a message in the conversation.

    Attributes:
        role: Role of the message sender (user, assistant, system, tool)
        content: Text content of the message
        metadata: Optional metadata dictionary
    """

    role: MessageRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate message role."""
        if self.role not in MessageRole:
            raise ValueError(
                f"role must be one of {list(MessageRole)}, got {self.role}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create Message from dictionary."""
        role_val = data["role"]
        role = MessageRole(role_val) if isinstance(role_val, str) else role_val
        return cls(
            role=role,
            content=data["content"],
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return f"[{self.role.value}] {self.content}"


@dataclass
class Document:
    """
    Represents a retrieved document.

    Attributes:
        page_content: The text content of the document
        metadata: Metadata dictionary with source, page, etc.
        score: Optional relevance score (0-1)
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create Document from dictionary."""
        return cls(
            page_content=data["page_content"],
            metadata=data.get("metadata", {}),
            score=data.get("score"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
            "score": self.score,
        }

    def __str__(self) -> str:
        return f"[{self.metadata.get('source', 'unknown')}] {self.page_content[:50]}..."


# LangGraph-specific TypedDict for state management
class GraphState(TypedDict):
    """
    State structure for LangGraph agent workflow.

    This state is used by LangGraph to track the agent's progress through
    the workflow. Fields can be appended to using LangGraph's annotated
    update mechanism.

    Attributes:
        query: The original user query
        messages: List of conversation messages (annotated for accumulation)
        documents: Retrieved documents
        context: Combined context from documents
        answer: Generated answer
        is_relevant: Evaluation result for document relevance
        should_search_again: Decision to perform another search
        search_query: Query used for search iterations
        search_results: Results from search operations
        validation_passed: Whether answer passed validation
        correction_triggered: Whether correction was applied
        hallucination_score: Score indicating potential hallucination (0-1)
        search_count: Number of searches performed
        iteration: Current iteration number
        error: Any error that occurred during execution
    """

    # Query and context
    query: str
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]
    context: str

    # Generated response
    answer: str

    # Evaluation results
    is_relevant: Optional[bool]
    should_search_again: Optional[bool]
    validation_passed: Optional[bool]
    correction_triggered: Optional[bool]
    hallucination_score: Optional[float]

    # Search metadata
    search_query: str
    search_results: List[dict[str, Any]]
    search_count: int
    iteration: int

    # Error handling
    error: Optional[str]


@dataclass
class AgentState:
    """
    State structure for LangGraph agent workflow.

    This state is updated throughout the agent's execution and contains:
    - Query and conversation history
    - Retrieved documents and context
    - Generated answers and validation results
    - Search metadata

    Attributes:
        query: The original user query
        messages: List of conversation messages
        documents: Retrieved documents
        context: Combined context from documents
        answer: Generated answer
        is_relevant: Evaluation result for document relevance
        should_search_again: Decision to perform another search
        search_query: Query used for search iterations
        search_results: Results from search operations
        validation_passed: Whether answer passed validation
        correction_triggered: Whether correction was applied
        hallucination_score: Score indicating potential hallucination (0-1)
        search_count: Number of searches performed
        iteration: Current iteration number
        error: Any error that occurred during execution
    """

    # Query and context
    query: str = ""
    messages: List[Message] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    context: str = ""

    # Generated response
    answer: str = ""

    # Evaluation results
    is_relevant: Optional[bool] = None
    should_search_again: Optional[bool] = None
    validation_passed: Optional[bool] = None
    correction_triggered: Optional[bool] = None
    hallucination_score: Optional[float] = None

    # Search metadata
    search_query: str = ""
    search_results: List[dict[str, Any]] = field(default_factory=list)
    search_count: int = 0
    iteration: int = 0

    # Error handling
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Create AgentState from dictionary."""
        # Convert messages
        messages = [
            Message.from_dict(msg) if isinstance(msg, dict) else msg
            for msg in data.get("messages", [])
        ]

        # Convert documents
        documents = [
            Document.from_dict(doc) if isinstance(doc, dict) else doc
            for doc in data.get("documents", [])
        ]

        return cls(
            query=data.get("query", ""),
            messages=messages,
            documents=documents,
            context=data.get("context", ""),
            answer=data.get("answer", ""),
            is_relevant=data.get("is_relevant"),
            should_search_again=data.get("should_search_again"),
            search_query=data.get("search_query", ""),
            search_results=data.get("search_results", []),
            validation_passed=data.get("validation_passed"),
            correction_triggered=data.get("correction_triggered"),
            hallucination_score=data.get("hallucination_score"),
            search_count=data.get("search_count", 0),
            iteration=data.get("iteration", 0),
            error=data.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "query": self.query,
            "messages": [msg.to_dict() for msg in self.messages],
            "documents": [doc.to_dict() for doc in self.documents],
            "context": self.context,
            "answer": self.answer,
            "is_relevant": self.is_relevant,
            "should_search_again": self.should_search_again,
            "search_query": self.search_query,
            "search_results": self.search_results,
            "validation_passed": self.validation_passed,
            "correction_triggered": self.correction_triggered,
            "hallucination_score": self.hallucination_score,
            "search_count": self.search_count,
            "iteration": self.iteration,
            "error": self.error,
        }

    def add_message(self, role: MessageRole, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append(Message(role=role, content=content))

    def clear_messages(self) -> None:
        """Clear conversation history while preserving query."""
        self.messages.clear()

    def add_document(self, doc: Document) -> None:
        """Add a document to the retrieved documents."""
        self.documents.append(doc)
        self.context += f"\n{doc.page_content}"

    def clear_documents(self) -> None:
        """Clear documents and context."""
        self.documents.clear()
        self.context = ""
