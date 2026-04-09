"""
LangGraph state definitions for Agentic RAG.

This module defines the state structures used by LangGraph to orchestrate
the agentic RAG workflow, including messages, documents, and evaluation results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field, ValidationInfo, field_validator
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


# Validation-related dataclasses
@dataclass
class ValidationDetail:
    """Detail of a single validation check."""

    field: str
    is_valid: bool
    message: str


@dataclass
class EvaluationResult:
    """
    Result of document relevance evaluation.

    Attributes:
        is_relevant: Whether documents are relevant to query
        reason: Explanation for relevance/irrelevance
        recommendation: Whether to search again
        quality_score: Overall quality score (0-1)
    """

    is_relevant: bool
    reason: str
    recommendation: bool
    quality_score: float = 0.5

    def to_dict(self) -> dict:
        return {
            "is_relevant": self.is_relevant,
            "reason": self.reason,
            "recommendation": self.recommendation,
            "quality_score": self.quality_score,
        }


class ValidationStatus(str, Enum):
    """Status of answer validation."""

    VALID = "VALID"
    PARTIALLY_VALID = "PARTIALLY_VALID"
    INVALID = "INVALID"
    HALLUCINATED = "HALLUCINATED"


# LangGraph-specific TypedDict for state management
class GraphState(TypedDict):
    """
    State structure for LangGraph agent workflow.

    This state is used by LangGraph to track the agent's progress through
    the workflow. Fields can be appended to using LangGraph's annotated
    update mechanism.
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


# Pydantic BaseModel for comprehensive state management (used by agent.py)
class DocumentMetadata(BaseModel):
    """Metadata for a document."""

    source: str = Field(..., description="Source of the document")
    page: int = Field(..., description="Page number")
    url: Optional[str] = Field(None, description="URL if available")


class SearchHistoryEntry(BaseModel):
    """Entry in search history tracking."""

    iteration: int
    query: str
    document_count: int
    evaluation: Optional[dict] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ValidationDetailModel(BaseModel):
    """Pydantic version of ValidationDetail."""

    field: str
    is_valid: bool
    message: str


class ValidationResultModel(BaseModel):
    """Validation result with structured output."""

    status: ValidationStatus
    quality_score: float
    validation_details: List[ValidationDetailModel]
    issues: List[str]
    corrective_action: Optional[str] = None
    answer: Optional[str] = None  # Added to match documentation schema


class AgenticRagState(BaseModel):
    """
    Comprehensive state structure for Agentic RAG workflow.

    This Pydantic-based state model provides type safety and validation
    for the entire RAG workflow. It tracks all aspects of the workflow
    from query to final answer.

    Attributes:
        query: The current query being processed
        original_query: The user's original query (for tracking query evolution)
        retrieved_documents: Documents retrieved from search
        search_history: Track all search attempts and their results
        evaluation_result: Latest evaluation of document relevance
        relevance_scores: List of scores from multiple evaluations
        generated_answer: The answer generated by the agent
        answer_quality_score: Quality score of the generated answer (0-1)
        validation_result: Structured validation result
        search_count: Number of searches performed
        max_searches: Maximum allowed searches
        should_rerun: Whether to restart the search process
        rerun_reason: Explanation if rerun is triggered
        session_id: Unique identifier for the session
        timestamps: Dictionary of key timestamps
    """

    # Core query state
    query: str = Field(..., description="The current query being processed")
    original_query: Optional[str] = Field(None, description="Original user query")

    # Retrieved documents
    retrieved_documents: List[Document] = Field(default_factory=list)

    # Search tracking
    search_history: List[SearchHistoryEntry] = Field(default_factory=list)
    relevance_scores: List[float] = Field(default_factory=list)
    search_count: int = Field(default=0, ge=0, le=100)  # Track search attempts
    max_searches: int = Field(default=3, ge=1, le=10)  # Configurable limit

    # Generated answer
    generated_answer: Optional[str] = Field(None, description="Generated answer")

    # Answer quality metrics
    answer_quality_score: Optional[float] = Field(None, description="Quality score 0-1")

    # Validation tracking
    validation_result: Optional[ValidationResultModel] = Field(None)

    # Rerun control
    should_rerun: bool = Field(default=False)
    rerun_reason: Optional[str] = Field(None)

    # Session tracking
    session_id: Optional[str] = Field(None, description="Unique session identifier")

    # Timestamps
    timestamps: Dict[str, str] = Field(default_factory=dict)

    @field_validator("answer_quality_score")
    @classmethod
    def validate_quality_score(cls, v: Optional[float]) -> Optional[float]:
        """Ensure quality score is between 0 and 1."""
        if v is not None and not 0 <= v <= 1:
            raise ValueError("answer_quality_score must be between 0 and 1")
        return v

    @field_validator("search_count")
    @classmethod
    def validate_search_count(cls, v: int, info: "ValidationInfo") -> int:
        """Ensure search count doesn't exceed max."""
        max_searches = info.data.get("max_searches")
        if max_searches and v > max_searches:
            raise ValueError(
                f"search_count cannot exceed max_searches ({max_searches})"
            )
        return v

    def update_timestamp(self, field_name: str) -> None:
        """Update timestamp for a specific field."""
        self.timestamps[field_name] = datetime.now().isoformat()

    def record_search(
        self, query: str, documents_count: int, evaluation: Optional[dict] = None
    ) -> "AgenticRagState":
        """Record a search attempt in history."""
        entry = SearchHistoryEntry(
            iteration=self.search_count + 1,
            query=query,
            document_count=documents_count,
            evaluation=evaluation,
        )
        self.search_history.append(entry)
        self.search_count += 1
        return self

    def set_answer_quality(
        self, score: float, validation: Optional[ValidationResultModel] = None
    ) -> "AgenticRagState":
        """Set answer quality score from validation."""
        if validation:
            self.answer_quality_score = validation.quality_score
            self.validation_result = validation
        else:
            self.answer_quality_score = score
        return self

    def trigger_rerun(self, reason: str) -> "AgenticRagState":
        """Mark state for rerun with reason."""
        self.should_rerun = True
        self.rerun_reason = reason
        return self

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return {
            "query": self.query,
            "original_query": self.original_query,
            "retrieved_documents": [doc.to_dict() for doc in self.retrieved_documents],
            "search_history": [entry.model_dump() for entry in self.search_history],
            "relevance_scores": self.relevance_scores,
            "generated_answer": self.generated_answer,
            "answer_quality_score": self.answer_quality_score,
            "validation_result": (
                self.validation_result.model_dump() if self.validation_result else None
            ),
            "search_count": self.search_count,
            "max_searches": self.max_searches,
            "should_rerun": self.should_rerun,
            "rerun_reason": self.rerun_reason,
            "session_id": self.session_id,
            "timestamps": self.timestamps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgenticRagState":
        """Create state from dictionary."""
        documents = [
            Document.from_dict(doc) if isinstance(doc, dict) else doc
            for doc in data.get("retrieved_documents", [])
        ]

        history_entries = []
        for entry in data.get("search_history", []):
            if isinstance(entry, dict):
                history_entries.append(SearchHistoryEntry.model_validate(entry))
            else:
                history_entries.append(entry)

        validation = None
        if data.get("validation_result"):
            validation = ValidationResultModel.model_validate(data["validation_result"])

        messages = []
        for msg in data.get("messages", []):
            if isinstance(msg, dict):
                messages.append(Message.from_dict(msg))
            elif isinstance(msg, Message):
                messages.append(msg)
            else:
                messages.append(msg)

        return cls.model_validate(
            {
                "query": data.get("query", ""),
                "original_query": data.get("original_query"),
                "retrieved_documents": documents,
                "search_history": history_entries,
                "relevance_scores": data.get("relevance_scores", []),
                "generated_answer": data.get("generated_answer"),
                "answer_quality_score": data.get("answer_quality_score"),
                "validation_result": validation,
                "search_count": data.get("search_count", 0),
                "max_searches": data.get("max_searches", 3),
                "should_rerun": data.get("should_rerun", False),
                "rerun_reason": data.get("rerun_reason"),
                "session_id": data.get("session_id"),
                "timestamps": data.get("timestamps", {}),
            }
        )


class AgentState(AgenticRagState):
    """
    Convenience wrapper around AgenticRagState for backwards compatibility.

    Maintains the same interface as the dataclass-based AgentState but
    inherits all the enhanced features from AgenticRagState.

    Messages are now a proper Pydantic field to ensure serialization works correctly.
    """

    messages: List[Message] = Field(default_factory=list)

    def __init__(self, **data):
        """Initialize AgentState with optional data."""
        # Handle backwards-compatible 'answer' field (maps to generated_answer)
        if "answer" in data:
            data["generated_answer"] = data.pop("answer")

        # Store GraphState-specific fields that aren't in AgenticRagState
        graph_state_fields = {
            "is_relevant": data.pop("is_relevant", None),
            "should_search_again": data.pop("should_search_again", None),
            "validation_passed": data.pop("validation_passed", None),
            "correction_triggered": data.pop("correction_triggered", None),
            "hallucination_score": data.pop("hallucination_score", None),
            "search_query": data.pop("search_query", None),
            "search_results": data.pop("search_results", None),
            "error": data.pop("error", None),
        }

        # Handle search_count - if provided, use it; otherwise let AgenticRagState default
        search_count_value = data.get("search_count")

        # Set default values for required fields if not provided
        if "query" not in data:
            data["query"] = ""
        if "original_query" not in data:
            data["original_query"] = None
        if "session_id" not in data:
            from datetime import datetime

            data["session_id"] = f"session_{datetime.now().isoformat()}"

        # Override search_count if it was provided (to handle GraphState compatibility)
        search_count_value = data.get("search_count")

        super().__init__(**data)

        # Track messages separately if provided
        messages_data = data.get("messages", [])
        if messages_data:
            for msg in messages_data:
                if isinstance(msg, dict):
                    self.messages.append(Message.from_dict(msg))
                elif isinstance(msg, Message):
                    self.messages.append(msg)
                else:
                    self.messages.append(msg)

        # Override search_count if it was provided (to handle GraphState compatibility)
        if search_count_value is not None:
            self.search_count = search_count_value

        # Store GraphState-specific fields as private attributes for direct access
        for key, value in graph_state_fields.items():
            if value is not None:
                object.__setattr__(self, f"_GraphState__{key}", value)

    def to_dict(self) -> dict:
        """Convert state to dictionary with messages included."""
        state_dict = super().to_dict()
        state_dict["messages"] = [msg.to_dict() for msg in self.messages]
        return state_dict

    @classmethod
    def from_dict(cls, data: dict) -> "AgentState":
        """Create AgentState from dictionary with messages."""
        # Extract messages first
        messages_data = data.get("messages", [])
        messages = []
        for msg in messages_data:
            if isinstance(msg, dict):
                messages.append(Message.from_dict(msg))
            elif isinstance(msg, Message):
                messages.append(msg)
            else:
                messages.append(msg)

        # Create state with messages
        state = cls.model_validate(
            {
                **data,
                "retrieved_documents": [
                    Document.from_dict(doc) if isinstance(doc, dict) else doc
                    for doc in data.get("retrieved_documents", [])
                ],
                "search_history": [
                    (
                        SearchHistoryEntry.model_validate(entry)
                        if isinstance(entry, dict)
                        else entry
                    )
                    for entry in data.get("search_history", [])
                ],
                "validation_result": (
                    ValidationResultModel.model_validate(data["validation_result"])
                    if data.get("validation_result")
                    else None
                ),
            }
        )

        # Update messages after validation
        state.messages = messages
        return state

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"AgenticRagState(query='{self.query[:50]}...', "
            f"search_count={self.search_count}, "
            f"documents={len(self.retrieved_documents)}, "
            f"quality_score={self.answer_quality_score})"
        )

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access (e.g., state['query'])."""
        return getattr(self, key, None)

    def get(self, key: str, default: Any = None) -> Any:
        """Support dictionary-style get() method."""
        value = getattr(self, key, None)
        return value if value is not None else default

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator (e.g., 'query' in state)."""
        return hasattr(self, key)

    def add_message(self, role: MessageRole, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: Role of the message sender (user, assistant, system, tool)
            content: Text content of the message
        """
        message = Message(role=role, content=content)
        self.messages.append(message)

    @property
    def documents(self) -> List[Document]:
        """Get documents from the state (alias for retrieved_documents)."""
        return self.retrieved_documents

    @documents.setter
    def documents(self, value: List[Document]):
        """Set documents (sets retrieved_documents)."""
        self.retrieved_documents = value

    @property
    def answer(self) -> Optional[str]:
        """Get answer from the state (alias for generated_answer)."""
        return self.generated_answer

    @answer.setter
    def answer(self, value: str):
        """Set answer (sets generated_answer)."""
        self.generated_answer = value

    @property
    def context(self) -> str:
        """Get context from documents."""
        return "\n\n".join([doc.page_content for doc in self.retrieved_documents])

    @property
    def is_relevant(self) -> Optional[bool]:
        """Get relevance flag (for backwards compatibility)."""
        if hasattr(self, "_is_relevant"):
            return self._is_relevant
        # Check for direct attribute (from GraphState dict updates)
        if hasattr(self, "_GraphState__is_relevant"):
            return self._GraphState__is_relevant
        return None

    @is_relevant.setter
    def is_relevant(self, value: bool):
        """Set relevance flag."""
        self._is_relevant = value

    @property
    def should_search_again(self) -> Optional[bool]:
        """Get should search again flag (for backwards compatibility)."""
        # Check for direct attribute (from GraphState dict updates)
        if hasattr(self, "_GraphState__should_search_again"):
            return self._GraphState__should_search_again
        if hasattr(self, "_should_search_again"):
            return self._should_search_again
        return self.should_rerun

    @should_search_again.setter
    def should_search_again(self, value: bool):
        """Set should search again flag."""
        self._GraphState__should_search_again = value
        self._should_search_again = value
        self.should_rerun = value

    @property
    def search_query(self) -> str:
        """Get search query (alias for query)."""
        return self.query

    @search_query.setter
    def search_query(self, value: str):
        """Set search query."""
        self.query = value

    @property
    def search_results(self) -> List[dict]:
        """Get search results (placeholder)."""
        if hasattr(self, "_search_results"):
            return self._search_results
        return []

    @search_results.setter
    def search_results(self, value: List[dict]):
        """Set search results."""
        self._search_results = value

    @property
    def validation_passed(self) -> Optional[bool]:
        """Get validation passed flag."""
        if hasattr(self, "_validation_passed"):
            return self._validation_passed
        if self.validation_result:
            return self.validation_result.status in [
                ValidationStatus.VALID,
                ValidationStatus.PARTIALLY_VALID,
            ]
        # Check for direct attribute (from GraphState dict updates)
        if hasattr(self, "_GraphState__validation_passed"):
            return self._GraphState__validation_passed
        return None

    @validation_passed.setter
    def validation_passed(self, value: bool):
        """Set validation passed flag."""
        self._validation_passed = value

    @property
    def correction_triggered(self) -> Optional[bool]:
        """Get correction triggered flag."""
        if hasattr(self, "_correction_triggered"):
            return self._correction_triggered
        # Check for direct attribute (from GraphState dict updates)
        if hasattr(self, "_GraphState__correction_triggered"):
            return self._GraphState__correction_triggered
        return None

    @correction_triggered.setter
    def correction_triggered(self, value: bool):
        """Set correction triggered flag."""
        self._correction_triggered = value

    @property
    def hallucination_score(self) -> Optional[float]:
        """Get hallucination score."""
        # Check for direct attribute (from GraphState dict updates)
        if hasattr(self, "_GraphState__hallucination_score"):
            return self._GraphState__hallucination_score
        return None

    @hallucination_score.setter
    def hallucination_score(self, value: float):
        """Set hallucination score."""
        self._GraphState__hallucination_score = value

    @property
    def error(self) -> Optional[str]:
        """Get error (for backwards compatibility)."""
        if hasattr(self, "_error"):
            return self._error
        return None

    @error.setter
    def error(self, value: str):
        """Set error."""
        self._error = value

    @property
    def iteration(self) -> int:
        """Get iteration count (alias for search_count)."""
        return self.search_count

    @iteration.setter
    def iteration(self, value: int):
        """Set iteration count."""
        self.search_count = value
