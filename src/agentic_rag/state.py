"""
Unified state definitions for Agentic RAG.

All state management uses Pydantic models for type safety,
validation, and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Valid message roles in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """
    Represents a message in the conversation.

    Attributes:
        role: Role of the message sender
        content: Text content of the message
        metadata: Optional metadata dictionary
    """

    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def __str__(self) -> str:
        return f"[{self.role.value}] {self.content}"


class Document(BaseModel):
    """
    Represents a retrieved document.

    Attributes:
        page_content: The text content of the document
        metadata: Metadata dictionary with source, page, etc.
        score: Optional relevance score (0-1)
    """

    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

    model_config = {"arbitrary_types_allowed": True}

    def __str__(self) -> str:
        return f"[{self.metadata.get('source', 'unknown')}] {self.page_content[:50]}..."


class ValidationStatus(str, Enum):
    """Status of answer validation."""

    VALID = "valid"
    PARTIALLY_VALID = "partially_valid"
    INVALID = "invalid"
    HALLUCINATED = "hallucinated"


class ValidationDetail(BaseModel):
    """Detail of a single validation check."""

    field: str
    is_valid: bool
    message: str


class ValidationResult(BaseModel):
    """Validation result with structured output."""

    status: ValidationStatus
    quality_score: float
    validation_details: List[ValidationDetail]
    issues: List[str] = Field(default_factory=list)
    corrective_action: Optional[str] = None
    answer: Optional[str] = None


class SearchHistoryEntry(BaseModel):
    """Entry in search history tracking."""

    iteration: int
    query: str
    document_count: int
    evaluation: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class EvaluationResult(BaseModel):
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


class AgenticRAGState(BaseModel):
    """
    Single unified state for all workflows.

    This model replaces GraphState, AgenticRagState, and AgentState,
    providing a single source of truth for state management across
    the entire RAG pipeline.

    Attributes:
        query: Current query being processed
        original_query: User's original query (for tracking evolution)
        messages: Conversation history
        documents: Retrieved documents
        context: Concatenated document content
        answer: Generated answer
        is_relevant: Whether documents are relevant
        should_rerun: Whether to restart search process
        validation_result: Structured validation result
        search_count: Number of searches performed
        session_id: Unique session identifier
        timestamps: Dictionary of key timestamps
    """

    # Core query state
    query: str = Field(default="", description="Current query being processed")
    original_query: Optional[str] = Field(None, description="Original user query")

    # Conversation
    messages: List[Message] = Field(default_factory=list)

    # Retrieved documents
    documents: List[Document] = Field(default_factory=list)
    context: str = Field(default="", description="Concatenated document content")

    # Generated answer
    answer: Optional[str] = Field(None, description="Generated answer")

    # Evaluation results
    is_relevant: Optional[bool] = None
    should_rerun: bool = Field(default=False)
    rerun_reason: Optional[str] = Field(None)

    # Validation
    validation_result: Optional[ValidationResult] = None
    validation_passed: Optional[bool] = None
    correction_triggered: Optional[bool] = None
    hallucination_score: Optional[float] = None
    answer_quality_score: Optional[float] = Field(None, ge=0, le=1)

    # Search tracking
    search_count: int = Field(default=0, ge=0, le=100)
    search_history: List[SearchHistoryEntry] = Field(default_factory=list)
    relevance_scores: List[float] = Field(default_factory=list)

    # Session tracking
    session_id: str = Field(
        default_factory=lambda: f"session_{datetime.now().isoformat()}"
    )
    timestamps: Dict[str, str] = Field(default_factory=dict)

    # Error handling
    error: Optional[str] = None

    # Iteration tracking (alias for search_count, kept for backwards compatibility)
    iteration: int = Field(default=0, ge=0)

    model_config = {"arbitrary_types_allowed": True}

    def add_message(self, role: MessageRole, content: str) -> None:
        """Add a message to the conversation history."""
        message = Message(role=role, content=content)
        self.messages.append(message)

    def update_timestamp(self, field_name: str) -> None:
        """Update timestamp for a specific field."""
        self.timestamps[field_name] = datetime.now().isoformat()

    def record_search(
        self,
        query: str,
        documents_count: int,
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a search attempt in history."""
        entry = SearchHistoryEntry(
            iteration=self.search_count + 1,
            query=query,
            document_count=documents_count,
            evaluation=evaluation,
        )
        self.search_history.append(entry)
        self.search_count += 1
        self.iteration = self.search_count

    def set_answer_quality(
        self, score: float, validation: Optional[ValidationResult] = None
    ) -> None:
        """Set answer quality score from validation."""
        self.answer_quality_score = score
        if validation:
            self.validation_result = validation

    def trigger_rerun(self, reason: str) -> None:
        """Mark state for rerun with reason."""
        self.should_rerun = True
        self.rerun_reason = reason

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"AgenticRAGState(query='{self.query[:50]}...', "
            f"search_count={self.search_count}, "
            f"documents={len(self.documents)}, "
            f"quality_score={self.answer_quality_score})"
        )

    # Backwards-compatibility properties
    @property
    def retrieved_documents(self) -> List[Document]:
        """Get retrieved documents (alias for documents)."""
        return self.documents

    @retrieved_documents.setter
    def retrieved_documents(self, value: List[Document]) -> None:
        """Set retrieved documents."""
        self.documents = value

    @property
    def generated_answer(self) -> Optional[str]:
        """Get generated answer (alias for answer)."""
        return self.answer

    @generated_answer.setter
    def generated_answer(self, value: Optional[str]) -> None:
        """Set generated answer."""
        self.answer = value

    @property
    def should_search_again(self) -> Optional[bool]:
        """Get should search again (alias for should_rerun)."""
        return self.should_rerun

    @should_search_again.setter
    def should_search_again(self, value: bool) -> None:
        """Set should search again."""
        self.should_rerun = value

    @property
    def search_query(self) -> str:
        """Get search query (alias for query)."""
        return self.query

    @search_query.setter
    def search_query(self, value: str) -> None:
        """Set search query."""
        self.query = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgenticRAGState":
        """Create state from dictionary representation."""
        return cls.model_validate(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Support dictionary-style get() method."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return getattr(self, key, None)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return hasattr(self, key)


# Compatibility aliases for gradual migration (remove after full migration)
AgentState = AgenticRAGState
AgenticRagState = AgenticRAGState
ValidationResultModel = ValidationResult
ValidationDetailModel = ValidationDetail
