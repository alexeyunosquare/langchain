"""
Agentic RAG Package - Self-Corrective RAG with LangGraph.

This package implements a self-corrective Retrieval-Augmented Generation (RAG)
system using LangGraph for orchestration. It includes:
- Document relevance evaluation
- LangGraph state machine orchestration
- CRAG (Corrective RAG) logic
- Tavily search integration
- Hallucination detection and correction
"""

from .agent import AgenticRAGAgent, AgentResult
from .config import AgenticRAGConfig
from .corrective import (
    AnswerValidator,
    CorrectionResult,
    CorrectionStrategy,
    CorrectiveRAG,
    ValidationResult,
)
from .evaluator import EvaluationResult, RelevanceEvaluator
from .search import SearchResults, TavilySearch
from .state import AgentState, Document, GraphState, Message

__all__ = [
    # Config
    "AgenticRAGConfig",
    # State
    "AgentState",
    "GraphState",
    "Message",
    "Document",
    # Evaluator
    "RelevanceEvaluator",
    "EvaluationResult",
    # Agent
    "AgenticRAGAgent",
    "AgentResult",
    # Corrective
    "CorrectiveRAG",
    "AnswerValidator",
    "CorrectionStrategy",
    "CorrectionResult",
    "ValidationResult",
    # Search
    "TavilySearch",
    "SearchResults",
]

__version__ = "0.1.0"
