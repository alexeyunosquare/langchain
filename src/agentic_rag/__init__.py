"""
Agentic RAG Package - Self-Corrective RAG with LangGraph.

This package implements a self-corrective Retrieval-Augmented Generation (RAG)
system using LangGraph for orchestration. It includes:
- Document relevance evaluation
- LangGraph state machine orchestration (Phase 5)
- CRAG (Corrective RAG) logic with validation schema
- Tavily search integration
- Hallucination detection and correction
- Phase 6: Integration and Optimization components
"""

from .agent import AgenticRAGAgent, AgentResult
from .config import AgenticRAGConfig
from .corrective import (
    AnswerValidator,
    CorrectionEngine,
    CorrectionResult,
    CorrectionStrategy,
    CorrectiveRAG,
)
from .evaluator import EvaluationResult, RelevanceEvaluator
from .factory import create_agentic_rag_agent, create_default_agentic_rag
from .graph import (
    LangGraphAgenticRAG,
    LangGraphNode,
    build_agentic_rag_graph,
    create_agentic_graph_workflow,
)
from .search import (
    DocumentResult,
    HybridRetrievalResult,
    HybridRetriever,
    QueryRefiner,
    SearchResults,
    TavilySearch,
    TavilySearchIntegration,
)
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
    # Agent (traditional)
    "AgenticRAGAgent",
    "AgentResult",
    # LangGraph Orchestration (Phase 5)
    "LangGraphAgenticRAG",
    "LangGraphNode",
    "build_agentic_rag_graph",
    "create_agentic_graph_workflow",
    # Corrective (Phase 4)
    "CorrectiveRAG",
    "AnswerValidator",
    "ValidationResult",
    "ValidationStatus",
    "ValidationDetail",
    "CorrectionEngine",
    "CorrectionResult",
    "CorrectionStrategy",
    # Search (Phase 5)
    "TavilySearch",
    "TavilySearchIntegration",
    "SearchResults",
    "DocumentResult",
    "HybridRetriever",
    "HybridRetrievalResult",
    "QueryRefiner",
    # Factory (Phase 6)
    "create_agentic_rag_agent",
    "create_default_agentic_rag",
]

__version__ = "0.1.0"
