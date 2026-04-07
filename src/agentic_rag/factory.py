"""
Factory functions for creating Agentic RAG components.

This module provides factory functions to easily create and configure
Agentic RAG agents with all required components.

Phase 6: Integration and Optimization
- Factory functions for easy agent creation
- Default configuration with environment variables
- Support for custom configurations
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever

from .agent import AgenticRAGAgent
from .config import AgenticRAGConfig
from .corrective import CorrectiveRAG
from .evaluator import RelevanceEvaluator
from .search import (
    HybridRetriever,
    QueryRefiner,
    TavilySearch,
    TavilySearchIntegration,
)


def create_agentic_rag_agent(
    llm: BaseLanguageModel,
    local_retriever: BaseRetriever,
    config: Optional[AgenticRAGConfig] = None,
    tavily_api_key: Optional[str] = None,
    use_hybrid_retrieval: bool = True,
    tavily_priority: float = 0.3,
    max_search_iterations: Optional[int] = None,
) -> AgenticRAGAgent:
    """
    Factory function to create a configured Agentic RAG agent.

    This function creates and configures all necessary components
    for a fully functional Agentic RAG agent.

    Args:
        llm: Language model for processing
        local_retriever: Local document retriever for search
        config: Optional configuration (uses defaults from env if None)
        tavily_api_key: Optional Tavily API key (from env if not provided)
        use_hybrid_retrieval: Whether to use hybrid local+Tavily retrieval
        tavily_priority: Weight for Tavily results (0-1)
        max_search_iterations: Override max search iterations (uses config if None)

    Returns:
        Configured AgenticRAGAgent instance

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_community.vectorstores import Chroma
        >>>
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> retriever = Chroma.as_retriever()
        >>>
        >>> agent = create_agentic_rag_agent(llm, retriever)
        >>> result = agent.run("What is LangChain?")
    """
    # Load or create config
    if config is None:
        config = AgenticRAGConfig.from_env()

    # Override max iterations if provided
    if max_search_iterations is not None:
        config.max_search_iterations = max_search_iterations

    # Initialize Tavily search if hybrid retrieval is enabled
    tavily_search: Optional[TavilySearch] = None
    if use_hybrid_retrieval:
        tavily_search = TavilySearch(api_key=tavily_api_key)

    # Create evaluator
    evaluator = RelevanceEvaluator(
        llm=llm,
        threshold=config.evaluation_threshold,
    )

    # Create corrective RAG
    corrective = CorrectiveRAG(
        llm=llm,
        correction_threshold=config.evaluation_threshold,
        max_correction_attempts=config.max_search_iterations,
    )

    # Create and return agent
    return AgenticRAGAgent(
        llm=llm,
        local_retriever=local_retriever,
        evaluator=evaluator,
        tavily_search=tavily_search,
        corrective=corrective,
        config=config,
        use_hybrid_retrieval=use_hybrid_retrieval,
        tavily_priority=tavily_priority,
        max_iterations=config.max_search_iterations,
    )


def create_default_agentic_rag(
    llm: BaseLanguageModel,
    local_retriever: BaseRetriever,
    config_dict: Optional[Dict[str, Any]] = None,
) -> AgenticRAGAgent:
    """
    Create agent with default configuration.

    This is a convenience function for quick setup using default
    configuration values.

    Args:
        llm: Language model for processing
        local_retriever: Local document retriever
        config_dict: Optional configuration dictionary

    Returns:
        Configured AgenticRAGAgent instance

    Example:
        >>> agent = create_default_agentic_rag(llm, retriever)
        >>> result = agent.run("Query here")
    """
    if config_dict:
        config = AgenticRAGConfig.from_dict(config_dict)
    else:
        config = AgenticRAGConfig.from_env()

    return create_agentic_rag_agent(
        llm=llm,
        local_retriever=local_retriever,
        config=config,
        use_hybrid_retrieval=True,
    )


def create_hybrid_retriever(
    local_retriever: BaseRetriever,
    tavily_search: TavilySearch,
    llm: Optional[BaseLanguageModel] = None,
    tavily_priority: float = 0.3,
    local_min_score: float = 0.5,
) -> HybridRetriever:
    """
    Create a hybrid retriever combining local and web search.

    Args:
        local_retriever: Local document retriever
        tavily_search: Tavily search integration
        llm: Language model for query refinement (optional)
        tavily_priority: Weight for Tavily results (0-1)
        local_min_score: Minimum local score to avoid Tavily

    Returns:
        Configured HybridRetriever instance
    """
    query_refiner = QueryRefiner(llm=llm) if llm else None

    return HybridRetriever(
        local_retriever=local_retriever,
        tavily_search=tavily_search,
        query_refiner=query_refiner,
        tavily_priority=tavily_priority,
        local_min_score=local_min_score,
    )


def create_evaluator(
    llm: BaseLanguageModel,
    threshold: float = 0.7,
) -> RelevanceEvaluator:
    """
    Create a document relevance evaluator.

    Args:
        llm: Language model for evaluation
        threshold: Threshold for relevance (0-1)

    Returns:
        Configured RelevanceEvaluator instance
    """
    return RelevanceEvaluator(llm=llm, threshold=threshold)


def create_corrective_rag(
    llm: BaseLanguageModel,
    quality_threshold: float = 0.7,
    max_attempts: int = 2,
) -> CorrectiveRAG:
    """
    Create a Corrective RAG component.

    Args:
        llm: Language model for correction
        quality_threshold: Threshold for triggering correction (0-1)
        max_attempts: Maximum correction attempts

    Returns:
        Configured CorrectiveRAG instance
    """
    return CorrectiveRAG(
        llm=llm,
        correction_threshold=quality_threshold,
        max_correction_attempts=max_attempts,
    )


def create_tavily_search(
    api_key: Optional[str] = None,
) -> TavilySearchIntegration:
    """
    Create a Tavily search integration.

    Args:
        api_key: Tavily API key (from env if not provided)

    Returns:
        Configured TavilySearchIntegration instance
    """
    tavily_search = TavilySearch(api_key=api_key)
    query_refiner = QueryRefiner(llm=None)  # Will be set when agent is created

    return TavilySearchIntegration(
        tavily_search=tavily_search,
        query_refiner=query_refiner,
    )


def merge_config_with_env(
    custom_config: Optional[AgenticRAGConfig] = None,
    **overrides: Any,
) -> AgenticRAGConfig:
    """
    Merge custom configuration with environment variables.

    Environment variables take precedence over custom config,
    except for explicitly provided overrides.

    Args:
        custom_config: Optional custom configuration
        **overrides: Override values (takes highest precedence)

    Returns:
        Merged AgenticRAGConfig instance
    """
    # Start with env-based config
    config = AgenticRAGConfig.from_env()

    # Apply custom config values
    if custom_config:
        for key, value in asdict(custom_config).items():
            # Don't override with None values
            if value is not None and key not in overrides:
                setattr(config, key, value)

    # Apply overrides (highest priority)
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
