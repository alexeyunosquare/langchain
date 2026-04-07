"""
Configuration management for Agentic RAG.

This module provides centralized configuration management for the
agentic RAG system, including thresholds, limits, and model parameters.
"""

from dataclasses import dataclass, field


@dataclass
class AgenticRAGConfig:
    """
    Configuration class for Agentic RAG system.

    Attributes:
        evaluation_threshold: Threshold for document relevance scoring (0-1)
        max_search_iterations: Maximum number of search iterations allowed
        temperature: LLM temperature for response generation (0-1)
        top_k: Number of top results to retrieve per search
        timeout: Request timeout in seconds
        include_domains: Optional list of domains to include in web search
        exclude_domains: Optional list of domains to exclude from web search
    """

    # Evaluation parameters
    evaluation_threshold: float = 0.7

    # Search parameters
    max_search_iterations: int = 3
    top_k: int = 5
    timeout: int = 30

    # LLM parameters
    temperature: float = 0.7

    # Search filtering
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)

    # Deprecated aliases for backwards compatibility
    similarity_threshold: float = 0.7
    max_iterations: int = 3
    retriever_type: str = "vectorstore"

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if not 0 <= self.evaluation_threshold <= 1:
            raise ValueError("evaluation_threshold must be between 0 and 1")

        if self.max_search_iterations < 1:
            raise ValueError("max_search_iterations must be positive")

        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")

        if self.timeout < 1:
            raise ValueError("timeout must be at least 1 second")

        # Validate deprecated aliases if provided
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        if self.top_k < 1:
            raise ValueError("top_k must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AgenticRAGConfig":
        """Create configuration from dictionary."""
        return cls(
            temperature=float(data.get("temperature", 0.7)),
            top_k=int(data.get("top_k", 5)),
            similarity_threshold=float(data.get("similarity_threshold", 0.7)),
            max_iterations=int(data.get("max_iterations", 3)),
            retriever_type=str(data.get("retriever_type", "vectorstore")),
            evaluation_threshold=float(data.get("evaluation_threshold", 0.7)),
            max_search_iterations=int(data.get("max_search_iterations", 3)),
            timeout=int(data.get("timeout", 30)),
            include_domains=list(data.get("include_domains", [])),
            exclude_domains=list(data.get("exclude_domains", [])),
        )

    @classmethod
    def from_env(cls, **overrides: dict[str, object]) -> "AgenticRAGConfig":
        """
        Create configuration from environment variables.

        Args:
            **overrides: Override values for specific configuration (highest priority)

        Returns:
            AgenticRAGConfig instance

        Example:
            >>> config = AgenticRAGConfig.from_env(
            ...     evaluation_threshold=0.8,
            ...     max_search_iterations=5
            ... )
        """
        import os

        config_dict = {
            "evaluation_threshold": float(os.getenv("RAG_EVALUATION_THRESHOLD", "0.7")),
            "max_search_iterations": int(os.getenv("RAG_MAX_SEARCH_ITERATIONS", "3")),
            "temperature": float(os.getenv("RAG_TEMPERATURE", "0.7")),
            "top_k": int(os.getenv("RAG_TOP_K", "5")),
            "timeout": int(os.getenv("RAG_TIMEOUT", "30")),
        }

        # Handle list environment variables
        if include_domains := os.getenv("RAG_INCLUDE_DOMAINS"):
            config_dict["include_domains"] = [
                d.strip() for d in include_domains.split(",")
            ]

        if exclude_domains := os.getenv("RAG_EXCLUDE_DOMAINS"):
            config_dict["exclude_domains"] = [
                d.strip() for d in exclude_domains.split(",")
            ]

        # Apply overrides last (highest priority)
        for key, value in overrides.items():
            if value is not None:
                config_dict[key] = value

        return cls(**config_dict)

    def to_dict(self) -> dict[str, object]:
        """Convert configuration to dictionary."""
        return {
            "evaluation_threshold": self.evaluation_threshold,
            "max_search_iterations": self.max_search_iterations,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "timeout": self.timeout,
            "include_domains": self.include_domains,
            "exclude_domains": self.exclude_domains,
        }
