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
        # Extract and validate each value
        temperature_val = data.get("temperature")
        temperature: float = float(temperature_val) if isinstance(temperature_val, (int, float)) else 0.7

        top_k_val = data.get("top_k")
        top_k: int = int(top_k_val) if isinstance(top_k_val, (int, float)) else 5

        similarity_threshold_val = data.get("similarity_threshold")
        similarity_threshold: float = (
            float(similarity_threshold_val)
            if isinstance(similarity_threshold_val, (int, float))
            else 0.7
        )

        max_iterations_val = data.get("max_iterations")
        max_iterations: int = int(max_iterations_val) if isinstance(max_iterations_val, (int, float)) else 3

        retriever_type_val = data.get("retriever_type")
        retriever_type: str = str(retriever_type_val) if isinstance(retriever_type_val, str) else "vectorstore"

        evaluation_threshold_val = data.get("evaluation_threshold")
        evaluation_threshold: float = (
            float(evaluation_threshold_val)
            if isinstance(evaluation_threshold_val, (int, float))
            else 0.7
        )

        max_search_iterations_val = data.get("max_search_iterations")
        max_search_iterations: int = (
            int(max_search_iterations_val)
            if isinstance(max_search_iterations_val, (int, float))
            else 3
        )

        timeout_val = data.get("timeout")
        timeout: int = int(timeout_val) if isinstance(timeout_val, (int, float)) else 30

        include_domains_val = data.get("include_domains")
        include_domains: list[str] = (
            list(include_domains_val) if isinstance(include_domains_val, list) else []
        )

        exclude_domains_val = data.get("exclude_domains")
        exclude_domains: list[str] = (
            list(exclude_domains_val) if isinstance(exclude_domains_val, list) else []
        )

        return cls(
            temperature=temperature,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            max_iterations=max_iterations,
            retriever_type=retriever_type,
            evaluation_threshold=evaluation_threshold,
            max_search_iterations=max_search_iterations,
            timeout=timeout,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
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

        # Get env values, using default if None
        eval_threshold = os.getenv("RAG_EVALUATION_THRESHOLD")
        max_iter = os.getenv("RAG_MAX_SEARCH_ITERATIONS")
        temp = os.getenv("RAG_TEMPERATURE")
        top_k_val = os.getenv("RAG_TOP_K")
        timeout_val = os.getenv("RAG_TIMEOUT")

        config_dict = {
            "evaluation_threshold": float(eval_threshold if eval_threshold is not None else "0.7"),
            "max_search_iterations": int(max_iter if max_iter is not None else "3"),
            "temperature": float(temp if temp is not None else "0.7"),
            "top_k": int(top_k_val if top_k_val is not None else "5"),
            "timeout": int(timeout_val if timeout_val is not None else "30"),
        }

        # Handle list environment variables
        include_domains = os.getenv("RAG_INCLUDE_DOMAINS")
        if include_domains:
            config_dict["include_domains"] = [
                d.strip() for d in include_domains.split(",")
            ]

        exclude_domains = os.getenv("RAG_EXCLUDE_DOMAINS")
        if exclude_domains:
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
