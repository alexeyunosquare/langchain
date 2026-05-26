"""
FastMCP server for Agentic RAG.

Exposes the agentic RAG system as MCP tools over Streamable HTTP transport
with bearer token authentication.

Environment variables:
    MCP_API_TOKEN: Bearer token for HTTP auth (required)
    MCP_HOST: HTTP bind host (default: 127.0.0.1)
    MCP_PORT: HTTP bind port (default: 8000)
    OPENAI_API_KEY: OpenAI API key
    LM_STUDIO_HOST: LM Studio host (default: localhost)
    LM_STUDIO_PORT: LM Studio port (default: 1234)
    TAVILY_API_KEY: Tavily web search API key
    CHROMA_PERSIST_DIR: Persistent vector store directory
    RAG_EVALUATION_THRESHOLD: Relevance threshold (default: 0.7)
    RAG_MAX_SEARCH_ITERATIONS: Max search loops (default: 3)
"""

import os
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.server.auth.providers.debug import DebugTokenVerifier

from agentic_rag import (
    AgenticRAGAgent,
    AgenticRAGConfig,
    create_default_agentic_rag,
)
from agentic_rag.state import Document
from conversational_rag import RAGChain, create_rag_chain

# ---------------------------------------------------------------------------
# Lazy agent / chain registry
# ---------------------------------------------------------------------------

class _AgentRegistry:
    """
    Singleton that lazily creates and caches the AgenticRAG agent and
    Conversational RAG chain on first tool call.

    Reuses the same instances across tool calls so conversation memory
    and the vector store are initialised only once.
    """

    _instance: Optional["_AgentRegistry"] = None
    _agent: Optional[AgenticRAGAgent] = None
    _rag_chain: Optional[RAGChain] = None

    def __new__(cls) -> "_AgentRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_agent(self) -> AgenticRAGAgent:
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    def get_rag_chain(self) -> RAGChain:
        if self._rag_chain is None:
            self._rag_chain = create_rag_chain()
        return self._rag_chain

    def _create_agent(self) -> AgenticRAGAgent:
        """Create the AgenticRAGAgent from environment configuration."""
        config = AgenticRAGConfig.from_env()
        rag_chain = self.get_rag_chain()
        return create_default_agentic_rag(
            llm=rag_chain.llm,
            local_retriever=rag_chain.retriever,
        )

    def reset(self) -> None:
        """Clear conversation memory and force re-initialisation."""
        if self._rag_chain is not None:
            self._rag_chain.clear_memory()


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

# Read bearer token from environment
_mcp_token = os.environ.get("MCP_API_TOKEN", "")

verifier = DebugTokenVerifier(
    validate=lambda token: token == _mcp_token,
)

mcp = FastMCP("agentic-rag", auth=verifier)


def _registry() -> _AgentRegistry:
    return _AgentRegistry()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def query(query: str, max_iterations: int = 3) -> dict:
    """Answer a question using agentic RAG with self-correction.

    Uses hybrid retrieval (local docs + Tavily web search),
    evaluates document relevance, and corrects hallucinations.

    Args:
        query: The question to answer.
        max_iterations: Maximum search iterations (1-5, default 3).

    Returns:
        Dictionary with answer, sources, and metadata.
    """
    agent = _registry().get_agent()
    result = agent.run(query, max_iterations=max_iterations)

    return {
        "answer": result.answer,
        "sources": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.score,
            }
            for doc in result.documents
        ],
        "search_iterations": result.search_iterations,
        "validation_passed": result.validation_passed,
        "hallucination_score": result.hallucination_score,
        "tavily_used": result.tavily_used,
        "total_documents": result.total_documents,
    }


@mcp.tool()
def search_documents(query: str, top_k: int = 5, use_web: bool = False) -> List[Dict[str, Any]]:
    """Search documents without generating an answer.

    Returns raw document snippets with metadata for inspection.

    Args:
        query: Search query text.
        top_k: Number of results to return (default 5).
        use_web: Include Tavily web search results (default False).

    Returns:
        List of document dictionaries with content and metadata.
    """
    agent = _registry().get_agent()

    if use_web and agent.hybrid_retriever:
        result = agent.hybrid_retriever.retrieve(query, max_local_results=top_k)
        documents = agent._convert_hybrid_to_documents(result)
    else:
        local_results = agent._retrieve_documents_local(query)
        documents = agent._convert_to_documents(local_results)

    return [doc.model_dump() for doc in documents[:top_k]]


@mcp.tool()
def validate_answer(answer: str, query: str, sources: List[str]) -> dict:
    """Validate an answer for hallucinations and quality.

    Uses the corrective RAG validator to check if the answer
    is supported by the provided source texts.

    Args:
        answer: The answer text to validate.
        query: The original question.
        sources: List of source document texts.

    Returns:
        Validation result with status, quality score, and issues.
    """
    agent = _registry().get_agent()

    from langchain_core.documents import Document as LC_Doc

    lc_docs = [
        LC_Doc(page_content=src) for src in sources
    ]

    validation = agent.corrective.answer_validator.validate(
        answer=answer,
        documents=lc_docs,
        query=query,
    )

    return {
        "status": validation.status.value if hasattr(validation.status, "value") else str(validation.status),
        "quality_score": validation.quality_score,
        "issues": validation.issues,
        "corrective_action": validation.corrective_action,
        "is_hallucinated": getattr(validation, "is_hallucinated", False),
    }


@mcp.tool()
def add_document(file_path: str) -> dict:
    """Load a text document and add it to the vector store.

    Args:
        file_path: Path to a text file to ingest.

    Returns:
        Status dictionary with number of chunks added.
    """
    chain = _registry().get_rag_chain()
    count = chain.load_and_store_document(file_path)
    return {
        "status": "ok",
        "chunks_added": count,
        "source": file_path,
    }


@mcp.tool()
def get_config() -> dict:
    """Return the current agentic RAG configuration.

    Shows evaluation threshold, max iterations, model settings, etc.
    """
    agent = _registry().get_agent()
    config = agent.config

    return {
        "evaluation_threshold": config.evaluation_threshold,
        "max_search_iterations": config.max_search_iterations,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "use_hybrid_retrieval": agent.use_hybrid_retrieval,
    }


@mcp.tool()
def reset_conversation() -> dict:
    """Clear conversation memory and reset the agent state.

    Returns:
        Status dictionary confirming reset.
    """
    _registry().reset()
    return {"status": "ok", "message": "Conversation memory cleared"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server."""
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", "8000"))

    if not _mcp_token:
        print("WARNING: MCP_API_TOKEN not set. Server accepts no auth.")

    mcp.run(transport="http", host=host, port=port)


def get_mcp_server() -> FastMCP:
    """Return the FastMCP server instance (for testing)."""
    return mcp


if __name__ == "__main__":
    main()