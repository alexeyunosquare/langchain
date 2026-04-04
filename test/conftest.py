"""
Pytest fixtures and configuration for Conversational RAG tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conversational_rag.rag_chain import RAGChain

# Configuration
LM_STUDIO_HOST = os.getenv("LM_STUDIO_HOST", "localhost")
LM_STUDIO_PORT = int(os.getenv("LM_STUDIO_PORT", "1234"))


@pytest.fixture(scope="session")
def sample_docs_path():
    """Return path to sample documents file."""
    return Path(__file__).parent / "sample_docs.txt"


@pytest.fixture(scope="session")
def lmstudio_connection_params():
    """Return LM Studio connection parameters."""
    return {
        "host": LM_STUDIO_HOST,
        "port": LM_STUDIO_PORT,
        "base_url": f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1",
    }


@pytest.fixture
def clean_chroma_collection():
    """Create a clean Chroma collection for each test using in-memory ChromaDB."""
    import chromadb
    from langchain_community.vectorstores import Chroma

    # Default: use in-memory client
    client = chromadb.EphemeralClient()

    # Create or reset collection
    collection_name = "test_rag_collection"
    _collection = client.get_or_create_collection(collection_name)

    # Wrap with LangChain Chroma
    vectorstore = Chroma(
        collection_name=collection_name,
        client=client,
        embedding_function=None,
    )

    yield vectorstore

    # Cleanup after test - delete all documents
    client.delete_collection(collection_name)


@pytest.fixture
def rag_chain(clean_chroma_collection, lmstudio_connection_params):
    """Create a RAG chain with memory for testing."""
    # Create RAG chain with the pre-created vectorstore
    rag_chain = RAGChain(
        lmstudio_host=lmstudio_connection_params["host"],
        lmstudio_port=lmstudio_connection_params["port"],
        vectorstore=clean_chroma_collection,
    )

    return rag_chain


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# ==================== Agentic RAG Fixtures ====================


@pytest.fixture
def mock_llm():
    """Mock LLM for agentic RAG unit tests."""
    from unittest.mock import MagicMock

    from src.agentic_rag.state import Message

    mock = MagicMock()
    mock.invoke.return_value = Message(role="assistant", content="Mock LLM response")
    return mock


@pytest.fixture
def mock_tavily_client():
    """Mock Tavily search client for testing."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.search.return_value = {
        "results": [
            {
                "url": "https://example.com/doc1",
                "title": "Document 1",
                "content": "Relevant content",
                "score": 0.9,
            }
        ]
    }
    return mock


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    from src.agentic_rag.state import Document

    return [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "langchain_docs.txt", "page": 1, "id": "doc1"},
        ),
        Document(
            page_content="Python is a high-level programming language known for its simplicity.",
            metadata={"source": "python_docs.txt", "page": 1, "id": "doc2"},
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "ml_docs.txt", "page": 1, "id": "doc3"},
        ),
    ]


@pytest.fixture
def mock_retriever():
    """Mock document retriever for testing."""
    from unittest.mock import MagicMock

    from langchain_core.retrievers import BaseRetriever

    mock = MagicMock(spec=BaseRetriever)
    mock.invoke.return_value = [
        {"id": "doc1", "content": "Relevant document content", "score": 0.9}
    ]
    return mock
