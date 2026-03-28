"""
Pytest fixtures and configuration for Conversational RAG tests.
"""
import pytest
import os
from pathlib import Path

from conversational_rag.rag_chain import RAGChain, create_rag_chain

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
    from langchain_community.vectorstores import Chroma
    
    import chromadb
    
    # Default: use in-memory client
    client = chromadb.EphemeralClient()
    
    # Create or reset collection
    collection_name = "test_rag_collection"
    collection = client.get_or_create_collection(collection_name)
    
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
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )