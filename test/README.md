# Conversational RAG Test Harness

Comprehensive test suite for testing Conversational Retrieval-Augmented Generation (RAG) with memory using pytest and Docker.

## Components

- **LLM**: LM Studio (OpenAI-compatible local LLM server)
- **Vector Database**: ChromaDB
- **Test Framework**: pytest
- **Orchestration**: Docker Compose

## Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ installed
- pytest installed (`pip install pytest`)
- LangChain installed (`pip install langchain langchain-community`)
- ChromaDB client installed (`pip install chromadb`)

## Quick Start

### 1. Start Services

```bash
cd test
docker-compose up -d
```

Wait for services to be healthy:
```bash
docker-compose ps
```

### 2. Run Tests

```bash
# Run all tests (excluding Docker-dependent tests)
pytest test/ -v

# Run integration tests (requires Docker services)
pytest test/ -v -m integration

# Run specific test class
pytest test/test_rag.py::TestBasicRAG -v

# Run with coverage
pytest test/ -v --cov=. --cov-report=html
```

## Test Categories

### Document Ingestion Tests (`TestDocumentIngestion`)
- `test_load_sample_documents`: Verifies document loading
- `test_ingest_documents_to_chroma`: Tests ChromaDB ingestion

### Basic RAG Tests (`TestBasicRAG`)
- `test_answer_question_about_python`: Q&A about Python
- `test_answer_question_about_ml`: Q&A about Machine Learning

### Conversation Memory Tests (`TestConversationMemory`)
- `test_memory_persists_across_turns`: Multi-turn conversation with pronouns
- `test_context_aware_followup`: Context retention in follow-ups

### Source Document Tests (`TestSourceDocuments`)
- `test_returns_source_documents`: Verifies source document retrieval

### Edge Cases (`TestEdgeCases`)
- `test_handles_unrelated_question`: Questions outside document scope
- `test_empty_question`: Empty question handling

### Performance Tests (`TestPerformance`)
- `test_response_time`: Response time benchmarking

## Pytest Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.integration` | Integration tests |
| `@pytest.mark.requires_docker` | Requires Docker services |
| `@pytest.mark.slow` | Slow-running tests |

## Customization

### Environment Variables

Set these before running tests:

```bash
export CHROMA_HOST=localhost
export CHROMA_PORT=8000
export LM_STUDIO_HOST=localhost
export LM_STUDIO_PORT=1234
```

### Test Fixtures

Custom fixtures in `conftest.py`:
- `sample_docs_path`: Path to sample documents
- `chroma_connection_params`: ChromaDB connection settings
- `lmstudio_connection_params`: LM Studio connection settings
- `clean_chroma_collection`: Fresh Chroma collection per test
- `rag_chain`: Pre-configured RAG chain with memory

## Adding New Tests

```python
from pathlib import Path
import pytest

class TestNewFeature:
    @pytest.mark.integration
    @pytest.mark.requires_docker
    def test_new_feature(self, rag_chain, sample_docs_path):
        # Load documents into rag_chain
        # Ask questions
        # Verify responses with semantic assertions
        assert "answer" in result
        assert len(result["answer"]) > 0
```

## Sample Documents

Sample documents in `sample_docs.txt` cover:
1. Python Programming Basics
2. Machine Learning Fundamentals
3. Docker Containerization
4. REST API Design
5. Testing Best Practices

## Running Without Docker

For unit tests without Docker:

```bash
# Run only document ingestion tests (no Docker needed)
pytest test/test_rag.py::TestDocumentIngestion -v
```

## Troubleshooting

### Services not healthy
```bash
docker-compose logs
```

### Connection errors
Check services are running:
```bash
curl http://localhost:1234/health  # LM Studio
curl http://localhost:8000/api/v1/heartbeat  # ChromaDB
```

### Test failures
```bash
pytest test/ -v --tb=short
```

## File Structure

```
test/
├── README.md              # This file
├── conftest.py            # Pytest fixtures and configuration
├── docker-compose.yml     # Docker service definitions
├── sample_docs.txt        # Sample documents for testing
└── test_rag.py            # Test suite
```

## Notes

- Tests use semantic assertions rather than exact string matching
- Follow-up questions rely on conversation memory (e.g., "What is X?" → "How does it work?")
- Source documents are verified for relevance
- Performance tests ensure responses are generated within acceptable time limits