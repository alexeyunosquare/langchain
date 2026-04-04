# Phase 1: Setup and Dependencies

## Objective

Establish the project foundation for Agentic RAG development with proper dependencies, directory structure, and initial test scaffolding.

## Prerequisites

- Poetry installed (for dependency management)
- Python 3.13+ environment
- Tavily API key (for enhanced search)
- Existing `conversational-rag` project setup

## Tasks

### 1.1 Add Dependencies

Add required packages to `pyproject.toml`:

```toml
[project.dependencies]
# Add to existing dependencies:
langgraph>=0.2.0          # For stateful agent workflows
tavily-python>=0.5.0      # For enhanced search capabilities
```

Add to dev dependencies if needed:
```toml
[dependency-groups.dev]
# Add mock libraries for testing
pytest-mock>=3.10.0
```

### 1.2 Create Directory Structure

Create the agentic_rag package structure:

```bash
mkdir -p src/agentic_rag
mkdir -p test/agentic_rag
touch src/agentic_rag/__init__.py
touch test/agentic_rag/__init__.py
```

### 1.3 Initialize Test Configuration

Update `test/conftest.py` with shared fixtures for agentic RAG:

```python
# Pseudo: Add fixtures for mock LLM, mock search, mock evaluator
@pytest.fixture
def mock_llm():
    """Mock LLM for unit tests"""
    ...

@pytest.fixture
def mock_tavily_client():
    """Mock Tavily search client"""
    ...

@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    ...
```

### 1.4 Create Initial Test Scaffolding

Create empty test files with structure (TDD: tests first!):

**test/agentic_rag/test_state.py:**
```python
# Pseudo: Test LangGraph state definitions
class TestAgentState:
    def test_state_initialization(self):
        """Test that agent state initializes with correct default values"""
        ...
    
    def test_state_serialization(self):
        """Test state can be serialized/deserialized"""
        ...
```

**test/agentic_rag/test_evaluator.py:**
```python
# Pseudo: Test document relevance evaluation
class TestRelevanceEvaluator:
    def test_evaluate_highly_relevant_document(self, mock_llm, sample_documents):
        """Test evaluator correctly identifies relevant documents"""
        ...
    
    def test_evaluate_irrelevant_document(self, mock_llm, sample_documents):
        """Test evaluator correctly identifies irrelevant documents"""
        ...
    
    def test_evaluate_edge_cases(self, mock_llm):
        """Test evaluator handles edge cases (empty docs, malformed input)"""
        ...
```

**test/agentic_rag/test_agent.py:**
```python
# Pseudo: Test agent orchestration
class TestAgenticRagAgent:
    def test_agent_initialization(self):
        """Test agent initializes with correct components"""
        ...
    
    def test_agent_workflow_execution(self, mock_llm, mock_tavily_client):
        """Test agent can execute complete workflow"""
        ...
    
    def test_agent_decision_to_rerun(self, mock_llm):
        """Test agent decides to re-search when documents are irrelevant"""
        ...
```

**test/agentic_rag/test_corrective.py:**
```python
# Pseudo: Test CRAG logic
class TestCorrectiveRag:
    def test_identify_hallucination(self, mock_llm):
        """Test detection of potential hallucinations"""
        ...
    
    def test_trigger_correction(self, mock_llm):
        """Test correction mechanism when hallucination detected"""
        ...
    
    def test_quality_score_calculation(self, mock_llm):
        """Test quality scoring of generated answers"""
        ...
```

**test/agentic_rag/test_search.py:**
```python
# Pseudo: Test Tavily integration
class TestTavilySearch:
    def test_search_execution(self, mock_tavily_client):
        """Test search returns results in expected format"""
        ...
    
    def test_search_with_filters(self, mock_tavily_client):
        """Test search with various query filters"""
        ...
    
    def test_search_error_handling(self):
        """Test graceful handling of search failures"""
        ...
```

### 1.5 Update Makefile

Add agentic RAG specific test targets:

```makefile
# Add to Makefile:
test-agentic:
	poetry run pytest test/agentic_rag/ -v --tb=short

test-agentic-fast:
	poetry run pytest test/agentic_rag/ -v --tb=short -m "not integration"
```

## Test Execution Strategy

1. Run all new tests - they should **fail** (red phase of TDD)
2. Verify test failures are meaningful (assertion errors, not import errors)
3. Document expected behavior in test docstrings
4. Commit test scaffolding before any implementation

## Success Criteria

- ✅ All dependencies added and installed successfully
- ✅ Directory structure created
- ✅ Test scaffolding in place with failing tests
- ✅ Makefile targets working
- ✅ Tests can be run independently (`make test-agentic`)

## Expected Test Output

```bash
$ make test-agentic
# Should show all tests failing (expected in TDD)
FAILED test/agentic_rag/test_state.py::TestAgentState::test_state_initialization
FAILED test/agentic_rag/test_evaluator.py::TestRelevanceEvaluator::test_evaluate_highly_relevant_document
...
```

## Next Steps

Once Phase 1 is complete, proceed to **Phase 2: Retrieval Evaluator** where we implement the document relevance evaluation component to make the first tests pass.

## Notes

- Keep tests focused and isolated
- Use mocking extensively for external dependencies
- Document expected behavior clearly in test docstrings
- Consider integration tests separately marked with `@pytest.mark.integration`
