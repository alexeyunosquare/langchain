# Phase 2: Retrieval Evaluator

## Objective

Implement the document relevance evaluation component that determines whether retrieved documents are relevant to the user's query. This is a critical component for the self-reflective nature of Agentic RAG.

## Component Overview

The **Relevance Evaluator** assesses retrieved documents and assigns relevance scores. It uses LLM-based evaluation to determine if documents should be used or if a new search is needed.

## Architecture

```
Query + Retrieved Documents → Evaluator → Relevance Score + Decision
                                           ↓
                                    ┌──────┴──────┐
                                    ↓             ↓
                              Relevant      Not Relevant
                              (use docs)    (re-search)
```

## Tasks

### 2.1 Define Evaluation State Schema

**File: `src/conversational_rag/agentic_rag/evaluator.py`**

Create data models for evaluation:

```python
# Pseudo code:
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional

class RelevanceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

class DocumentEvaluation(BaseModel):
    document_id: str
    relevance_score: float  # 0.0 to 1.0
    relevance_level: RelevanceLevel
    reasoning: str  # LLM explanation
    
class EvaluationResult(BaseModel):
    query: str
    documents_evaluated: int
    relevant_documents: List[DocumentEvaluation]
    overall_relevance: RelevanceLevel
    should_rerun_search: bool
    rerun_reason: Optional[str]
```

### 2.2 Implement Relevance Evaluator Class

**File: `src/conversational_rag/agentic_rag/evaluator.py`**

```python
# Pseudo code:
class RelevanceEvaluator:
    def __init__(
        self,
        llm: BaseLanguageModel,
        relevance_threshold: float = 0.7,
        min_relevant_docs: int = 2
    ):
        self.llm = llm
        self.relevance_threshold = relevance_threshold
        self.min_relevant_docs = min_relevant_docs
        self.evaluation_chain = self._build_evaluation_chain()
    
    def _build_evaluation_chain(self) -> Runnable:
        """Build LLM chain for document evaluation"""
        # Pseudo: Use LangChain prompt template + LLM
        # Prompt should evaluate document relevance to query
        # Output structured as DocumentEvaluation
        ...
    
    def evaluate(self, query: str, documents: List[Document]) -> EvaluationResult:
        """
        Evaluate relevance of documents to query
        
        Args:
            query: User's query
            documents: Retrieved documents
            
        Returns:
            EvaluationResult with scores and decisions
        """
        # Pseudo:
        # 1. Evaluate each document individually
        # 2. Aggregate results
        # 3. Determine if re-search needed
        # 4. Return structured result
        ...
    
    def _evaluate_single_document(
        self, 
        query: str, 
        document: Document
    ) -> DocumentEvaluation:
        """Evaluate single document relevance"""
        # Pseudo: Call evaluation chain with query + document content
        ...
    
    def _should_rerun_search(self, evaluations: List[DocumentEvaluation]) -> bool:
        """Determine if search should be rerun"""
        # Pseudo:
        # - Check if enough relevant documents
        # - Check average relevance score
        # - Return decision
        ...
```

### 2.3 Implement Tests (TDD - Make Tests Pass)

**File: `test/agentic_rag/test_evaluator.py`**

Now implement the actual test logic to make scaffolding tests pass:

```python
# Pseudo code structure:

class TestRelevanceEvaluator:
    
    def test_initialization(self):
        """Test evaluator initializes with correct defaults"""
        evaluator = RelevanceEvaluator(llm=mock_llm)
        assert evaluator.relevance_threshold == 0.7
        assert evaluator.min_relevant_docs == 2
    
    def test_evaluate_highly_relevant_document(self, mock_llm, sample_documents):
        """Test evaluator correctly identifies relevant documents"""
        # Pseudo:
        # - Create query and highly relevant document
        # - Call evaluator.evaluate()
        # - Assert relevance_score > threshold
        # - Assert relevance_level == HIGH
        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = evaluator.evaluate(query="Python programming", docs=[relevant_doc])
        assert result.overall_relevance == RelevanceLevel.HIGH
        assert not result.should_rerun_search
    
    def test_evaluate_irrelevant_document(self, mock_llm, sample_documents):
        """Test evaluator correctly identifies irrelevant documents"""
        # Pseudo:
        # - Create query and irrelevant document
        # - Call evaluator.evaluate()
        # - Assert relevance_score < threshold
        # - Assert should_rerun_search == True
        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = evaluator.evaluate(query="Python", docs=[irrelevant_doc])
        assert result.should_rerun_search == True
        assert result.rerun_reason is not None
    
    def test_evaluate_mixed_relevance(self, mock_llm, sample_documents):
        """Test evaluator handles mixed relevance documents"""
        # Pseudo: Mix of relevant and irrelevant docs
        # Should identify relevant ones and decide based on count
        ...
    
    def test_evaluate_empty_documents(self, mock_llm):
        """Test evaluator handles empty document list"""
        # Pseudo: Edge case handling
        evaluator = RelevanceEvaluator(llm=mock_llm)
        result = evaluator.evaluate(query="test", docs=[])
        assert result.should_rerun_search == True
    
    def test_threshold_configuration(self, mock_llm):
        """Test custom threshold configuration"""
        # Pseudo: Test with different thresholds
        evaluator = RelevanceEvaluator(llm=mock_llm, relevance_threshold=0.9)
        assert evaluator.relevance_threshold == 0.9
    
    def test_min_relevant_docs_logic(self, mock_llm, sample_documents):
        """Test min_relevant_docs decision logic"""
        # Pseudo: Even if docs are relevant, but below minimum count
        # Should trigger re-search
        ...
```

### 2.4 Create Mock Fixtures

**File: `test/conftest.py`**

Add fixtures needed for evaluator tests:

```python
# Pseudo:
@pytest.fixture
def sample_documents():
    """Return list of sample documents with varying relevance"""
    return [
        Document(page_content="Python is a programming language...", metadata={"id": "1"}),
        Document(page_content="Cooking recipes for dinner...", metadata={"id": "2"}),
        # More documents...
    ]

@pytest.fixture
def mock_llm_with_evaluator():
    """Mock LLM configured for evaluation tasks"""
    # Pseudo: Configure mock to return structured evaluation results
    ...
```

### 2.5 Integration with Existing RAG Chain

Update existing RAG chain to use evaluator:

**File: `src/conversational_rag/rag_chain.py`** (optional integration point)

```python
# Pseudo: Show integration point (not full implementation)
# In later phases, the evaluator will be called after retrieval
# to decide if documents should be used or re-search triggered
```

## Test Coverage Requirements

- ✅ 90%+ code coverage for evaluator module
- ✅ All edge cases tested (empty docs, malformed input)
- ✅ Threshold configuration tested
- ✅ Decision logic (rerun vs. use) thoroughly tested

## Success Criteria

- ✅ All tests in `test_evaluator.py` pass
- ✅ Evaluator correctly identifies relevant vs. irrelevant documents
- ✅ Relevance scores are consistent and meaningful
- ✅ Decision to re-search is based on configurable thresholds
- ✅ Integration points with existing RAG chain identified

## Performance Considerations

- Evaluation should be fast (< 500ms for typical document set)
- Consider caching for repeated queries
- Batch evaluation for multiple documents when possible

## Next Steps

After completing the evaluator, proceed to **Phase 3: LangGraph Agent State** where we build the state machine that orchestrates the evaluation and decision-making process.

## Notes

- Use LangChain's structured output for evaluation results
- Consider using a separate, smaller LLM for evaluation (cost optimization)
- Log evaluation decisions for debugging and metrics
- The evaluator is the "brain" that makes Agentic RAG self-reflective
