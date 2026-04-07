# Phase 3: LangGraph Agent State

## Objective

Implement the LangGraph state machine that orchestrates the Agentic RAG workflow. This phase defines the agent's state, nodes, and edges that enable autonomous decision-making.

## Component Overview

LangGraph provides a stateful, multi-node architecture for building agents. The Agentic RAG workflow will have multiple states and decision points:

```
[Receive Query] → [Retrieve Documents] → [Evaluate Relevance]
                                                   ↓
                                            ┌──────┴───────┐
                                            ↓             ↓
                                     Relevant       Not Relevant
                                            ↓             ↓
                                     [Generate        [Re-search]
                                      Answer]            │
                                            ↓            │
                                     [Validate] <────────┘
                                            ↓
                                     [Return Answer]
```

## Tasks

### 3.1 Define Agent State Schema

**File: `src/agentic_rag/state.py`**

```python
# Pseudo code:
from typing import List, Optional, Annotated
from langgraph.graph import StateGraph
from pydantic import BaseModel
from .evaluator import EvaluationResult

class AgenticRagState(BaseModel):
    """State schema for Agentic RAG workflow"""
    
    # Input
    query: str
    original_query: Optional[str] = None  # Track query evolution
    
    # Retrieval state
    retrieved_documents: List[Document] = []
    search_history: List[dict] = []  # Track search attempts
    
    # Evaluation state
    evaluation_result: Optional[EvaluationResult] = None
    relevance_scores: List[float] = []
    
    # Answer state
    generated_answer: Optional[str] = None
    answer_quality_score: Optional[float] = None
    validation_result: Optional[dict] = None
    
    # Control flow
    search_count: int = 0
    max_searches: int = 3
    should_rerun: bool = False
    rerun_reason: Optional[str] = None
    
    # Metadata
    session_id: Optional[str] = None
    timestamps: dict = {}

# Define state updates using LangGraph's annotated updates
# Pseudo:
# messages = Annotated[List[Message], add_messages]
# documents = Annotated[List[Document], add_documents]
```

### 3.2 Implement Node Functions

**File: `src/agentic_rag/agent.py`**

Each node in the graph performs a specific task:

```python
# Pseudo code:
from langgraph.graph import StateGraph, END
from .state import AgenticRagState
from .evaluator import RelevanceEvaluator

def retrieve_documents_node(state: AgenticRagState) -> dict:
    """Node: Retrieve documents based on query"""
    # Pseudo:
    # - Use existing RAG chain or vector store
    # - Store retrieved documents in state
    # - Increment search_count
    # - Log search attempt
    return {
        "retrieved_documents": documents,
        "search_count": state.search_count + 1,
        "search_history": [...new_search...]
    }

def evaluate_relevance_node(state: AgenticRagState) -> dict:
    """Node: Evaluate relevance of retrieved documents"""
    # Pseudo:
    # - Call RelevanceEvaluator
    # - Store evaluation_result in state
    # - Set should_rerun based on evaluation
    evaluator = RelevanceEvaluator(llm=state.llm)
    result = evaluator.evaluate(state.query, state.retrieved_documents)
    return {
        "evaluation_result": result,
        "should_rerun": result.should_rerun_search,
        "rerun_reason": result.rerun_reason
    }

def generate_answer_node(state: AgenticRagState) -> dict:
    """Node: Generate answer using LLM and relevant documents"""
    # Pseudo:
    # - Filter documents by relevance
    # - Call LLM with query + documents
    # - Store generated_answer
    return {
        "generated_answer": answer,
        "timestamps": {"answer_generated": datetime.now()}
    }

def validate_answer_node(state: AgenticRagState) -> dict:
    """Node: Validate answer quality and check for hallucinations"""
    # Pseudo:
    # - Compare answer against source documents
    # - Calculate quality score
    # - Flag potential hallucinations
    return {
        "answer_quality_score": score,
        "validation_result": validation_data
    }

def rerun_search_node(state: AgenticRagState) -> dict:
    """Node: Reformulate query and re-search"""
    # Pseudo:
    # - Analyze why previous search failed
    # - Reformulate query (more specific, different keywords)
    # - Return modified query for new search
    return {
        "query": reformulated_query,
        "original_query": state.query if state.original_query is None else state.original_query
    }
```

### 3.3 Build LangGraph Workflow

**File: `src/agentic_rag/agent.py`**

```python
# Pseudo code:
class AgenticRagAgent:
    def __init__(self, llm: BaseLanguageModel, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Pseudo:
        workflow = StateGraph(AgenticRagState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve_documents_node)
        workflow.add_node("evaluate", evaluate_relevance_node)
        workflow.add_node("generate", generate_answer_node)
        workflow.add_node("validate", validate_answer_node)
        workflow.add_node("rerun_search", rerun_search_node)
        
        # Add edges (conditional logic)
        workflow.set_entry_point("retrieve")
        
        # retrieve -> evaluate
        workflow.add_edge("retrieve", "evaluate")
        
        # evaluate -> conditional
        workflow.add_conditional_edges(
            "evaluate",
            self._should_rerun_condition,
            {
                "rerun": "rerun_search",
                "proceed": "generate"
            }
        )
        
        # rerun_search -> retrieve (if under max searches)
        workflow.add_conditional_edges(
            "rerun_search",
            self._check_max_searches,
            {
                "continue": "retrieve",
                "give_up": "generate"  # Generate with best available
            }
        )
        
        # generate -> validate -> END
        workflow.add_edge("generate", "validate")
        workflow.add_edge("validate", END)
        
        return workflow.compile()
    
    def _should_rerun_condition(self, state: AgenticRagState) -> str:
        """Condition: Should we rerun search?"""
        # Pseudo:
        if state.should_rerun:
            return "rerun"
        return "proceed"
    
    def _check_max_searches(self, state: AgenticRagState) -> str:
        """Condition: Have we exceeded max searches?"""
        # Pseudo:
        if state.search_count < state.max_searches:
            return "continue"
        return "give_up"
    
    def invoke(self, query: str, config: Optional[dict] = None) -> dict:
        """Invoke the agent with a query"""
        # Pseudo:
        initial_state = AgenticRagState(query=query)
        result = self.graph.invoke(initial_state, config)
        return result
```

### 3.4 Implement Tests

**File: `test/agentic_rag/test_state.py`**

```python
# Pseudo code:
class TestAgentState:
    
    def test_state_initialization(self):
        """Test state initializes with correct defaults"""
        state = AgenticRagState(query="test query")
        assert state.query == "test query"
        assert state.search_count == 0
        assert state.max_searches == 3
        assert state.retrieved_documents == []
    
    def test_state_updates(self):
        """Test state can be updated correctly"""
        state = AgenticRagState(query="test")
        # Pseudo: Test state update mechanics
        ...
    
    def test_state_serialization(self):
        """Test state can be serialized/deserialized"""
        state = AgenticRagState(query="test")
        # Pseudo: Test JSON serialization for checkpointing
        ...

class TestAgentNodes:
    
    def test_retrieve_node(self, mock_retriever):
        """Test retrieve node populates documents"""
        # Pseudo: Mock retriever, run node, check state update
        ...
    
    def test_evaluate_node(self, mock_evaluator):
        """Test evaluate node calls evaluator correctly"""
        # Pseudo: Check evaluation_result is populated
        ...
    
    def test_rerun_condition_logic(self):
        """Test conditional edge logic for rerun decision"""
        # Pseudo: Test both branches of condition
        ...
    
    def test_max_searches_limit(self):
        """Test agent respects max_searches limit"""
        # Pseudo: Verify loop terminates after max searches
        ...
```

### 3.5 Add State Persistence Support

Optional: Add checkpointing for long-running workflows:

```python
# Pseudo: In agent.py
from langgraph.checkpoint.memory import MemorySaver

# Add to AgenticRagAgent.__init__:
checkpoint = MemorySaver()
self.graph = workflow.compile(checkpointer=checkpoint)
```

## Test Coverage Requirements

- ✅ All state fields tested for initialization
- ✅ All node functions tested individually
- ✅ Conditional edge logic thoroughly tested
- ✅ State transitions validated
- ✅ Max searches limit enforced

## Success Criteria

- ✅ LangGraph workflow compiles without errors
- ✅ All node functions execute correctly
- ✅ Conditional edges route based on evaluation
- ✅ Agent can be invoked with a query and produces result
- ✅ State is properly maintained across nodes
- ✅ All tests in `test_state.py` and `test_agent.py` pass

## Integration Points

- Evaluator from Phase 2 is called in `evaluate_relevance_node`
- Existing RAG chain used in `retrieve_documents_node`
- CRAG logic (Phase 4) will extend `validate_answer_node`
- Tavily search (Phase 5) will enhance `retrieve_documents_node`

## Performance Considerations

- State should be lightweight to enable fast serialization
- Consider streaming output for long workflows
- Implement timeouts for each node to prevent hangs

## Next Steps

After completing the LangGraph state machine, proceed to **Phase 4: Corrective RAG Logic** where we implement the self-correction mechanisms that make the agent truly autonomous.

## Notes

- LangGraph's checkpointing enables resuming interrupted workflows
- Consider adding human-in-the-loop approval nodes for critical decisions
- Log state transitions for debugging
- The graph structure makes the workflow observable and testable
