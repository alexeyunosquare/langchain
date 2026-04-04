# Phase 5: Tavily Search Integration

## Objective

Integrate Tavily's search API to enhance the Agentic RAG system with external knowledge retrieval. This enables the agent to access up-to-date information beyond the local document store.

## Component Overview

Tavily provides AI-optimized search with features ideal for RAG:
- Search results optimized for LLM consumption
- Support for search depth (basic vs. advanced)
- URL filtering and content extraction
- Fresh, web-based information

Integration points:
- Fallback when local retrieval fails
- Supplement local documents with web results
- Re-search mechanism when evaluation fails
- Fresh data for time-sensitive queries

```
[Query] → [Local Retrieval] → [Evaluate]
                        ↓
              Insufficient? → [Tavily Search]
                        ↓
              [Merge Results] → [Generate Answer]
```

## Tasks

### 5.1 Configure Tavily Client

**File: `src/conversational_rag/agentic_rag/search.py`**

```python
# Pseudo code:
from tavily import TavilyClient
from typing import List, Optional
from pydantic import BaseModel

class SearchConfig(BaseModel):
    """Configuration for Tavily search"""
    api_key: str
    search_depth: str = "advanced"  # "basic" or "advanced"
    max_results: int = 5
    include_answer: bool = True
    include_raw_content: bool = False
    days_limit: Optional[int] = None  # Recency filter

class TavilySearchResult(BaseModel):
    """Structured Tavily search result"""
    query: str
    results: List[dict]
    answer: Optional[str] = None
    related_queries: List[str] = []
    search_time: float = 0.0

class TavilySearchIntegration:
    """Tavily search integration for Agentic RAG"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.client = TavilyClient(api_key=config.api_key)
    
    def search(
        self,
        query: str,
        depth: Optional[str] = None,
        max_results: Optional[int] = None,
        days_limit: Optional[int] = None
    ) -> TavilySearchResult:
        """
        Perform Tavily search
        
        Args:
            query: Search query
            depth: Search depth (overrides config)
            max_results: Max results (overrides config)
            days_limit: Recency filter (overrides config)
            
        Returns:
            TavilySearchResult with structured results
        """
        # Pseudo:
        # - Call Tavily API with parameters
        # - Parse and structure results
        # - Convert to Document format for compatibility
        # - Return with timing metadata
        ...
    
    def search_to_documents(
        self, 
        search_result: TavilySearchResult
    ) -> List[Document]:
        """Convert Tavily results to LangChain Document format"""
        # Pseudo:
        # - Extract content from each result
        # - Create Document with proper metadata
        # - Include source URL for attribution
        ...
    
    def refine_search(
        self,
        original_query: str,
        search_history: List[dict],
        evaluation_feedback: dict
    ) -> str:
        """
        Refine search query based on previous failures
        
        Args:
            original_query: Initial query
            search_history: Previous search attempts
            evaluation_feedback: Why previous results were insufficient
            
        Returns:
            Refined query for re-search
        """
        # Pseudo:
        # - Analyze why previous search failed
        # - Generate alternative query (different keywords, more specific)
        # - Return refined query
        ...
```

### 5.2 Integrate with Retrieval Node

**File: `src/conversational_rag/agentic_rag/agent.py`**

Update retrieval to support Tavily:

```python
# Pseudo code update:
from .search import TavilySearchIntegration

def retrieve_documents_node(state: AgenticRagState) -> dict:
    """Node: Retrieve documents (local + Tavily)"""
    # Pseudo:
    # 1. Try local retrieval first
    # 2. Evaluate local results
    # 3. If insufficient, call Tavily
    # 4. Merge results
    # 5. Update state
    
    local_docs = state.local_retriever.get_relevant_documents(state.query)
    
    # Check if we need Tavily
    need_tavily = (
        len(local_docs) == 0 or 
        state.search_count > 1 or
        state.evaluation_result.should_rerun_search if state.evaluation_result else False
    )
    
    if need_tavily and state.tavily_search:
        tavily_result = state.tavily_search.search(state.query)
        tavily_docs = state.tavily_search.search_to_documents(tavily_result)
        # Merge with local docs (priority to local)
        all_docs = local_docs + tavily_docs
    else:
        all_docs = local_docs
    
    return {
        "retrieved_documents": all_docs,
        "search_count": state.search_count + 1,
        "search_history": [...],
        "used_tavily": need_tavily
    }
```

### 5.3 Add Query Refinement Strategy

**File: `src/conversational_rag/agentic_rag/search.py`**

```python
# Pseudo code:
class QueryRefiner:
    """Refines queries for better search results"""
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.refinement_chain = self._build_refinement_chain()
    
    def refine(
        self,
        original_query: str,
        search_history: List[dict],
        failure_reason: str
    ) -> str:
        """
        Refine query based on search failures
        
        Args:
            original_query: Original query
            search_history: Previous search attempts and results
            failure_reason: Why previous search failed (from evaluator)
            
        Returns:
            Refined query
        """
        # Pseudo:
        # - Use LLM to analyze failure
        # - Generate alternative query
        # - Avoid repeating same keywords if they failed
        ...
    
    def _suggest_alternative_keywords(self, query: str) -> List[str]:
        """Suggest alternative keywords for query"""
        # Pseudo: Expand query with synonyms, related terms
        ...
    
    def _make_query_more_specific(self, query: str, context: str) -> str:
        """Add specificity to vague queries"""
        # Pseudo: Narrow down broad queries
        ...
```

### 5.4 Implement Hybrid Retrieval Strategy

**File: `src/conversational_rag/agentic_rag/search.py`**

```python
# Pseudo code:
class HybridRetriever:
    """Combines local vector search with Tavily web search"""
    
    def __init__(
        self,
        local_retriever: BaseRetriever,
        tavily_search: TavilySearchIntegration,
        query_refiner: QueryRefiner,
        use_tavily_fallback: bool = True,
        tavily_priority: float = 0.3  # Weight for Tavily results
    ):
        self.local_retriever = local_retriever
        self.tavily_search = tavily_search
        self.query_refiner = query_refiner
        self.use_tavily_fallback = use_tavily_fallback
        self.tavily_priority = tavily_priority
    
    def retrieve(
        self,
        query: str,
        search_history: List[dict] = None,
        eval_feedback: dict = None
    ) -> List[Document]:
        """
        Perform hybrid retrieval
        
        Args:
            query: Search query
            search_history: Previous searches (for refinement)
            eval_feedback: Feedback from evaluation
            
        Returns:
            Combined and ranked documents
        """
        # Pseudo:
        # 1. Local retrieval
        # 2. Decide if Tavily needed
        # 3. If needed, refine query if re-search
        # 4. Tavily search
        # 5. Merge and rank results
        # 6. Return combined documents
        ...
    
    def _merge_and_rank(
        self,
        local_docs: List[Document],
        tavily_docs: List[Document]
    ) -> List[Document]:
        """Merge results with appropriate ranking"""
        # Pseudo:
        # - Score local docs by relevance
        # - Score Tavily docs by relevance + freshness
        # - Combine with configured priority
        # - Return ranked list
        ...
```

### 5.5 Implement Tests

**File: `test/agentic_rag/test_search.py`**

```python
# Pseudo code:
class TestTavilySearchIntegration:
    
    def test_search_execution(self, mock_tavily_client):
        """Test search returns results in expected format"""
        # Pseudo:
        # - Mock Tavily API response
        # - Call search method
        # - Verify TavilySearchResult structure
        ...
    
    def test_search_to_documents_conversion(self, mock_tavily_client):
        """Test conversion to LangChain Document format"""
        # Pseudo:
        # - Verify Document has content and metadata
        # - Check source URL is preserved
        ...
    
    def test_search_with_filters(self, mock_tavily_client):
        """Test search with various filters"""
        # Pseudo: Test days_limit, max_results, depth parameters
        ...
    
    def test_search_error_handling(self):
        """Test graceful handling of search failures"""
        # Pseudo:
        # - Mock API error
        # - Verify appropriate error handling
        # - Return empty results or fallback
        ...
    
    def test_api_key_configuration(self):
        """Test API key is properly configured"""
        # Pseudo: Test config validation
        ...


class TestQueryRefiner:
    
    def test_refine_after_failure(self, mock_llm):
        """Test query refinement after search failure"""
        # Pseudo:
        # - Provide failed search history
        # - Verify refined query is different
        # - Check it addresses failure reason
        ...
    
    def test_alternative_keyword_generation(self, mock_llm):
        """Test alternative keyword suggestions"""
        # Pseudo: Verify diverse keyword suggestions
        ...
    
    def test_specificity_improvement(self, mock_llm):
        """Test making vague queries more specific"""
        # Pseudo: Verify query becomes more targeted
        ...


class TestHybridRetriever:
    
    def test_hybrid_retrieval(self, mock_local_retriever, mock_tavily):
        """Test combined local + Tavily retrieval"""
        # Pseudo:
        # - Mock both retrievers
        # - Verify results are merged
        # - Check ranking is correct
        ...
    
    def test_local_only_mode(self, mock_local_retriever):
        """Test retrieval without Tavily"""
        # Pseudo: When Tavily is disabled
        ...
    
    def test_tavily_fallback(self, mock_local_retriever, mock_tavily):
        """Test Tavily as fallback when local fails"""
        # Pseudo:
        # - Local returns empty
        # - Tavily should be triggered
        ...
    
    def test_merge_and_ranking(self, mock_local_retriever, mock_tavily):
        """Test document merging and ranking logic"""
        # Pseudo: Verify ranking algorithm
        ...
    
    def test_query_refinement_integration(self, mock_local_retriever, mock_tavily):
        """Test query refinement on re-search"""
        # Pseudo:
        # - Multiple search attempts
        # - Verify query is refined between attempts
        ...
```

### 5.6 Add Environment Configuration

**File: `.env.example`** (add to project root)

```bash
# Tavily API Configuration
TAVILY_API_KEY=your_tavily_api_key_here

# Search Configuration
TAVILY_SEARCH_DEPTH=advanced
TAVILY_MAX_RESULTS=5
TAVILY_DAYS_LIMIT=30
```

**Update `pyproject.toml`:**

```toml
# Add environment variable loading
[project.optional-dependencies]
tavily = [
    "python-dotenv>=1.0.0",
]
```

## Test Coverage Requirements

- ✅ 90%+ code coverage for search module
- ✅ All Tavily API parameters tested
- ✅ Error handling for API failures tested
- ✅ Query refinement tested with various scenarios
- ✅ Hybrid retrieval tested with different configurations

## Success Criteria

- ✅ Tavily search returns properly formatted results
- ✅ Results can be converted to LangChain Documents
- ✅ Query refinement improves search quality
- ✅ Hybrid retrieval combines local + web results
- ✅ All tests in `test_search.py` pass
- ✅ Integration with agent workflow verified

## Performance Considerations

- Tavily search adds latency (~1-3s per search)
- Consider caching search results for repeated queries
- Limit Tavily usage to when necessary (cost optimization)
- Implement timeouts to prevent hangs
- Batch search requests when possible

## Cost Considerations

- Tavily has usage limits based on plan
- Track search counts to monitor costs
- Use "basic" depth when "advanced" is not needed
- Cache results to avoid redundant searches
- Consider rate limiting in high-traffic scenarios

## Integration Points

- Replaces/augments existing retriever in `retrieve_documents_node`
- Provides fallback when local retrieval fails
- Enables re-search in `rerun_search_node`
- Works with evaluator to decide when web search is needed

## Next Steps

After Tavily integration, proceed to **Phase 6: Integration and Optimization** to bring all components together and optimize the complete system.

## Notes

- Tavily is optimized for AI workflows (better than generic search for RAG)
- Consider other search providers as alternatives (Serper, Google Custom Search)
- Search quality directly impacts RAG quality
- Monitor search success rate and adjust strategy accordingly
- The hybrid approach provides best of both worlds (local knowledge + fresh web data)
