# Phase 4: Corrective RAG (CRAG) Logic

## Objective

Implement Corrective RAG techniques that enable the agent to identify and correct its own mistakes, particularly hallucinations. This is the core "self-correcting" capability of Agentic RAG.

## Component Overview

CRAG adds a validation and correction loop to the RAG pipeline:

```
[Generate Answer] → [Validate Against Sources] → [Quality Check]
                                              ↓
                                      ┌───────┴────────┐
                                      ↓               ↓
                              High Quality      Low Quality
                                      ↓               ↓
                              [Return Answer]  [Correct/Regenerate]
                                                  ↓
                                          [Reformulate & Retry]
```

## CRAG Techniques to Implement

1. **Relevance Assessment** - Already covered in Phase 2
2. **Answer Validation** - Verify answer is grounded in retrieved documents
3. **Hallucination Detection** - Identify claims not supported by sources
4. **Corrective Actions** - Regenerate, re-search, or fallback strategies
5. **Quality Scoring** - Quantitative measure of answer quality

## Tasks

### 4.1 Define Validation Schema

**File: `src/conversational_rag/agentic_rag/corrective.py`**

```python
# Pseudo code:
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional

class ValidationStatus(Enum):
    VALID = "valid"
    PARTIALLY_VALID = "partially_valid"
    INVALID = "invalid"
    HALLUCINATED = "hallucinated"

class ValidationDetail(BaseModel):
    claim: str
    is_supported: bool
    supporting_document_id: Optional[str]
    confidence: float
    issue_type: Optional[str]  # "hallucination", "misinterpretation", etc.

class ValidationResult(BaseModel):
    answer: str
    status: ValidationStatus
    quality_score: float  # 0.0 to 1.0
    validation_details: List[ValidationDetail]
    issues: List[str]
    corrective_action: Optional[str]  # "regenerate", "re-search", "accept"

class CorrectionResult(BaseModel):
    original_answer: str
    corrected_answer: str
    correction_type: str  # "regeneration", "re-search", "hybrid"
    quality_improvement: float
    iterations: int
```

### 4.2 Implement Answer Validator

**File: `src/conversational_rag/agentic_rag/corrective.py`**

```python
# Pseudo code:
class AnswerValidator:
    """Validates generated answers against source documents"""
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.validator_chain = self._build_validator_chain()
    
    def _build_validator_chain(self) -> Runnable:
        """Build LLM chain for answer validation"""
        # Pseudo: Create prompt that compares answer to documents
        # Extract claims from answer
        # Verify each claim against documents
        # Output structured validation result
        ...
    
    def validate(
        self, 
        answer: str, 
        documents: List[Document],
        query: str
    ) -> ValidationResult:
        """
        Validate answer against source documents
        
        Args:
            answer: Generated answer to validate
            documents: Source documents used for generation
            query: Original query for context
            
        Returns:
            ValidationResult with quality assessment
        """
        # Pseudo:
        # 1. Extract key claims from answer
        # 2. For each claim, check if supported by documents
        # 3. Calculate overall quality score
        # 4. Determine validation status
        # 5. Suggest corrective action if needed
        ...
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from answer"""
        # Pseudo: Use LLM to identify declarative statements
        ...
    
    def _verify_claim(
        self, 
        claim: str, 
        documents: List[Document]
    ) -> ValidationDetail:
        """Verify single claim against documents"""
        # Pseudo: Check if claim is supported by any document
        ...
    
    def _calculate_quality_score(
        self, 
        validation_details: List[ValidationDetail]
    ) -> float:
        """Calculate overall quality score"""
        # Pseudo:
        # - Weight supported claims higher
        # - Penalize hallucinations heavily
        # - Consider answer completeness
        ...
```

### 4.3 Implement Correction Engine

**File: `src/conversational_rag/agentic_rag/corrective.py`**

```python
# Pseudo code:
class CorrectionEngine:
    """Handles corrective actions when validation fails"""
    
    def __init__(
        self, 
        llm: BaseLanguageModel,
        quality_threshold: float = 0.8,
        max_correction_attempts: int = 2
    ):
        self.llm = llm
        self.quality_threshold = quality_threshold
        self.max_correction_attempts = max_correction_attempts
    
    def should_correct(self, validation_result: ValidationResult) -> bool:
        """Determine if correction is needed"""
        # Pseudo:
        # - Check if quality_score < threshold
        # - Check if status is INVALID or HALLUCINATED
        # - Return decision
        ...
    
    def correct(
        self,
        validation_result: ValidationResult,
        query: str,
        documents: List[Document],
        search_available: bool = True
    ) -> CorrectionResult:
        """
        Apply corrective actions
        
        Args:
            validation_result: Result from validator
            query: Original query
            documents: Current documents
            search_available: Can we re-search?
            
        Returns:
            CorrectionResult with improved answer
        """
        # Pseudo:
        # 1. Analyze validation issues
        # 2. Choose correction strategy
        # 3. Apply correction (regenerate or re-search)
        # 4. Validate corrected answer
        # 5. Return result with quality improvement metric
        ...
    
    def _choose_correction_strategy(
        self, 
        validation_result: ValidationResult,
        search_available: bool
    ) -> str:
        """Choose appropriate correction strategy"""
        # Pseudo:
        # - If hallucination detected and search available → re-search
        # - If misinterpretation → regenerate with better prompt
        # - If partial validity → hybrid approach
        # - If no search available → regenerate with constraints
        ...
    
    def _regenerate_with_constraints(
        self,
        query: str,
        documents: List[Document],
        validation_issues: List[str]
    ) -> str:
        """Regenerate answer with constraints from validation"""
        # Pseudo:
        # - Use validation issues to create constraints
        # - Prompt LLM to address specific issues
        # - Ensure answer stays grounded in documents
        ...
    
    def _iterate_corrections(
        self,
        query: str,
        documents: List[Document],
        initial_validation: ValidationResult
    ) -> CorrectionResult:
        """Iteratively correct until quality threshold met or max attempts reached"""
        # Pseudo:
        # Loop:
        #   - Apply correction
        #   - Validate new answer
        #   - Check if quality improved
        #   - Stop if threshold met or max attempts reached
        ...
```

### 4.4 Integrate with Agent Workflow

**File: `src/conversational_rag/agentic_rag/agent.py`**

Update the validate_answer_node to use CRAG:

```python
# Pseudo code update:
def validate_answer_node(state: AgenticRagState) -> dict:
    """Node: Validate answer with CRAG techniques"""
    # Pseudo:
    validator = AnswerValidator(llm=state.llm)
    validator_result = validator.validate(
        answer=state.generated_answer,
        documents=state.retrieved_documents,
        query=state.query
    )
    
    # Check if correction needed
    correction_engine = CorrectionEngine(llm=state.llm)
    if correction_engine.should_correct(validator_result):
        # Apply correction
        correction = correction_engine.correct(
            validation_result=validator_result,
            query=state.query,
            documents=state.retrieved_documents,
            search_available=state.search_count < state.max_searches
        )
        return {
            "generated_answer": correction.corrected_answer,
            "answer_quality_score": correction.quality_improvement,
            "validation_result": {...}
        }
    else:
        return {
            "answer_quality_score": validator_result.quality_score,
            "validation_result": validator_result.model_dump()
        }
```

### 4.5 Implement Tests

**File: `test/agentic_rag/test_corrective.py`**

```python
# Pseudo code:
class TestAnswerValidator:
    
    def test_validate_perfect_answer(self, mock_llm, sample_documents):
        """Test validator approves answer fully supported by documents"""
        # Pseudo:
        # - Create answer that is 100% grounded in documents
        # - Validate should return VALID status
        # - Quality score should be high (> 0.9)
        ...
    
    def test_validate_hallucinated_answer(self, mock_llm, sample_documents):
        """Test validator detects hallucinated claims"""
        # Pseudo:
        # - Create answer with claims not in documents
        # - Validator should mark as HALLUCINATED
        # - Validation details should identify unsupported claims
        ...
    
    def test_validate_partial_answer(self, mock_llm, sample_documents):
        """Test validator handles partially supported answers"""
        # Pseudo: Mix of supported and unsupported claims
        # Status should be PARTIALLY_VALID
        ...
    
    def test_quality_score_calculation(self, mock_llm):
        """Test quality score reflects validation results"""
        # Pseudo: Verify scoring formula
        ...


class TestCorrectionEngine:
    
    def test_should_correct_decision(self, mock_llm):
        """Test correction decision logic"""
        # Pseudo: Test threshold-based decision
        ...
    
    def test_correction_regeneration(self, mock_llm, sample_documents):
        """Test regeneration correction strategy"""
        # Pseudo:
        # - Provide invalid answer
        # - Apply correction with regeneration
        # - Verify corrected answer is improved
        ...
    
    def test_correction_with_research(self, mock_llm, sample_documents):
        """Test correction that triggers new search"""
        # Pseudo:
        # - When hallucination detected and search available
        # - Should choose re-search strategy
        ...
    
    def test_max_correction_attempts(self, mock_llm):
        """Test engine respects max correction attempts"""
        # Pseudo: Verify loop termination
        ...
    
    def test_quality_improvement_tracking(self, mock_llm):
        """Test correction tracks quality improvement"""
        # Pseudo: Verify quality_improvement metric
        ...


class TestCragIntegration:
    
    def test_full_validation_correction_loop(self, mock_llm, sample_documents):
        """Test complete validation → correction → re-validation flow"""
        # Pseudo: End-to-end CRAG workflow
        ...
    
    def test_crag_reduces_hallucinations(self, mock_llm):
        """Test that CRAG actually reduces hallucination rate"""
        # Pseudo: Comparative test with/without CRAG
        ...
```

### 4.6 Add Metrics and Logging

**File: `src/conversational_rag/agentic_rag/corrective.py`**

```python
# Pseudo: Add metrics collection
import logging
from typing import Optional

class MetricsCollector:
    def __init__(self):
        self.hallucination_count = 0
        self.correction_count = 0
        self.avg_quality_score: Optional[float] = None
    
    def record_validation(self, result: ValidationResult):
        """Record validation metrics"""
        # Pseudo: Update metrics
        if result.status == ValidationStatus.HALLUCINATED:
            self.hallucination_count += 1
        ...
    
    def get_statistics(self) -> dict:
        """Return aggregated statistics"""
        ...
```

## Test Coverage Requirements

- ✅ 90%+ code coverage for corrective module
- ✅ All validation statuses tested
- ✅ All correction strategies tested
- ✅ Edge cases (no documents, empty answer) handled
- ✅ Quality scoring formula validated

## Success Criteria

- ✅ Validator correctly identifies hallucinations
- ✅ Correction engine improves answer quality
- ✅ Quality scores are meaningful and consistent
- ✅ Correction loop terminates correctly
- ✅ All tests in `test_corrective.py` pass
- ✅ Hallucination rate reduced compared to baseline

## Performance Considerations

- Validation should be fast (< 1s typical)
- Limit correction iterations to prevent long loops
- Consider caching validation results for similar queries
- Track correction success rate for optimization

## Metrics to Track

- Hallucination detection rate
- Correction success rate
- Average quality score improvement
- Time spent in correction loop
- Most common validation issues

## Next Steps

After implementing CRAG logic, proceed to **Phase 5: Tavily Search Integration** to enhance the search capabilities with external knowledge sources.

## Notes

- CRAG is the key differentiator that makes Agentic RAG "self-correcting"
- Consider human feedback loop for continuous improvement
- Log all corrections for analysis and model fine-tuning
- The quality threshold should be configurable per use case
