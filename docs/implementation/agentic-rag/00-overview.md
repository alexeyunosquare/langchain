# Agentic RAG Implementation - Overview

## Project Context

This implementation plan covers the development of a **Self-Reflective Agentic RAG** system for the `conversational-rag` project. The current project is a Conversational RAG system built with LangChain and ChromaDB.

## Goal

Implement a RAG system using **LangGraph** that:
- Evaluates the relevance of retrieved documents
- Makes autonomous decisions to search again or rely on the LLM
- Incorporates **CRAG (Corrective RAG)** techniques for hallucination reduction
- Builds self-correcting autonomous agents

## Key Components

1. **LangGraph** - For building stateful, multi-agent workflows
2. **Tavily Python** - For enhanced search capabilities
3. **CRAG Techniques** - Corrective RAG for quality improvement
4. **Existing RAG Chain** - Building upon `src/conversational_rag/rag_chain.py`

## TDD Methodology

This implementation follows **Test-Driven Development (TDD)**:
1. Write failing tests first
2. Implement minimum code to pass tests
3. Refactor while keeping tests green
4. Progress incrementally through phases

## Implementation Phases

The implementation is divided into 6 phases:

| Phase | File | Focus |
|-------|------|-------|
| 1 | [01-setup-and-dependencies.md](01-setup-and-dependencies.md) | Project setup, dependencies, initial test scaffolding |
| 2 | [02-retrieval-evaluator.md](02-retrieval-evaluator.md) | Document relevance evaluation component |
| 3 | [03-langgraph-agent-state.md](03-langgraph-agent-state.md) | LangGraph state machine and agent orchestration |
| 4 | [04-corrective-rag-logic.md](04-corrective-rag-logic.md) | CRAG techniques and self-correction mechanisms |
| 5 | [05-tavily-search-integration.md](05-tavily-search-integration.md) | Enhanced search with Tavily integration |
| 6 | [06-integration-and-optimization.md](06-integration-and-optimization.md) | Full integration, testing, and optimization |

## Directory Structure

```
src/conversational_rag/
├── agentic_rag/
│   ├── __init__.py
│   ├── state.py              # LangGraph state definitions
│   ├── evaluator.py          # Document relevance evaluator
│   ├── agent.py              # Main agent orchestration
│   ├── corrective.py         # CRAG logic
│   └── search.py             # Tavily search integration
├── rag_chain.py              # Existing RAG chain
└── __init__.py

test/agentic_rag/
├── __init__.py
├── test_state.py
├── test_evaluator.py
├── test_agent.py
├── test_corrective.py
└── test_search.py
```

## Success Criteria

- ✅ All unit tests pass (TDD requirement)
- ✅ Integration tests validate end-to-end workflow
- ✅ Hallucination reduction compared to baseline
- ✅ Agent can autonomously decide to re-search when needed
- ✅ Performance metrics meet targets (latency, accuracy)

## Next Steps

Begin with **Phase 1: Setup and Dependencies** to establish the foundation for TDD development.
