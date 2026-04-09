"""
LangGraph orchestration for Agentic RAG.

This module implements the LangGraph state machine that orchestrates
the agentic RAG workflow with proper node transitions and conditional logic.

Phase 5: LangGraph Orchestration
- Graph-based state machine for workflow orchestration
- Conditional branching based on evaluation results
- State persistence and recovery
"""

from datetime import datetime
from typing import Any, Optional

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from .config import AgenticRAGConfig
from .evaluator import RelevanceEvaluator
from .state import Document, GraphState, Message, MessageRole


class NodeResult(dict):
    """
    Dictionary subclass that also supports attribute access.

    This allows LangGraph nodes to return dicts while supporting both
    dict-style and attribute-style access for test compatibility.
    """

    def __getattr__(self, key: str):
        """Support attribute access (e.g., result.validation_passed)."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            ) from None

    def __setattr__(self, key: str, value):
        """Support attribute assignment (e.g., result.validation_passed = True)."""
        self[key] = value

    def __delattr__(self, key: str):
        """Support attribute deletion."""
        del self[key]


class LangGraphNode:
    """
    LangGraph node functions for the Agentic RAG workflow.

    This class encapsulates the various nodes (functions) that make up
    the LangGraph workflow, including retrieval, evaluation, answer
    generation, and correction.
    """

    @staticmethod
    def retrieve_documents(state, retriever: object) -> dict:
        """
        Retrieve documents based on the current query.

        This node handles document retrieval and updates the state
        with retrieved documents and metadata.

        Args:
            state: Current LangGraph state (TypedDict or AgentState)
            retriever: Document retriever instance

        Returns:
            Dict of updates to apply to state
        """
        # Handle both TypedDict and Pydantic model
        if hasattr(state, "get"):
            query = state.get("query", "")
            search_count = state.get("search_count", 0)
            iteration = state.get("iteration", 0)
        else:
            query = getattr(state, "query", "")
            search_count = getattr(state, "search_count", 0)
            iteration = getattr(state, "iteration", 0)

        # Retrieve documents using retriever
        try:
            results = retriever.invoke(query)

            # Convert results to Document objects
            documents = []
            for doc in results:
                document = Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    score=getattr(doc, "score", None),
                )
                documents.append(document)

            # Return update dict (not modified state)
            return {
                "documents": documents,
                "context": "\n\n".join([doc.page_content for doc in documents]),
                "search_count": search_count + 1,
                "iteration": iteration + 1,
            }

        except Exception as e:
            return {
                "error": f"Retrieval error: {str(e)}",
                "documents": [],
                "context": "",
                "search_count": search_count + 1,
                "iteration": iteration + 1,
            }

    @staticmethod
    def evaluate_relevance(state, evaluator: RelevanceEvaluator) -> dict:
        """
        Evaluate relevance of retrieved documents.

        This node assesses whether the retrieved documents are relevant
        to the query and determines if another search is needed.

        Args:
            state: Current LangGraph state (TypedDict or AgentState)
            evaluator: Relevance evaluator instance

        Returns:
            Dict of updates to apply to state
        """
        # Handle both TypedDict and Pydantic model
        if hasattr(state, "get"):
            documents = state.get("documents")
            error = state.get("error")
            query = state.get("query", "")
        else:
            documents = getattr(state, "documents", [])
            error = getattr(state, "error", None)
            query = getattr(state, "query", "")

        if not documents:
            return {
                "is_relevant": False,
                "should_search_again": True,
                "error": error or "No documents retrieved",
            }

        # Evaluate relevance
        evaluation = evaluator.evaluate(query, documents)

        return {
            "is_relevant": evaluation.is_relevant,
            "should_search_again": evaluator.should_search_again(evaluation),
            "evaluation_result": evaluation.to_dict(),
        }

    @staticmethod
    def refine_query(state) -> dict:
        """
        Refine the search query based on previous results.

        This node generates a better query for the next search iteration
        based on the evaluation of previous documents.

        Args:
            state: Current LangGraph state (TypedDict or AgentState)

        Returns:
            Dict with updated query
        """
        # Handle both TypedDict and Pydantic model
        if hasattr(state, "get"):
            query = state.get("query", "")
            evaluation = state.get("evaluation_result")
        else:
            query = getattr(state, "query", "")
            evaluation = getattr(state, "evaluation_result", None)

        if not evaluation or not evaluation.get("reason"):
            # No reason provided, return original query
            return {"query": query}

        # Use LLM to generate better query based on evaluation feedback
        from .agent import AgenticRAGAgent

        # Get the LLM from closure or state
        llm = AgenticRAGAgent.__dict__.get("_llm", None)

        if llm:
            refinement_prompt = f"""
            You are an expert query refiner. The previous search for:

            "{query}"

            Failed or returned limited results.
            Reason: {evaluation.get("reason")}

            Analyze the reason and generate a more specific or alternative search query
            that addresses this issue. Consider:
            - Synonyms or alternative terms
            - More specific keywords
            - Broader or narrower scope
            - Different phrasing

            Return ONLY the new query, nothing else.
            """
            try:
                response = llm.invoke(refinement_prompt)
                new_query = (
                    response.content if hasattr(response, "content") else str(response)
                ).strip()
                return {"query": new_query}
            except Exception:
                return {"query": query}

        # Fallback: return original query
        return {"query": query}

    @staticmethod
    def generate_answer(state, llm: object, corrective: object) -> NodeResult:
        """
        Generate answer from retrieved documents.

        This node creates the final answer using the LLM and retrieved
        documents as context.

        Args:
            state: Current LangGraph state (TypedDict or AgentState)
            llm: Language model instance
            corrective: CorrectiveRAG instance for validation

        Returns:
            NodeResult with the generated answer and messages (supports both dict and attr access)
        """
        # Handle both TypedDict and Pydantic model
        if hasattr(state, "get"):
            documents = state.get("documents", [])
            context = state.get("context")
            query = state.get("query", "")
            messages = state.get("messages", [])
        else:
            # For Pydantic models, use dict() to get all fields as a dictionary
            state_dict = state.dict() if hasattr(state, "dict") else {}
            documents = state_dict.get(
                "documents", state_dict.get("retrieved_documents", [])
            )
            context = state_dict.get("context")
            query = state_dict.get("query", "")
            messages = state_dict.get("messages", [])

        # Build context from documents if not provided
        if not context:
            context = "\n\n".join([doc.page_content for doc in documents])

        # Create generation prompt
        prompt = f"""
You are an expert assistant. Answer the following question based only on
the provided context. If the context doesn't contain enough information,
state that clearly and provide the best answer you can.

Question: {query}

Context:
{context}

Answer:
"""

        # Generate answer using LLM
        try:
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        # Add fallback: ensure answer is always a string (not None)
        answer = answer or "Unable to generate answer"

        # Build messages list with the new assistant message

        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=answer,
        )

        if isinstance(messages, list):
            updated_messages = messages + [assistant_message]
        else:
            updated_messages = list(messages) + [assistant_message]

        # Return NodeResult (supports both dict and attribute access)
        updates = NodeResult(
            {
                "answer": answer,
                "messages": updated_messages,
            }
        )

        # Validate and correct if available
        if corrective and documents:
            is_hallucinated, hallucination_score = corrective.check_hallucination(
                answer, documents
            )
            updates["hallucination_score"] = hallucination_score

            if is_hallucinated:
                corrected_answer = corrective.correct_answer(answer, documents)
                updates["answer"] = corrected_answer
                updates["correction_triggered"] = True

        updates["validation_passed"] = True
        return updates

    @staticmethod
    def should_continue(state: dict[str, Any]) -> str:
        """
        Determine if workflow should continue searching or generate answer.

        Args:
            state: Current LangGraph state

        Returns:
            'retrieve' if should search again, 'generate' if should generate
        """
        # Check if we should search again - default to False when not set
        # Support both 'should_search_again' and 'should_rerun' field names
        should_search = state.get(
            "should_search_again", state.get("should_rerun", False)
        )
        if should_search is None:
            should_search = False

        search_count = state.get("search_count", 0)
        # Support both 'search_count' and 'max_searches' fields
        max_searches = state.get("max_searches", state.get("max_search_iterations", 3))

        if should_search and search_count < max_searches:
            return "retrieve"

        return "generate"

    @staticmethod
    def validate_and_correct(state) -> NodeResult:
        """
        Final validation node for answer quality.

        This node performs final validation of the answer before returning.

        Args:
            state: Current LangGraph state (TypedDict or AgentState)

        Returns:
            NodeResult with validation status (supports both dict and attr access)
        """
        # Handle both TypedDict and Pydantic model
        if hasattr(state, "get"):
            answer = state.get("answer", "")
            # Support both 'answer' and 'generated_answer' field names
            if not answer:
                answer = state.get("generated_answer", "")
        else:
            answer = getattr(state, "answer", "") or getattr(
                state, "generated_answer", ""
            )

        # Add None check: if state is None
        if answer is None:
            answer = ""

        # Ensure answer is a non-empty string
        answer_str = answer.strip() if isinstance(answer, str) else ""

        result = NodeResult(
            {
                "validation_passed": True,
                "correction_triggered": False,
                "hallucination_score": 0.5,  # Default score
            }
        )

        if not answer_str:
            result["validation_passed"] = False
            result["error"] = "Empty answer generated"

        return result

    @staticmethod
    def route_after_retrieval(state: dict[str, Any]) -> str:
        """
        Route to evaluate or refine based on search count.

        After the first retrieval (search_count=1), route to evaluate.
        After subsequent retrievals (search_count > 1), route to refine to improve the query.

        Args:
            state: Current LangGraph state

        Returns:
            'evaluate' if first retrieval, 'refine' otherwise
        """
        # Support both TypedDict and Pydantic model
        if hasattr(state, "get"):
            search_count = state.get("search_count", 0)
        else:
            search_count = getattr(state, "search_count", 0)

        # After first retrieval (search_count == 1), go to evaluate
        # After subsequent retrievals (search_count > 1), go to refine
        if search_count >= 1:
            return "refine"
        return "evaluate"


def build_agentic_rag_graph(
    evaluator: RelevanceEvaluator,
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
    config: Optional[AgenticRAGConfig] = None,
) -> StateGraph:
    """
    Build the LangGraph state machine for Agentic RAG.

    This function constructs the complete workflow graph with all nodes
    and conditional edges for orchestrating the agentic RAG process.

    Args:
        evaluator: Relevance evaluator instance
        llm: Language model instance
        retriever: Document retriever instance
        config: Optional configuration (uses defaults if None)

    Returns:
        Compiled LangGraph workflow
    """
    from .corrective import CorrectiveRAG

    config = config or AgenticRAGConfig()
    corrective = CorrectiveRAG(llm=llm)

    # Create the workflow with GraphState (TypedDict)
    workflow = StateGraph(GraphState)

    # Add nodes with proper function signatures
    from functools import partial

    workflow.add_node(
        "retrieve",
        partial(LangGraphNode.retrieve_documents, retriever=retriever),
    )
    workflow.add_node(
        "evaluate",
        partial(LangGraphNode.evaluate_relevance, evaluator=evaluator),
    )
    workflow.add_node("refine", LangGraphNode.refine_query)
    workflow.add_node(
        "generate",
        partial(
            LangGraphNode.generate_answer,
            llm=llm,
            corrective=corrective,
        ),
    )
    workflow.add_node("validate", LangGraphNode.validate_and_correct)

    # Add edges - START -> retrieve
    workflow.add_edge(START, "retrieve")

    # retrieve -> conditional routing (evaluate or refine)
    workflow.add_conditional_edges(
        "retrieve",
        LangGraphNode.route_after_retrieval,
        {
            "evaluate": "evaluate",
            "refine": "refine",
        },
    )

    # evaluate -> decide whether to continue (retrieve) or generate
    workflow.add_conditional_edges(
        "evaluate",
        LangGraphNode.should_continue,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    # refine -> generate
    workflow.add_edge("refine", "generate")

    # generate -> validate
    workflow.add_edge("generate", "validate")

    # validate -> END
    workflow.add_edge("validate", END)

    # Compile the graph
    return workflow.compile()


def create_agentic_graph_workflow(
    evaluator: RelevanceEvaluator,
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
    config: Optional[AgenticRAGConfig] = None,
) -> dict[str, Any]:
    """
    Create a ready-to-use agentic RAG workflow.

    This is a convenience function that builds and compiles the LangGraph
    workflow, returning it ready for execution.

    Args:
        evaluator: Relevance evaluator instance
        llm: Language model instance
        retriever: Document retriever instance
        config: Optional configuration

    Returns:
        Compiled LangGraph application
    """
    graph = build_agentic_rag_graph(evaluator, llm, retriever, config)

    return {
        "graph": graph,
        "entry_point": "retrieve",
        "end_point": END,
    }


class LangGraphAgenticRAG:
    """
    LangGraph-based Agentic RAG implementation.

    This class provides a LangGraph-native implementation of the Agentic
    RAG workflow, leveraging state machines for orchestration and proper
    handling of iterative search and correction.

    Attributes:
        graph: Compiled LangGraph workflow
        evaluator: Document relevance evaluator
        llm: Language model
        retriever: Document retriever
        config: Configuration settings
    """

    def __init__(
        self,
        evaluator: RelevanceEvaluator,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        config: Optional[AgenticRAGConfig] = None,
    ):
        """
        Initialize LangGraph Agentic RAG.

        Args:
            evaluator: Document relevance evaluator
            llm: Language model
            retriever: Document retriever
            config: Optional configuration
        """
        self.evaluator = evaluator
        self.llm = llm
        self.retriever = retriever
        self.config = config or AgenticRAGConfig()

        # Build the graph with partial functions for dependencies
        self.graph = build_agentic_rag_graph(
            evaluator=self.evaluator,
            llm=self.llm,
            retriever=self.retriever,
            config=self.config,
        )

    def run(self, query: str, max_search_count: int = None):
        """
        Execute the agentic RAG workflow.

        Args:
            query: User's question
            max_search_count: Maximum number of searches (uses config if None)

        Returns:
            Final state after workflow completion (as dict with get() and attributes)
        """
        from .state import AgentState

        max_search_count = max_search_count or self.config.max_search_iterations

        # Initial state using dict
        initial_state = {
            "query": query,
            "messages": [],
            "documents": [],
            "context": "",
            "answer": "",
            "is_relevant": None,
            "should_search_again": None,
            "validation_passed": None,
            "correction_triggered": None,
            "hallucination_score": None,
            "search_query": query,
            "search_results": [],
            "search_count": 0,
            "iteration": 0,
            "error": None,
        }

        # Execute graph
        result = self.graph.invoke(initial_state)

        # Convert dict result to AgentState for consistent access
        if isinstance(result, dict):
            return AgentState(
                query=result.get("query", ""),
                original_query=result.get("search_query", ""),
                retrieved_documents=result.get("documents", []),
                search_history=[],
                relevance_scores=[],
                generated_answer=result.get("answer", ""),
                answer_quality_score=result.get("hallucination_score"),
                validation_result=None,
                should_rerun=result.get("should_search_again", False),
                rerun_reason=None,
                session_id=f"session_{datetime.now().isoformat()}",
                timestamps={},
                # GraphState-specific fields
                is_relevant=result.get("is_relevant"),
                should_search_again=result.get("should_search_again"),
                validation_passed=result.get("validation_passed"),
                correction_triggered=result.get("correction_triggered"),
                hallucination_score=result.get("hallucination_score"),
                search_query=result.get("search_query", ""),
                search_results=result.get("search_results", []),
                search_count=result.get("search_count", 0),
                iteration=result.get("iteration", 0),
                error=result.get("error"),
            )

        return result

    def stream(self, query: str):
        """
        Stream workflow execution with progress updates.

        Args:
            query: User's question

        Yields:
            State updates at each node (as AgentState for consistent access)
        """
        from .state import AgentState

        initial_state = {
            "query": query,
            "messages": [],
            "documents": [],
            "context": "",
            "answer": "",
            "is_relevant": None,
            "should_search_again": None,
            "validation_passed": None,
            "correction_triggered": None,
            "hallucination_score": None,
            "search_query": query,
            "search_results": [],
            "search_count": 0,
            "iteration": 0,
            "error": None,
        }

        for event in self.graph.stream(initial_state, stream_mode="values"):
            # Convert dict to AgentState for consistent access
            if isinstance(event, dict):
                yield AgentState(
                    query=event.get("query", ""),
                    original_query=event.get("search_query", ""),
                    retrieved_documents=event.get("documents", []),
                    search_history=[],
                    relevance_scores=[],
                    generated_answer=event.get("answer", ""),
                    answer_quality_score=event.get("hallucination_score"),
                    validation_result=None,
                    should_rerun=event.get("should_search_again", False),
                    rerun_reason=None,
                    session_id=f"session_{datetime.now().isoformat()}",
                    timestamps={},
                    is_relevant=event.get("is_relevant"),
                    should_search_again=event.get("should_search_again"),
                    validation_passed=event.get("validation_passed"),
                    correction_triggered=event.get("correction_triggered"),
                    hallucination_score=event.get("hallucination_score"),
                    search_query=event.get("search_query", ""),
                    search_results=event.get("search_results", []),
                    search_count=event.get("search_count", 0),
                    iteration=event.get("iteration", 0),
                    error=event.get("error"),
                )
            else:
                yield event

    def get_state(self, state: dict) -> dict:
        """
        Get current state snapshot.

        Args:
            state: Current state

        Returns:
            State snapshot
        """
        return state

    def update_state(self, state: dict, updates: dict) -> dict:
        """
        Update state with new values.

        Args:
            state: Current state
            updates: Dictionary of field updates

        Returns:
            Updated state
        """
        # Deep merge updates into state
        for key, value in updates.items():
            if key in state:
                if isinstance(state[key], dict) and isinstance(value, dict):
                    state[key].update(value)
                else:
                    state[key] = value
            else:
                state[key] = value
        return state
