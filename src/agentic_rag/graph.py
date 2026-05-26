"""
LangGraph orchestration for Agentic RAG.

This module implements the LangGraph state machine that orchestrates
the agentic RAG workflow with proper node transitions and conditional logic.

Phase 5: LangGraph Orchestration
- Graph-based state machine for workflow orchestration
- Conditional branching based on evaluation results
- State persistence and recovery
"""

from typing import Any, Optional

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from .config import AgenticRAGConfig
from .evaluator import RelevanceEvaluator
from .state import Document, Message, MessageRole


class LangGraphNode:
    """
    LangGraph node functions for the Agentic RAG workflow.

    This class encapsulates the various nodes (functions) that make up
    the LangGraph workflow, including retrieval, evaluation, answer
    generation, and correction.
    """

    @staticmethod
    def retrieve_documents(state: dict, retriever: BaseRetriever) -> dict:
        """
        Retrieve documents based on the current query.

        This node handles document retrieval and updates the state
        with retrieved documents and metadata.

        Args:
            state: Current LangGraph state (dict)
            retriever: Document retriever instance

        Returns:
            Dict of updates to apply to state
        """
        query = state.get("query", "")
        search_count = state.get("search_count", 0)
        iteration = state.get("iteration", 0)

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
    def evaluate_relevance(state: dict, evaluator: RelevanceEvaluator) -> dict:
        """
        Evaluate relevance of retrieved documents.

        This node assesses whether the retrieved documents are relevant
        to the query and determines if another search is needed.

        Args:
            state: Current LangGraph state (dict)
            evaluator: Relevance evaluator instance

        Returns:
            Dict of updates to apply to state
        """
        documents = state.get("documents")
        error = state.get("error")
        query = state.get("query", "")

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
            "evaluation_result": evaluation.model_dump(),
        }

    @staticmethod
    def refine_query(state: dict) -> dict:
        """
        Refine the search query based on previous results.

        This node generates a better query for the next search iteration
        based on the evaluation of previous documents.

        Args:
            state: Current LangGraph state (dict)

        Returns:
            Dict with updated query
        """
        query = state.get("query", "")
        evaluation = state.get("evaluation_result")

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
    def generate_answer(state: dict, llm: BaseLanguageModel, corrective: Any) -> dict:
        """
        Generate answer from retrieved documents.

        This node creates the final answer using the LLM and retrieved
        documents as context.

        Args:
            state: Current LangGraph state (dict)
            llm: Language model instance
            corrective: CorrectiveRAG instance for validation

        Returns:
            Dict with the generated answer and messages
        """
        documents = state.get("documents", [])
        context = state.get("context")
        query = state.get("query", "")
        messages = state.get("messages", [])

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

        # Return dict with answer and messages
        updates: dict = {
            "answer": answer,
            "messages": updated_messages,
        }

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
    def validate_and_correct(state: dict) -> dict:
        """
        Final validation node for answer quality.

        This node performs final validation of the answer before returning.

        Args:
            state: Current LangGraph state (dict)

        Returns:
            Dict with validation status
        """
        answer = state.get("answer", "")

        # Ensure answer is a non-empty string
        answer_str = answer.strip() if isinstance(answer, str) else ""

        result: dict = {
            "validation_passed": True,
            "correction_triggered": False,
            "hallucination_score": 0.5,  # Default score
        }

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
            state: Current LangGraph state (dict)

        Returns:
            'evaluate' if first retrieval, 'refine' otherwise
        """
        search_count = state.get("search_count", 0)

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
) -> Any:  # type: ignore[return-value]
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

    # Create the workflow with dict state (LangGraph native)
    workflow = StateGraph(dict)  # type: ignore[arg-type]

    # Add nodes with proper function signatures
    from functools import partial

    workflow.add_node(
        "retrieve",
        partial(LangGraphNode.retrieve_documents, retriever=retriever),  # type: ignore[arg-type]
    )
    workflow.add_node(
        "evaluate",
        partial(LangGraphNode.evaluate_relevance, evaluator=evaluator),  # type: ignore[arg-type]
    )
    workflow.add_node("refine", LangGraphNode.refine_query)  # type: ignore[arg-type]
    workflow.add_node(
        "generate",
        partial(
            LangGraphNode.generate_answer,
            llm=llm,
            corrective=corrective,
        ),  # type: ignore[arg-type]
    )
    workflow.add_node("validate", LangGraphNode.validate_and_correct)  # type: ignore[arg-type]

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

    def run(self, query: str, max_search_count: Optional[int] = None):
        """
        Execute the agentic RAG workflow.

        Args:
            query: User's question
            max_search_count: Maximum number of searches (uses config if None)

        Returns:
            Final state after workflow completion (as dict with get() and attributes)
        """
        from .state import AgenticRAGState

        max_search_count = max_search_count or self.config.max_search_iterations

        # Initial state using dict
        initial_state = {
            "query": query,
            "messages": [],
            "documents": [],
            "context": "",
            "answer": None,
            "is_relevant": None,
            "should_search_again": None,
            "should_rerun": False,
            "validation_passed": None,
            "correction_triggered": None,
            "hallucination_score": None,
            "search_query": query,
            "search_results": [],
            "search_count": 0,
            "iteration": 0,
            "error": None,
            "original_query": query,
        }

        # Execute graph
        result = self.graph.invoke(initial_state)

        # Convert dict result to AgenticRAGState for consistent access
        if isinstance(result, dict):
            return AgenticRAGState(
                query=result.get("query", ""),
                original_query=result.get("original_query", result.get("search_query", "")),
                documents=result.get("documents", []),
                search_history=[],
                relevance_scores=[],
                answer=result.get("answer"),
                answer_quality_score=result.get("hallucination_score"),
                validation_result=None,
                should_rerun=result.get("should_search_again", result.get("should_rerun", False)),
                rerun_reason=None,
                is_relevant=result.get("is_relevant"),
                validation_passed=result.get("validation_passed"),
                correction_triggered=result.get("correction_triggered"),
                hallucination_score=result.get("hallucination_score"),
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
            State updates at each node (as AgenticRAGState for consistent access)
        """
        from .state import AgenticRAGState

        initial_state = {
            "query": query,
            "messages": [],
            "documents": [],
            "context": "",
            "answer": None,
            "is_relevant": None,
            "should_search_again": None,
            "should_rerun": False,
            "validation_passed": None,
            "correction_triggered": None,
            "hallucination_score": None,
            "search_query": query,
            "search_results": [],
            "search_count": 0,
            "iteration": 0,
            "error": None,
            "original_query": query,
        }

        for event in self.graph.stream(initial_state, stream_mode="values"):
            # Convert dict to AgenticRAGState for consistent access
            if isinstance(event, dict):
                yield AgenticRAGState(
                    query=event.get("query", ""),
                    original_query=event.get("original_query", event.get("search_query", "")),
                    documents=event.get("documents", []),
                    search_history=[],
                    relevance_scores=[],
                    answer=event.get("answer"),
                    answer_quality_score=event.get("hallucination_score"),
                    validation_result=None,
                    should_rerun=event.get("should_search_again", event.get("should_rerun", False)),
                    rerun_reason=None,
                    is_relevant=event.get("is_relevant"),
                    validation_passed=event.get("validation_passed"),
                    correction_triggered=event.get("correction_triggered"),
                    hallucination_score=event.get("hallucination_score"),
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
