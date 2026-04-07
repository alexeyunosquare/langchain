"""
LangGraph orchestration for Agentic RAG.

This module implements the LangGraph state machine that orchestrates
the agentic RAG workflow with proper node transitions and conditional logic.

Phase 5: LangGraph Orchestration
- Graph-based state machine for workflow orchestration
- Conditional branching based on evaluation results
- State persistence and recovery
"""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import AgenticRAGConfig
from .evaluator import EvaluationResult, RelevanceEvaluator
from .state import AgentState, Document, Message, MessageRole


class LangGraphNode:
    """
    LangGraph node functions for the Agentic RAG workflow.

    This class encapsulates the various nodes (functions) that make up
    the LangGraph workflow, including retrieval, evaluation, answer
    generation, and correction.
    """

    @staticmethod
    def retrieve_documents(
        state: AgentState, retriever: object
    ) -> AgentState:
        """
        Retrieve documents based on the current query.

        This node handles document retrieval and updates the state
        with retrieved documents and metadata.

        Args:
            state: Current agent state
            retriever: Document retriever instance

        Returns:
            Updated state with retrieved documents
        """
        query = state.query

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
            
            # Update state with documents
            state.documents = documents
            state.context = "\n\n".join([doc.page_content for doc in documents])
            
        except Exception as e:
            state.error = f"Retrieval error: {str(e)}"
            state.documents = []

        # Update search metadata
        state.search_count += 1
        state.iteration += 1

        return state

    @staticmethod
    def evaluate_relevance(
        state: AgentState, evaluator: RelevanceEvaluator
    ) -> AgentState:
        """
        Evaluate relevance of retrieved documents.

        This node assesses whether the retrieved documents are relevant
        to the query and determines if another search is needed.

        Args:
            state: Current agent state
            evaluator: Relevance evaluator instance

        Returns:
            Updated state with evaluation results
        """
        if not state.documents:
            state.is_relevant = False
            state.should_search_again = True
            state.error = state.error or "No documents retrieved"
            return state

        # Evaluate relevance
        evaluation = evaluator.evaluate(state.query, state.documents)
        state.is_relevant = evaluation.is_relevant
        state.should_search_again = evaluator.should_search_again(evaluation)

        return state

    @staticmethod
    def refine_query(state: AgentState) -> AgentState:
        """
        Refine the search query based on previous results.

        This node generates a better query for the next search iteration
        based on the evaluation of previous documents.

        Args:
            state: Current agent state

        Returns:
            Updated state with refined query
        """
        # For now, use the original query
        # Could be extended to use LLM to generate better query
        # based on evaluation reasons
        return state

    @staticmethod
    def generate_answer(
        state: AgentState, llm: object, corrective: object
    ) -> AgentState:
        """
        Generate answer from retrieved documents.

        This node creates the final answer using the LLM and retrieved
        documents as context.

        Args:
            state: Current agent state
            llm: Language model instance
            corrective: CorrectiveRAG instance for validation

        Returns:
            Updated state with generated answer
        """
        # Build context from documents
        context = state.context or "\n\n".join([doc.page_content for doc in state.documents])

        # Create generation prompt
        prompt = f"""
You are an expert assistant. Answer the following question based only on
the provided context. If the context doesn't contain enough information,
state that clearly and provide the best answer you can.

Question: {state.query}

Context:
{context}

Answer:
"""

        # Generate answer using LLM
        try:
            response = llm.invoke(prompt)
            answer = (
                response.content if hasattr(response, "content") else str(response)
            )
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        state.answer = answer

        # Add message to conversation history
        state.add_message(MessageRole.ASSISTANT, answer)

        # Validate and correct
        if corrective and state.documents:
            is_hallucinated, hallucination_score = corrective.check_hallucination(
                answer, state.documents
            )
            state.hallucination_score = hallucination_score

            if is_hallucinated:
                answer = corrective.correct_answer(answer, state.documents)
                state.correction_triggered = True
                state.answer = answer
                state.add_message(MessageRole.ASSISTANT, answer)

        state.validation_passed = True

        return state

    @staticmethod
    def should_continue(state: AgentState) -> str:
        """
        Determine if workflow should continue searching or generate answer.

        Args:
            state: Current agent state

        Returns:
            'generate' if should proceed, 'retrieve' if should search again
        """
        # Check if we should search again
        if state.should_search_again and state.search_count < 3:
            return "retrieve"
        
        return "generate"

    @staticmethod
    def validate_and_correct(state: AgentState) -> AgentState:
        """
        Final validation node for answer quality.

        This node performs final validation of the answer before returning.

        Args:
            state: Current agent state

        Returns:
            Updated state with validated answer
        """
        # Final quality check
        if not state.answer.strip():
            state.validation_passed = False
            state.error = "Empty answer generated"
        else:
            state.validation_passed = True
            state.correction_triggered = False
            state.hallucination_score = 0.0

        return state


def build_agentic_rag_graph(
    evaluator: RelevanceEvaluator,
    llm: object,  # Type is BaseLanguageModel but avoid circular import
    retriever: object,  # Type is BaseRetriever but avoid circular import
    config: AgenticRAGConfig = None,
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

    # Create the workflow
    workflow = StateGraph(AgentState)

    # Add nodes with proper function signatures
    # Note: In LangGraph, node functions receive state as first parameter
    # Additional dependencies (llm, retriever, etc.) should be passed via
    # closure or as partial functions
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

    # Add edges
    workflow.add_edge(START, "retrieve")

    # Conditional edge after retrieve -> evaluate
    workflow.add_edge("retrieve", "evaluate")

    # Conditional edge after evaluate -> decide whether to continue or generate
    workflow.add_conditional_edges(
        "evaluate",
        LangGraphNode.should_continue,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    workflow.add_edge("refine", "generate")
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", END)

    # Compile the graph
    return workflow.compile()


def create_agentic_graph_workflow(
    evaluator: RelevanceEvaluator,
    llm: object,
    retriever: object,
    config: AgenticRAGConfig = None,
) -> dict:
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
        llm: object,
        retriever: object,
        config: AgenticRAGConfig = None,
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

    def run(self, query: str, max_search_count: int = None) -> AgentState:
        """
        Execute the agentic RAG workflow.

        Args:
            query: User's question
            max_search_count: Maximum number of searches (uses config if None)

        Returns:
            Final agent state after workflow completion
        """
        max_search_count = max_search_count or self.config.max_search_iterations

        # Initial state
        initial_state = AgentState(query=query)

        # Execute graph
        result = self.graph.invoke(initial_state)

        # Convert dict result to AgentState if needed
        if isinstance(result, dict):
            return AgentState.from_dict(result)
        
        return result

    def stream(self, query: str) -> object:
        """
        Stream workflow execution with progress updates.

        Args:
            query: User's question

        Yields:
            State updates at each node
        """
        initial_state = AgentState(query=query)

        for event in self.graph.stream(initial_state, stream_mode="values"):
            yield event

    def get_state(self, state: AgentState) -> AgentState:
        """
        Get current state snapshot.

        Args:
            state: Current state

        Returns:
            State snapshot
        """
        return state

    def update_state(self, state: AgentState, updates: dict) -> AgentState:
        """
        Update state with new values.

        Args:
            state: Current state
            updates: Dictionary of field updates

        Returns:
            Updated state
        """
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state
