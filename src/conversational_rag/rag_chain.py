"""
Conversational RAG Chain Implementation

This module provides a production-ready RAG (Retrieval-Augmented Generation) chain
with conversation memory support, using LM Studio's OpenAI-compatible API and ChromaDB.
Uses modern LCEL (LangChain Expression Language) pattern.
"""

import os
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Manual prompt template definition
QA_SYSTEM_PROMPT = """You are a helpful assistant. Use the following context to answer the question. If you don't know the answer, just say you don't know.

Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}"""


class RAGChain:
    """
    A conversational RAG chain that maintains conversation history.

    Uses modern LCEL (LangChain Expression Language) pattern.

    Attributes:
        llm: The language model instance
        vectorstore: Chroma vector store for document retrieval
        retriever: Document retriever with configurable k
        chain: The LCEL runnable chain
        history: List of message tuples for conversation history
    """

    def __init__(
        self,
        collection_name: str = None,
        host: str = None,
        port: int = None,
        lmstudio_host: str = None,
        lmstudio_port: int = None,
        embedding_function: Optional[Any] = None,
        search_k: int = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        vectorstore: Optional[Chroma] = None,
        persistent: bool = False,
        persist_dir: Optional[str] = None,
    ):
        """
        Initialize the RAG chain.

        All parameters can be overridden via environment variables:
        - CHROMA_COLLECTION_NAME: Collection name (default: "rag_collection")
        - CHROMA_HOST: ChromaDB host address (default: "localhost")
        - CHROMA_PORT: ChromaDB port (default: 8000)
        - CHROMA_PERSIST_DIR: If set, use persistent storage instead of in-memory
        - LM_STUDIO_HOST: LM Studio host address (default: "localhost")
        - LM_STUDIO_PORT: LM Studio port (default: 1234)
        - CHROMA_SEARCH_K: Number of documents to retrieve (default: 3)
        - LM_STUDIO_MODEL: Model name for LM Studio (default: "llama")
        - LM_STUDIO_TEMPERATURE: Temperature for generation (default: 0.7)
        - LM_STUDIO_MAX_TOKENS: Maximum tokens to generate (default: 1024)

        Args:
            collection_name: Name of the ChromaDB collection
            host: ChromaDB host address (for HttpClient)
            port: ChromaDB port (for HttpClient)
            lmstudio_host: LM Studio host address
            lmstudio_port: LM Studio port
            embedding_function: Optional custom embedding function
            search_k: Number of documents to retrieve
            model: Model name for LM Studio
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            vectorstore: Pre-created Chroma vectorstore (for in-memory usage)
            persistent: Whether to use persistent storage (default: False for in-memory)
            persist_dir: Directory for persistent storage (ignored if persistent=False)
        """
        # Use environment variables with fallback to parameters
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_collection"
        )
        self.lmstudio_host = lmstudio_host or os.getenv("LM_STUDIO_HOST", "localhost")
        self.lmstudio_port = lmstudio_port or int(os.getenv("LM_STUDIO_PORT", "1234"))
        self.search_k = search_k or int(os.getenv("CHROMA_SEARCH_K", "3"))
        self.model = model or os.getenv("LM_STUDIO_MODEL", "llama")
        self.temperature = temperature or float(
            os.getenv("LM_STUDIO_TEMPERATURE", "0.7")
        )
        self.max_tokens = max_tokens or int(os.getenv("LM_STUDIO_MAX_TOKENS", "1024"))
        self.persistent = persistent or (
            os.getenv("CHROMA_PERSISTENT", "false").lower() == "true"
        )
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", None)

        # ChromaDB connection parameters (for backward compatibility)
        self.host = host or os.getenv("CHROMA_HOST", "localhost")
        self.port = port or int(os.getenv("CHROMA_PORT", "8000"))

        # Initialize LLM using OpenAI-compatible client for LM Studio
        self.llm = ChatOpenAI(
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=f"http://{self.lmstudio_host}:{self.lmstudio_port}/v1",
            api_key="lm-studio",  # Dummy key for LM Studio
        )

        # Initialize vector store
        if vectorstore:
            # Use pre-created vectorstore (for in-memory testing)
            self.vectorstore = vectorstore
        elif self.persistent or self.persist_dir:
            # Use persistent storage
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_dir)

            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedding_function,
                client=client,
            )
        else:
            # Use in-memory ChromaDB (default)
            import chromadb

            client = chromadb.EphemeralClient()

            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedding_function,
                client=client,
            )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.search_k}
        )

        # Initialize conversation history
        self.history: List[tuple] = []

        # Create the RAG chain using LCEL pattern
        self._build_chain()

    def _build_chain(self) -> None:
        """Build the LCEL retrieval chain with conversation history."""
        # Use manual prompt template instead of hub.pull()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        # Build the chain using LCEL - don't include retriever here
        self.chain = prompt | self.llm | StrOutputParser()

    def __call__(self, question: str) -> Dict[str, Any]:
        """
        Process a question and return the answer with source documents.

        Args:
            question: The question to answer

        Returns:
            Dictionary containing 'answer' and 'source_documents'
        """
        # Get documents first (outside of chain to avoid nesting)
        documents = self.retriever.invoke(question)

        # Create context string from documents
        context = "\n".join([doc.page_content for doc in documents])

        # Prepare chat history for prompt
        chat_history = []
        for i in range(0, len(self.history), 2):
            if i < len(self.history):
                chat_history.append(HumanMessage(content=self.history[i][1]))
            if i + 1 < len(self.history):
                chat_history.append(AIMessage(content=self.history[i + 1][1]))

        # Run the chain
        result = self.chain.invoke(
            {"input": question, "context": context, "chat_history": chat_history}
        )

        # Update history
        self.history.append(("human", question))
        self.history.append(("ai", result))

        return {"answer": result, "source_documents": documents}

    def load_and_store_document(
        self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> int:
        """
        Load a document and store it in the vector store.

        Args:
            file_path: Path to the text file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Returns:
            Number of chunks added to the vector store
        """
        # Load documents
        loader = TextLoader(str(file_path))
        docs = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(docs)

        # Add chunks to vector store
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": str(file_path), "chunk_id": i} for i in range(len(chunks))
        ]

        self.vectorstore.add_documents(
            documents=chunks, ids=chunk_ids, metadatas=metadatas
        )

        return len(chunks)

    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.history.clear()

    @property
    def chat_history(self) -> List[tuple]:
        """Get the current chat history."""
        return self.history.copy()

    def get_conversation_messages(self) -> List[Any]:
        """
        Get conversation history as LangChain Message objects.

        Returns:
            List of HumanMessage and AIMessage objects
        """
        messages = []
        for i in range(0, len(self.history), 2):
            if i < len(self.history):
                messages.append(HumanMessage(content=self.history[i][1]))
            if i + 1 < len(self.history):
                messages.append(AIMessage(content=self.history[i + 1][1]))
        return messages


def create_rag_chain(
    collection_name: str = None,
    host: str = None,
    port: int = None,
    lmstudio_host: str = None,
    lmstudio_port: int = None,
    search_k: int = None,
    model: str = None,
    temperature: float = None,
) -> RAGChain:
    """
    Factory function to create a RAG chain instance.

    All parameters can be overridden via environment variables:
    - CHROMA_COLLECTION_NAME: Collection name (default: "rag_collection")
    - CHROMA_HOST: ChromaDB host address (default: "localhost")
    - CHROMA_PORT: ChromaDB port (default: 8000)
    - LM_STUDIO_HOST: LM Studio host address (default: "localhost")
    - LM_STUDIO_PORT: LM Studio port (default: 1234)
    - CHROMA_SEARCH_K: Number of documents to retrieve (default: 3)
    - LM_STUDIO_MODEL: Model name for LM Studio (default: "llama")
    - LM_STUDIO_TEMPERATURE: Temperature for generation (default: 0.7)

    Args:
        collection_name: Name of the ChromaDB collection
        host: ChromaDB host address
        port: ChromaDB port
        lmstudio_host: LM Studio host address
        lmstudio_port: LM Studio port
        search_k: Number of documents to retrieve
        model: Model name for LM Studio
        temperature: Temperature for generation

    Returns:
        Configured RAGChain instance
    """
    return RAGChain(
        collection_name=collection_name,
        host=host,
        port=port,
        lmstudio_host=lmstudio_host,
        lmstudio_port=lmstudio_port,
        search_k=search_k,
        model=model,
        temperature=temperature,
    )
