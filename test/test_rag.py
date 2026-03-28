"""
Comprehensive test suite for Conversational RAG with Memory.

Tests cover:
- Document ingestion and retrieval
- Basic Q&A functionality
- Conversation memory persistence
- Follow-up question handling
- Edge cases and error handling
"""
import pytest
import os
from pathlib import Path


class TestDocumentIngestion:
    """Tests for document loading and vector storage."""
    
    @pytest.mark.integration
    def test_load_sample_documents(self, sample_docs_path):
        """Test that sample documents can be loaded."""
        assert sample_docs_path.exists(), "Sample documents file should exist"
        content = sample_docs_path.read_text()
        assert len(content) > 0, "Sample documents should not be empty"
        assert "Python" in content, "Should contain Python documentation"
        assert "Machine Learning" in content, "Should contain ML documentation"
    
    @pytest.mark.integration
    def test_ingest_documents_to_chroma(self, clean_chroma_collection, sample_docs_path):
        """Test document ingestion into ChromaDB."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader
        
        # Load and split documents
        loader = TextLoader(str(sample_docs_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        chunks = text_splitter.split_documents(docs)
        
        # Add to collection
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(sample_docs_path), "chunk_id": i} for i in range(len(chunks))]
        
        clean_chroma_collection.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        # Verify ingestion
        result = clean_chroma_collection._collection.count()
        assert result > 0, "Should have ingested at least one document chunk"


class TestBasicRAG:
    """Tests for basic RAG functionality."""
    
    @pytest.mark.integration
    def test_answer_question_about_python(self, rag_chain, sample_docs_path):
        """Test RAG can answer questions about Python."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader
        
        # Load and ingest documents
        loader = TextLoader(str(sample_docs_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(docs)
        
        # Add all chunks at once
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(sample_docs_path), "chunk_id": i} for i in range(len(chunks))]
        rag_chain.vectorstore.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        # Ask question
        result = rag_chain("What is Python?")
        
        # Verify response
        assert "answer" in result, "Response should contain an answer"
        assert len(result["answer"]) > 0, "Answer should not be empty"
        
        # Semantic assertion - answer should mention key concepts
        answer_lower = result["answer"].lower()
        assert any(word in answer_lower for word in ["programming", "language", "python"]), \
            f"Answer should mention Python or programming. Got: {result['answer']}"
    
    @pytest.mark.integration
    def test_answer_question_about_ml(self, rag_chain, sample_docs_path):
        """Test RAG can answer questions about Machine Learning."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader
        
        # Load and ingest documents
        loader = TextLoader(str(sample_docs_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(docs)
        
        # Add all chunks at once
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(sample_docs_path), "chunk_id": i} for i in range(len(chunks))]
        rag_chain.vectorstore.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        # Ask question
        result = rag_chain("What are the types of Machine Learning?")
        
        # Verify response
        assert "answer" in result
        assert len(result["answer"]) > 0
        
        # Semantic assertion
        answer_lower = result["answer"].lower()
        assert any(word in answer_lower for word in ["supervised", "unsupervised", "learning", "types"]), \
            f"Answer should mention ML types. Got: {result['answer']}"


class TestConversationMemory:
    """Tests for conversation memory and context retention."""
    
    @pytest.mark.integration
    def test_memory_persists_across_turns(self, rag_chain, sample_docs_path):
        """Test that conversation history is maintained across multiple turns."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader
        
        # Load and ingest documents
        loader = TextLoader(str(sample_docs_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(docs)
        
        # Add all chunks at once
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(sample_docs_path), "chunk_id": i} for i in range(len(chunks))]
        rag_chain.vectorstore.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        # First question
        result1 = rag_chain("What is Docker?")
        assert "answer" in result1
        answer1 = result1["answer"].lower()
        assert any(word in answer1 for word in ["docker", "container", "platform"]), \
            f"First answer should be about Docker. Got: {result1['answer']}"
        
        # Follow-up question using pronoun (relies on memory)
        result2 = rag_chain("What are its benefits?")
        assert "answer" in result2
        answer2 = result2["answer"].lower()
        
        # Verify memory is working - answer should relate to Docker benefits
        assert any(word in answer2 for word in ["benefit", "consistency", "efficiency", "deployment", "scalability"]), \
            f"Second answer should reference Docker benefits. Got: {result2['answer']}"
    
    @pytest.mark.integration
    def test_context_aware_followup(self, rag_chain, sample_docs_path):
        """Test that follow-up questions use previous context."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader
        
        # Load and ingest documents
        loader = TextLoader(str(sample_docs_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(docs)
        
        # Add all chunks at once
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(sample_docs_path), "chunk_id": i} for i in range(len(chunks))]
        rag_chain.vectorstore.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        # First question about REST APIs
        result1 = rag_chain("What is a REST API?")
        assert "answer" in result1
        
        # Follow-up using pronoun
        result2 = rag_chain("What HTTP methods does it support?")
        assert "answer" in result2
        answer2 = result2["answer"].lower()
        assert any(word in answer2 for word in ["http", "method", "get", "post", "put", "delete"]), \
            f"Answer should mention HTTP methods. Got: {result2['answer']}"


class TestSourceDocuments:
    """Tests for source document retrieval."""
    
    @pytest.mark.integration
    def test_returns_source_documents(self, rag_chain, sample_docs_path):
        """Test that RAG returns source documents with answers."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader
        
        # Load and ingest documents
        loader = TextLoader(str(sample_docs_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(docs)
        
        # Add all chunks at once
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(sample_docs_path), "chunk_id": i} for i in range(len(chunks))]
        rag_chain.vectorstore.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        # Ask question
        result = rag_chain("What is Test-Driven Development?")
        
        # Verify source documents are returned
        assert "source_documents" in result, "Response should include source documents"
        assert len(result["source_documents"]) > 0, "Should return at least one source document"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.integration
    def test_handles_unrelated_question(self, rag_chain, sample_docs_path):
        """Test RAG handles questions outside document scope."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader
        
        # Load and ingest documents
        loader = TextLoader(str(sample_docs_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(docs)
        
        # Add all chunks at once
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(sample_docs_path), "chunk_id": i} for i in range(len(chunks))]
        rag_chain.vectorstore.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        # Ask question unrelated to sample docs
        result = rag_chain("What is quantum computing?")
        
        # Should still return a response (even if not in docs)
        assert "answer" in result
        assert len(result["answer"]) > 0
    
    @pytest.mark.integration
    def test_empty_question(self, rag_chain, sample_docs_path):
        """Test RAG handles empty questions gracefully."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader
        
        # Load and ingest documents
        loader = TextLoader(str(sample_docs_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(docs)
        
        # Add all chunks at once
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(sample_docs_path), "chunk_id": i} for i in range(len(chunks))]
        rag_chain.vectorstore.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        # Ask empty question
        result = rag_chain("")
        
        # Should return a response
        assert "answer" in result


class TestPerformance:
    """Performance tests for RAG chain."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_response_time(self, rag_chain, sample_docs_path):
        """Test that responses are generated within acceptable time limits."""
        import time
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import TextLoader
        
        # Load and ingest documents
        loader = TextLoader(str(sample_docs_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(docs)
        
        # Add all chunks at once
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(sample_docs_path), "chunk_id": i} for i in range(len(chunks))]
        rag_chain.vectorstore.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        # Measure response time
        start = time.time()
        result = rag_chain("What is Python?")
        elapsed = time.time() - start
        
        # Should respond within 30 seconds (generous for local LLM)
        assert elapsed < 30, f"Response took too long: {elapsed:.2f}s"
        assert "answer" in result
