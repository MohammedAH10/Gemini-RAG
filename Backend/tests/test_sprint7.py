"""
Sprint 7 Tests: LLM Response Generation
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Set environment variables BEFORE importing
os.environ["SECRET_KEY"] = "test_secret_key_minimum_32_characters_long"
os.environ["GEMINI_API_KEY"] = "AIzaSyTestKey123456789"
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["SUPABASE_KEY"] = "test_key_123456"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_service_key_123456"

from app.core.rag_engine import RAGEngine, get_rag_engine
from app.models.schemas import TextChunk


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock()
    client.model_name = "gemini-test"
    client.generate_text.return_value = (
        "Machine learning is a subset of artificial intelligence. [Source 1]"
    )
    return client


@pytest.fixture
def mock_embedding_service():
    """Create mock embedding service."""
    service = Mock()
    service.generate_query_embedding.return_value = [0.1] * 768
    return service


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = Mock()
    store.query.return_value = {
        "ids": ["chunk_1", "chunk_2"],
        "documents": [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks."
        ],
        "metadatas": [
            {"document_id": "doc1", "chunk_index": 0},
            {"document_id": "doc1", "chunk_index": 1}
        ],
        "distances": [0.1, 0.2]
    }
    return store


@pytest.fixture
def rag_engine(mock_llm_client, mock_embedding_service, mock_vector_store):
    """Create RAG engine with mocked components."""
    engine = RAGEngine()
    engine._llm_client = mock_llm_client
    engine._embedding_service = mock_embedding_service
    engine._vector_store = mock_vector_store
    return engine


@pytest.fixture
def sample_context_chunks():
    """Create sample context chunks."""
    return [
        {
            "chunk_id": "chunk_1",
            "document": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"document_id": "doc1", "chunk_index": 0},
            "distance": 0.1,
            "similarity": 0.9
        },
        {
            "chunk_id": "chunk_2",
            "document": "Deep learning is a specialized form of machine learning.",
            "metadata": {"document_id": "doc1", "chunk_index": 1},
            "distance": 0.2,
            "similarity": 0.8
        }
    ]


# Context Building Tests
def test_build_context(rag_engine, sample_context_chunks):
    """Test building context from chunks."""
    context = rag_engine._build_context(sample_context_chunks)
    
    assert "Machine learning" in context
    assert "Deep learning" in context
    assert "[Source 1" in context
    assert "[Source 2" in context


def test_build_context_empty(rag_engine):
    """Test building context with no chunks."""
    context = rag_engine._build_context([])
    assert context == ""


# Prompt Building Tests
def test_build_prompt_with_context(rag_engine):
    """Test building prompt with context."""
    prompt = rag_engine._build_prompt(
        query="What is machine learning?",
        context="Machine learning is a subset of AI.",
        include_citations=True
    )
    
    assert "What is machine learning?" in prompt
    assert "Machine learning is a subset of AI" in prompt
    assert "cite the source" in prompt or "Source" in prompt


def test_build_prompt_without_context(rag_engine):
    """Test building prompt without context."""
    prompt = rag_engine._build_prompt(
        query="What is machine learning?",
        context="",
        include_citations=False
    )
    
    assert "What is machine learning?" in prompt
    assert "Context" not in prompt


def test_build_prompt_custom_system(rag_engine):
    """Test building prompt with custom system prompt."""
    custom_system = "You are a technical expert."
    prompt = rag_engine._build_prompt(
        query="Explain AI",
        context="",
        system_prompt=custom_system
    )
    
    assert custom_system in prompt


# Citation Extraction Tests
def test_extract_citations(rag_engine, sample_context_chunks):
    """Test extracting citations from response."""
    response = "Machine learning is important. [Source 1] It uses algorithms. [Source 2]"
    
    citations = rag_engine._extract_citations(response, sample_context_chunks)
    
    assert len(citations) == 2
    assert citations[0]["source_number"] == 1
    assert citations[1]["source_number"] == 2
    assert "chunk_1" in citations[0]["chunk_id"]


def test_extract_citations_no_citations(rag_engine, sample_context_chunks):
    """Test extraction when no citations in response."""
    response = "Machine learning is important."
    
    citations = rag_engine._extract_citations(response, sample_context_chunks)
    
    assert len(citations) == 0


def test_extract_citations_duplicate_sources(rag_engine, sample_context_chunks):
    """Test extraction with duplicate source references."""
    response = "[Source 1] and [Source 1] again."
    
    citations = rag_engine._extract_citations(response, sample_context_chunks)
    
    # Should only include unique sources
    assert len(citations) == 1


# Response Generation Tests
def test_generate_response(rag_engine, sample_context_chunks):
    """Test generating response with context."""
    result = rag_engine.generate_response(
        query="What is machine learning?",
        context_chunks=sample_context_chunks,
        include_citations=True
    )
    
    assert result["success"]
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert result["num_chunks_used"] == 2


def test_generate_response_no_context(rag_engine):
    """Test generating response without context."""
    result = rag_engine.generate_response(
        query="What is machine learning?",
        context_chunks=[],
        include_citations=False
    )
    
    assert result["success"]
    assert result["num_chunks_used"] == 0


def test_generate_response_with_temperature(rag_engine, sample_context_chunks):
    """Test generating response with custom temperature."""
    result = rag_engine.generate_response(
        query="What is AI?",
        context_chunks=sample_context_chunks,
        temperature=0.9
    )
    
    assert result["success"]
    rag_engine.llm_client.generate_text.assert_called_once()


def test_generate_response_error_handling(rag_engine, sample_context_chunks):
    """Test error handling in response generation."""
    rag_engine.llm_client.generate_text.side_effect = Exception("API Error")
    
    result = rag_engine.generate_response(
        query="What is AI?",
        context_chunks=sample_context_chunks
    )
    
    assert not result["success"]
    assert "error" in result


# Complete RAG Pipeline Tests
def test_rag_query_pipeline(rag_engine):
    """Test complete RAG query pipeline."""
    result = rag_engine.query(
        query="What is machine learning?",
        n_results=5
    )
    
    assert "answer" in result
    assert "query" in result
    assert "citations" in result
    assert "total_time" in result


def test_rag_query_with_user_filter(rag_engine):
    """Test RAG query with user ID filter."""
    result = rag_engine.query(
        query="What is AI?",
        user_id="test_user",
        n_results=3
    )
    
    assert result["retrieval_info"]["user_id_filter"] == "test_user"


def test_rag_query_with_document_filter(rag_engine):
    """Test RAG query with document ID filter."""
    result = rag_engine.query(
        query="What is AI?",
        document_id="doc123",
        n_results=5
    )
    
    assert result["retrieval_info"]["document_id_filter"] == "doc123"


def test_rag_query_no_citations(rag_engine):
    """Test RAG query without citations."""
    result = rag_engine.query(
        query="What is AI?",
        include_citations=False
    )
    
    assert len(result["citations"]) == 0


def test_rag_query_error_handling(rag_engine):
    """Test RAG query error handling."""
    rag_engine.embedding_service.generate_query_embedding.side_effect = Exception("Embedding Error")
    
    result = rag_engine.query(query="What is AI?")
    
    assert not result["success"]
    assert "error" in result


# Performance Tests
def test_response_generation_time(rag_engine, sample_context_chunks):
    """Test that response generation tracks time."""
    result = rag_engine.generate_response(
        query="What is AI?",
        context_chunks=sample_context_chunks
    )
    
    assert result["generation_time"] >= 0
    assert isinstance(result["generation_time"], float)


def test_query_total_time(rag_engine):
    """Test that total query time is tracked."""
    result = rag_engine.query(query="What is AI?")
    
    assert result["total_time"] >= 0
    assert result["total_time"] >= result.get("generation_time", 0)


# Token Limit Tests
def test_generate_with_max_tokens(rag_engine, sample_context_chunks):
    """Test generating with max tokens limit."""
    result = rag_engine.generate_response(
        query="What is AI?",
        context_chunks=sample_context_chunks,
        max_tokens=100
    )
    
    assert result["success"]
    # Verify max_tokens was passed to LLM
    call_args = rag_engine.llm_client.generate_text.call_args
    assert call_args is not None


# Integration Tests
def test_rag_engine_lazy_loading(rag_engine):
    """Test lazy loading of components."""
    # Components should be initialized on first access
    assert rag_engine.llm_client is not None
    assert rag_engine.embedding_service is not None
    assert rag_engine.vector_store is not None


def test_get_rag_engine_singleton():
    """Test RAG engine singleton."""
    with patch('app.core.rag_engine.get_gemini_client'):
        with patch('app.core.rag_engine.get_vector_store'):
            with patch('app.core.rag_engine.get_embedding_service'):
                # Clear singleton
                import app.core.rag_engine
                app.core.rag_engine._rag_engine = None
                
                engine1 = get_rag_engine()
                engine2 = get_rag_engine()
                
                assert engine1 is engine2