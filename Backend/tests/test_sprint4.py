"""
Sprint 4 Tests: Embedding Generation
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Set environment variables BEFORE importing app modules
os.environ["SECRET_KEY"] = "test_secret_key_minimum_32_characters_long"
os.environ["GEMINI_API_KEY"] = "AIzaSyTestKey123456789"
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["SUPABASE_KEY"] = "test_key_123456"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_service_key_123456"

from app.core.llm_client import GeminiClient
from app.core.embeddings import EmbeddingService, get_embedding_service
from app.models.schemas import TextChunk


# Mock embedding for testing
MOCK_EMBEDDING_768 = [0.1] * 768
MOCK_EMBEDDING_DIMENSIONS = 768


@pytest.fixture
def mock_gemini_client():
    """Create a mock Gemini client."""
    client = Mock(spec=GeminiClient)
    client.embedding_dimensions = MOCK_EMBEDDING_DIMENSIONS
    client.embedding_model = "models/text-embedding-004"
    client.generate_embedding.return_value = MOCK_EMBEDDING_768
    client.generate_embeddings_batch.return_value = [MOCK_EMBEDDING_768] * 3
    return client


@pytest.fixture
def embedding_service(mock_gemini_client):
    """Create embedding service with mock client."""
    return EmbeddingService(gemini_client=mock_gemini_client)


@pytest.fixture
def sample_chunks():
    """Create sample text chunks."""
    return [
        TextChunk(
            chunk_id=f"chunk_{i}",
            document_id="test_doc",
            text=f"This is test chunk number {i}.",
            chunk_index=i,
            start_char=i * 50,
            end_char=(i + 1) * 50,
            token_count=10
        )
        for i in range(3)
    ]


# Gemini Client Tests
@patch('app.core.llm_client.genai.Client')
def test_gemini_client_initialization(mock_client):
    """Test Gemini client initialization."""
    # Mock the client instance
    mock_client_instance = Mock()
    mock_client.return_value = mock_client_instance

    client = GeminiClient(api_key="test_key")

    assert client.api_key == "test_key"
    assert client.embedding_dimensions > 0
    mock_client.assert_called_once_with(api_key="test_key")


# @patch('app.core.llm_client.genai.configure')
@patch('app.core.llm_client.genai.Client')
def test_get_embedding_dimensions(mock_client):
    """Test embedding dimensions detection."""
    mock_client_instance = Mock()
    mock_client.return_value = mock_client_instance

    client = GeminiClient(
        api_key="test_key",
        embedding_model="models/text-embedding-004"
    )

    # The embedding dimensions are set in __init__ based on the model name
    # For text-embedding-004, it should be 768 (as per your code)
    assert client.embedding_dimensions == 768


# Embedding Service Tests
def test_embedding_service_initialization(mock_gemini_client):
    """Test embedding service initialization."""
    service = EmbeddingService(gemini_client=mock_gemini_client)
    
    assert service.client is not None
    assert service.embedding_dimensions == MOCK_EMBEDDING_DIMENSIONS


def test_generate_single_embedding(embedding_service):
    """Test generating embedding for single text."""
    result = embedding_service.generate_embedding(
        text="Test text for embedding",
        chunk_id="test_chunk_1"
    )
    
    assert result.success
    assert result.chunk_id == "test_chunk_1"
    assert len(result.embedding) == MOCK_EMBEDDING_DIMENSIONS
    assert result.processing_time >= 0


def test_generate_embedding_empty_text(embedding_service):
    """Test generating embedding with empty text."""
    embedding_service.client.generate_embedding.side_effect = ValueError("Text cannot be empty")
    
    result = embedding_service.generate_embedding(
        text="",
        chunk_id="empty_chunk"
    )
    
    assert not result.success
    assert result.error_message is not None


def test_generate_embeddings_for_chunks(embedding_service, sample_chunks):
    """Test generating embeddings for multiple chunks."""
    result = embedding_service.generate_embeddings_for_chunks(sample_chunks)
    
    assert result.total_chunks == len(sample_chunks)
    assert result.successful >= 0
    assert len(result.embeddings) == len(sample_chunks)
    assert result.total_time >= 0


def test_batch_embedding_with_batch_size(embedding_service, sample_chunks):
    """Test batch embedding with custom batch size."""
    result = embedding_service.generate_embeddings_for_chunks(
        sample_chunks,
        batch_size=2
    )
    
    assert result.total_chunks == len(sample_chunks)
    assert len(result.embeddings) == len(sample_chunks)


def test_generate_query_embedding(embedding_service):
    """Test generating embedding for search query."""
    query = "What is machine learning?"
    
    embedding = embedding_service.generate_query_embedding(query)
    
    assert len(embedding) == MOCK_EMBEDDING_DIMENSIONS
    embedding_service.client.generate_embedding.assert_called_once()


def test_validate_embedding_valid(embedding_service):
    """Test validating a valid embedding."""
    valid_embedding = [0.1] * MOCK_EMBEDDING_DIMENSIONS
    
    is_valid = embedding_service.validate_embedding(valid_embedding)
    
    assert is_valid


def test_validate_embedding_wrong_dimensions(embedding_service):
    """Test validating embedding with wrong dimensions."""
    wrong_embedding = [0.1] * 512  # Wrong size
    
    is_valid = embedding_service.validate_embedding(wrong_embedding)
    
    assert not is_valid


def test_validate_embedding_zero_vector(embedding_service):
    """Test validating zero vector."""
    zero_embedding = [0.0] * MOCK_EMBEDDING_DIMENSIONS
    
    is_valid = embedding_service.validate_embedding(zero_embedding)
    
    assert not is_valid


def test_validate_embedding_empty(embedding_service):
    """Test validating empty embedding."""
    is_valid = embedding_service.validate_embedding([])
    
    assert not is_valid


def test_get_embedding_info(embedding_service):
    """Test getting embedding service info."""
    info = embedding_service.get_embedding_info()
    
    assert "embedding_dimensions" in info
    assert "model" in info
    assert info["embedding_dimensions"] == MOCK_EMBEDDING_DIMENSIONS


def test_embedding_service_singleton():
    """Test that get_embedding_service returns singleton."""
    with patch('app.core.embeddings.get_gemini_client'):
        # Clear the global singleton first
        import app.core.embeddings
        app.core.embeddings._embedding_service = None
        
        service1 = get_embedding_service()
        service2 = get_embedding_service()
        
        assert service1 is service2


# Error Handling Tests
def test_embedding_generation_failure_handling(embedding_service):
    """Test handling of embedding generation failures."""
    # Simulate API failure
    embedding_service.client.generate_embedding.side_effect = Exception("API Error")
    
    result = embedding_service.generate_embedding(
        text="Test text",
        chunk_id="fail_chunk"
    )
    
    assert not result.success
    assert result.error_message is not None
    assert len(result.embedding) == MOCK_EMBEDDING_DIMENSIONS  # Zero vector fallback


def test_batch_embedding_fallback(embedding_service, sample_chunks):
    """Test fallback to individual processing when batch fails."""
    # First call fails (batch), subsequent calls succeed (individual)
    embedding_service.client.generate_embeddings_batch.side_effect = Exception("Batch failed")
    embedding_service.client.generate_embedding.return_value = MOCK_EMBEDDING_768
    
    result = embedding_service.generate_embeddings_for_chunks(sample_chunks)
    
    # Should fall back to individual processing
    assert embedding_service.client.generate_embedding.call_count == len(sample_chunks)


# Integration Tests
def test_end_to_end_embedding_generation(sample_chunks):
    """Test end-to-end embedding generation with mocks."""
    with patch('app.core.embeddings.get_gemini_client') as mock_get_client:
        mock_client = Mock(spec=GeminiClient)
        mock_client.embedding_dimensions = 768
        mock_client.generate_embeddings_batch.return_value = [[0.1] * 768] * 3
        mock_get_client.return_value = mock_client
        
        # Clear singleton
        import app.core.embeddings
        app.core.embeddings._embedding_service = None
        
        service = EmbeddingService()
        result = service.generate_embeddings_for_chunks(sample_chunks)
        
        assert result.total_chunks == 3
        assert result.successful == 3
        assert result.failed == 0

    
# Async Tests
@pytest.mark.asyncio
async def test_generate_embedding_async(embedding_service):
    """Test async embedding generation."""
    async def mock_async_return(*args, **kwargs):  # ← FIXED: Accept arguments
        return MOCK_EMBEDDING_768
    
    embedding_service.client.generate_embedding_async = Mock(side_effect=mock_async_return)
    
    result = await embedding_service.generate_embedding_async(
        text="Async test text",
        chunk_id="async_chunk"
    )
    
    assert result.success
    assert len(result.embedding) == MOCK_EMBEDDING_DIMENSIONS