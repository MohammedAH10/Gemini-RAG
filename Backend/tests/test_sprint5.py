"""
Sprint 5 Tests: Vector Database Integration
"""

import os

# Set environment variables BEFORE importing app modules
os.environ["SECRET_KEY"] = "test_secret_key_minimum_32_characters_long"
os.environ["GEMINI_API_KEY"] = "AIzaSyTestKey123456789"
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["SUPABASE_KEY"] = "test_key_123456"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_service_key_123456"

import pytest
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

from app.core.vector_store import VectorStore, get_vector_store
from app.models.schemas import TextChunk


@pytest.fixture
def temp_chroma_dir():
    """Create temporary directory for ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def vector_store(temp_chroma_dir):
    """Create vector store with temporary directory."""
    return VectorStore(
        collection_name="test_collection",
        persist_directory=temp_chroma_dir
    )


@pytest.fixture
def sample_chunks():
    """Create sample text chunks."""
    return [
        TextChunk(
            chunk_id=f"test_chunk_{i}",
            document_id="test_doc",
            text=f"This is test chunk number {i} about machine learning.",
            chunk_index=i,
            start_char=i * 50,
            end_char=(i + 1) * 50,
            token_count=10
        )
        for i in range(3)
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings."""
    return [[0.1 + i * 0.1] * 768 for i in range(3)]


# Initialization Tests
def test_vector_store_initialization(temp_chroma_dir):
    """Test vector store initialization."""
    store = VectorStore(
        collection_name="test_collection",
        persist_directory=temp_chroma_dir
    )
    
    assert store.collection_name == "test_collection"
    assert store.persist_directory == temp_chroma_dir
    assert store.collection is not None


def test_collection_creation(vector_store):
    """Test that collection is created."""
    assert vector_store.collection is not None
    stats = vector_store.get_collection_stats()
    assert stats["collection_name"] == "test_collection"


# Adding Embeddings Tests
def test_add_embeddings(vector_store, sample_chunks, sample_embeddings):
    """Test adding embeddings to vector store."""
    result = vector_store.add_embeddings(
        embeddings=sample_embeddings,
        chunks=sample_chunks,
        user_id="test_user",
        document_id="test_doc"
    )
    
    assert result["success"]
    assert result["chunks_added"] == 3
    assert len(result["chunk_ids"]) == 3


def test_add_embeddings_mismatch_length(vector_store, sample_chunks):
    """Test error when embeddings and chunks length mismatch."""
    with pytest.raises(ValueError, match="must match"):
        vector_store.add_embeddings(
            embeddings=[[0.1] * 768],  # Only 1 embedding
            chunks=sample_chunks,  # 3 chunks
            user_id="test_user",
            document_id="test_doc"
        )


def test_add_empty_embeddings(vector_store):
    """Test error when adding empty embeddings."""
    with pytest.raises(ValueError, match="No embeddings"):
        vector_store.add_embeddings(
            embeddings=[],
            chunks=[],
            user_id="test_user",
            document_id="test_doc"
        )


# Query Tests
def test_query_vector_store(vector_store, sample_chunks, sample_embeddings):
    """Test querying vector store."""
    # Add embeddings first
    vector_store.add_embeddings(
        embeddings=sample_embeddings,
        chunks=sample_chunks,
        user_id="test_user",
        document_id="test_doc"
    )
    
    # Query
    results = vector_store.query(
        query_embedding=[0.1] * 768,
        n_results=2,
        user_id="test_user"
    )
    
    assert results["total_results"] > 0
    assert len(results["ids"]) <= 2


def test_query_with_document_filter(vector_store, sample_chunks, sample_embeddings):
    """Test querying with document ID filter."""
    vector_store.add_embeddings(
        embeddings=sample_embeddings,
        chunks=sample_chunks,
        user_id="test_user",
        document_id="test_doc"
    )
    
    results = vector_store.query(
        query_embedding=[0.1] * 768,
        n_results=5,
        document_id="test_doc"
    )
    
    assert results["total_results"] > 0


# Retrieval Tests
def test_get_by_id(vector_store, sample_chunks, sample_embeddings):
    """Test getting chunk by ID."""
    vector_store.add_embeddings(
        embeddings=sample_embeddings,
        chunks=sample_chunks,
        user_id="test_user",
        document_id="test_doc"
    )
    
    chunk = vector_store.get_by_id("test_chunk_0")
    
    assert chunk is not None
    assert chunk["id"] == "test_chunk_0"
    assert "document" in chunk
    assert "metadata" in chunk


def test_get_nonexistent_id(vector_store):
    """Test getting non-existent chunk."""
    chunk = vector_store.get_by_id("nonexistent_id")
    assert chunk is None


# Deletion Tests
def test_delete_by_document_id(vector_store, sample_chunks, sample_embeddings):
    """Test deleting document by ID."""
    vector_store.add_embeddings(
        embeddings=sample_embeddings,
        chunks=sample_chunks,
        user_id="test_user",
        document_id="test_doc"
    )
    
    deleted_count = vector_store.delete_by_document_id("test_doc")
    
    assert deleted_count == 3


def test_delete_nonexistent_document(vector_store):
    """Test deleting non-existent document."""
    deleted_count = vector_store.delete_by_document_id("nonexistent_doc")
    assert deleted_count == 0


def test_delete_by_user_id(vector_store, sample_chunks, sample_embeddings):
    """Test deleting all data for a user."""
    vector_store.add_embeddings(
        embeddings=sample_embeddings,
        chunks=sample_chunks,
        user_id="test_user",
        document_id="test_doc"
    )
    
    deleted_count = vector_store.delete_by_user_id("test_user")
    
    assert deleted_count == 3


# Document Management Tests
def test_list_documents(vector_store, sample_chunks, sample_embeddings):
    """Test listing documents for a user."""
    vector_store.add_embeddings(
        embeddings=sample_embeddings,
        chunks=sample_chunks,
        user_id="test_user",
        document_id="test_doc"
    )
    
    documents = vector_store.list_documents("test_user")
    
    assert len(documents) == 1
    assert documents[0]["document_id"] == "test_doc"
    assert documents[0]["chunk_count"] == 3


def test_list_documents_empty(vector_store):
    """Test listing documents for user with no documents."""
    documents = vector_store.list_documents("nonexistent_user")
    assert len(documents) == 0


def test_check_document_exists(vector_store, sample_chunks, sample_embeddings):
    """Test checking if document exists."""
    vector_store.add_embeddings(
        embeddings=sample_embeddings,
        chunks=sample_chunks,
        user_id="test_user",
        document_id="test_doc"
    )
    
    exists = vector_store.check_document_exists("test_doc", "test_user")
    assert exists
    
    not_exists = vector_store.check_document_exists("other_doc", "test_user")
    assert not not_exists


# Statistics Tests
def test_get_collection_stats(vector_store, sample_chunks, sample_embeddings):
    """Test getting collection statistics."""
    vector_store.add_embeddings(
        embeddings=sample_embeddings,
        chunks=sample_chunks,
        user_id="test_user",
        document_id="test_doc"
    )
    
    stats = vector_store.get_collection_stats()
    
    assert stats["total_chunks"] == 3
    assert stats["has_data"]
    assert stats["collection_name"] == "test_collection"


def test_empty_collection_stats(vector_store):
    """Test stats for empty collection."""
    stats = vector_store.get_collection_stats()
    
    assert stats["total_chunks"] == 0
    assert not stats["has_data"]


# Persistence Tests
def test_persistence(temp_chroma_dir, sample_chunks, sample_embeddings):
    """Test that data persists across instances."""
    # Create store and add data
    store1 = VectorStore(
        collection_name="persist_test",
        persist_directory=temp_chroma_dir
    )
    store1.add_embeddings(
        embeddings=sample_embeddings,
        chunks=sample_chunks,
        user_id="test_user",
        document_id="test_doc"
    )
    
    # Create new store instance (should load existing data)
    store2 = VectorStore(
        collection_name="persist_test",
        persist_directory=temp_chroma_dir
    )
    
    stats = store2.get_collection_stats()
    assert stats["total_chunks"] == 3


# User Isolation Tests
def test_user_isolation(vector_store, sample_chunks, sample_embeddings):
    """Test that users' data is isolated."""
    # Add data for user1
    vector_store.add_embeddings(
        embeddings=sample_embeddings,
        chunks=sample_chunks,
        user_id="user1",
        document_id="doc1"
    )
    
    # Add data for user2
    chunks_user2 = [
        TextChunk(
            chunk_id="user2_chunk_0",
            document_id="doc2",
            text="User 2 content",
            chunk_index=0,
            start_char=0,
            end_char=50,
            token_count=5
        )
    ]
    vector_store.add_embeddings(
        embeddings=[sample_embeddings[0]],
        chunks=chunks_user2,
        user_id="user2",
        document_id="doc2"
    )
    
    # Query for user1 only
    results = vector_store.query(
        query_embedding=[0.1] * 768,
        n_results=10,
        user_id="user1"
    )
    
    # Should only get user1's data
    assert all(meta["user_id"] == "user1" for meta in results["metadatas"])


# Singleton Tests
def test_get_vector_store_singleton():
    """Test that get_vector_store returns singleton."""
    with patch('app.core.vector_store.VectorStore'):
        # Clear singleton
        import app.core.vector_store
        app.core.vector_store._vector_store = None
        
        store1 = get_vector_store()
        store2 = get_vector_store()
        
        assert store1 is store2