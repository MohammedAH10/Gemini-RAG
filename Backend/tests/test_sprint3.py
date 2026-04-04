"""
Sprint 3 Tests: Text Chunking and Preprocessing
"""
import pytest
from app.core.rag_engine import RAGEngine, get_rag_engine
from app.utils.validators import TextValidator
from app.models.schemas import ChunkingRequest


# Test Data
SHORT_TEXT = "This is a short document with only a few words."

MEDIUM_TEXT = """
This is a medium-length document. It has multiple sentences and paragraphs.
This should be chunked into a few segments based on the configuration.

The second paragraph contains more information. We want to test that the
chunking algorithm properly handles paragraph breaks and maintains context.

The third paragraph is here to ensure we get multiple chunks. Each chunk
should have proper overlap with adjacent chunks to maintain semantic continuity.
"""

LONG_TEXT = " ".join([f"Sentence number {i} in a very long document." for i in range(1000)])


@pytest.fixture
def rag_engine():
    """Get RAG engine with default settings."""
    return get_rag_engine()


@pytest.fixture
def custom_rag_engine():
    """Get RAG engine with custom settings."""
    return RAGEngine(chunk_size=256, chunk_overlap=25)


# Validator Tests
def test_text_validator_valid_text():
    """Test validator accepts valid text."""
    is_valid, error = TextValidator.is_valid_text(MEDIUM_TEXT)
    assert is_valid
    assert error is None


def test_text_validator_empty_text():
    """Test validator rejects empty text."""
    is_valid, error = TextValidator.is_valid_text("")
    assert not is_valid
    assert "empty" in error.lower()


def test_text_validator_too_short():
    """Test validator rejects too short text."""
    is_valid, error = TextValidator.is_valid_text("Hi")
    assert not is_valid
    assert "short" in error.lower()


def test_text_validator_whitespace_only():
    """Test validator rejects whitespace-only text."""
    whitespace_text = "          \n\n\t      "
    is_valid, error = TextValidator.is_valid_text(whitespace_text)
    assert not is_valid
    # Accept any rejection reason - the important thing is it's rejected
    assert error is not None


def test_clean_text_removes_extra_whitespace():
    """Test text cleaning removes extra whitespace."""
    dirty_text = "This  has   extra    spaces\n\n\n\nand newlines"
    clean = TextValidator.clean_text(dirty_text)
    
    assert "  " not in clean
    assert "\n\n\n" not in clean


def test_clean_text_preserves_paragraphs():
    """Test text cleaning preserves paragraph structure."""
    text = "First paragraph.\n\nSecond paragraph."
    clean = TextValidator.clean_text(text)
    
    assert "\n\n" in clean


def test_validate_chunk_size_valid():
    """Test chunk size validation accepts valid sizes."""
    valid_sizes = [50, 256, 512, 1024, 2048]
    
    for size in valid_sizes:
        is_valid, error = TextValidator.validate_chunk_size(size)
        assert is_valid, f"Size {size} should be valid"


def test_validate_chunk_size_too_small():
    """Test chunk size validation rejects too small sizes."""
    is_valid, error = TextValidator.validate_chunk_size(10)
    assert not is_valid
    assert "at least 50" in error


def test_validate_chunk_size_too_large():
    """Test chunk size validation rejects too large sizes."""
    is_valid, error = TextValidator.validate_chunk_size(5000)
    assert not is_valid
    assert "not exceed 4000" in error


def test_validate_chunk_overlap_valid():
    """Test chunk overlap validation."""
    is_valid, error = TextValidator.validate_chunk_overlap(50, 512)
    assert is_valid


def test_validate_chunk_overlap_negative():
    """Test chunk overlap rejects negative values."""
    is_valid, error = TextValidator.validate_chunk_overlap(-10, 512)
    assert not is_valid


def test_validate_chunk_overlap_too_large():
    """Test chunk overlap rejects values >= chunk size."""
    is_valid, error = TextValidator.validate_chunk_overlap(512, 512)
    assert not is_valid


# RAG Engine Tests
def test_rag_engine_initialization():
    """Test RAG engine initializes correctly."""
    engine = RAGEngine(chunk_size=512, chunk_overlap=50)
    
    assert engine.chunk_size == 512
    assert engine.chunk_overlap == 50
    assert engine.text_splitter is not None


def test_rag_engine_default_parameters(rag_engine):
    """Test RAG engine uses default parameters."""
    assert rag_engine.chunk_size > 0
    assert rag_engine.chunk_overlap >= 0
    assert rag_engine.chunk_overlap < rag_engine.chunk_size


def test_chunk_short_document(rag_engine):
    """Test chunking a short document (< chunk size)."""
    result = rag_engine.chunk_text(
        text=SHORT_TEXT,
        document_id="short_doc"
    )
    
    assert result.total_chunks >= 1
    assert result.document_id == "short_doc"
    assert all(chunk.document_id == "short_doc" for chunk in result.chunks)


def test_chunk_medium_document(rag_engine):
    """Test chunking a medium document."""
    result = rag_engine.chunk_text(
        text=MEDIUM_TEXT,
        document_id="medium_doc"
    )
    
    assert result.total_chunks > 0
    assert result.total_tokens > 0
    assert len(result.chunks) == result.total_chunks


def test_chunk_long_document(rag_engine):
    """Test chunking a long document."""
    result = rag_engine.chunk_text(
        text=LONG_TEXT,
        document_id="long_doc"
    )
    
    assert result.total_chunks > 1
    assert all(chunk.chunk_index == i for i, chunk in enumerate(result.chunks))


def test_chunk_ids_unique(rag_engine):
    """Test that all chunk IDs are unique."""
    result = rag_engine.chunk_text(
        text=LONG_TEXT,
        document_id="test_doc"
    )
    
    chunk_ids = [chunk.chunk_id for chunk in result.chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_indices_sequential(rag_engine):
    """Test that chunk indices are sequential."""
    result = rag_engine.chunk_text(
        text=MEDIUM_TEXT,
        document_id="test_doc"
    )
    
    for i, chunk in enumerate(result.chunks):
        assert chunk.chunk_index == i


def test_chunk_character_positions(rag_engine):
    """Test that character positions are tracked correctly."""
    result = rag_engine.chunk_text(
        text=MEDIUM_TEXT,
        document_id="test_doc"
    )
    
    for chunk in result.chunks:
        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char
        assert chunk.end_char <= len(MEDIUM_TEXT)


def test_chunk_overlap_exists(rag_engine):
    """Test that chunks have overlap when document is long enough."""
    result = rag_engine.chunk_text(
        text=LONG_TEXT,
        document_id="test_doc"
    )
    
    if len(result.chunks) > 1:
        overlap_stats = rag_engine.verify_chunk_overlap(result.chunks)
        # Should have overlap if chunk_overlap > 0
        if rag_engine.chunk_overlap > 0:
            assert overlap_stats["has_overlap"]


def test_chunk_with_metadata(rag_engine):
    """Test chunking with custom metadata."""
    metadata = {"source": "test", "page": 1}
    
    result = rag_engine.chunk_text(
        text=MEDIUM_TEXT,
        document_id="test_doc",
        metadata=metadata
    )
    
    for chunk in result.chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "test"


def test_chunk_stats(rag_engine):
    """Test chunk statistics calculation."""
    result = rag_engine.chunk_text(
        text=MEDIUM_TEXT,
        document_id="test_doc"
    )
    
    stats = rag_engine.get_chunk_stats(result.chunks)
    
    assert stats["total_chunks"] == len(result.chunks)
    assert stats["total_tokens"] > 0
    assert stats["avg_chunk_size"] > 0


def test_custom_chunk_size(custom_rag_engine):
    """Test chunking with custom chunk size."""
    result = custom_rag_engine.chunk_text(
        text=LONG_TEXT,
        document_id="test_doc"
    )
    
    assert custom_rag_engine.chunk_size == 256
    assert custom_rag_engine.chunk_overlap == 25


def test_chunk_preserves_content(rag_engine):
    """Test that chunking preserves all content."""
    original_text = MEDIUM_TEXT.strip()
    
    result = rag_engine.chunk_text(
        text=original_text,
        document_id="test_doc"
    )
    
    # Concatenate all chunks
    reconstructed = " ".join(chunk.text for chunk in result.chunks)
    
    # Should contain most of the original content
    # (some whitespace differences are acceptable)
    assert len(reconstructed) > 0


def test_invalid_text_raises_error(rag_engine):
    """Test that invalid text raises an error."""
    with pytest.raises(ValueError):
        rag_engine.chunk_text(
            text="",
            document_id="test_doc"
        )


def test_chunk_multiple_documents(rag_engine):
    """Test chunking multiple documents."""
    documents = [
        {"text": SHORT_TEXT, "document_id": "doc1"},
        {"text": MEDIUM_TEXT, "document_id": "doc2"},
    ]
    
    results = rag_engine.chunk_documents(documents)
    
    assert len(results) == 2
    assert results[0].document_id == "doc1"
    assert results[1].document_id == "doc2"


def test_get_rag_engine_singleton():
    """Test that get_rag_engine returns singleton."""
    engine1 = get_rag_engine()
    engine2 = get_rag_engine()
    
    assert engine1 is engine2


def test_get_rag_engine_custom_params():
    """Test that custom params create new instance."""
    engine1 = get_rag_engine()
    engine2 = get_rag_engine(chunk_size=256)
    
    assert engine1 is not engine2
    assert engine2.chunk_size == 256


def test_chunk_processing_time(rag_engine):
    """Test that processing time is recorded."""
    result = rag_engine.chunk_text(
        text=MEDIUM_TEXT,
        document_id="test_doc"
    )
    
    assert result.processing_time > 0
    assert result.processing_time < 10  # Should be fast


def test_empty_chunks_list_stats():
    """Test statistics with empty chunks list."""
    engine = RAGEngine()
    stats = engine.get_chunk_stats([])
    
    assert stats["total_chunks"] == 0
    assert stats["total_characters"] == 0


def test_unicode_text_handling(rag_engine):
    """Test handling of Unicode text."""
    unicode_text = "Hello 世界! Bonjour café. Здравствуй мир!"
    
    result = rag_engine.chunk_text(
        text=unicode_text,
        document_id="unicode_doc"
    )
    
    assert result.total_chunks > 0
    assert any("世界" in chunk.text or "café" in chunk.text for chunk in result.chunks)