# Backend/app/models/schemas.py
"""
Pydantic schemas for document handling.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    EPUB = "epub"
    TXT = "txt"
    DOCX = "docx"
    MOBI = "mobi"
    AZW = "azw"
    AZW3 = "azw3"


class DocumentStatus(str, Enum):
    """Document processing status."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Document metadata."""

    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    page_count: Optional[int] = Field(None, description="Number of pages")
    word_count: Optional[int] = Field(None, description="Number of words")
    file_size: int = Field(..., description="File size in bytes")
    file_type: DocumentType = Field(..., description="Document type")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")
    language: Optional[str] = Field(None, description="Document language")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(use_enum_values=True)


class DocumentUploadRequest(BaseModel):
    """Request model for document upload metadata."""

    title: Optional[str] = Field(
        None, max_length=255, description="Custom document title"
    )
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: List[str]) -> List[str]:
        """Validate and clean tags."""
        if not tags:
            return []

        # Clean and deduplicate tags
        cleaned_tags = [tag.strip().lower() for tag in tags if tag.strip()]
        return list(set(cleaned_tags))[:10]  # Max 10 tags


class DocumentUploadResponse(BaseModel):
    """Response model for successful document upload."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: DocumentType = Field(..., description="Document type")
    file_size: int = Field(..., description="File size in bytes")
    status: DocumentStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    metadata: DocumentMetadata = Field(..., description="Document metadata")

    model_config = ConfigDict(use_enum_values=True)


class DocumentExtractionResult(BaseModel):
    """Result of document text extraction."""

    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: DocumentType = Field(..., description="Document type")
    text_content: str = Field(..., description="Extracted text content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    extraction_time: float = Field(..., description="Time taken to extract (seconds)")
    success: bool = Field(True, description="Extraction success status")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    model_config = ConfigDict(use_enum_values=True)


class DocumentProcessingError(BaseModel):
    """Error response for document processing."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    filename: Optional[str] = Field(None, description="Filename that caused error")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )


class DocumentInfo(BaseModel):
    """Document information for listing."""

    document_id: str
    filename: str
    file_type: DocumentType
    file_size: int
    status: DocumentStatus
    title: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime

    model_config = ConfigDict(use_enum_values=True)


class TextChunk(BaseModel):
    """Individual text chunk."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    text: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Position in document (0-based)")
    start_char: int = Field(..., description="Starting character position in original text")
    end_char: int = Field(..., description="Ending character position in original text")
    token_count: int = Field(..., description="Approximate token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        json_schema_extra = {
            "example":{
                "chunk_id": "doc123_chunk_0",
                "document_id": "doc123",
                "text": "This is the first chunk of text...",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 512,
                "token_count": 120,
                "metadata": {"page": 1}
            }
        }
    )
        
class ChunkingResult(BaseModel):
    """ Result of chunking operation """
    document_id: str = Field(..., description="Document Identifier")
    chunks: List[TextChunk] = Field(..., description="List of text chunks")
    total_chunks: int = Field(..., description="Total number of chunks")
    total_tokens: int = Field(..., description="Total tokens across all chunks")
    chunk_size: int = Field(..., description="Configured chunk size")
    chunk_overlap: int = Field(..., description="Configured chunk overlap")
    processing_time: float = Field(..., description="Time taken to chunk (seconds)")
    
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "document_id": "doc123",
                "chunks": [],
                "total_chunks": 10,
                "total_tokens": 1200,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "processing_time": 0.15
            }
        }
    )

class ChunkingRequest(BaseModel):
    """Request for text chunking."""
    text: str = Field(..., description="Text to chunk")
    document_id: str = Field(..., description="Document identifier")
    chunk_size: Optional[int] = Field(None, description="Custom chunk size")
    chunk_overlap: Optional[int] = Field(None, description="Custom chunk overlap")
    
    @field_validator("text")
    @classmethod
    def validate_text_content(cls, text: str) -> str:
        """Validate text is not empty."""
        if not text or len(text.strip()) < 10:
            raise ValueError("Text must be at least 10 characters")
        return text
        

"""Embeddings"""
class EmbeddingRequest(BaseModel):
    """Request for embedding generation."""
    text: str = Field(..., description="Text to embed", min_length=1)
    chunk_id: Optional[str] = Field(None, description="Optional chunk identifier")
    task_type: Optional[str] = Field(
        "retrieval_document",
        description="Task type: retrieval_document, retrieval_query, semantic_similarity"
    )


class EmbeddingResponse(BaseModel):
    """Response for embedding generation."""
    chunk_id: str = Field(..., description="Chunk identifier")
    embedding: List[float] = Field(..., description="Embedding vector")
    dimensions: int = Field(..., description="Number of dimensions")
    processing_time: float = Field(..., description="Processing time in seconds")
    success: bool = Field(..., description="Whether generation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class BatchEmbeddingRequest(BaseModel):
    """Request for batch embedding generation."""
    texts: List[str] = Field(..., description="List of texts to embed", min_length=1)
    chunk_ids: Optional[List[str]] = Field(None, description="Optional chunk IDs")
    task_type: Optional[str] = Field("retrieval_document", description="Task type")
    batch_size: Optional[int] = Field(100, description="Batch size", ge=1, le=100)


class BatchEmbeddingResponse(BaseModel):
    """Response for batch embedding generation."""
    embeddings: List[EmbeddingResponse] = Field(..., description="List of embeddings")
    total_chunks: int = Field(..., description="Total number of chunks")
    successful: int = Field(..., description="Number of successful embeddings")
    failed: int = Field(..., description="Number of failed embeddings")
    total_time: float = Field(..., description="Total processing time")
    average_time: float = Field(..., description="Average time per embedding")

class AddEmbeddingsRequest(BaseModel):
    """Request to add embeddings to the vector store."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    chunks: List[TextChunk] = Field(..., description="List of text chunks (must match embeddings length)")
    user_id: str = Field(..., description="Owner of the document")
    document_id: str = Field(..., description="Document identifier")

    
""" Schemas for vector store operations """
class VectorStoreAddRequest(BaseModel):
    """Request for adding embeddings to vector store."""
    document_id: str = Field(..., description="Document identifier")
    user_id: str = Field(..., description="User identifier")
    chunks: List[TextChunk] = Field(..., description="Text chunks with embeddings")
    embeddings: List[List[float]] = Field(..., description="Embedding vectors")


class VectorStoreAddResponse(BaseModel):
    """Response for adding embeddings."""
    success: bool = Field(..., description="Whether operation succeeded")
    document_id: str = Field(..., description="Document identifier")
    chunks_added: int = Field(..., description="Number of chunks added")
    chunk_ids: List[str] = Field(..., description="List of chunk IDs")


class VectorStoreQueryRequest(BaseModel):
    """Request for querying vector store."""
    query_text: Optional[str] = Field(None, description="Query text (will be embedded)")
    query_embedding: Optional[List[float]] = Field(None, description="Pre-computed query embedding")
    n_results: int = Field(5, description="Number of results", ge=1, le=100)
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    document_id: Optional[str] = Field(None, description="Filter by document ID")
    min_similarity: Optional[float] = Field(None, description="Minimum similarity threshold (0-1", ge=0.0, le=1.0)


class VectorStoreQueryResult(BaseModel):
    """Single query result."""
    chunk_id: str = Field(..., description="Chunk identifier")
    document: str = Field(..., description="Chunk text")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")
    distance: float = Field(..., description="Distance score (lower is better)")
    similarity: float = Field(..., description="Similarity score (higher is better)")


class VectorStoreQueryResponse(BaseModel):
    """Response for vector store query."""
    results: List[VectorStoreQueryResult] = Field(..., description="Query results")
    total_results: int = Field(..., description="Total results returned")
    query_time: float = Field(..., description="Query time in seconds")


class VectorStoreStatsResponse(BaseModel):
    """Response for collection statistics."""
    collection_name: str = Field(..., description="Collection name")
    total_chunks: int = Field(..., description="Total chunks in collection")
    persist_directory: str = Field(..., description="Persistence directory")
    has_data: bool = Field(..., description="Whether collection has data")


class DocumentExistsResponse(BaseModel):
    """Response for document existence check."""
    document_id: str = Field(..., description="Document identifier")
    exists: bool = Field(..., description="Whether document exists")
    chunk_count: Optional[int] = Field(None, description="Number of chunks if exists")
    
    
""" ------------- Schemas for RAG LLM Response --------------- """
class RAGQueryRequest(BaseModel):
    """Request for RAG query."""
    query: str = Field(..., description="User query", min_length=1)
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    document_id: Optional[str] = Field(None, description="Filter by specific document")
    n_results: int = Field(5, description="Number of chunks to retrieve", ge=1, le=20)
    temperature: Optional[float] = Field(None, description="LLM temperature", ge=0.0, le=2.0)
    include_citations: bool = Field(True, description="Include source citations")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")


class CitationInfo(BaseModel):
    """Citation information for a source."""
    source_number: int = Field(..., description="Source number in response")
    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    text: str = Field(..., description="Excerpt from source (truncated)")
    similarity: float = Field(..., description="Similarity score")


class RAGQueryResponse(BaseModel):
    """Response for RAG query."""
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original query")
    citations: List[CitationInfo] = Field(default_factory=list, description="Source citations")
    num_chunks_used: int = Field(..., description="Number of context chunks used")
    generation_time: float = Field(..., description="Time to generate response")
    total_time: float = Field(..., description="Total query time")
    model: str = Field(..., description="Model used for generation")
    success: bool = Field(..., description="Whether generation succeeded")
    retrieval_info: Optional[Dict[str, Any]] = Field(None, description="Retrieval metadata")
    error: Optional[str] = Field(None, description="Error message if failed")


class GenerateResponseRequest(BaseModel):
    """Request for generating response with custom context."""
    query: str = Field(..., description="User query")
    context: str = Field(..., description="Context to use for generation")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens")
    include_citations: bool = Field(True, description="Include citations")


class TokenUsageStats(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(..., description="Tokens in prompt")
    completion_tokens: int = Field(..., description="Tokens in completion")
    total_tokens: int = Field(..., description="Total tokens used")
    estimated_cost: float = Field(..., description="Estimated cost in USD")
    
    
"""" --------Authentication related Schemas -----------------"""

class UserSignUpRequest(BaseModel):
    """Request for user registration."""
    email: str = Field(..., description="User email", pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    password: str = Field(..., description="User password", min_length=8)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional user metadata")


class UserSignInRequest(BaseModel):
    """Request for user login."""
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")


class AuthResponse(BaseModel):
    """Response for authentication operations."""
    success: bool = Field(..., description="Whether operation succeeded")
    user_id: Optional[str] = Field(None, description="User ID")
    email: Optional[str] = Field(None, description="User email")
    access_token: Optional[str] = Field(None, description="Access token")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    expires_in: Optional[int] = Field(None, description="Token expiry in seconds")
    message: Optional[str] = Field(None, description="Status message")
    error: Optional[str] = Field(None, description="Error message if failed")


class RefreshTokenRequest(BaseModel):
    """Request for refreshing access token."""
    refresh_token: str = Field(..., description="Refresh token")


class UserInfoResponse(BaseModel):
    """Response for user information."""
    user_id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    created_at: str = Field(..., description="Account creation date")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="User metadata")


class RateLimitInfo(BaseModel):
    """Rate limit information."""
    current_requests: int = Field(..., description="Current request count")
    max_requests: int = Field(..., description="Maximum allowed requests")
    remaining: int = Field(..., description="Remaining requests")
    window_seconds: int = Field(..., description="Time window in seconds")
    reset_at: Optional[str] = Field(None, description="When limit resets")


class PasswordResetRequest(BaseModel):
    """Request for password reset."""
    email: str = Field(..., description="User email")
    
    
class OAuthProviderResponse(BaseModel):
    """Response for OAuth provider information."""
    provider: str = Field(..., description="OAuth provider name")
    oauth_url: str = Field(..., description="OAuth authorization URL")
    message: str = Field(..., description="Instructions")
    
    
# Add these query-related schemas

class QueryRequest(BaseModel):
    """Enhanced request for RAG query."""
    query: str = Field(..., description="User query", min_length=1, max_length=1000)
    document_ids: Optional[List[str]] = Field(None, description="Filter by specific documents")
    n_results: int = Field(5, description="Number of chunks to retrieve", ge=1, le=20)
    temperature: Optional[float] = Field(0.7, description="LLM temperature", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    include_citations: bool = Field(True, description="Include source citations")
    min_similarity: Optional[float] = Field(None, description="Minimum similarity threshold", ge=0.0, le=1.0)


class SourceDocument(BaseModel):
    """Source document information."""
    document_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    chunk_id: str = Field(..., description="Chunk identifier")
    chunk_text: str = Field(..., description="Chunk text excerpt")
    similarity: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QueryResponse(BaseModel):
    """Enhanced response for RAG query."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents")
    citations: List[CitationInfo] = Field(default_factory=list, description="Citations in answer")
    confidence_score: Optional[float] = Field(None, description="Answer confidence score")
    
    # Performance metrics
    retrieval_time: float = Field(..., description="Time for retrieval in seconds")
    generation_time: float = Field(..., description="Time for generation in seconds")
    total_time: float = Field(..., description="Total processing time in seconds")
    
    # Statistics
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    chunks_used: int = Field(..., description="Number of chunks used in answer")
    tokens_used: Optional[int] = Field(None, description="Estimated tokens used")
    
    # Metadata
    model: str = Field(..., description="Model used for generation")
    success: bool = Field(..., description="Whether query succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchQueryRequest(BaseModel):
    """Request for batch queries."""
    queries: List[str] = Field(..., description="List of queries", min_length=1, max_length=10)
    n_results: int = Field(5, description="Number of chunks per query")
    temperature: float = Field(0.7, description="LLM temperature")


class BatchQueryResponse(BaseModel):
    """Response for batch queries."""
    results: List[QueryResponse] = Field(..., description="Query results")
    total_queries: int = Field(..., description="Total queries processed")
    successful: int = Field(..., description="Successful queries")
    failed: int = Field(..., description="Failed queries")
    total_time: float = Field(..., description="Total processing time")


class QueryStatsResponse(BaseModel):
    """Response for query statistics."""
    total_queries: int = Field(..., description="Total queries made")
    avg_response_time: float = Field(..., description="Average response time")
    avg_chunks_retrieved: float = Field(..., description="Average chunks retrieved")
    documents_indexed: int = Field(..., description="Number of documents indexed")
    total_chunks: int = Field(..., description="Total chunks in vector store")