"""
Database models and schemas for Supabase.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentDB(BaseModel):
    """Document database model."""
    id: str = Field(..., description="Document ID (UUID)")
    user_id: str = Field(..., description="Owner user ID")
    title: str = Field(..., description="Document title")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type (pdf, txt, etc)")
    file_size: int = Field(..., description="File size in bytes")
    file_path: str = Field(..., description="Storage path")
    
    # Processing metadata
    status: DocumentStatus = Field(..., description="Processing status")
    chunk_count: int = Field(0, description="Number of chunks")
    total_tokens: int = Field(0, description="Total tokens")
    
    # Additional metadata
    tags: List[str] = Field(default_factory=list, description="Document tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    indexed_at: Optional[datetime] = Field(None, description="Indexing completion timestamp")
    
    # Error tracking
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(0, description="Number of retry attempts")


class DocumentCreate(BaseModel):
    """Schema for creating a document."""
    user_id: str = Field(..., description="Owner user ID")
    title: str = Field(..., description="Document title")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type")
    file_size: int = Field(..., description="File size in bytes")
    file_path: str = Field(..., description="Storage path")
    tags: List[str] = Field(default_factory=list, description="Tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""
    title: Optional[str] = Field(None, description="Updated title")
    tags: Optional[List[str]] = Field(None, description="Updated tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    status: Optional[DocumentStatus] = Field(None, description="Updated status")
    chunk_count: Optional[int] = Field(None, description="Chunk count")
    total_tokens: Optional[int] = Field(None, description="Total tokens")
    error_message: Optional[str] = Field(None, description="Error message")


class DocumentListResponse(BaseModel):
    """Response for document list."""
    documents: List[DocumentDB] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")
    total_pages: int = Field(..., description="Total pages")


class DocumentDetailResponse(BaseModel):
    """Response for single document details."""
    document: DocumentDB = Field(..., description="Document details")
    chunks_preview: Optional[List[str]] = Field(None, description="Preview of chunks")


class DocumentDeleteResponse(BaseModel):
    """Response for document deletion."""
    success: bool = Field(..., description="Whether deletion succeeded")
    document_id: str = Field(..., description="Deleted document ID")
    chunks_deleted: int = Field(..., description="Number of chunks deleted")
    message: str = Field(..., description="Deletion message")