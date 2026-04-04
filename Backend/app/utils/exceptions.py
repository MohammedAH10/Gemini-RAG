"""
Custom exceptions for the application.
"""
from typing import Optional, Any, Dict
from fastapi import status


class RAGException(Exception):
    """Base exception for RAG application."""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class DocumentNotFoundError(RAGException):
    """Document not found."""
    
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document not found: {document_id}",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"document_id": document_id}
        )


class UnauthorizedAccessError(RAGException):
    """Unauthorized access to resource."""
    
    def __init__(self, resource: str):
        super().__init__(
            message=f"Unauthorized access to {resource}",
            status_code=status.HTTP_403_FORBIDDEN,
            details={"resource": resource}
        )


class DocumentProcessingError(RAGException):
    """Document processing failed."""
    
    def __init__(self, document_id: str, error: str):
        super().__init__(
            message=f"Failed to process document: {error}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"document_id": document_id, "error": error}
        )


class EmbeddingGenerationError(RAGException):
    """Embedding generation failed."""
    
    def __init__(self, error: str):
        super().__init__(
            message=f"Failed to generate embeddings: {error}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": error}
        )


class QueryProcessingError(RAGException):
    """Query processing failed."""
    
    def __init__(self, query: str, error: str):
        super().__init__(
            message=f"Failed to process query: {error}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"query": query, "error": error}
        )


class RateLimitExceededError(RAGException):
    """Rate limit exceeded."""
    
    def __init__(self, limit: int, window: int):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window} seconds",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={"limit": limit, "window": window}
        )


class ValidationError(RAGException):
    """Validation error."""
    
    def __init__(self, field: str, message: str):
        super().__init__(
            message=f"Validation error for {field}: {message}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"field": field, "validation_error": message}
        )