"""
FastAPI dependencies for API routes.
"""
from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from loguru import logger

from app.utils.auth import get_current_user
from app.core.rag_engine import get_rag_engine, RAGEngine
from app.core.vector_store import get_vector_store, VectorStore
from app.core.embeddings import get_embedding_service, EmbeddingService
from app.services.supabase_service import get_supabase_service, SupabaseService
from app.services.indexing_service import get_indexing_service, IndexingService


# Common dependencies
def get_user_id(user_id: str = Depends(get_current_user)) -> str:
    """Get current user ID."""
    return user_id


def get_rag(rag_engine: RAGEngine = Depends(get_rag_engine)) -> RAGEngine:
    """Get RAG engine."""
    return rag_engine


def get_vectors(vector_store: VectorStore = Depends(get_vector_store)) -> VectorStore:
    """Get vector store."""
    return vector_store


def get_embeddings(service: EmbeddingService = Depends(get_embedding_service)) -> EmbeddingService:
    """Get embedding service."""
    return service


def get_supabase(service: SupabaseService = Depends(get_supabase_service)) -> SupabaseService:
    """Get Supabase service."""
    return service


def get_indexing(service: IndexingService = Depends(get_indexing_service)) -> IndexingService:
    """Get indexing service."""
    return service


# Request validation
async def validate_content_type(request: Request):
    """Validate request content type."""
    content_type = request.headers.get("content-type", "")
    
    if request.method in ["POST", "PUT", "PATCH"]:
        if not content_type.startswith("application/json") and not content_type.startswith("multipart/form-data"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Content-Type must be application/json or multipart/form-data"
            )


# Pagination
class PaginationParams:
    """Pagination parameters."""
    
    def __init__(self, page: int = 1, page_size: int = 20):
        self.page = max(1, page)
        self.page_size = min(max(1, page_size), 100)
        self.offset = (self.page - 1) * self.page_size