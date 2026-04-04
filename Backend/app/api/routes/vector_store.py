# Backend/app/api/routes/vector_store.py
"""
API routes for vector store operations.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Optional, Dict, Any
import time

from loguru import logger

from app.core.vector_store import get_vector_store, VectorStore
from app.core.embeddings import get_embedding_service, EmbeddingService
from app.models.schemas import VectorStoreQueryRequest, VectorStoreQueryResponse
from app.models.schemas import VectorStoreQueryResult,VectorStoreStatsResponse,DocumentExistsResponse,AddEmbeddingsRequest

from app.services.query_service import get_query_service, QueryServiceError, QueryService

router = APIRouter()


@router.post(
    "/query",
    response_model=VectorStoreQueryResponse,
    summary="Query vector store",
    description="Search for similar documents in the vector store"
)
async def query_vector_store(request: VectorStoreQueryRequest,query_service: QueryService = Depends(get_query_service)):
    """
    Query the vector store for similar documents.

    - If `query_text` is provided, it will be embedded automatically.
    - If `query_embedding` is provided, it will be used directly.
    - Optional filters: `user_id`, `document_id`.
    - Optional `min_similarity` threshold (0-1).
    """
    try:
        if request.query_text:
            result = query_service.search(
                query_text=request.query_text,
                top_k=request.n_results,
                user_id=request.user_id,
                document_id=request.document_id,
                min_similarity=request.min_similarity,
            )
        elif request.query_embedding:
            result = query_service.search_by_embedding(
                query_embedding=request.query_embedding,
                top_k=request.n_results,
                user_id=request.user_id,
                document_id=request.document_id,
                min_similarity=request.min_similarity,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either query_text or query_embedding must be provided"
            )

        return VectorStoreQueryResponse(
            results=result["results"],
            total_results=result["total_results"],
            query_time=result["query_time"],
        )

    except QueryServiceError as e:
        logger.error(f"Query service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/stats",
    response_model=VectorStoreStatsResponse,
    summary="Get collection statistics",
    description="Get statistics about the vector store collection"
)
async def get_collection_stats(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Get vector store collection statistics."""
    try:
        stats = vector_store.get_collection_stats()
        
        return VectorStoreStatsResponse(
            collection_name=stats["collection_name"],
            total_chunks=stats["total_chunks"],
            persist_directory=stats["persist_directory"],
            has_data=stats["has_data"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/documents/{document_id}/exists",
    response_model=DocumentExistsResponse,
    summary="Check if document exists",
    description="Check if a document exists in the vector store"
)
async def check_document_exists(
    document_id: str,
    user_id: Optional[str] = None,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Check if a document exists in the vector store."""
    try:
        exists = vector_store.check_document_exists(document_id, user_id)
        
        # Get chunk count if exists
        chunk_count = None
        if exists:
            docs = vector_store.list_documents(user_id) if user_id else []
            for doc in docs:
                if doc["document_id"] == document_id:
                    chunk_count = doc["chunk_count"]
                    break
        
        return DocumentExistsResponse(
            document_id=document_id,
            exists=exists,
            chunk_count=chunk_count
        )
        
    except Exception as e:
        logger.error(f"Failed to check document existence: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete(
    "/documents/{document_id}",
    summary="Delete document from vector store",
    description="Delete all chunks for a document"
)
async def delete_document(
    document_id: str,
    user_id: Optional[str] = None,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Delete a document from the vector store."""
    try:
        deleted_count = vector_store.delete_by_document_id(document_id, user_id)
        
        return {
            "success": True,
            "document_id": document_id,
            "chunks_deleted": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/users/{user_id}/documents",
    summary="List user documents",
    description="List all documents for a user"
)
async def list_user_documents(
    user_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """List all documents for a user."""
    try:
        documents = vector_store.list_documents(user_id)
        
        return {
            "user_id": user_id,
            "total_documents": len(documents),
            "documents": documents
        }
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
        
@router.post("/add", response_model=Dict[str, Any], summary="Add embeddings to vector store")
async def add_embeddings(
    request: AddEmbeddingsRequest,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Add embeddings and corresponding chunks to the vector store.
    
    - **embeddings**: list of embedding vectors (each a list of floats)
    - **chunks**: list of TextChunk objects (must match embeddings length)
    - **user_id**: owner of the document
    - **document_id**: document identifier
    
    Returns a dictionary with:
    - success: bool
    - document_id: str
    - chunks_added: int
    - chunk_ids: List[str]
    """
    logger.info(f"Adding {len(request.embeddings)} embeddings for document {request.document_id} (user: {request.user_id})")
    
    try:
        result = vector_store.add_embeddings(
            embeddings=request.embeddings,
            chunks=request.chunks,
            user_id=request.user_id,
            document_id=request.document_id
        )
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to add embeddings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while adding embeddings")