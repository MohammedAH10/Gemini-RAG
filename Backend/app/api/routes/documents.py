"""
API routes for document management.
"""
from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form, Query
from typing import Optional, List
from pathlib import Path
import uuid
from datetime import datetime

from loguru import logger

from app.services.document_processor import DocumentProcessor
from app.services.indexing_service import get_indexing_service, IndexingService
from app.core.vector_store import get_vector_store, VectorStore
from app.utils.auth import get_current_user, check_rate_limit
from app.models.database import (
    DocumentDB,
    DocumentStatus,
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentDeleteResponse,
    DocumentUpdate
)
from app.config import get_settings

settings = get_settings()
router = APIRouter()

# In-memory document storage (replace with Supabase in production)
_documents_db: dict = {}


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index document",
    description="Upload a document file and automatically index it for RAG"
)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user),
    indexing_service: IndexingService = Depends(get_indexing_service)
):
    """
    Upload and index a document.
    
    Args:
        file: Document file
        title: Optional document title
        tags: Comma-separated tags
        user_id: Current user ID
        indexing_service: Indexing service
        
    Returns:
        Document upload and indexing result
    """
    # Check rate limit
    await check_rate_limit(user_id, "upload")
    
    logger.info(f"Upload request from user {user_id}: {file.filename}")
    
    try:
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Process document
        processor = DocumentProcessor()
        
        # Validate file
        is_valid, error = processor.validate_file(file.filename, file.size)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error
            )
        
        # Save file
        file_path = processor.save_file(file, document_id)
        
        # Create document record
        doc_title = title or file.filename
        doc_tags = tags.split(",") if tags else []
        
        document = DocumentDB(
            id=document_id,
            user_id=user_id,
            title=doc_title,
            filename=file.filename,
            file_type=Path(file.filename).suffix.lstrip("."),
            file_size=file.size,
            file_path=str(file_path),
            status=DocumentStatus.PROCESSING,
            tags=doc_tags,
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store in database
        _documents_db[document_id] = document
        
        # Index document
        index_result = indexing_service.index_document(
            document_id=document_id,
            user_id=user_id,
            file_path=str(file_path),
            title=doc_title
        )
        
        # Update document status
        if index_result["success"]:
            document.status = DocumentStatus.COMPLETED
            document.chunk_count = index_result["chunk_count"]
            document.total_tokens = index_result["total_tokens"]
            document.indexed_at = datetime.now()
        else:
            document.status = DocumentStatus.FAILED
            document.error_message = index_result.get("error")
        
        document.updated_at = datetime.now()
        _documents_db[document_id] = document
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "title": doc_title,
            "status": document.status,
            "chunk_count": document.chunk_count,
            "total_tokens": document.total_tokens,
            "indexing_time": index_result.get("total_time", 0),
            "message": "Document uploaded and indexed successfully" if index_result["success"] else "Indexing failed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List user documents",
    description="Get paginated list of documents for the current user"
)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[DocumentStatus] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search in title"),
    user_id: str = Depends(get_current_user)
):
    """
    List documents for current user with pagination.
    
    Args:
        page: Page number
        page_size: Items per page
        status_filter: Optional status filter
        search: Optional search query
        user_id: Current user ID
        
    Returns:
        Paginated document list
    """
    logger.info(f"List documents for user {user_id}, page {page}")
    
    try:
        # Filter user's documents
        user_docs = [
            doc for doc in _documents_db.values()
            if doc.user_id == user_id
        ]
        
        # Apply filters
        if status_filter:
            user_docs = [doc for doc in user_docs if doc.status == status_filter]
        
        if search:
            search_lower = search.lower()
            user_docs = [
                doc for doc in user_docs
                if search_lower in doc.title.lower() or search_lower in doc.filename.lower()
            ]
        
        # Sort by created_at descending
        user_docs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Pagination
        total = len(user_docs)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_docs = user_docs[start_idx:end_idx]
        total_pages = (total + page_size - 1) // page_size
        
        return DocumentListResponse(
            documents=paginated_docs,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details",
    description="Get detailed information about a specific document"
)
async def get_document(
    document_id: str,
    user_id: str = Depends(get_current_user),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Get document details.
    
    Args:
        document_id: Document ID
        user_id: Current user ID
        vector_store: Vector store
        
    Returns:
        Document details with chunks preview
    """
    logger.info(f"Get document {document_id} for user {user_id}")
    
    # Check if document exists
    if document_id not in _documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    document = _documents_db[document_id]
    
    # Check authorization
    if document.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this document"
        )
    
    # Get chunks preview from vector store
    chunks_preview = None
    try:
        query_result = vector_store.query(
            query_embedding=[0.0] * 768,  # Dummy query just to get chunks
            n_results=3,
            document_id=document_id
        )
        
        chunks_preview = query_result.get("documents", [])[:3]
    except Exception as e:
        logger.warning(f"Failed to get chunks preview: {e}")
    
    return DocumentDetailResponse(
        document=document,
        chunks_preview=chunks_preview
    )


@router.patch(
    "/{document_id}",
    summary="Update document metadata",
    description="Update document title, tags, or metadata"
)
async def update_document(
    document_id: str,
    update_data: DocumentUpdate,
    user_id: str = Depends(get_current_user)
):
    """
    Update document metadata.
    
    Args:
        document_id: Document ID
        update_data: Update data
        user_id: Current user ID
        
    Returns:
        Updated document
    """
    logger.info(f"Update document {document_id} for user {user_id}")
    
    # Check if document exists
    if document_id not in _documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    document = _documents_db[document_id]
    
    # Check authorization
    if document.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this document"
        )
    
    # Update fields
    if update_data.title is not None:
        document.title = update_data.title
    
    if update_data.tags is not None:
        document.tags = update_data.tags
    
    if update_data.metadata is not None:
        document.metadata.update(update_data.metadata)
    
    document.updated_at = datetime.now()
    _documents_db[document_id] = document
    
    return {
        "success": True,
        "document_id": document_id,
        "message": "Document updated successfully",
        "document": document
    }


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete document",
    description="Delete document from database and vector store"
)
async def delete_document(
    document_id: str,
    user_id: str = Depends(get_current_user),
    indexing_service: IndexingService = Depends(get_indexing_service)
):
    """
    Delete document.
    
    Args:
        document_id: Document ID
        user_id: Current user ID
        indexing_service: Indexing service
        
    Returns:
        Deletion confirmation
    """
    logger.info(f"Delete document {document_id} for user {user_id}")
    
    # Check if document exists
    if document_id not in _documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    document = _documents_db[document_id]
    
    # Check authorization
    if document.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this document"
        )
    
    try:
        # Delete from vector store
        delete_result = indexing_service.delete_document_index(
            document_id=document_id,
            user_id=user_id
        )
        
        # Delete file from storage
        try:
            Path(document.file_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete file: {e}")
        
        # Delete from database
        del _documents_db[document_id]
        
        return DocumentDeleteResponse(
            success=True,
            document_id=document_id,
            chunks_deleted=delete_result.get("chunks_deleted", 0),
            message="Document deleted successfully"
        )
        
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {str(e)}"
        )


@router.get(
    "/{document_id}/stats",
    summary="Get document statistics",
    description="Get indexing and usage statistics for a document"
)
async def get_document_stats(
    document_id: str,
    user_id: str = Depends(get_current_user),
    indexing_service: IndexingService = Depends(get_indexing_service)
):
    """
    Get document statistics.
    
    Args:
        document_id: Document ID
        user_id: Current user ID
        indexing_service: Indexing service
        
    Returns:
        Document statistics
    """
    # Check if document exists
    if document_id not in _documents_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    document = _documents_db[document_id]
    
    # Check authorization
    if document.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this document"
        )
    
    # Get indexing stats
    stats = indexing_service.get_document_stats(document_id, user_id)
    
    return {
        "document_id": document_id,
        "title": document.title,
        "status": document.status,
        "chunk_count": document.chunk_count,
        "total_tokens": document.total_tokens,
        "file_size": document.file_size,
        "created_at": document.created_at,
        "indexed_at": document.indexed_at,
        "vector_store_stats": stats
    }