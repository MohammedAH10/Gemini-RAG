# Backend/app/api/routes/embeddings.py
"""
API routes for embedding generation.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from typing import List

from loguru import logger

from app.core.embeddings import get_embedding_service, EmbeddingService
from app.models.schemas import EmbeddingRequest, EmbeddingResponse, BatchEmbeddingRequest, BatchEmbeddingResponse

router = APIRouter()


@router.post(
    "/generate",
    response_model=EmbeddingResponse,
    summary="Generate embedding for text",
    description="Generate vector embedding for a single text input"
)

async def generate_embedding(request: EmbeddingRequest, service: EmbeddingService = Depends(get_embedding_service)):
    """
    Generate embedding for a single text.
    
    Args:
        request: Embedding request
        service: Embedding service
        
    Returns:
        EmbeddingResponse with vector and metadata
    """
    logger.info(f"Embedding request for chunk: {request.chunk_id or 'unknown'}")
    
    try:
        result = service.generate_embedding(
            text=request.text,
            chunk_id=request.chunk_id,
            task_type=request.task_type
        )
        
        response = EmbeddingResponse(
            chunk_id=result.chunk_id,
            embedding=result.embedding,
            dimensions=result.dimensions,
            processing_time=result.processing_time,
            success=result.success,
            error_message=result.error_message
        )
        
        if not result.success:
            logger.warning(f"Embedding generation failed: {result.error_message}")
        
        return response
        
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embedding: {str(e)}"
        )


@router.post(
    "/generate/batch",
    response_model=BatchEmbeddingResponse,
    summary="Generate embeddings for multiple texts",
    description="Generate vector embeddings for a batch of texts"
)


async def generate_embeddings_batch(request: BatchEmbeddingRequest, service: EmbeddingService = Depends(get_embedding_service)):
    """
    Generate embeddings for multiple texts.
    
    Args:
        request: Batch embedding request
        service: Embedding service
        
    Returns:
        BatchEmbeddingResponse with all embeddings
    """
    logger.info(f"Batch embedding request for {len(request.texts)} texts")
    
    if request.chunk_ids and len(request.chunk_ids) != len(request.texts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Number of chunk_ids must match number of texts"
        )
    
    try:
        # Create mock TextChunk objects
        from app.models.schemas import TextChunk
        
        chunks = []
        for i, text in enumerate(request.texts):
            chunk_id = (
                request.chunk_ids[i] if request.chunk_ids
                else f"chunk_{i}"
            )
            
            chunk = TextChunk(
                chunk_id=chunk_id,
                document_id="batch_request",
                text=text,
                chunk_index=i,
                start_char=0,
                end_char=len(text),
                token_count=len(text.split())
            )
            chunks.append(chunk)
        
        # Generate embeddings
        result = service.generate_embeddings_for_chunks(
            chunks=chunks,
            task_type=request.task_type,
            batch_size=request.batch_size
        )
        
        # Convert to response format
        embedding_responses = [
            EmbeddingResponse(
                chunk_id=emb.chunk_id,
                embedding=emb.embedding,
                dimensions=emb.dimensions,
                processing_time=emb.processing_time,
                success=emb.success,
                error_message=emb.error_message
            )
            for emb in result.embeddings
        ]
        
        response = BatchEmbeddingResponse(
            embeddings=embedding_responses,
            total_chunks=result.total_chunks,
            successful=result.successful,
            failed=result.failed,
            total_time=result.total_time,
            average_time=result.average_time
        )
        
        logger.info(
            f"Batch complete: {result.successful}/{result.total_chunks} successful"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}"
        )


@router.get(
    "/info",
    summary="Get embedding service information",
    description="Get information about the embedding service and model"
)


async def get_embedding_info(service: EmbeddingService = Depends(get_embedding_service)):
    """Get embedding service information."""
    return service.get_embedding_info()