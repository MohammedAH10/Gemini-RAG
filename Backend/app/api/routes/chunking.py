# Backend/app/api/routes/chunking.py
"""
API routes for text chunking operations.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Optional

from loguru import logger

from app.core.rag_engine import get_rag_engine, RAGEngine
from app.models.schemas import ChunkingRequest, ChunkingResult, TextChunk

router = APIRouter()

@router.post("/chunk", response_model=ChunkingResult, summary="Chunk text into segments", description="Split text into smaller chunks for embedding and retrieval")

async def chunk_text(request: ChunkingRequest, rag_engine: RAGEngine = Depends(get_rag_engine)):
    """
    Chunk text into segments.
    Args:
        request: Chunking request with text and parameters
        rag_engine: RAG engine instance
        
    Returns:
        ChunkingResult with all chunks
    """
    logger.info(f"Chunking request for document {request.document_id}")
    
    try:
        # Get custom parameters or use defaults
        if request.chunk_size or request.chunk_overlap:
            rag_engine = get_rag_engine(chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap)
        
        # Chunk text
        result = rag_engine.chunk_text(text=request.text, document_id=request.document_id)
        
        logger.info(f"Chunking complete: {result.total_chunks} chunks created")
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to chunk text: {str(e)}")


@router.post("/chunk/stats",summary="Get chunking statistics", description="Get statistics about chunks without returning full chunk data")
async def get_chunk_statistics(request: ChunkingRequest, rag_engine: RAGEngine = Depends(get_rag_engine)):
    """
    Get chunking statistics.
    Args:
        request: Chunking request
        rag_engine: RAG engine instance
        
    Returns:
        Statistics about the chunking operation
    """
    try:
        # Chunk text
        result = rag_engine.chunk_text(text=request.text, document_id=request.document_id)
        
        # Get statistics
        stats = rag_engine.get_chunk_stats(result.chunks)
        overlap_stats = rag_engine.verify_chunk_overlap(result.chunks)
        
        return {
            "document_id": result.document_id,
            "chunk_stats": stats,
            "overlap_stats": overlap_stats,
            "configuration": {
                "chunk_size": result.chunk_size,
                "chunk_overlap": result.chunk_overlap
            },
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/config", summary="Get chunking configuration", description="Get current chunking configuration")
async def get_chunking_config():
    """Get current chunking configuration."""
    rag_engine = get_rag_engine()
    
    return {
        "chunk_size": rag_engine.chunk_size,
        "chunk_overlap": rag_engine.chunk_overlap,
        "text_splitter": "SentenceSplitter"
    }