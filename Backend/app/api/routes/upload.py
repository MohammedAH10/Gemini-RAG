"""
API routes for document upload and processing.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Optional, List
import json

from loguru import logger

from app.services.document_processor import get_document_processor
from app.models.schemas import (DocumentUploadResponse, DocumentExtractionResult, DocumentStatus)

router = APIRouter()

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a document",
    description="Upload a document file (PDF, EPUB, TXT, DOCX, MOBI, AZW) for text extraction"
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    title: Optional[str] = Form(None, description="Custom document title"),
    tags: Optional[str] = Form(None, description="Comma-separated tags")
):
    """
    Upload and process a document file.
    
    Supported formats: PDF, EPUB, TXT, DOCX, MOBI, AZW, AZW3
    
    Args:
        file: Document file
        title: Optional custom title
        tags: Optional comma-separated tags
        
    Returns:
        DocumentUploadResponse with processing status
    """
    logger.info(f"Upload request received: {file.filename}")
    
    try:
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Get processor
        processor = get_document_processor()
        
        # Process document
        result = await processor.process_document(
            file=file,
            title=title,
            tags=tag_list
        )
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message or "Document processing failed"
            )
        
        # Create response
        response = DocumentUploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            file_type=result.file_type,
            file_size=result.metadata.file_size,
            status=DocumentStatus.COMPLETED,
            message="Document uploaded and processed successfully",
            metadata=result.metadata
        )
        
        logger.info(f"Upload successful: {result.document_id}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@router.post(
    "/upload/batch",
    response_model=List[DocumentUploadResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload multiple documents",
    description="Upload multiple document files at once"
)
async def upload_documents_batch(
    files: List[UploadFile] = File(..., description="Multiple document files")
):
    """
    Upload and process multiple documents.
    
    Args:
        files: List of document files
        
    Returns:
        List of DocumentUploadResponse
    """
    logger.info(f"Batch upload request received: {len(files)} files")
    
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 files allowed per batch upload"
        )
    
    processor = get_document_processor()
    responses = []
    
    for file in files:
        try:
            result = await processor.process_document(file=file)
            
            if result.success:
                response = DocumentUploadResponse(
                    document_id=result.document_id,
                    filename=result.filename,
                    file_type=result.file_type,
                    file_size=result.metadata.file_size,
                    status=DocumentStatus.COMPLETED,
                    message="Document uploaded and processed successfully",
                    metadata=result.metadata
                )
            else:
                response = DocumentUploadResponse(
                    document_id="",
                    filename=result.filename,
                    file_type=result.file_type,
                    file_size=0,
                    status=DocumentStatus.FAILED,
                    message=result.error_message or "Processing failed",
                    metadata=result.metadata
                )
            
            responses.append(response)
            
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            # Continue with other files
            continue
    
    logger.info(f"Batch upload complete: {len(responses)} processed")
    
    return responses


@router.get(
    "/supported-formats",
    summary="Get supported file formats",
    description="Get list of supported document formats"
)

async def get_supported_formats():
    """Get list of supported document formats."""
    processor = get_document_processor()
    
    return {
        "supported_formats": list(processor.SUPPORTED_FORMATS.keys()),
        "max_file_size_mb": processor.max_file_size / (1024 * 1024),
        "details": processor.SUPPORTED_FORMATS
    }