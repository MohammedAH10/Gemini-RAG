"""
Document indexing service.
Orchestrates the complete document processing pipeline.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

from loguru import logger

from app.core.rag_engine import get_rag_engine, RAGEngine
from app.core.embeddings import get_embedding_service, EmbeddingService
from app.core.vector_store import get_vector_store, VectorStore
from app.services.document_processor import DocumentProcessor
from app.models.database import DocumentStatus


class IndexingService:
    """
    Service for indexing documents into the RAG system.
    Handles: chunking → embedding → vector storage
    """
    
    def __init__(self):
        """Initialize indexing service."""
        self.rag_engine: RAGEngine = get_rag_engine()
        self.embedding_service: EmbeddingService = get_embedding_service()
        self.vector_store: VectorStore = get_vector_store()
        
        logger.info("IndexingService initialized")
    
    def index_document(
        self,
        document_id: str,
        user_id: str,
        file_path: str,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index a document through the complete pipeline.
        
        Args:
            document_id: Document identifier
            user_id: User identifier
            file_path: Path to document file
            title: Optional document title
            
        Returns:
            Indexing result with statistics
        """
        start_time = time.time()
        
        logger.info(f"Starting indexing for document {document_id}")
        
        try:
            # Step 1: Extract text from document
            processor = DocumentProcessor()
            
            doc_metadata = processor.process_document(file_path)
            text = doc_metadata["text"]
            
            # Step 2: Chunk the text
            chunk_result = self.rag_engine.chunk_text(
                text=text,
                document_id=document_id
            )
            
            chunks = chunk_result["chunks"]
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No chunks generated from document",
                    "status": DocumentStatus.FAILED
                }
            
            # Step 3: Generate embeddings
            embedding_result = self.embedding_service.generate_embeddings_for_chunks(
                chunks=chunks,
                task_type="retrieval_document"
            )
            
            if embedding_result.failed > 0:
                logger.warning(
                    f"{embedding_result.failed}/{embedding_result.total_chunks} "
                    f"embeddings failed"
                )
            
            # Step 4: Store in vector database
            embeddings = [emb.embedding for emb in embedding_result.embeddings]
            
            vector_result = self.vector_store.add_embeddings(
                embeddings=embeddings,
                chunks=chunks,
                user_id=user_id,
                document_id=document_id
            )
            
            total_time = time.time() - start_time
            
            logger.info(
                f"Indexing complete for {document_id}: "
                f"{len(chunks)} chunks in {total_time:.2f}s"
            )
            
            return {
                "success": True,
                "document_id": document_id,
                "status": DocumentStatus.COMPLETED,
                "chunk_count": len(chunks),
                "total_tokens": sum(chunk.token_count for chunk in chunks),
                "embedding_stats": {
                    "successful": embedding_result.successful,
                    "failed": embedding_result.failed,
                    "total_time": embedding_result.total_time
                },
                "vector_stats": vector_result,
                "total_time": total_time,
                "metadata": {
                    "word_count": doc_metadata.get("word_count", 0),
                    "page_count": doc_metadata.get("page_count", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Indexing failed for {document_id}: {e}")
            
            return {
                "success": False,
                "document_id": document_id,
                "status": DocumentStatus.FAILED,
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    def reindex_document(
        self,
        document_id: str,
        user_id: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Reindex a document (delete old data and reindex).
        
        Args:
            document_id: Document identifier
            user_id: User identifier
            file_path: Path to document file
            
        Returns:
            Reindexing result
        """
        logger.info(f"Reindexing document {document_id}")
        
        try:
            # Delete existing data
            self.delete_document_index(document_id, user_id)
            
            # Reindex
            return self.index_document(document_id, user_id, file_path)
            
        except Exception as e:
            logger.error(f"Reindexing failed for {document_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_document_index(
        self,
        document_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Delete document from vector store.
        
        Args:
            document_id: Document identifier
            user_id: User identifier
            
        Returns:
            Deletion result
        """
        logger.info(f"Deleting index for document {document_id}")
        
        try:
            chunks_deleted = self.vector_store.delete_by_document_id(
                document_id=document_id,
                user_id=user_id
            )
            
            return {
                "success": True,
                "document_id": document_id,
                "chunks_deleted": chunks_deleted
            }
            
        except Exception as e:
            logger.error(f"Index deletion failed for {document_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_document_stats(
        self,
        document_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics for a document.
        
        Args:
            document_id: Document identifier
            user_id: User identifier
            
        Returns:
            Document statistics
        """
        try:
            # Check if document exists in vector store
            exists = self.vector_store.check_document_exists(document_id, user_id)
            
            if not exists:
                return {
                    "exists": False,
                    "chunk_count": 0
                }
            
            # Get chunks
            docs = self.vector_store.list_documents(user_id)
            
            for doc in docs:
                if doc["document_id"] == document_id:
                    return {
                        "exists": True,
                        "chunk_count": doc["chunk_count"]
                    }
            
            return {
                "exists": False,
                "chunk_count": 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for {document_id}: {e}")
            return {
                "exists": False,
                "error": str(e)
            }


# Global instance
_indexing_service: Optional[IndexingService] = None


def get_indexing_service() -> IndexingService:
    """
    Get or create global indexing service instance.
    
    Returns:
        IndexingService instance
    """
    global _indexing_service
    
    if _indexing_service is None:
        _indexing_service = IndexingService()
    
    return _indexing_service