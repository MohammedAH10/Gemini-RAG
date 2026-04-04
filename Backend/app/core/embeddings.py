"""
Embedding generation and management service.
"""
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from loguru import logger

from app.core.llm_client import get_gemini_client, GeminiClient
from app.models.schemas import TextChunk


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    chunk_id: str
    embedding: List[float]
    dimensions: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding generation."""
    embeddings: List[EmbeddingResult]
    total_chunks: int
    successful: int
    failed: int
    total_time: float
    average_time: float


class EmbeddingService:
    """
    Service for generating and managing embeddings.
    """
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        Initialize embedding service.
        
        Args:
            gemini_client: Gemini client instance (creates new if not provided)
        """
        self.client = gemini_client or get_gemini_client()
        self.embedding_dimensions = self.client.embedding_dimensions
        
        logger.info(f"EmbeddingService initialized with {self.embedding_dimensions}D embeddings")
    
    def generate_embedding(self, text: str, chunk_id: Optional[str] = None, task_type: str = "retrieval_document") -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            chunk_id: Optional chunk identifier
            task_type: Task type for embedding
            
        Returns:
            EmbeddingResult with embedding and metadata
        """
        start_time = time.time()
        chunk_id = chunk_id or "unknown"
        
        try:
            embedding = self.client.generate_embedding(text, task_type)
            processing_time = time.time() - start_time
            
            logger.info(
                f"Generated embedding for {chunk_id}: "
                f"{len(embedding)}D in {processing_time:.3f}s"
            )
            
            return EmbeddingResult(
                chunk_id=chunk_id,
                embedding=embedding,
                dimensions=len(embedding),
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to generate embedding for {chunk_id}: {e}")
            
            return EmbeddingResult(
                chunk_id=chunk_id,
                embedding=[0.0] * self.embedding_dimensions,
                dimensions=self.embedding_dimensions,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def generate_embeddings_for_chunks(self, chunks: List[TextChunk], task_type: str = "retrieval_document", batch_size: int = 100) -> BatchEmbeddingResult:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of TextChunk objects
            task_type: Task type for embedding
            batch_size: Batch size for processing
            
        Returns:
            BatchEmbeddingResult with all embeddings and statistics
        """
        start_time = time.time()
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts and chunk IDs
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Generate embeddings in batches
        try:
            embeddings = self.client.generate_embeddings_batch(
                texts=texts,
                task_type=task_type,
                batch_size=batch_size
            )
            
            # Create results
            results = []
            for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
                # Check if embedding is valid (not a zero vector from error)
                is_zero = all(x == 0.0 for x in embedding)
                success = not is_zero
                
                result = EmbeddingResult(
                    chunk_id=chunk_id,
                    embedding=embedding,
                    dimensions=len(embedding),
                    processing_time=0,  # Individual time not tracked in batch
                    success=success,
                    error_message=None if success else "Batch processing error"
                )
                results.append(result)
            
            total_time = time.time() - start_time
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            logger.info(
                f"Batch embedding complete: {successful}/{len(chunks)} successful "
                f"in {total_time:.2f}s"
            )
            
            return BatchEmbeddingResult(
                embeddings=results,
                total_chunks=len(chunks),
                successful=successful,
                failed=failed,
                total_time=total_time,
                average_time=total_time / len(chunks) if chunks else 0
            )
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            
            # Fall back to individual processing
            return self._generate_individually(chunks, task_type, start_time)
    
    def _generate_individually(self, chunks: List[TextChunk], task_type: str, start_time: float) -> BatchEmbeddingResult:
        """
        Generate embeddings one by one (fallback method).
        
        Args:
            chunks: List of chunks
            task_type: Task type
            start_time: Start timestamp
            
        Returns:
            BatchEmbeddingResult
        """
        logger.info("Falling back to individual embedding generation")
        
        results = []
        for chunk in chunks:
            result = self.generate_embedding(
                text=chunk.text,
                chunk_id=chunk.chunk_id,
                task_type=task_type
            )
            results.append(result)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        return BatchEmbeddingResult(
            embeddings=results,
            total_chunks=len(chunks),
            successful=successful,
            failed=failed,
            total_time=total_time,
            average_time=total_time / len(chunks) if chunks else 0
        )
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector
        """
        logger.info(f"Generating query embedding: {query[:50]}...")
        
        embedding = self.client.generate_embedding(
            text=query,
            task_type="retrieval_query"
        )
        
        return embedding
    
    async def generate_embedding_async(self, text: str, chunk_id: Optional[str] = None, task_type: str = "retrieval_document") -> EmbeddingResult:
        """
        Generate embedding asynchronously.
        
        Args:
            text: Text to embed
            chunk_id: Optional chunk ID
            task_type: Task type
            
        Returns:
            EmbeddingResult
        """
        start_time = time.time()
        chunk_id = chunk_id or "unknown"
        
        try:
            embedding = await self.client.generate_embedding_async(text, task_type)
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                chunk_id=chunk_id,
                embedding=embedding,
                dimensions=len(embedding),
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                chunk_id=chunk_id,
                embedding=[0.0] * self.embedding_dimensions,
                dimensions=self.embedding_dimensions,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate embedding vector.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not embedding:
            return False
        
        if len(embedding) != self.embedding_dimensions:
            logger.warning(
                f"Invalid embedding dimensions: expected {self.embedding_dimensions}, "
                f"got {len(embedding)}"
            )
            return False
        
        # Check if it's not a zero vector
        if all(x == 0.0 for x in embedding):
            logger.warning("Embedding is a zero vector")
            return False
        
        # Check for NaN or Inf values
        if any(not isinstance(x, (int, float)) or x != x or abs(x) == float('inf') for x in embedding):
            logger.warning("Embedding contains invalid values (NaN or Inf)")
            return False
        
        return True
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get embedding service information.
        
        Returns:
            Dictionary with service info
        """
        return {
            "embedding_dimensions": self.embedding_dimensions,
            "model": self.client.embedding_model,
            "service": "Google Gemini"
        }


# Global service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create global embedding service instance.
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    
    return _embedding_service