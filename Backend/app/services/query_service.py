"""
Query service for semantic search and retrieval.
Combines embedding generation and vector store querying.
"""
import time
from typing import List, Optional, Dict, Any

from loguru import logger

from app.core.embeddings import get_embedding_service, EmbeddingService
from app.core.vector_store import get_vector_store, VectorStore, VectorStoreError
from app.models.schemas import VectorStoreQueryResult


class QueryServiceError(Exception):
    """Custom exception for query service errors."""
    pass


class QueryService:
    """
    Service for performing semantic search queries.
    """

    def __init__(self,embedding_service: Optional[EmbeddingService] = None,vector_store: Optional[VectorStore] = None,):
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()

    def search(self,query_text: str,top_k: int = 5,user_id: Optional[str] = None,
        document_id: Optional[str] = None,min_similarity: Optional[float] = None,) -> Dict[str, Any]:
        """
        Perform semantic search for the given query text.

        Args:
            query_text: The search query.
            top_k: Number of results to return.
            user_id: Optional filter by user.
            document_id: Optional filter by document.
            min_similarity: Optional minimum similarity threshold (0-1).

        Returns:
            Dictionary with:
                - results: List of VectorStoreQueryResult
                - total_results: int
                - query_time: float (seconds)
        """
        start_time = time.time()
        logger.info(f"Query search: '{query_text[:50]}...' (top_k={top_k})")

        # 1. Generate query embedding
        try:
            query_embedding = self.embedding_service.generate_query_embedding(query_text)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise QueryServiceError(f"Embedding generation failed: {e}")

        # 2. Query vector store
        try:
            raw_results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=top_k,
                user_id=user_id,
                document_id=document_id,
            )
        except VectorStoreError as e:
            logger.error(f"Vector store query failed: {e}")
            raise QueryServiceError(f"Vector store query failed: {e}")

        # 3. Format results and apply similarity threshold
        results = []
        for i in range(len(raw_results["ids"])):
            distance = raw_results["distances"][i]
            # Convert distance to similarity (0-1, higher is better)
            # Using 1/(1+distance) gives values in (0,1] and is monotonic.
            similarity = 1.0 / (1.0 + distance)

            if min_similarity is not None and similarity < min_similarity:
                continue

            result = VectorStoreQueryResult(
                chunk_id=raw_results["ids"][i],
                document=raw_results["documents"][i],
                metadata=raw_results["metadatas"][i],
                distance=distance,
                similarity=similarity,
            )
            results.append(result)

        query_time = time.time() - start_time
        logger.info(f"Query returned {len(results)} results in {query_time:.3f}s")

        return {
            "results": results,
            "total_results": len(results),
            "query_time": query_time,
        }

    def search_by_embedding(self,query_embedding: List[float],top_k: int = 5,user_id: Optional[str] = None,
        document_id: Optional[str] = None,min_similarity: Optional[float] = None,) -> Dict[str, Any]:
        """
        Search using a pre‑computed query embedding.
        """
        start_time = time.time()
        try:
            raw_results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=top_k,
                user_id=user_id,
                document_id=document_id,
            )
        except VectorStoreError as e:
            logger.error(f"Vector store query failed: {e}")
            raise QueryServiceError(f"Vector store query failed: {e}")

        results = []
        for i in range(len(raw_results["ids"])):
            distance = raw_results["distances"][i]
            similarity = 1.0 / (1.0 + distance)
            if min_similarity is not None and similarity < min_similarity:
                continue
            result = VectorStoreQueryResult(
                chunk_id=raw_results["ids"][i],
                document=raw_results["documents"][i],
                metadata=raw_results["metadatas"][i],
                distance=distance,
                similarity=similarity,
            )
            results.append(result)

        query_time = time.time() - start_time
        return {
            "results": results,
            "total_results": len(results),
            "query_time": query_time,
        }


# Global singleton instance
_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """Get or create the global QueryService instance."""
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service