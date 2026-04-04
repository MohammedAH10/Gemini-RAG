# Backend/app/core/vector_store.py
"""
Vector store service using ChromaDB for embedding storage and retrieval.
"""
import uuid
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from loguru import logger

from app.config import get_settings
from app.models.schemas import TextChunk

settings = get_settings()

class VectorStoreError(Exception):
    pass

class VectorStore:
    """
    Vector store for managing embeddings in ChromaDB.
    Handles storage, retrieval, and management of document embeddings.
    """
    
    def __init__(self,collection_name: Optional[str] = None,persist_directory: Optional[str] = None):
        """
        Initialize vector store with ChromaDB.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or str(
            settings.get_chroma_persist_directory()
        )
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(
            f"VectorStore initialized: collection='{self.collection_name}', "
            f"persist_dir='{self.persist_directory}'"
        )
    
    def _get_or_create_collection(self):
        """
        Get existing collection or create new one.
        
        Returns:
            ChromaDB collection
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document embeddings"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection
    
    def add_embeddings(self,embeddings: List[List[float]],chunks: List[TextChunk],
        user_id: str,document_id: str) -> Dict[str, Any]:
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of TextChunk objects
            user_id: User identifier
            document_id: Document identifier
            
        Returns:
            Dictionary with storage confirmation
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        if not embeddings:
            raise ValueError("No embeddings provided")
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "user_id": user_id,
                "document_id": document_id,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        try:
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(
                f"Added {len(embeddings)} embeddings for document {document_id} "
                f"(user: {user_id})"
            )
            
            return {
                "success": True,
                "document_id": document_id,
                "chunks_added": len(embeddings),
                "chunk_ids": ids
            }
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise
    
    def query(self,query_embedding: List[float],n_results: int = 5,user_id: Optional[str] = None,
        document_id: Optional[str] = None,filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the vector store for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            user_id: Optional user ID filter
            document_id: Optional document ID filter
            filter_metadata: Additional metadata filters
            
        Returns:
            Dictionary with query results
        """
        # Build where clause for filtering
        conditions = []
        
        if user_id:
            conditions.append({"user_id": {"$eq": user_id}})
        
        if document_id:
            conditions.append({"document_id": {"$eq": document_id}})
        
        if filter_metadata:
            for key, value in filter_metadata.items():
                conditions.append({key: {"$eq": value}})
        
        if len(conditions) == 0:
            where = None
        elif len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding], 
                n_results=n_results,
                where=where if where else None,
                include=["embeddings", "documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "total_results": len(results["ids"][0]) if results["ids"] else 0
            }
            
            logger.info(
                f"Query returned {formatted_results['total_results']} results"
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
        
        Returns:
            Chunk data or None if not found
        """
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["embeddings", "documents", "metadatas"]
            )
        
            # Check if results exist
            if not result["ids"] or len(result["ids"]) == 0:
                return None
        
            return {
                "id": result["ids"][0],
                "document": result["documents"][0] if result["documents"] else None,
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
                "embedding": result["embeddings"][0] if result["embeddings"] is not None and len(result["embeddings"]) > 0 else None
            }
        
        except Exception as e:
            logger.error(f"Failed to get chunk by ID: {e}")
            return None
    
    def delete_by_document_id(self, document_id: str, user_id: Optional[str] = None) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document identifier
            user_id: Optional user ID for additional filtering
            
        Returns:
            Number of chunks deleted
        """
        try:
            #  build where clause with proper chroma syntax
            if user_id:
                where = {
                    "$and": [
                        {"document_id": {"$eq": document_id}},
                        {"user_id": {"$eq": user_id}}
                    ]   
                }
            else:
                where = {"document_id": {"$eq": document_id}}
            
            # Get IDs to delete
            results = self.collection.get(
                where=where,
                include=[]
            )
            
            ids_to_delete = results["ids"]
            
            if not ids_to_delete:
                logger.info(f"No chunks found for document {document_id}")
                return 0
            
            # Delete
            self.collection.delete(ids=ids_to_delete)
            
            logger.info(f"Deleted {len(ids_to_delete)} chunks for document {document_id}")
            
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise
    
    def delete_by_user_id(self, user_id: str) -> int:
        """
        Delete all chunks for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of chunks deleted
        """
        try:
            # Get IDs to delete
            results = self.collection.get(
                where={"user_id": user_id},
                include=[]
            )
            
            ids_to_delete = results["ids"]
            
            if not ids_to_delete:
                logger.info(f"No chunks found for user {user_id}")
                return 0
            
            # Delete
            self.collection.delete(ids=ids_to_delete)
            
            logger.info(f"Deleted {len(ids_to_delete)} chunks for user {user_id}")
            
            return len(ids_to_delete)
            
        except Exception as e:
            logger.error(f"Failed to delete user data: {e}")
            raise
    
    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all documents for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of document information
        """
        try:
            # Get all chunks for user
            results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            if not results["metadatas"]:
                return []
            
            # Group by document_id
            documents = {}
            for metadata in results["metadatas"]:
                doc_id = metadata.get("document_id")
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "user_id": user_id,
                        "chunk_count": 0
                    }
                if doc_id:
                    documents[doc_id]["chunk_count"] += 1
            
            document_list = list(documents.values())
            
            logger.info(f"Found {len(document_list)} documents for user {user_id}")
            
            return document_list
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample to check structure
            sample = self.collection.peek(limit=1)
            
            stats = {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": self.persist_directory,
                "has_data": count > 0
            }
            
            logger.info(f"Collection stats: {count} chunks")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise
    
    def check_document_exists(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """
        Check if a document exists in the vector store.
        """
        try:
            # ChromaDB requires AND operator for multiple conditions
            if user_id:
                where = {
                    "$and": [
                        {"document_id": {"$eq": document_id}},
                        {"user_id": {"$eq": user_id}}
                    ]
                }
            else:
                where = {"document_id": {"$eq": document_id}}
            
            results = self.collection.get(
                where=where,
                limit=1,
                include=[]
            )
            
            exists = len(results["ids"]) > 0
            
            return exists
            
        except Exception as e:
            logger.error(f"Failed to check document existence: {e}")
            return False
        
        
    def reset_collection(self):
        """
        Delete and recreate the collection.
        WARNING: This deletes all data!
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.warning(f"Collection '{self.collection_name}' has been reset")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise


# Global instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get or create global vector store instance.
    
    Returns:
        VectorStore instance
    """
    global _vector_store
    
    if _vector_store is None:
        _vector_store = VectorStore()
    
    return _vector_store