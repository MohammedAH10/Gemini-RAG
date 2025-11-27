import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from config.constants import CHROMA_COLLECTION_NAME, CHROMA_DIR
from config.settings import settings

logger = logging.getLogger(__name__)


class ChromaService:
    """Service for ChromaDB vector database operations"""

    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_function = None

        self._initialize_chroma()

    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
            )

            # Initialize embedding function (using default for now)
            # In production, you might want to use Gemini embeddings
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                f"ChromaService initialized with collection: {CHROMA_COLLECTION_NAME}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaService: {e}")
            raise

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add documents to ChromaDB collection

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs

        Returns:
            Operation result
        """
        try:
            if not documents:
                return {"success": False, "error": "No documents provided"}

            # Generate IDs if not provided
            if not ids:
                ids = [str(uuid.uuid4()) for _ in documents]

            # Prepare metadata
            if not metadatas:
                metadatas = [{} for _ in documents]

            # Ensure all metadata has timestamp
            for metadata in metadatas:
                if "timestamp" not in metadata:
                    metadata["timestamp"] = datetime.utcnow().isoformat()

            # Add to collection
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

            logger.info(f"Added {len(documents)} documents to ChromaDB")

            return {"success": True, "document_count": len(documents), "ids": ids}

        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return {"success": False, "error": str(e)}

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Query the ChromaDB collection

        Args:
            query_text: Query string
            n_results: Number of results to return
            where: Optional metadata filter
            where_document: Optional document content filter

        Returns:
            Query results
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                where_document=where_document,
            )

            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ):
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "document": doc,
                            "metadata": metadata or {},
                            "distance": distance,
                            "score": 1
                            - distance,  # Convert distance to similarity score
                        }
                    )

            return {
                "success": True,
                "results": formatted_results,
                "query": query_text,
                "result_count": len(formatted_results),
            }

        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return {"success": False, "error": str(e), "results": []}

    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Get a specific document by ID

        Args:
            document_id: Document ID

        Returns:
            Document data
        """
        try:
            results = self.collection.get(ids=[document_id])

            if not results["documents"]:
                return {"success": False, "error": "Document not found"}

            return {
                "success": True,
                "document": results["documents"][0],
                "metadata": results["metadatas"][0] if results["metadatas"] else {},
                "id": document_id,
            }

        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            return {"success": False, "error": str(e)}

    def update_document(
        self, document_id: str, document: str, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Update a document in the collection

        Args:
            document_id: Document ID to update
            document: New document text
            metadata: New metadata

        Returns:
            Operation result
        """
        try:
            if metadata is None:
                metadata = {}

            # Update timestamp
            metadata["updated_at"] = datetime.utcnow().isoformat()

            self.collection.update(
                ids=[document_id], documents=[document], metadatas=[metadata]
            )

            logger.info(f"Updated document: {document_id}")

            return {"success": True, "id": document_id}

        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return {"success": False, "error": str(e)}

    def delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Delete documents from the collection

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Operation result
        """
        try:
            self.collection.delete(ids=document_ids)

            logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")

            return {
                "success": True,
                "deleted_count": len(document_ids),
                "deleted_ids": document_ids,
            }

        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return {"success": False, "error": str(e)}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()

            # Get some sample documents to analyze
            sample_results = self.collection.get(limit=min(10, count))

            avg_length = 0
            if sample_results["documents"]:
                avg_length = sum(len(doc) for doc in sample_results["documents"]) / len(
                    sample_results["documents"]
                )

            return {
                "success": True,
                "collection_name": CHROMA_COLLECTION_NAME,
                "document_count": count,
                "average_document_length": avg_length,
                "persistence_path": str(CHROMA_DIR),
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"success": False, "error": str(e)}

    def clear_collection(self) -> Dict[str, Any]:
        """Clear all documents from the collection"""
        try:
            # Get all document IDs first
            all_docs = self.collection.get()
            document_ids = all_docs["ids"]

            if document_ids:
                self.collection.delete(ids=document_ids)

            logger.info(f"Cleared collection, removed {len(document_ids)} documents")

            return {"success": True, "cleared_count": len(document_ids)}

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return {"success": False, "error": str(e)}

    def search_by_metadata(
        self, metadata_filter: Dict, n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search documents by metadata filter

        Args:
            metadata_filter: Metadata filter dictionary
            n_results: Number of results to return

        Returns:
            Search results
        """
        try:
            results = self.collection.get(where=metadata_filter, limit=n_results)

            formatted_results = []
            for i, (doc, metadata) in enumerate(
                zip(results["documents"], results["metadatas"])
            ):
                formatted_results.append(
                    {
                        "id": results["ids"][i],
                        "document": doc,
                        "metadata": metadata or {},
                    }
                )

            return {
                "success": True,
                "results": formatted_results,
                "result_count": len(formatted_results),
                "filter": metadata_filter,
            }

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return {"success": False, "error": str(e), "results": []}
