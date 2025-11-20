import chromadb
from chromadb.config import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import Document as LlamaDocument
from typing import List, Optional, Dict, Any
import logging

from config.settings import settings
from config.constants import CHROMA_DIR, CHROMA_COLLECTION_NAME

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB vector store management"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.vector_store = None
        self.storage_context = None
        self.index = None
        self.initialized = False
        
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize LlamaIndex ChromaVectorStore
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            self.initialized = True
            logger.info("ChromaDB vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, documents: List[LlamaDocument]) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of LlamaIndex Document objects
            
        Returns:
            List of document IDs
        """
        if not self.initialized:
            raise RuntimeError("Vector store not initialized")
        
        try:
            # Create index with documents
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                show_progress=True
            )
            
            # Get document IDs (this is a simplified approach)
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
            logger.info(f"Added {len(documents)} documents to vector store")
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def query(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents
        
        Args:
            query_text: The query string
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.initialized:
            raise RuntimeError("Vector store not initialized")
        
        if top_k is None:
            top_k = settings.SIMILARITY_TOP_K
        
        try:
            # If index doesn't exist, create an empty one
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(
                    [],  # Empty documents list
                    storage_context=self.storage_context
                )
            
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                similarity_cutoff=settings.SIMILARITY_CUTOFF
            )
            
            # Execute query
            response = query_engine.query(query_text)
            
            # Format results
            results = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    results.append({
                        'text': node.node.get_content(),
                        'score': node.score,
                        'metadata': node.node.metadata,
                        'node_id': node.node_id
                    })
            
            logger.info(f"Vector store query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection"""
        if not self.initialized:
            return {"error": "Vector store not initialized"}
        
        try:
            count = self.collection.count()
            return {
                "collection_name": CHROMA_COLLECTION_NAME,
                "document_count": count,
                "persistence_path": str(CHROMA_DIR)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        if not self.initialized:
            return False
        
        try:
            self.client.delete_collection(CHROMA_COLLECTION_NAME)
            
            self.collection = self.client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME
            )
            
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.index = None
            
            logger.info("Vector store collection cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False