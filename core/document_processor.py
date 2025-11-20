import os
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode
from llama_index.readers.file import PDFReader, DocxReader, UnstructuredReader
from llama_index.readers.file import FlatReader  # For txt files

from config.settings import settings
from config.constants import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handle document uploads and processing for RAG"""
    
    def __init__(self):
        self.supported_extensions = SUPPORTED_EXTENSIONS
        self.max_file_size = MAX_FILE_SIZE
        
        # Initialize node parser for chunking
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP
        )
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate uploaded file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Validation result dictionary
        """
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return {"valid": False, "error": "File does not exist"}
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_extensions:
                return {
                    "valid": False, 
                    "error": f"Unsupported file type: {file_path.suffix}. Supported: {', '.join(self.supported_extensions)}"
                }
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                return {
                    "valid": False,
                    "error": f"File too large: {file_size} bytes. Maximum: {self.max_file_size} bytes"
                }
            
            if file_size == 0:
                return {"valid": False, "error": "File is empty"}
            
            return {"valid": True, "size": file_size}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def load_document(self, file_path: str) -> Optional[List[LlamaDocument]]:
        """
        Load document using appropriate reader based on file type
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of LlamaIndex Document objects or None if failed
        """
        validation = self.validate_file(file_path)
        if not validation["valid"]:
            logger.error(f"File validation failed: {validation['error']}")
            return None
        
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            # Select appropriate reader based on file type
            if file_extension == '.pdf':
                reader = PDFReader()
            elif file_extension in ['.docx', '.doc']:
                reader = DocxReader()
            elif file_extension == '.txt':
                reader = FlatReader()
            else:
                # Use UnstructuredReader for other file types
                reader = UnstructuredReader()
            
            # Load documents
            documents = reader.load_data(file_path)
            
            # Add file metadata to each document
            for doc in documents:
                doc.metadata.update({
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_size": validation["size"],
                    "file_type": file_extension
                })
            
            logger.info(f"Successfully loaded {len(documents)} documents from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return None
    
    def process_documents(self, file_paths: List[str]) -> List[LlamaDocument]:
        """
        Process multiple documents and return chunked nodes
        
        Args:
            file_paths: List of paths to documents
            
        Returns:
            List of processed Document objects
        """
        all_documents = []
        
        for file_path in file_paths:
            documents = self.load_document(file_path)
            if documents:
                all_documents.extend(documents)
        
        logger.info(f"Processed {len(all_documents)} total documents from {len(file_paths)} files")
        return all_documents
    
    def chunk_documents(self, documents: List[LlamaDocument]) -> List[BaseNode]:
        """
        Chunk documents into smaller nodes for vector storage
        
        Args:
            documents: List of LlamaIndex Document objects
            
        Returns:
            List of chunked nodes
        """
        if not documents:
            return []
        
        try:
            nodes = self.node_parser.get_nodes_from_documents(documents)
            logger.info(f"Chunked {len(documents)} documents into {len(nodes)} nodes")
            return nodes
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            return []
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_extensions)
    
    def cleanup_file(self, file_path: str) -> bool:
        """
        Clean up temporary file
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            Success status
        """
        try:
            file_path = Path(file_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Cleaned up file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {e}")
            return False