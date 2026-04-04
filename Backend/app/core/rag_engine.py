"""
RAG Engine for text chunking and retrieval using LlamaIndex.
"""

import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from loguru import logger

from app.config import get_settings
from app.core.embeddings import EmbeddingService, get_embedding_service
from app.core.llm_client import GeminiClient, get_gemini_client
from app.core.vector_store import VectorStore, get_vector_store
from app.models.schemas import ChunkingResult, TextChunk
from app.utils.validators import TextValidator

settings = get_settings()


class RAGEngine:
    """
    RAG Engine for document processing and retrieval.
    Handles text chunking, embedding, and retrieval.
    """

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        """
        Initialize RAG Engine.
        Args:
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        self._llm_client: Optional[GeminiClient] = None
        self._vector_store: Optional[VectorStore] = None
        self._embedding_service: Optional[EmbeddingService] = None

        # Validate parameters
        is_valid_size, size_error = TextValidator.validate_chunk_size(self.chunk_size)
        if not is_valid_size:
            raise ValueError(f"Invalid chunk size: {size_error}")

        is_valid_overlap, overlap_error = TextValidator.validate_chunk_overlap(self.chunk_overlap, self.chunk_size)
        
        if not is_valid_overlap:
            raise ValueError(f"Invalid chunk overlap: {overlap_error}")

        # Initialize sentence splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
        )

        logger.info(f"RAG Engine initialized with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    @property
    def llm_client(self) -> GeminiClient:
        """Get or Create llm Client"""
        if self._llm_client is None:
            self._llm_client = get_gemini_client()
        return self._llm_client

    @property
    def vector_store(self) -> VectorStore:
        """Get or Create vector store"""
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    @property
    def embedding_service(self) -> EmbeddingService:
        """Get or Create embeddings client"""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def validate_and_clean_text(self, text: str) -> str:
        """
        Validate and clean text before chunking.
        Args:
            text: Raw text to validate and clean
        Returns:
            Cleaned text
        Raises:
            ValueError: If text is invalid
        """
        # Validate
        is_valid, error_msg = TextValidator.is_valid_text(text)
        if not is_valid:
            raise ValueError(f"Text validation failed: {error_msg}")

        # Clean
        text = TextValidator.clean_text(text)
        text = TextValidator.normalize_unicode(text)
        text = TextValidator.remove_extra_whitespace(text)

        return text

    def chunk_text(
        self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Chunk text into smaller segments for processing.
        Args:
            text: Text to chunk
            document_id: Document identifier
            metadata: Optional metadata to attach to chunks
        Returns:
            ChunkingResult with all chunks and metadata
        """
        start_time = time.time()
        logger.info(f"Chunking document {document_id}: {len(text)} characters")

        # Validate and clean text
        try:
            text = self.validate_and_clean_text(text)
        except ValueError as e:
            logger.error(f"Text validation failed: {e}")
            raise

        # Create LlamaIndex Document
        doc = Document(text=text, doc_id=document_id, metadata=metadata or {})

        # Split into nodes (chunks)
        nodes = self.text_splitter.get_nodes_from_documents([doc])

        # Convert nodes to TextChunk objects
        chunks: List[TextChunk] = []
        total_tokens = 0

        for idx, node in enumerate(nodes):
            # Generate chunk ID
            chunk_id = f"{document_id}_chunk_{idx}"

            # Estimate token count (rough approximation)
            token_count = len(node.text.split())
            total_tokens += token_count

            # Get character positions
            start_char = node.start_char_idx if node.start_char_idx is not None else 0
            end_char = (
                node.end_char_idx if node.end_char_idx is not None else len(node.text)
            )

            # Create chunk metadata
            chunk_metadata = {
                **(metadata or {}),
                "chunk_method": "sentence_splitter",
                "original_length": len(text),
            }

            # Create TextChunk
            chunk = TextChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=node.text,
                chunk_index=idx,
                start_char=start_char,
                end_char=end_char,
                token_count=token_count,
                metadata=chunk_metadata,
            )

            chunks.append(chunk)

        processing_time = time.time() - start_time

        logger.info(
            f"Chunking complete: {len(chunks)} chunks, "
            f"{total_tokens} tokens, {processing_time:.2f}s"
        )

        # Create result
        result = ChunkingResult(
            document_id=document_id,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            processing_time=processing_time,
        )

        return result

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[ChunkingResult]:
        """
        Chunk multiple documents.
        Args:
            documents: List of dicts with 'text', 'document_id', and optional 'metadata'
        Returns:
            List of ChunkingResult objects
        """
        results = []

        for doc in documents:
            try:
                result = self.chunk_text(
                    text=doc["text"],
                    document_id=doc["document_id"],
                    metadata=doc.get("metadata"),
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to chunk document {doc.get('document_id')}: {e}")
                continue

        return results

    def get_chunk_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        Get statistics about chunks.
        Args:
            chunks: List of text chunks
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "total_tokens": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        chunk_sizes = [len(chunk.text) for chunk in chunks]
        token_counts = [chunk.token_count for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "total_tokens": sum(token_counts),
            "avg_chunk_size": sum(chunk_sizes) / len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks),
        }

    def verify_chunk_overlap(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        Verify that chunks have proper overlap.
        Args:
            chunks: List of text chunks
        Returns:
            Dictionary with overlap statistics
        """
        if len(chunks) < 2:
            return {"has_overlap": False, "overlap_count": 0, "avg_overlap_chars": 0}

        overlaps = []

        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]

            # Check if there's overlap in character positions
            if chunk1.end_char > chunk2.start_char:
                overlap_chars = chunk1.end_char - chunk2.start_char
                overlaps.append(overlap_chars)

        return {
            "has_overlap": len(overlaps) > 0,
            "overlap_count": len(overlaps),
            "avg_overlap_chars": sum(overlaps) / len(overlaps) if overlaps else 0,
            "min_overlap_chars": min(overlaps) if overlaps else 0,
            "max_overlap_chars": max(overlaps) if overlaps else 0,
        }

    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_citations: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate response using LLM with retrieved context.
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            system_prompt: Optional custom system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            include_citations: Whether to include source citations

        Returns:
            Dictionary with generated response and metadata
        """

        start_time = time.time()

        # build context from chunk
        context = self._build_context(context_chunks)

        #  build prompt
        prompt = self._build_prompt(
            query=query,
            context=context,
            system_prompt=system_prompt,
            include_citations=include_citations,
        )

        logger.info(f"Generating response:{query[:100]}....")

        try:
            # Generate response
            response_text = self.llm_client.generate_text(
                prompt=prompt,
                temperature=temperature or settings.temperature,
                max_tokens=max_tokens or settings.max_tokens,
            )
            #  Extract citations if included
            citations = []
            if include_citations and context_chunks:
                citations = self._extract_citations(response_text, context_chunks)

            generation_time = time.time() - start_time
            logger.info(f"Response generated in {generation_time:.2f}s")

            return {
                "answer": response_text,
                "query": query,
                "citations": citations,
                "num_chunks_used": len(context_chunks),
                "generation_time": generation_time,
                "model": self.llm_client.model_name,
                "temperature": temperature or settings.temperature,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": "",
                "query": query,
                "citations": [],
                "num_chunks_used": 0,
                "generation_time": time.time() - start_time,
                "model": self.llm_client.model_name,
                "success": False,
                "error": str(e),
            }

    # def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
    #     """ "
    #     Build context from retrieved chunks

    #     Args:
    #         chunks: List of retrieved chunks

    #     Returns:
    #         formatted context string
    #     """
    #     if not chunks:
    #         return ""

    #     context_parts = []
    #     for i, chunk in enumerate(chunks):
    #         text = chunk.get("document", chunk.get("text", ""))
    #         metadata = chunk.get("metadata", {})
    #         doc_id = metadata.get("document_id", "unknown")

    #         context_parts.append(f"[Source {i} - Document {doc_id}]\n\{text}\n")

    #     return "\n".join(context_parts)
        
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return ""
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):  # start at 1 for source numbering
            text = chunk.get("document", chunk.get("text", ""))
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("document_id", "unknown")
            context_parts.append(f"[Source {i} - Document {doc_id}]\n{text}\n")  # fixed
        return "\n".join(context_parts)

    def _build_prompt(self, query: str, context: str,system_prompt: Optional[str] = None, include_citations: bool = True) -> str:
        """
        Build prompt for LLM.
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional custom system prompt
            include_citations: Whether to request citations
        Returns:
            Complete prompt string
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        citations_instructions = ""
        if include_citations and context:
            citations_instructions = (
                "\n\nWhen using information from the context, cite the source number "
                "in square brackets like [Source 1] or [Source 2]."
            )

        if context:
            prompt = f"""{system_prompt}

Context Information:
{context}

Question: {query}

Answer the question based on the context provided above. If the context does not contain enough information to answer the question, say so clearly {citations_instructions}

Answer:"""
        else:
            prompt = f"""{system_prompt}
Question: {query}

Answer:"""
        return prompt

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt"""

        return (
            "You are a helpful AI assistant that answers questions based on provided context."
            "Provide accurate, concise, and well structured answers."
            "If you are not sure how to answer, say so clearly instead of guessing."
        )

    def _extract_citations(
        self, response: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract citations from response
        Args:
            response: The response text
            chunks: The list of chunks used to generate the response
        Returns:
            A list of citation dictionaries
        """
        citations = []

        # Find all [Source N] references in response
        citation_pattern = r"\[Source (\d+)\]"
        matches = re.findall(citation_pattern, response)

        # Get unique source numbers
        source_numbers = set(int(match) for match in matches)

        for source_number in source_numbers:
            # source number are 1 -indexed
            idx = source_number - 1

            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                metadata = chunk.get("metadata", {})

                citations.append(
                    {
                        "source_number": source_number,
                        "chunk_id": chunk.get("chunk_id", metadata.get("chunk_id")),
                        "document_id": metadata.get("document_id", "unknown"),
                        "text": chunk.get("document", chunk.get("text", ""))[:200]
                        + "...",
                        "similarity": chunk.get("similarity", chunk.get("distance", 0)),
                    }
                )

        return citations

    def query(self,query: str,user_id: Optional[str] = None,document_id: Optional[str] = None,
        n_results: int = 5,temperature: Optional[float] = None,include_citations: bool = True,) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve context and generate response.

        Args:
            query: User query
            user_id: Optional user ID filter
            document_id: Optional document ID filter
            n_results: Number of chunks to retrieve
            temperature: LLM temperature
            include_citations: Include source citations

        Returns:
            Complete RAG response with answer and sources
        """
        start_time = time.time()

        logger.info(f"RAG query: {query[:100]}...")

        try:
            # Step 1: Generate query embedding
            query_embedding = self.embedding_service.generate_query_embedding(query)

            # Step 2: Retrieve relevant chunks
            retrieval_results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=n_results,
                user_id=user_id,
                document_id=document_id,
            )

            # Format chunks for response generation
            context_chunks = []
            for i in range(len(retrieval_results["ids"])):
                context_chunks.append(
                    {
                        "chunk_id": retrieval_results["ids"][i],
                        "document": retrieval_results["documents"][i],
                        "metadata": retrieval_results["metadatas"][i],
                        "distance": retrieval_results["distances"][i],
                        "similarity": 1.0 / (1.0 + retrieval_results["distances"][i]),
                    }
                )

            # Step 3: Generate response
            response = self.generate_response(
                query=query,
                context_chunks=context_chunks,
                temperature=temperature,
                include_citations=include_citations,
            )

            total_time = time.time() - start_time

            # Add retrieval info
            response["retrieval_info"] = {
                "chunks_retrieved": len(context_chunks),
                "user_id_filter": user_id,
                "document_id_filter": document_id,
            }
            response["total_time"] = total_time

            logger.info(f"RAG pipeline complete in {total_time:.2f}s")

            return response

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "answer": f"I encountered an error processing your query: {str(e)}",
                "query": query,
                "citations": [],
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time,
            }


# Global instance
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine(
    chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None
) -> RAGEngine:
    """
    Get or create RAG engine instance.
    Args:
        chunk_size: Optional custom chunk size
        chunk_overlap: Optional custom chunk overlap
    Returns:
        RAGEngine instance
    """
    global _rag_engine

    # If custom parameters provided, create new instance
    if chunk_size is not None or chunk_overlap is not None:
        return RAGEngine(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Otherwise use singleton
    if _rag_engine is None:
        _rag_engine = RAGEngine()

    return _rag_engine
