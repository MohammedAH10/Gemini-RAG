import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from services.gemini_client import GeminiClient
from services.chroma_service import ChromaService
from services.search_services import SearchService

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

from config.settings import settings

from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .web_retriever import WebRetriever

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Retrieval mode for RAG system"""

    DOCUMENTS_ONLY = "documents_only"
    WEB_ONLY = "web_only"
    HYBRID = "hybrid"


class RAGEngine:
    """Main RAG orchestration engine"""

    def __init__(self):

        
        self.gemini_client = GeminiClient()
        self.chroma_service = ChromaService()
        self.search_service = SearchService()
        
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
        self.web_retriever = WebRetriever()

        # Initialize Gemini LLM
        self.llm = Gemini(
            model=settings.GEMINI_MODEL,
            temperature=settings.GEMINI_TEMPERATURE,
            max_tokens=settings.GEMINI_MAX_TOKENS,
        )

        # Initialize Gemini embeddings
        self.embed_model = GeminiEmbedding(model_name="models/embedding-001")

        self.retrieval_mode = RetrievalMode.HYBRID
        self.conversation_history = []

    def set_retrieval_mode(self, mode: RetrievalMode):
        """Set the retrieval mode for the RAG system"""
        self.retrieval_mode = mode
        logger.info(f"Retrieval mode set to: {mode.value}")

    def process_uploaded_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process and index uploaded documents

        Args:
            file_paths: List of paths to uploaded documents

        Returns:
            Processing results
        """
        try:
            # Process documents
            documents = self.document_processor.process_documents(file_paths)
            if not documents:
                return {"success": False, "error": "No valid documents processed"}

            # Add to vector store
            doc_ids = self.vector_store.add_documents(documents)

            return {
                "success": True,
                "document_count": len(documents),
                "file_count": len(file_paths),
                "document_ids": doc_ids,
            }

        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return {"success": False, "error": str(e)}

    def _format_context(self, documents: List[Dict], web_results: List[Dict]) -> str:
        """Format retrieved context for the LLM"""
        context_parts = []

        # Add document context
        if documents:
            context_parts.append("## Relevant Document Information:")
            for i, doc in enumerate(documents, 1):
                context_parts.append(f"{i}. {doc['text'][:500]}...")
                if "metadata" in doc and "file_name" in doc["metadata"]:
                    context_parts.append(f"   Source: {doc['metadata']['file_name']}")
                context_parts.append("")

        # Add web context
        if web_results:
            context_parts.append("## Web Search Results:")
            for i, result in enumerate(web_results, 1):
                context_parts.append(f"{i}. {result.get('title', 'No title')}")
                content = result.get("content") or result.get("snippet", "")
                context_parts.append(f"   {content[:300]}...")
                context_parts.append(f"   Source: {result.get('source', 'unknown')}")
                context_parts.append("")

        return (
            "\n".join(context_parts) if context_parts else "No relevant context found."
        )

    def _build_prompt(
        self, query: str, context: str, history: List[ChatMessage]
    ) -> List[ChatMessage]:
        """Build the prompt for the LLM"""
        system_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content="""You are a helpful AI assistant. Use the provided context to answer the user's question.
            If the context doesn't contain relevant information, use your general knowledge but indicate this.
            Be concise and accurate in your responses. Cite your sources when possible.""",
        )

        if context.strip():
            user_content = f"""Context Information:
{context}

User Question: {query}

Please answer the question based on the context above. If the context doesn't fully answer the question, you can use your general knowledge but please indicate what information comes from the context vs general knowledge."""
        else:
            user_content = f"User Question: {query}"

        user_message = ChatMessage(role=MessageRole.USER, content=user_content)

        # Combine history with current messages
        messages = [system_message]
        if history:
            messages.extend(history[-6:])  # Last 3 exchanges
        messages.append(user_message)

        return messages

    def query(
        self,
        question: str,
        session_id: str = None,
        use_documents: bool = None,
        use_web: bool = None,
    ) -> Dict[str, Any]:
        """
        Main query method for the RAG system

        Args:
            question: User's question
            session_id: Optional session ID for conversation history
            use_documents: Override for document retrieval
            use_web: Override for web retrieval

        Returns:
            Response dictionary with answer and metadata
        """
        try:
            # Determine retrieval strategy
            if use_documents is None or use_web is None:
                use_documents = self.retrieval_mode in [
                    RetrievalMode.DOCUMENTS_ONLY,
                    RetrievalMode.HYBRID,
                ]
                use_web = self.retrieval_mode in [
                    RetrievalMode.WEB_ONLY,
                    RetrievalMode.HYBRID,
                ]

            # Retrieve relevant information
            document_results = []
            web_results = []

            if use_documents:
                document_results = self.vector_store.query(question)

            if use_web and self.web_retriever.is_configured():
                web_results = self.web_retriever.search_web(question)

            # Format context
            context = self._format_context(document_results, web_results)

            # Get conversation history for this session
            history = self._get_conversation_history(session_id)

            # Build prompt
            messages = self._build_prompt(question, context, history)

            # Generate response
            response = self.llm.chat(messages)
            answer = (
                response.message.content
                if hasattr(response, "message")
                else str(response)
            )

            # Update conversation history
            self._update_conversation_history(session_id, question, answer)

            # Prepare sources
            sources = []
            for doc in document_results:
                sources.append(
                    {
                        "type": "document",
                        "content": doc["text"][:200] + "...",
                        "metadata": doc.get("metadata", {}),
                        "score": doc.get("score", 0),
                    }
                )

            for web in web_results:
                sources.append(
                    {
                        "type": "web",
                        "title": web.get("title", ""),
                        "source": web.get("source", ""),
                        "url": web.get("link", ""),
                        "content": (web.get("content") or web.get("snippet", ""))[:200]
                        + "...",
                    }
                )

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "context_used": bool(document_results or web_results),
                "document_results_count": len(document_results),
                "web_results_count": len(web_results),
                "retrieval_mode": self.retrieval_mode.value,
            }

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "I encountered an error while processing your request. Please try again.",
                "sources": [],
            }

    def _get_conversation_history(self, session_id: str = None) -> List[ChatMessage]:
        """Get conversation history for a session"""
        # This is a simplified implementation
        # In production, you'd want to persist this properly
        return self.conversation_history

    def _update_conversation_history(
        self, session_id: str = None, question: str = None, answer: str = None
    ):
        """Update conversation history for a session"""
        if question and answer:
            # Add to history (simplified - in production, use proper session management)
            self.conversation_history.extend(
                [
                    ChatMessage(role=MessageRole.USER, content=question),
                    ChatMessage(role=MessageRole.ASSISTANT, content=answer),
                ]
            )

            # Keep only last 10 messages (5 exchanges)
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and configuration"""
        vector_store_info = self.vector_store.get_collection_info()

        return {
            "retrieval_mode": self.retrieval_mode.value,
            "llm_model": settings.GEMINI_MODEL,
            "vector_store": vector_store_info,
            "web_search_enabled": self.web_retriever.is_configured(),
            "document_processing_enabled": True,
            "supported_file_types": self.document_processor.get_supported_formats(),
        }

    def clear_conversation_history(self, session_id: str = None):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")


help(RAGEngine)
