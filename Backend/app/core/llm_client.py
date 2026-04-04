# Backend/app/core/llm_client.py
"""
LLM client for Google Gemini API and Embedding operations.
Uses the new Google GenAI SDK (July 2025+)
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings

settings = get_settings()


class GeminiClient:
    """
    Client for interacting with Google Gemini API.
    Handles both text generation and embedding generation using the new GenAI SDK.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, embedding_model: Optional[str] = None):
        """
        Initialize Gemini Client.

        Args:
            api_key: Gemini API Key (uses settings if not provided)
            model_name: Model for text generation
            embedding_model: Model for embeddings
        """
        self.api_key = (api_key or getattr(settings, "gemini_api_key", None) or getattr(settings, "google_api_key", None))
        self.model_name = model_name or settings.gemini_model
        self.embedding_model = embedding_model or settings.gemini_embedding_model

        # Initialize the new GenAI Client
        self.client = genai.Client(api_key=self.api_key)

        # Embedding dimensions (768 for gemini-embedding-001)
        self.embedding_dimensions = 768

        logger.info(f"GeminiClient initialized with embedding model: {self.embedding_model}, dimensions: {self.embedding_dimensions}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    
    def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            task_type: Task type for embedding
                - "RETRIEVAL_DOCUMENT": For document chunks
                - "RETRIEVAL_QUERY": For search queries
                - "SEMANTIC_SIMILARITY": For similarity tasks

        Returns:
            Embedding vector as list of floats

        Raises:
            ValueError: If text is empty
            Exception: If API call fails after retries
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.embedding_dimensions,
                ),
            )

            embedding = result.embeddings[0].values

            # Validate embedding
            if not embedding or len(embedding) != self.embedding_dimensions:
                raise ValueError(
                    f"Invalid embedding dimensions: expected {self.embedding_dimensions}, "
                    f"got {len(embedding) if embedding else 0}"
                )

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT", batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            task_type: Task type for embedding
            batch_size: Maximum texts per batch (Gemini limit is 100)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            logger.info(f"Processing batch {i // batch_size + 1}: {len(batch)} texts")

            try:
                batch_embeddings = self._generate_batch(batch, task_type)
                embeddings.extend(batch_embeddings)

                # Rate limiting: small delay between batches
                if i + batch_size < len(texts):
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Try individual embeddings for this batch
                for text in batch:
                    try:
                        emb = self.generate_embedding(text, task_type)
                        embeddings.append(emb)
                    except Exception as e2:
                        logger.error(f"Individual embedding failed: {e2}")
                        # Add zero vector as placeholder
                        embeddings.append([0.0] * self.embedding_dimensions)

        return embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    
    def _generate_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """
        Generate embeddings for a batch (internal method with retry).

        Args:
            texts: Batch of texts
            task_type: Task type of embedding

        Returns:
            List of embeddings
        """
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.embedding_dimensions,
            ),
        )

        return [emb.values for emb in result.embeddings]

    async def generate_embedding_async(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """
        Generate embedding asynchronously.

        Args:
            text: Text to embed
            task_type: Task type

        Returns:
            Embedding vector
        """
        # Use the async client
        result = await self.client.aio.models.embed_content(
            model=self.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.embedding_dimensions,
            ),
        )

        return result.embeddings[0].values

    async def generate_embeddings_batch_async(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT", batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts asynchronously.

        Args:
            texts: List of texts
            task_type: Task type
            batch_size: Batch size

        Returns:
            List of embeddings
        """
        result = await self.client.aio.models.embed_content(
            model=self.embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.embedding_dimensions,
            ),
        )

        return [emb.values for emb in result.embeddings]

    def generate_text(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """
        Generate text using Gemini LLM.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        config_dict = {}
        if temperature is not None:
            config_dict["temperature"] = temperature
        if max_tokens is not None:
            config_dict["max_output_tokens"] = max_tokens

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_dict)
                if config_dict
                else None,
            )

            return response.text

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about configured models.

        Returns:
            Dictionary with model information
        """
        return {
            "llm_model": self.model_name,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "api_configured": bool(self.api_key),
        }

# Global client instance
_gemini_client: Optional[GeminiClient] = None

def get_gemini_client() -> GeminiClient:
    """
    Get or create global Gemini client instance.

    Returns:
        GeminiClient instance
    """
    global _gemini_client

    if _gemini_client is None:
        _gemini_client = GeminiClient()

    return _gemini_client