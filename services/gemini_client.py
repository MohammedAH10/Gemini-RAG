import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from config.settings import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for Google Gemini 1.5 Flash API"""

    def __init__(self):
        self.model_name = settings.GEMINI_MODEL
        self.temperature = settings.GEMINI_TEMPERATURE
        self.max_tokens = settings.GEMINI_MAX_TOKENS
        self.client = None
        self.model = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Gemini client"""
        try:
            if not settings.GEMINI_API_KEY:
                raise ValueError("Gemini API key is not configured")

            genai.configure(api_key=settings.GEMINI_API_KEY)

            # Initialize the model
            generation_config = {
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": self.max_tokens,
            }

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
            ]

            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            logger.info(f"Gemini client initialized with model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini 1.5 Flash

        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            temperature: Optional temperature override

        Returns:
            Response dictionary
        """
        try:
            start_time = time.time()

            # Prepare generation config
            generation_config = {
                "temperature": temperature or self.temperature,
                "max_output_tokens": self.max_tokens,
            }

            # Create model instance with system instruction if provided
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                )
            else:
                model = self.model

            # Generate content
            response = model.generate_content(prompt)

            processing_time = time.time() - start_time

            # Extract response text
            if response.parts:
                text = "".join(part.text for part in response.parts)
            else:
                text = response.text

            # Check for safety blocks
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(
                    f"Prompt blocked: {response.prompt_feedback.block_reason}"
                )
                return {
                    "success": False,
                    "error": f"Prompt blocked: {response.prompt_feedback.block_reason}",
                    "text": "",
                    "processing_time": processing_time,
                }

            result = {
                "success": True,
                "text": text,
                "processing_time": processing_time,
                "model": self.model_name,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Add token counts if available
            if hasattr(response, "usage_metadata"):
                result["token_count"] = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "candidates_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            logger.info(f"Gemini generated content in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error generating content with Gemini: {e}")
            return {"success": False, "error": str(e), "text": "", "processing_time": 0}

    def chat(
        self, messages: List[Dict[str, str]], temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Chat completion with conversation history

        Args:
            messages: List of messages in format [{"role": "user", "content": "hello"}, ...]
            temperature: Optional temperature override

        Returns:
            Response dictionary
        """
        try:
            start_time = time.time()

            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                role = "user" if msg["role"] in ["user", "human"] else "model"
                gemini_messages.append({"role": role, "parts": [msg["content"]]})

            # Start chat session
            chat_session = self.model.start_chat(history=gemini_messages[:-1])

            # Get the last message (user input)
            last_message = gemini_messages[-1]["parts"][0]

            # Send message
            response = chat_session.send_message(last_message)

            processing_time = time.time() - start_time

            # Extract response
            if response.parts:
                text = "".join(part.text for part in response.parts)
            else:
                text = response.text

            result = {
                "success": True,
                "text": text,
                "processing_time": processing_time,
                "model": self.model_name,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Add token counts if available
            if hasattr(response, "usage_metadata"):
                result["token_count"] = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "candidates_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            logger.info(f"Gemini chat completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error in Gemini chat: {e}")
            return {"success": False, "error": str(e), "text": "", "processing_time": 0}

    def get_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get embeddings for a list of texts

        Args:
            texts: List of texts to embed

        Returns:
            Embeddings result
        """
        try:
            start_time = time.time()

            # Use the embedding model
            embedding_model = genai.embed_content(
                model="models/embedding-001",
                content=texts,
                task_type="retrieval_document",
            )

            processing_time = time.time() - start_time

            result = {
                "success": True,
                "embeddings": embedding_model["embedding"],
                "processing_time": processing_time,
                "model": "models/embedding-001",
                "text_count": len(texts),
            }

            logger.info(
                f"Generated embeddings for {len(texts)} texts in {processing_time:.2f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return {
                "success": False,
                "error": str(e),
                "embeddings": [],
                "processing_time": 0,
            }

    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models"""
        try:
            models = genai.list_models()
            model_names = [model.name for model in models]
            return model_names
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check Gemini API health"""
        try:
            # Simple health check by listing models
            models = self.get_available_models()
            return {
                "status": "healthy" if models else "unhealthy",
                "model_count": len(models),
                "configured_model": self.model_name,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
