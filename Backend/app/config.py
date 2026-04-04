"""
Configuration management for RAG Startup Books application.
Loads and validates environment variables using Pydantic Settings.
"""

# pyright: reportArgumentType=false
from pathlib import Path
from typing import List, Optional

from loguru import logger
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Settings
    app_name: str = Field(default="Gemini Startup Book RAG", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")

    # Server Configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    reload: bool = Field(default=True, alias="RELOAD")

    # Security
    secret_key: str = Field(..., alias="SECRET_KEY")
    algorithm: str = Field(default="HS256", alias="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Google Gemini API
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field(default="gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL")

    # Supabase Configuration
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_key: str = Field(..., alias="SUPABASE_KEY")
    supabase_service_role_key: Optional[str] = Field(default=None, alias="SUPABASE_SERVICE_ROLE_KEY")
    
    # OAuth Configuration
    oauth_redirect_url: Optional[str] = Field(default=None, alias="OAUTH_REDIRECT_URL")
    oauth_google_client_id: Optional[str] = Field(default=None, alias="OAUTH_GOOGLE_CLIENT_ID")

    # ChromaDB Configuration
    chroma_persist_directory: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="startup_books", alias="CHROMA_COLLECTION_NAME")

    # PDF Storage
    pdf_storage_path: str = Field(default="./data", alias="PDF_STORAGE_PATH")
    # allowed_extensions: List[str] = Field(default="pdf,epub,txt,docx,mobi,azw,azw3", alias="ALLOWED_EXTENSIONS")
    allowed_extensions: str = Field(default="pdf,epub,txt,docx,mobi,azw,azw3", alias="ALLOWED_EXTENSIONS")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")

    # RAG Configuration
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    max_tokens: int = Field(default=2048, alias="MAX_TOKENS")

    # CORS Settings
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:5173", alias="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: str = Field(default="*", alias="CORS_ALLOW_METHODS")
    cors_allow_headers: str = Field(default="*", alias="CORS_ALLOW_HEADERS")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", alias="LOG_FILE")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, secret_key: str) -> str:
        """Validate that secret key is set and not the default."""
        if not secret_key or len(secret_key) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        if secret_key == "your-secret-key-here-change-in-production":
            logger.warning("Using default SECRET_KEY - change this in production!")
        return secret_key

    @field_validator("gemini_api_key")
    @classmethod
    def validate_google_api_key(cls, api_key: str) -> str:
        """Validate Google API key format."""
        if not api_key or api_key == "your-gemini-api-key-here":
            raise ValueError("GEMINI_API_KEY must be set")
        if not api_key.startswith("AIza"):
            logger.warning(
                "GEMINI_API_KEY does not match expected format (should start with 'AIza')"
            )
        return api_key

    @field_validator("supabase_url")
    @classmethod
    def validate_supabase_url(cls, supabase_url: str) -> str:
        """Validate Supabase URL format."""
        if not supabase_url or supabase_url == "your-supabase-project-url":
            raise ValueError("SUPABASE_URL must be set")
        if not supabase_url.startswith("https://"):
            raise ValueError("SUPABASE_URL must be a valid HTTPS URL")
        return supabase_url

    @field_validator("supabase_key")
    @classmethod
    def validate_supabase_key(cls, supabase_anon_key: str) -> str:
        """Validate Supabase key is set."""
        if not supabase_anon_key or supabase_anon_key == "your-supabase-anon-key":
            raise ValueError("SUPABASE_KEY must be set")
        return supabase_anon_key

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, chunk_size: int) -> int:
        """Validate chunk size is reasonable."""
        if chunk_size < 100 or chunk_size > 2000:
            raise ValueError("CHUNK_SIZE must be between 100 and 2000")
        return chunk_size

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, chunk_overlap: int) -> int:
        """Validate chunk overlap is reasonable."""
        if chunk_overlap < 0 or chunk_overlap > 500:
            raise ValueError("CHUNK_OVERLAP must be between 0 and 500")
        return chunk_overlap

    @field_validator("top_k_results")
    @classmethod
    def validate_top_k(cls, top_k: int) -> int:
        """Validate top_k is reasonable."""
        if top_k < 1 or top_k > 20:
            raise ValueError("TOP_K_RESULTS must be between 1 and 20")
        return top_k

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, tmprture: float) -> float:
        """Validate temperature is in valid range."""
        if tmprture < 0.0 or tmprture > 2.0:
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0")
        return tmprture

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins string into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        """Convert max file size from MB to bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def allowed_extensions_list(self) -> List[str]:
        """Parse allowed extensions string into a list."""
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    def get_pdf_storage_path(self) -> Path:
        """Get PDF storage path as Path object."""
        return Path(self.pdf_storage_path)

    def get_chroma_persist_directory(self) -> Path:
        """Get ChromaDB persist directory as Path object."""
        return Path(self.chroma_persist_directory)

    def get_log_file_path(self) -> Path:
        """Get log file path as Path object."""
        return Path(self.log_file)

    def ensure_directories_exist(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.get_pdf_storage_path(),
            self.get_chroma_persist_directory(),
            self.get_log_file_path().parent,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")

    def display_config(self) -> dict:
        """Return a safe version of config for display (without sensitive data)."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment,
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "gemini_model": self.gemini_model,
            "gemini_embedding_model": self.gemini_embedding_model,
            "chroma_collection_name": self.chroma_collection_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k_results": self.top_k_results,
            "temperature": self.temperature,
            "max_file_size_mb": self.max_file_size_mb,
            "cors_origins": self.cors_origins_list,
        }


def load_settings() -> Settings:
    """
    Load and validate settings from environment.

    Returns:
        Settings: Validated settings object

    Raises:
        ValidationError: If required environment variables are missing or invalid
    """
    try:
        settings = Settings()  # pyright: ignore[reportCallIssue]
        settings.ensure_directories_exist()
        logger.info(
            f"Configuration loaded successfully for environment: {settings.environment}"
        )
        return settings
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


# Global settings instance
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    Creates it if it doesn't exist.

    Returns:
        Settings: Global settings instance
    """
    global settings
    if settings is None:
        settings = load_settings()
    return settings
