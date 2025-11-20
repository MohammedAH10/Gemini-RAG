import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Settings:
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
    GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
    
    SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "3"))
    SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.7"))
    
    ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "True").lower() == "true"
    
    def validate_settings(self):
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        
        if self.ENABLE_WEB_SEARCH and (not self.GOOGLE_SEARCH_API_KEY or not self.GOOGLE_SEARCH_ENGINE_ID):
            print("Warning: Web search is enabled but GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_ENGINE_ID is missing")
    
    @property
    def is_development(self):
        return self.FLASK_ENV == "development"
    
    @property
    def is_production(self):
        return self.FLASK_ENV == "production"

settings = Settings()

try:
    settings.validate_settings()
except ValueError as e:
    print(f"Configuration error: {e}")
