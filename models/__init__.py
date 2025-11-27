# Models package for data structures and schemas
from .chat_models import (
    ChatMessage, 
    ChatResponse, 
    ChatRequest,
    SourceDocument,
    WebSource,
    ConversationHistory
)
from .schemas import (
    FileUploadResponse,
    ProcessingResult,
    SystemStatus,
    HealthCheck,
    ErrorResponse,
    SessionInfo
)

__all__ = [
    # Chat models
    'ChatMessage',
    'ChatResponse', 
    'ChatRequest',
    'SourceDocument',
    'WebSource',
    'ConversationHistory',
    
    # API schemas
    'FileUploadResponse',
    'ProcessingResult',
    'SystemStatus',
    'HealthCheck',
    'ErrorResponse',
    'SessionInfo'
]