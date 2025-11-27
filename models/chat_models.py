from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    """Role of the message sender"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class RetrievalMode(str, Enum):
    """Retrieval mode for RAG system"""
    DOCUMENTS_ONLY = "documents_only"
    WEB_ONLY = "web_only"
    HYBRID = "hybrid"
    AUTO = "auto"

class SourceType(str, Enum):
    """Type of source document"""
    DOCUMENT = "document"
    WEB = "web"
    WIKIPEDIA = "wikipedia"
    GOOGLE = "google"

class ChatMessage(BaseModel):
    """Individual chat message"""
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_id: Optional[str] = Field(None, description="Unique message ID")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SourceDocument(BaseModel):
    """Source document from vector store"""
    type: SourceType = Field(..., description="Type of source")
    content: str = Field(..., description="Content snippet")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = Field(None, description="Similarity score")
    node_id: Optional[str] = Field(None, description="Vector store node ID")
    
    class Config:
        use_enum_values = True

class WebSource(BaseModel):
    """Web search result source"""
    type: SourceType = Field(..., description="Type of web source")
    title: str = Field(..., description="Page title")
    content: str = Field(..., description="Content snippet")
    source: str = Field(..., description="Source name (google, wikipedia, etc.)")
    url: Optional[str] = Field(None, description="Source URL")
    relevance_score: Optional[float] = Field(None, description="Relevance score")
    
    class Config:
        use_enum_values = True

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    retrieval_mode: RetrievalMode = Field(RetrievalMode.AUTO, description="Retrieval strategy")
    use_documents: Optional[bool] = Field(None, description="Override document retrieval")
    use_web: Optional[bool] = Field(None, description="Override web retrieval")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="Max response tokens")
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "message": "What is machine learning?",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "retrieval_mode": "auto",
                "temperature": 0.1
            }
        }

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="AI response message")
    sources: List[Any] = Field(default_factory=list, description="Source documents and web results")
    context_used: bool = Field(..., description="Whether context was used in response")
    document_results_count: int = Field(0, description="Number of document results used")
    web_results_count: int = Field(0, description="Number of web results used")
    retrieval_mode: RetrievalMode = Field(..., description="Retrieval mode used")
    session_id: Optional[str] = Field(None, description="Session ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage statistics")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConversationHistory(BaseModel):
    """Complete conversation history for a session"""
    session_id: str = Field(..., description="Unique session identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="Chat messages")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_message(self, role: MessageRole, content: str):
        """Add a message to the conversation"""
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    
    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """Get most recent messages"""
        return self.messages[-count:] if self.messages else []
    
    def clear_messages(self):
        """Clear all messages"""
        self.messages.clear()
        self.updated_at = datetime.utcnow()
    
    @property
    def message_count(self) -> int:
        """Get total message count"""
        return len(self.messages)
    
    @property
    def user_message_count(self) -> int:
        """Get user message count"""
        return len([msg for msg in self.messages if msg.role == MessageRole.USER])