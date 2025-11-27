from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class FileUploadStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class UploadedFile(BaseModel):
    """Uploaded file information"""
    filename: str = Field(..., description="Original filename")
    saved_path: str = Field(..., description="Server file path")
    size: int = Field(..., ge=0, description="File size in bytes")
    file_type: str = Field(..., description="File extension")
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = Field(False, description="Whether file was processed successfully")
    error: Optional[str] = Field(None, description="Processing error if any")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class FileUploadResponse(BaseModel):
    """Response model for file upload endpoint"""
    status: FileUploadStatus = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")
    files: List[UploadedFile] = Field(default_factory=list, description="Uploaded files")
    total_files: int = Field(0, description="Total files processed")
    successful_uploads: int = Field(0, description="Number of successful uploads")
    failed_uploads: int = Field(0, description="Number of failed uploads")
    processing_result: Optional[Dict[str, Any]] = Field(None, description="Document processing results")
    
    class Config:
        use_enum_values = True

class ProcessingResult(BaseModel):
    """Document processing result"""
    success: bool = Field(..., description="Processing success status")
    document_count: int = Field(0, description="Number of documents processed")
    file_count: int = Field(0, description="Number of files processed")
    document_ids: List[str] = Field(default_factory=list, description="Vector store document IDs")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        use_enum_values = True

class VectorStoreInfo(BaseModel):
    """Vector store information"""
    collection_name: str = Field(..., description="ChromaDB collection name")
    document_count: int = Field(0, description="Number of documents in collection")
    persistence_path: str = Field(..., description="Storage path")
    status: str = Field(..., description="Vector store status")

class SystemStatus(BaseModel):
    """System status and configuration"""
    retrieval_mode: str = Field(..., description="Current retrieval mode")
    llm_model: str = Field(..., description="LLM model name")
    vector_store: VectorStoreInfo = Field(..., description="Vector store information")
    web_search_enabled: bool = Field(False, description="Web search availability")
    document_processing_enabled: bool = Field(True, description="Document processing status")
    supported_file_types: List[str] = Field(default_factory=list, description="Supported file formats")
    system_health: HealthStatus = Field(..., description="Overall system health")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HealthCheck(BaseModel):
    """Health check response"""
    status: HealthStatus = Field(..., description="System health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    environment: str = Field(..., description="Application environment")
    version: Optional[str] = Field(None, description="Application version")
    uptime: Optional[float] = Field(None, description="System uptime in seconds")
    dependencies: Dict[str, HealthStatus] = Field(default_factory=dict, description="Dependency health")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SessionInfo(BaseModel):
    """Session information"""
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    message_count: int = Field(0, description="Total messages in session")
    uploaded_files: List[str] = Field(default_factory=list, description="Uploaded filenames")
    file_count: int = Field(0, description="Number of uploaded files")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SearchQuery(BaseModel):
    """Search query model"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    max_results: int = Field(5, ge=1, le=20, description="Maximum results to return")
    use_google: bool = Field(True, description="Use Google search")
    use_wikipedia: bool = Field(True, description="Use Wikipedia search")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "max_results": 5,
                "use_google": True,
                "use_wikipedia": True
            }
        }

class SystemConfig(BaseModel):
    """System configuration update model"""
    retrieval_mode: Optional[str] = Field(None, description="Retrieval mode to set")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM temperature")
    similarity_top_k: Optional[int] = Field(None, ge=1, le=10, description="Similarity top K")
    similarity_cutoff: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity cutoff")
    enable_web_search: Optional[bool] = Field(None, description="Enable/disable web search")
    
    class Config:
        schema_extra = {
            "example": {
                "retrieval_mode": "hybrid",
                "temperature": 0.1,
                "similarity_top_k": 3,
                "enable_web_search": True
            }
        }