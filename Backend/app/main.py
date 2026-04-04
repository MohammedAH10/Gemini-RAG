"""
Main FastAPI application with middleware.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from loguru import logger

from app.config import get_settings
from app.middleware import (
    RequestLoggingMiddleware,
    ExceptionHandlerMiddleware,
    RateLimitMiddleware
)
from app.api.routes import (
    upload,
    chunking,
    embeddings,
    vector_store,
    query,
    auth,
    documents
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Ensure directories exist
    settings.ensure_directories_exist()
    logger.info("All directories initialized")
    
    logger.info(f"Server ready at http://{settings.host}:{settings.port}")
    logger.info(f"API docs at http://{settings.host}:{settings.port}/docs")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG system with Gemini and Supabase",
    lifespan=lifespan
)

# Add middleware (order matters!)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(ExceptionHandlerMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(chunking.router, prefix="/api/v1/chunking", tags=["chunking"])
app.include_router(embeddings.router, prefix="/api/v1/embeddings", tags=["embeddings"])
app.include_router(vector_store.router, prefix="/api/v1/vector-store", tags=["vector-store"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])


# Health check
@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    }


# Root endpoint
@app.get("/", tags=["system"])
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }




# # Backend/app/main.py
# """
# FastAPI application main entry point.
# """

# from contextlib import asynccontextmanager

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.gzip import GZipMiddleware
# from fastapi.responses import JSONResponse

# from loguru import logger

# from app.config import get_settings
# from app.middleware import RequestLoggingMiddleware, ExceptionHandlerMiddleware, RateLimitMiddleware
# from app.api.routes import upload, chunking, embeddings, vector_store, query, auth, documents

# # load settings
# settings = get_settings()


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Application lifespane event:
#     Run on startup and shutdown
#     """
#     # Startup
#     logger.info(f"Starting {settings.app_name} v{settings.app_version}")
#     logger.info(f"Environment: {settings.environment}")
#     logger.info(f"Debug mode: {settings.debug}")

#     # make sure dirs are created in get_settings()
#     logger.info("All directories initialized")

#     logger.info(f"Server ready at http://{settings.host}:{settings.port}")
#     logger.info(f"API docs at http://{settings.host}:{settings.port}/docs")

#     yield

#     # Shutdown
#     logger.info("Shutting Down App")


# # Create FastAPI app
# app = FastAPI(
#     title=settings.app_name,
#     version=settings.app_version,
#     debug=settings.debug,
#     lifespan=lifespan,
#     docs_url="/docs",
#     redoc_url="/redoc",
#     openapi_url="/api/v1/openapi.json",
# )


# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.cors_origins_list,
#     allow_credentials=settings.cors_allow_credentials,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include routers
 
# # upload router
# app.include_router(
#     upload.router,
#     prefix="/api/v1/documents",
#     tags=['documents']
# )

# # chunking router
# app.include_router(
#     chunking.router,
#     prefix="/api/v1/chunking",
#     tags=['chunking']
# )

# # embeddings router
# app.include_router(
#     embeddings.router,
#     prefix="/api/v1/embeddings",
#     tags=['embeddings']
# )

# @app.get("/")
# async def root():
#     """Root endpoint - basic API information"""
#     logger.info("Root endpoint accessed")
#     return {
#         "message": f"Welcome to {settings.app_name}",
#         "version": settings.app_version,
#         "status": "healthy",
#         "environment": settings.environment,
#         "docs": "/docs",
#     }


# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "environment": settings.environment,
#         "version": settings.app_version,
#         "debug": settings.debug,
#     }


# @app.get("/config")
# async def get_config():
#     """
#     Get non-sensitive configuration
#     Only available in development mode
#     """
#     if settings.environment == "production" and not settings.debug:
#         return {"error": "Configuration settings disabled in production"}

#     return settings.display_config()

# app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
# app.include_router(upload.router, prefix="/api/v1/documents", tags=["documents"])
# app.include_router(chunking.router, prefix="/api/v1/chunking", tags=["chunking"])
# app.include_router(embeddings.router, prefix="/api/v1/embeddings", tags=["embeddings"])
# app.include_router(vector_store.router, prefix="/api/v1/vector-store", tags=["vector-store"])
# app.include_router(query.router, prefix="/api/v1/query", tags=['query'])

# # from app.api.routes import auth, documents, query, upload
# # app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
# # app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
# # app.include_router(query.router, prefix="/api/v1", tags=["query"])
# # app.include_router(auth.router, prefix="/api/v1", tags=["auth"])
