"""
Custom middleware for the application.
"""
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from loguru import logger

from app.utils.exceptions import RAGException


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} | "
            f"ID: {request_id} | "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )

        # Time the request
        start_time = time.time()

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                f"Response: {response.status_code} | "
                f"ID: {request_id} | "
                f"Duration: {duration:.3f}s"
            )

            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{duration:.3f}"

            return response

        except Exception as e:
            duration = time.time() - start_time

            logger.error(
                f"Request failed: {request.method} {request.url.path} | "
                f"ID: {request_id} | "
                f"Duration: {duration:.3f}s | "
                f"Error: {str(e)}"
            )

            raise


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Handle exceptions globally."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)

        except RAGException as e:
            # Our custom exceptions
            logger.warning(f"RAG Exception: {e.message} | Details: {e.details}")

            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.message,
                    "details": e.details,
                    "request_id": getattr(request.state, "request_id", None)
                }
            )

        except Exception as e:
            # Unexpected exceptions
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(e) if logger.level == "DEBUG" else "An unexpected error occurred",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Global rate limiting."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = request.client.host if request.client else "unknown"

        # Check rate limit
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > minute_ago
            ]
        else:
            self.requests[client_id] = []

        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_id}")

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute"
                }
            )

        # Record request
        self.requests[client_id].append(now)

        return await call_next(request)
