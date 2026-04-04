"""
Authentication utilities and middleware.
"""
from typing import Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import time

from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from loguru import logger

from app.services.supabase_service import get_supabase_service, SupabaseService


# HTTP Bearer token scheme
security = HTTPBearer()


# Rate limiting storage
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.limits = {
            "query": {"requests": 20, "window": 3600},  # 20 queries per hour
            "upload": {"requests": 10, "window": 3600},  # 10 uploads per hour
            "embedding": {"requests": 100, "window": 3600},  # 100 embeddings per hour
            "default": {"requests": 100, "window": 3600}  # 100 requests per hour
        }
    
    def is_allowed(self,user_id: str,endpoint_type: str = "default") -> Tuple[bool, Optional[str]]:
        """
        Check if request is allowed under rate limits.
        
        Args:
            user_id: User identifier
            endpoint_type: Type of endpoint (query, upload, embedding, default)
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        now = time.time()
        key = f"{user_id}:{endpoint_type}"
        
        # Get limit config
        limit_config = self.limits.get(endpoint_type, self.limits["default"])
        max_requests = limit_config["requests"]
        window_seconds = limit_config["window"]
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < window_seconds
        ]
        
        # Check limit
        if len(self.requests[key]) >= max_requests:
            window_minutes = window_seconds // 60
            return False, f"Rate limit exceeded: {max_requests} requests per {window_minutes} minutes"
        
        # Record request
        self.requests[key].append(now)
        return True, None
    
    def get_usage(self, user_id: str, endpoint_type: str = "default") -> dict:
        """
        Get current usage stats for a user.
        
        Args:
            user_id: User identifier
            endpoint_type: Type of endpoint
            
        Returns:
            Usage statistics
        """
        now = time.time()
        key = f"{user_id}:{endpoint_type}"
        
        limit_config = self.limits.get(endpoint_type, self.limits["default"])
        window_seconds = limit_config["window"]
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < window_seconds
        ]
        
        current_requests = len(self.requests[key])
        max_requests = limit_config["requests"]
        
        return {
            "current_requests": current_requests,
            "max_requests": max_requests,
            "remaining": max_requests - current_requests,
            "window_seconds": window_seconds,
            "reset_at": datetime.fromtimestamp(
                now + window_seconds
            ).isoformat() if self.requests[key] else None
        }


# Global rate limiter
rate_limiter = RateLimiter()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: SupabaseService = Depends(get_supabase_service)) -> str:
    """
    Get current authenticated user ID from JWT token.
    
    Args:
        credentials: HTTP authorization credentials
        supabase: Supabase service
        
    Returns:
        User ID
        
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    
    try:
        # Verify token with Supabase
        is_valid, user_id, error = supabase.verify_token(token)
        
        if not is_valid or not user_id:
            logger.warning(f"Invalid token: {error}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        logger.debug(f"Authenticated user: {user_id}")
        return user_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_optional_user(request: Request, supabase: SupabaseService = Depends(get_supabase_service)) -> Optional[str]:
    """
    Get current user ID if authenticated, None otherwise.
    
    Args:
        request: FastAPI request
        supabase: Supabase service
        
    Returns:
        User ID or None
    """
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.replace("Bearer ", "")
    
    try:
        is_valid, user_id, _ = supabase.verify_token(token)
        return user_id if is_valid else None
    except Exception:
        return None


async def check_rate_limit(user_id: str = Depends(get_current_user), endpoint_type: str = "default") -> str:
    """
    Check rate limit for user.
    
    Args:
        user_id: User identifier
        endpoint_type: Type of endpoint
        
    Returns:
        User ID if allowed
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    is_allowed, error = rate_limiter.is_allowed(user_id, endpoint_type)
    
    if not is_allowed:
        logger.warning(f"Rate limit exceeded for user {user_id}: {endpoint_type}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=error
        )
    
    return user_id

def require_auth(func):
    """Decorator to require authentication."""
    async def wrapper(*args, user_id: str = Depends(get_current_user), **kwargs):
        return await func(*args, user_id=user_id, **kwargs)
    return wrapper